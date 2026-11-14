"""Ranking test R (docs/hypotheses-reproduce.md, 2026-11-14 section "T1 withdrawn; T2 replaced by ranking test R"):
what would v4b gain if one kind of decision were made the way Balatro Lab's bot makes it?

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e): its seeded
Run (through twin.py) and balatro.strategy.ValueBot. Nothing of Balatro Lab is copied here.

Arms: A0 v4b alone; A1 hand play (play, discard); A2 shop; A3 pack picks; A4 blind select/skip; A5 all of them.
v4b plays as in its evals (train/rollout.py play_games, mode "eval": its phase networks, offer values, sampling
from random.Random("eval-<seed>")). At a handed-over decision Balatro Lab's bot chooses on a copy of the twin's
state (for hand play after the twin is lost: on a Run built from the game state, state.py), and its choice is
played in the game as the matching harness move (then passed to the twin from the
harness move, as for v4b's moves). Its reorderings of jokers or hand cards are not carried over (the game's order
stays; the twin is put in the game's order before each decision). If its choice has no matching legal move (the
action set offers it differently), v4b makes that move: counted as `unmapped`. If the twin can't follow the game,
v4b plays the rest of that game: counted as `lost_twin` with the move number.

    python3 rank.py --weights W.pt --lab <balatro-lab> --src <source> --arms A0,A1,A2,A3,A4,A5 --seeds 300 \
                    --workers 8 --out rank.jsonl
    python3 rank.py --report rank.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import random
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
import mine as M  # noqa: E402  (twin, harness and train paths)
import twin as TW  # noqa: E402

ARMS = {"A0": set(), "A1": {"round"}, "A2": {"shop"}, "A3": {"pack"}, "A4": {"blind"},
        "A5": {"round", "shop", "pack", "blind"}}
PACK_STATES = ("TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK")


def kind_now(G, state_name):
    if state_name == "SELECTING_HAND":
        return "round"
    if state_name == "SHOP":
        return "shop"
    if state_name in PACK_STATES:
        return "pack"
    if state_name == "BLIND_SELECT" and str(G.GAME.blind_on_deck) != "Boss":
        return "blind"
    return None


def lab_choice(bot, kind, run, extra):
    """Balatro Lab's bot's next move here (reorders skipped), on a copy, as (action, copy run, copy extra)."""
    agent = run.agent
    run.agent = None
    try:
        r, x = copy.deepcopy((run, extra))
    finally:
        run.agent = agent
    r.agent = bot
    if kind == "blind":
        blind, tag = x
        return ("skip_blind",) if bot.skip_blind(r, blind, tag) else ("select_blind",), r, x
    decide = {"round": bot.round_action, "shop": bot.shop_action, "pack": bot.pack_action}[kind]
    for _ in range(30):
        a = decide(r) if kind == "round" else decide(r, x)
        if a and a[0] in ("order_jokers", "order_hand", "sort_hand"):
            r.free_action(a)
            continue
        return a, r, x
    return None, r, x


def to_desc(G, a, r, x, descs):
    """Balatro Lab's move -> index of the matching legal harness move, or None."""
    k = a[0] if a else None
    hand_sigs = [TW.g_card(c) for c in M.lua_list(G.hand.cards)]

    def positions(idx):
        want = [TW.l_card(r.hand[i]) for i in idx]
        pos, free = [], list(range(len(hand_sigs)))
        for s in want:
            p = next((p for p in free if hand_sigs[p] == s), None)
            if p is None:
                return None
            free.remove(p)
            pos.append(p + 1)
        return pos

    def find(kind, i=None, cards=None, lead=None):
        best = None
        for n, d in enumerate(descs):
            dk, di, dj, dc, dl = M.parse_desc(d)
            if dk != kind or (i is not None and di != i):
                continue
            if cards is not None and sorted(dc) != sorted(cards):
                continue
            if best is None or (lead is not None and dl == lead):
                best = n
        return best

    if k in ("skip_blind", "select_blind"):
        return find(k)
    if k in ("play", "discard"):
        idx = r.selection(a[1])
        pos = positions(idx)
        if pos is None:
            return None
        return find(k, cards=pos, lead=pos[0] if k == "play" else None)
    if k == "buy":
        where, i = (a[1], a[2]) if len(a) >= 3 else ("card", a[1])
        return find({"card": "buy", "voucher": "redeem", "pack": "open_pack"}[where], i=i + 1)
    tg = lambda: positions(a[2]) if len(a) > 2 and a[2] else None  # noqa: E731
    if k == "buy_use":
        return find("buy_and_use", i=a[1] + 1, cards=tg())
    if k == "pick":
        return find("pick", i=a[1] + 1, cards=tg())
    if k == "use":
        return find("use_consumable", i=a[1] + 1, cards=tg())
    if k in ("sell_joker", "sell_consumable"):
        return find(k, i=a[1] + 1)
    return find({"reroll": "reroll", "leave": "end_shop", "skip": "skip_pack"}.get(k, "?"))


def play_arm(sim, model, collate, seed, arm, bot_factory, args):
    import torch
    delegated = ARMS[arm]
    o = sim.reset(seed, "b_red", 1)
    G = sim.env.lua.globals().G
    if not TW.LOC:
        L = G.localization.descriptions
        for s in ("Joker", "Tarot", "Planet", "Spectral", "Voucher", "Tag", "Blind"):
            TW.LOC[s] = L[s]
    srng = random.Random(f"eval-{seed}")
    tw = TW.Twin(seed) if delegated else None
    bot = bot_factory() if delegated else None
    held = None                  # the twin's open question: (kind, run, extra), answered by the next real move
    stats = {"theirs": 0, "theirs_no_twin": 0, "unmapped": 0, "lost_twin": None, "resyncs": 0, "mirrored": {}}

    def pull():
        nonlocal held
        if held is None:
            held = tw.next()
        return held

    def check(kind):
        got, run, extra = pull()
        if got != kind:
            raise RuntimeError(f"twin at {got}, game at {kind}")
        M.order_jokers_like_game(G, run)
        if got == "pack" and M.lua_list(G.hand.cards) and getattr(extra.pack, "kind", "") in ("Arcana", "Spectral"):
            TW.resync_hand(G, run)
        d = TW.diff(TW.game_view(G, got), TW.lab_view(run, got, extra))
        if d and got == "pack" and set(d) == {"pack"}:
            M.resync_pack(G, run, extra)
            stats["resyncs"] += 1
        if d and got == "shop" and set(d) == {"cards"} and M.resync_shop_cards(G, run, extra):
            stats["resyncs"] += 1
        if d and got == "round" and set(d) <= {"hand", "next"} and run.blind.boss is not None \
                and run.blind.boss.name == "The Hook" and TW.resync_hand(G, run):
            stats["resyncs"] += 1
        d = TW.diff(TW.game_view(G, got), TW.lab_view(run, got, extra))
        if d:
            mirror(G, run, d)
            M.order_jokers_like_game(G, run)
            d = TW.diff(TW.game_view(G, got), TW.lab_view(run, got, extra))
        if d:
            raise RuntimeError("diverged: " + json.dumps(d, default=str)[:200])
        if got == "round":
            M.order_hand_like_game(G, run)
        return run, extra

    def mirror(G, run, d):
        """Chance outcomes the twin rolled its own way: copy the game's (the game is the truth here)."""
        from balatro.shop import CONSUMABLES
        if "score" in d:
            run.round_score = int(G.GAME.chips)
        if "money" in d:
            run.money = int(G.GAME.dollars)
        if "levels" in d:
            for k, h in G.GAME.hands.items():
                run.hand_levels[str(k)] = int(h.level)
        if "jokers" in d or "joker_state" in d:
            game = [(TW.g_name(c), TW.edition_of(c), c) for c in M.lua_list(G.jokers.cards)]
            if sorted(n for n, _, _ in game) == sorted(j.name for j in run.jokers):
                free = list(run.jokers)
                for n, ed, c in game:
                    j = next(j for j in free if j.name == n)
                    free.remove(j)
                    j.edition = ed
                    for k, v in TW.joker_state(n, c.ability, G).items():
                        setattr(j, k, v)
        if "consumables" in d:
            run.consumables = [CONSUMABLES[TW.g_name(c)]() for c in M.lua_list(G.consumeables.cards)
                               if TW.g_name(c) in CONSUMABLES]
        if "hand" in d or "next" in d:
            TW.resync_hand(G, run)
        for k in d:
            stats["mirrored"][k] = stats["mirrored"].get(k, 0) + 1

    done = False
    while not done and sim.steps < args.max_steps:
        st = sim.H.state_name()
        kind = kind_now(G, st)
        descs = [sim.H.action_desc(i + 1) for i in range(len(o["act"]))]
        a = None
        if tw is not None and kind is not None:
            try:
                run, extra = check(kind)
                if kind in delegated:
                    act, r, x = lab_choice(bot, kind, run, extra)
                    a = to_desc(G, act, r, x, descs) if act is not None else None
                    stats["unmapped" if a is None else "theirs"] += 1
            except (RuntimeError, queue.Empty) as e:
                stats["lost_twin"] = {"move": sim.steps, "why": str(e)[:200]}
                tw = None
        if a is None and tw is None and kind == "round" and "round" in delegated:
            try:                                            # twin lost: their bot on the game state itself
                run0 = TW.lab_run(G)
                run0.agent = bot
                act, r, x = lab_choice(bot, "round", run0, None)
                a = to_desc(G, act, r, x, descs) if act is not None else None
                stats["unmapped" if a is None else "theirs_no_twin"] += 1
            except Exception as e:
                stats["no_twin_errors"] = stats.get("no_twin_errors", 0) + 1
                stats["no_twin_last"] = str(e)[:150]
                a = None
        if a is None:                                       # v4b, exactly as its evals sample
            o["member"] = M.phase_member(st)
            with torch.no_grad():
                logits, _ = model(*collate([o]))
            p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
            a = min(int(np.searchsorted(np.cumsum(p), srng.random() * p.sum())), len(p) - 1)
        if tw is not None:                                  # the twin makes the same move
            dk, di, dj, dc, dl = M.parse_desc(descs[a])
            try:
                if dk in M.FREE:                            # sell / use: straight onto the twin's run
                    got, run, extra = pull()
                    M.order_jokers_like_game(G, run)
                    if got == "round":
                        M.order_hand_like_game(G, run)
                    try:
                        act_t = M.twin_action(G, run, dk, di, dc, dl)
                    except StopIteration:
                        TW.resync_hand(G, run)
                        act_t = M.twin_action(G, run, dk, di, dc, dl)
                    run.free_action(act_t)
                elif dk in M.REAL_KIND or (dk in ("skip_blind", "select_blind") and kind == "blind"):
                    got, run, extra = pull()
                    held = None
                    if got == "blind":
                        tw.answer(dk == "skip_blind")
                    else:
                        try:
                            act_t = M.twin_action(G, run, dk, di, dc, dl)
                        except StopIteration:
                            TW.resync_hand(G, run)
                            act_t = M.twin_action(G, run, dk, di, dc, dl)
                        tw.answer(act_t)
            except (StopIteration, queue.Empty, RuntimeError) as e:
                stats["lost_twin"] = {"move": sim.steps, "why": f"passing {descs[a]}: {str(e)[:150]}"}
                tw = None
        _, prog, done = sim.step(a)
        if not done:
            o = sim.observe()
    prog = sim.progress
    return {"seed": seed, "arm": arm, "won": bool(prog["won"]), "blinds": int(prog["blinds_beaten"]),
            "ante": int(prog["ante"]), "moves": sim.steps, **stats}


def worker(a):
    import torch
    from common import collate
    from model import phase_member
    M.phase_member = phase_member
    os.environ.setdefault("VALUEBOT_WEIGHTS", "hand")
    TW.TIMEOUT = 3600
    TW.load_lab(a.lab)
    from balatro.strategy import ValueBot
    sim, _route = M.load_ours(a.weights, a.src)
    sim.build_phi = False                                   # as in evals (rollout.py)
    w = torch.load(a.weights, map_location="cpu", weights_only=True)
    from model import make_model
    cfg = dict(w["trained"]["provenance"]["config"])
    model = make_model(argparse.Namespace(**cfg), len(w["vocab"]))
    model.load_state_dict(w["state_dict"])
    model.eval()
    from common import eval_seeds
    seeds = eval_seeds(600)[a.first:a.first + a.seeds]
    arms = a.arms.split(",")
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["seed"], json.loads(l)["arm"]) for l in open(a.out)}
    with open(a.out, "a") as out:
        for seed in seeds:
            for arm in arms:
                if (seed, arm) in done:
                    continue
                t = time.time()
                try:
                    rec = play_arm(sim, model, collate, seed, arm, ValueBot, a)
                except Exception:
                    rec = {"seed": seed, "arm": arm, "error": traceback.format_exc(limit=4)}
                rec["seconds"] = round(time.time() - t, 1)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(json.dumps(rec)[:240], flush=True)


def report(path):
    rows = [json.loads(l) for l in open(path) if "error" not in json.loads(l)]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    base = by.get("A0", {})
    rng = np.random.default_rng(0)
    print(f"{'arm':<4} {'games':>5} {'wins':>5} {'blinds/g':>9}  {'vs A0 (paired) blinds':>28}  {'wins':>6}  "
          f"{'theirs/g':>8} {'unmapped/g':>10} {'lost twin':>9}")
    for arm in sorted(by):
        g = by[arm]
        n, w = len(g), sum(r["won"] for r in g.values())
        b = np.mean([r["blinds"] for r in g.values()])
        common = sorted(set(g) & set(base))
        line = f"{arm:<4} {n:>5} {w:>5} {b:>9.2f}"
        if arm != "A0" and common:
            diff = np.array([g[s]["blinds"] - base[s]["blinds"] for s in common], float)
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            dw = sum(g[s]["won"] for s in common) - sum(base[s]["won"] for s in common)
            line += f"  {diff.mean():+7.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}] n={len(common):<4}  {dw:+6d}"
        else:
            line += " " * 44
        line += f"  {np.mean([r['theirs'] for r in g.values()]):8.1f} {np.mean([r['unmapped'] for r in g.values()]):10.1f}" \
                f" {sum(bool(r['lost_twin']) for r in g.values()):9d}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--lab")
    ap.add_argument("--src")
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4,A5")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--out", default="rank.jsonl")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return 0
    if a.workers > 1:
        per = -(-a.seeds // a.workers)
        procs = []
        for k in range(a.workers):
            n = min(per, a.seeds - k * per)
            if n <= 0:
                break
            procs.append(subprocess.Popen([sys.executable, __file__, "--weights", a.weights, "--lab", a.lab,
                                           "--src", a.src, "--arms", a.arms, "--seeds", str(n),
                                           "--first", str(a.first + k * per), "--max-steps", str(a.max_steps),
                                           "--out", a.out.replace(".jsonl", f"_w{k}.jsonl")]))
        for p in procs:
            p.wait()
        with open(a.out, "a") as out:
            for k in range(len(procs)):
                f = a.out.replace(".jsonl", f"_w{k}.jsonl")
                if os.path.exists(f):
                    out.write(open(f).read())
                    os.remove(f)
        report(a.out)
        return 0
    worker(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
