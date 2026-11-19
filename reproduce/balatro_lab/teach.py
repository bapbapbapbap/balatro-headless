"""Model v6's teacher data (docs/hypotheses-v6.md, stages 0-1): Balatro Lab's bot plays our game through the bridge,
and every decision is written down as the student sees it.

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e): its value bot
(balatro.strategy.ValueBot, the bot behind its 70.8% reproduction) through bridge.py, and, with --check, its seeded
Run through twin.py. Nothing of Balatro Lab is copied here; every recorded move is its bot's, credited as such in
each game's record ("teacher").

For each seed (fresh seeds from random.Random("v6-teach"), never one of the 600 eval seeds):
- The game runs in the harness with model v5's observation (configs/v5.toml: ordered-capped actions, encoding 2,
  offer values, play values). At each decision the bridge builds Balatro Lab's state from the game, the teacher
  chooses, and its move is played as the matching legal harness move.
- Written per decision: the observation (as the trainer's collate reads it), the phase network that decides it,
  the teacher's move (index into the legal moves) and the progress after it (for returns under our reward).
- When the teacher's move has no legal match (`unmapped`), a fallback move is played (cash out, select the blind,
  end the shop, skip the pack, else the first legal move) and that decision is marked unlabeled.
- --check (B1 in the cloud, and the bridge's own test): Balatro Lab's seeded twin follows the game as in R; while
  it is in sync, the teacher also decides on the twin's state, and the two decisions are compared (`agree`,
  `disagree` with both moves). The bridge's state is what is played either way.

    python3 teach.py --lab <balatro-lab> --src <source> --games 600 --workers 12 --out <dir> [--check]
    python3 teach.py --report <dir>
    --leak (B2): every hand decision is asked again with the draw pile shuffled; an honest teacher answers the same.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import queue
import random
import subprocess
import sys
import time
import traceback
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
import twin as TW  # noqa: E402  (harness paths, the LuaJIT check, state.py)
import bridge as B  # noqa: E402
from state import lua_list  # noqa: E402

CHARS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
FALLBACK = ("cash_out", "select_blind", "end_shop", "skip_pack")


def teach_seeds(n, first=0):
    """Fresh seeds for teacher games: a fixed generator, skipping the 600 eval seeds."""
    from common import eval_seeds
    held = set(eval_seeds(600))
    rng, out = random.Random("v6-teach"), []
    while len(out) < first + n:
        s = "".join(rng.choice(CHARS) for _ in range(8))
        if s not in held:
            out.append(s)
    return out[first:]


def seeds_from(path):
    return [l.strip() for l in open(path) if l.strip()]


def make_sim(src):
    from sim import Sim
    return Sim(src, fast=True, mode="ordered-capped", enc=2, build_phi=True, offer_values=True, play_values=True)


def to_desc(G, a, run, descs):
    """The teacher's move -> index of the matching legal harness move (rank.to_desc, plus the moves R never
    handed over: cash out, reroll the boss)."""
    import rank
    if a and a[0] in ("cash_out", "reroll_boss"):
        return next((n for n, d in enumerate(descs) if d.split()[0] == a[0]), None)
    if a and a[0] == "move_joker":
        want = f"move_joker {a[1]}>{a[2]}"
        return next((n for n, d in enumerate(descs) if d.split()[:2] == want.split()), None)
    if a and a[0] == "use" and len(a) > 2 and a[2]:
        # F1b (docs/hypotheses-v6.md, 2026-11-19): the game always selects Cerulean Bell's forced card, so a targeting
        # consumable's legal targets include it; Balatro Lab applies the Bell to plays and discards only. Put the
        # forced card in place of the last target, as its Run.selection does for plays.
        boss = run.blind.boss if run.blind is not None else None
        forced = getattr(boss, "forced", None) if boss is not None and not boss.disabled else None
        fi = next((i for i, c in enumerate(run.hand) if c is forced), None) if forced is not None else None
        if fi is not None and fi not in a[2]:
            a = (a[0], a[1], sorted(list(a[2])[:-1] + [fi]))
    return rank.to_desc(G, a, run, None, descs)


class Student:
    """Stage 2 (DAgger, docs/hypotheses-v6.md 2026-11-19): a v6 checkpoint that plays the moves while the teacher only
    labels them. Samples at temperature 1 like the trainer's evals, with a fixed RNG per seed."""

    def __init__(self, path, sim):
        import torch
        import types
        sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
        from model import make_model
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["vocab"] != sim.vocab:
            raise SystemExit(f"{path}: trained on a different game vocabulary")
        self.model = make_model(types.SimpleNamespace(**ck["arch"]), len(ck["vocab"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        torch.set_num_threads(1)
        self.path, self.iteration = path, ck.get("iteration")

    def reset(self, seed):
        self.rng = random.Random(f"dagger-{seed}")

    def move(self, o, member):
        import numpy as np
        import torch
        from common import collate
        o = dict(o, member=member)
        with torch.no_grad():
            logits, _ = self.model(*collate([o]))
        p = torch.softmax(logits[0].double(), -1).numpy()
        return min(int(np.searchsorted(np.cumsum(p), self.rng.random() * p.sum())), len(p) - 1)


def play_game(sim, seed, bot_factory, check=False, max_steps=1500, leak=False, student=None):
    import mine as M
    from model import phase_member
    o = sim.reset(seed, "b_red", 1)
    G = sim.env.lua.globals().G
    if not TW.LOC:
        L = G.localization.descriptions
        for s in ("Joker", "Tarot", "Planet", "Spectral", "Voucher", "Tag", "Blind"):
            TW.LOC[s] = L[s]
    sh = B.Shadow(seed)
    bot = bot_factory()
    tw = TW.Twin(seed) if check else None
    tbot = bot_factory() if check else None
    held = None
    if student is not None:
        student.reset(seed)
    st = {"unmapped": Counter(), "reorders": Counter(), "agree": 0, "disagree": [], "lost_twin": None,
          "no_kind": Counter(), "errors": Counter()}
    decisions = []
    done = False
    while not done and sim.steps < max_steps:
        sname = sim.H.state_name()
        kind = B.decision_kind(G, sname)
        descs = [sim.H.action_desc(i + 1) for i in range(len(o["act"]))]
        a, labeled, act = None, False, None
        if kind is None:
            st["no_kind"][sname] += 1
        else:
            sh.before(G, kind)
            try:
                run, extra = sh.state(G, kind, sname)
                drags = any(d.startswith("move_joker") for d in descs)
                act = B.teacher_move(bot, kind, run, extra, st["reorders"], drags)
                a = to_desc(G, act, run, descs) if act is not None else None
                labeled = a is not None
                if leak and kind == "round" and act is not None and act[0] != "move_joker" and len(run.draw_pile) > 1:
                    # B2: the same decision with the unseen cards in another order must not change
                    run2, extra2 = sh.state(G, kind, sname)
                    random.Random(f"leak:{seed}:{sim.steps}").shuffle(run2.draw_pile)
                    act2 = B.teacher_move(bot_factory(), kind, run2, extra2, Counter(), drags)
                    a2 = to_desc(G, act2, run2, descs) if act2 is not None else None
                    st["leak_same" if a2 == a else "leak_diff"] = st.get("leak_same" if a2 == a else "leak_diff", 0) + 1
                if a is None:
                    st["unmapped"][f"{kind}:{act[0] if act else None}"] += 1
            except Exception as e:
                st["errors"][f"{kind}: {type(e).__name__}: {str(e)[:80]}"] += 1
        # --check: the same decision on the twin's state, while the twin is in sync
        if tw is not None and kind in ("round", "shop", "pack", "blind"):
            try:
                if held is None:
                    held = tw.next()
                got, trun, textra = held
                if got != kind:
                    raise RuntimeError(f"twin at {got}, game at {kind}")
                M.order_jokers_like_game(G, trun)
                if got == "round":
                    M.order_hand_like_game(G, trun)
                if got == "pack" and lua_list(G.hand.cards) and getattr(textra.pack, "kind", "") in ("Arcana", "Spectral"):
                    TW.resync_hand(G, trun)          # its pack hand is dealt by its own randomness (as rank.py)
                d = TW.diff(TW.game_view(G, got), TW.lab_view(trun, got, textra))
                if d:
                    raise RuntimeError("diverged: " + json.dumps(d, default=str)[:200])
                import rank
                tact, r, x = rank.lab_choice(tbot, kind, trun, textra)
                ta = rank.to_desc(G, tact, r, x, descs) if tact is not None else None
                if act is not None and act[0] == "move_joker":
                    st["drag_not_compared"] = st.get("drag_not_compared", 0) + 1
                elif ta == a:
                    st["agree"] += 1
                else:
                    st["disagree"].append({"move": sim.steps, "kind": kind, "bridge": descs[a] if a is not None else None,
                                           "twin": descs[ta] if ta is not None else None})
            except (RuntimeError, queue.Empty) as e:
                st["lost_twin"] = {"move": sim.steps, "why": str(e)[:200]}
                tw = None
        label = a if labeled else None
        if student is not None:
            a = student.move(o, phase_member(sname))       # DAgger: the student plays; the teacher's move is the label
        elif a is None:
            a = next((n for n, d in enumerate(descs) if d.split()[0] in FALLBACK), 0)
        dk, di, dj, dc, dl = M.parse_desc(descs[a])
        sh.after(dk, len(dc))
        # the twin makes the same move (rank.play_arm's passing, unchanged)
        if tw is not None:
            try:
                if dk in M.FREE:
                    got, trun, textra = held if held is not None else tw.next()
                    held = (got, trun, textra)
                    M.order_jokers_like_game(G, trun)
                    if got == "round":
                        M.order_hand_like_game(G, trun)
                    try:
                        act_t = M.twin_action(G, trun, dk, di, dc, dl)
                    except StopIteration:
                        TW.resync_hand(G, trun)
                        act_t = M.twin_action(G, trun, dk, di, dc, dl)
                    trun.free_action(act_t)
                elif dk in M.REAL_KIND or (dk in ("skip_blind", "select_blind") and kind == "blind"):
                    got, trun, textra = held if held is not None else tw.next()
                    held = None
                    if got == "blind":
                        tw.answer(dk == "skip_blind")
                    else:
                        try:
                            act_t = M.twin_action(G, trun, dk, di, dc, dl)
                        except StopIteration:
                            TW.resync_hand(G, trun)
                            act_t = M.twin_action(G, trun, dk, di, dc, dl)
                        tw.answer(act_t)
            except (StopIteration, queue.Empty, RuntimeError) as e:
                st["lost_twin"] = {"move": sim.steps, "why": f"passing {descs[a]}: {str(e)[:150]}"}
                tw = None
        # "a" is what imitation trains on: the teacher's move when labeled. "played" is the move made (the
        # student's in DAgger, else the teacher's or the fallback), which the game's progress follows.
        rec = {"obs": o, "member": phase_member(sname), "a": int(label if labeled else a), "labeled": labeled,
               "kind": kind, "desc": descs[a], "played": int(a)}
        _, prog, done = sim.step(a)
        rec["progress"] = prog
        decisions.append(rec)
        if not done:
            o = sim.observe()
    prog = sim.progress
    out = {"seed": seed, "teacher": "Balatro Lab ValueBot (KvashaIhor/balatro-lab 3e09d47, VALUEBOT_WEIGHTS=hand)",
           "won": bool(prog["won"]), "blinds": int(prog["blinds_beaten"]), "ante": int(prog["ante"]),
           "moves": sim.steps, "labeled": sum(d["labeled"] for d in decisions),
           "unmapped": dict(st["unmapped"]), "reorders": dict(st["reorders"]), "no_kind": dict(st["no_kind"]),
           "errors": dict(st["errors"]), "bridge_unmapped": dict(sh.unmapped)}
    if student is not None:
        out.update(student=student.path, student_iteration=student.iteration,
                   played_teacher=sum(1 for d in decisions if d["labeled"] and d["a"] == d["played"]))
    if leak:
        out.update(leak_same=st.get("leak_same", 0), leak_diff=st.get("leak_diff", 0))
    if check:
        out.update(agree=st["agree"], disagree=st["disagree"], lost_twin=st["lost_twin"],
                   drag_not_compared=st.get("drag_not_compared", 0))
    return out, decisions


def worker(a):
    os.environ.setdefault("VALUEBOT_WEIGHTS", "hand")
    TW.TIMEOUT = 3600
    TW.load_lab(a.lab)
    from balatro.strategy import ValueBot
    sim = make_sim(a.src)
    student = Student(a.student, sim) if a.student else None
    os.makedirs(os.path.join(a.out, "games"), exist_ok=True)
    log = os.path.join(a.out, f"games_w{a.first}.jsonl")
    done = set()
    for f in os.listdir(a.out):
        if f.startswith("games_w") and f.endswith(".jsonl"):
            done |= {json.loads(l)["seed"] for l in open(os.path.join(a.out, f))}
    with open(log, "a") as out:
        seeds = seeds_from(a.seed_file)[a.first:a.first + a.games] if a.seed_file else teach_seeds(a.games, a.first)
        for seed in seeds:
            if seed in done:
                continue
            t = time.time()
            try:
                rec, decisions = play_game(sim, seed, ValueBot, a.check, a.max_steps, a.leak, student)
                with gzip.open(os.path.join(a.out, "games", f"{seed}.pkl.gz"), "wb") as f:
                    pickle.dump(decisions, f, protocol=4)
            except Exception:
                rec = {"seed": seed, "error": traceback.format_exc(limit=5)}
            rec["seconds"] = round(time.time() - t, 1)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps({k: rec.get(k) for k in ("seed", "won", "blinds", "ante", "moves", "labeled", "unmapped",
                                                      "agree", "lost_twin", "seconds", "error")})[:300], flush=True)


def report(d):
    rows = []
    for f in sorted(os.listdir(d)):
        if f.startswith("games_w") and f.endswith(".jsonl"):
            rows += [json.loads(l) for l in open(os.path.join(d, f))]
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    if not n:
        print(f"{len(rows)} games, all errors" if rows else "no games yet")
        return
    w = sum(r["won"] for r in ok)
    moves = sum(r["moves"] for r in ok)
    lab = sum(r["labeled"] for r in ok)
    um, ro, nk, er, bu = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in ok:
        um.update(r["unmapped"]); ro.update(r["reorders"]); nk.update(r["no_kind"]); er.update(r["errors"])
        bu.update(r["bridge_unmapped"])
    se = (w / n * (1 - w / n) / n) ** 0.5
    print(f"games {n} (errors {len(rows) - n}); teacher wins {w} = {100 * w / n:.1f}% (+-{196 * se:.1f}); "
          f"blinds/game {sum(r['blinds'] for r in ok) / n:.2f}; mean s/game {sum(r['seconds'] for r in ok) / n:.0f}")
    print(f"decisions {moves}, labeled {lab} ({100 * lab / max(1, moves):.1f}%); unmapped/game "
          f"{sum(um.values()) / n:.2f}: {dict(um.most_common(8))}")
    print(f"reorders (joker ones played as drags): {dict(ro)}; states with no teacher decision: {dict(nk)}")
    if er:
        print(f"errors: {dict(er.most_common(8))}")
    if bu:
        print(f"bridge couldn't carry: {dict(bu.most_common(10))}")
    chk = [r for r in ok if "agree" in r]
    if chk:
        ag = sum(r["agree"] for r in chk)
        dis = sum(len(r["disagree"]) for r in chk)
        kinds = Counter(x["kind"] for r in chk for x in r["disagree"])
        print(f"check: bridge vs twin decisions agree {ag}/{ag + dis} ({100 * ag / max(1, ag + dis):.1f}%); "
              f"disagreements by kind {dict(kinds)}; twin lost in {sum(bool(r['lost_twin']) for r in chk)}/{len(chk)}")
    lk = [r for r in ok if "leak_same" in r]
    if lk:
        same, diff = sum(r["leak_same"] for r in lk), sum(r["leak_diff"] for r in lk)
        print(f"B2 leak: hand decisions unchanged with the draw pile shuffled {same}/{same + diff}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab")
    ap.add_argument("--src")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--out", default="teach")
    ap.add_argument("--check", action="store_true", help="compare with Balatro Lab's seeded twin while in sync")
    ap.add_argument("--leak", action="store_true", help="B2: every hand decision again with the draw pile shuffled")
    ap.add_argument("--report")
    ap.add_argument("--student", help="stage 2 (DAgger): a v6 checkpoint plays; the teacher labels every state it reaches")
    ap.add_argument("--seed-file", help="play these seeds (one per line) instead of the teacher seeds; --first/--games "
                                        "index into it (re-measuring chosen games, e.g. after a bridge fix)")
    a = ap.parse_args()
    if a.seed_file and a.workers > 1:
        a.games = min(a.games, len(seeds_from(a.seed_file)) - a.first)
    if a.report:
        report(a.report)
        return 0
    os.makedirs(a.out, exist_ok=True)
    if a.workers > 1:
        per = -(-a.games // a.workers)
        procs = []
        for k in range(a.workers):
            n = min(per, a.games - k * per)
            if n <= 0:
                break
            procs.append(subprocess.Popen([sys.executable, __file__, "--lab", a.lab, "--src", a.src, "--games", str(n),
                                           "--first", str(a.first + k * per), "--max-steps", str(a.max_steps),
                                           "--out", a.out] + (["--seed-file", a.seed_file] if a.seed_file else [])
                                          + (["--student", a.student] if a.student else [])
                                          + (["--check"] if a.check else []) + (["--leak"] if a.leak else [])))
        for p in procs:
            p.wait()
        report(a.out)
        return 0
    worker(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
