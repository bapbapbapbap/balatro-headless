"""AnteZero as teacher, stage 0: its bot plays fresh seeds in its simulator; every game is replayed in our harness and
written down as our student sees it (docs/hypotheses-reproduce.md, 2026-11-27 "AnteZero as teacher, stage 0").

**AnteZero is not ours.** namecallfilter's (https://github.com/namecallfilter/antezero, MIT), pinned at
4fe07f771104a811ca8452efd4831b0dbf097e6e; model antezero-v0.2.0.pt. None of its code is in this repository. From a
clone passed as --az this script calls `antezero_agent.runtime.load_model` and `make_env`, `antezero_agent.model.
obs_from_numpy`, `Net.act`; from `antezero_sim`: `Game`, `VecEnv.set_game / step / game / pop_episodes`,
`Game.factored_to_code / clone / resample_hidden / step / state_json`. Every recorded move is its bot's, credited as
such in each game's record ("teacher").

**Ours** (Claude's, reviewed by chud): this file.

    python3 teach_az.py record --az <clone> --ckpt antezero-v0.2.0.pt --per-cell 5 --stakes 1,8 --out <dir> [--leak 30]
    python3 teach_az.py label --src <game source> --dir <dir> --workers N
    python3 teach_az.py report --dir <dir>

record writes <dir>/teacher_games.jsonl (seed, deck, stake, the teacher's action codes, its result in its simulator,
the leak probe). label writes teach.py's format into <dir>: games_w0.jsonl and games/<seed>.pkl.gz, which
train/imitate.py reads unchanged. Both resume.
"""
import argparse, gzip, json, os, pickle, random, sys, time, traceback
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))
import lockstep as L  # noqa: E402

TEACHER = "AnteZero v0.2.0 (namecallfilter/antezero 4fe07f7, antezero-v0.2.0.pt sha256 ea28506d), greedy, no search"
Z3_SEEDS_FILE = os.path.join(HERE, "results", "2026-11-27", "z3_games.jsonl.gz")
PACK_STATES = L.PACK_STATES


def teach_seeds(n):
    """Fresh seeds: random.Random("az-teach"), skipping our 600 eval seeds and the 20 Z3 seeds; never repeated."""
    from common import eval_seeds
    held = set(eval_seeds(600))
    if os.path.exists(Z3_SEEDS_FILE):
        held |= {json.loads(l)["seed"] for l in gzip.open(Z3_SEEDS_FILE, "rt")}
    rng, out, seen = random.Random("az-teach"), [], set()
    while len(out) < n:
        s = "".join(rng.choice(L.SEED_CHARS) for _ in range(8))
        if s not in held and s not in seen:
            seen.add(s); out.append(s)
    return out


def jobs_for(per_cell, backs, stakes):
    cells = [(b, s) for b in backs for s in stakes]
    seeds = teach_seeds(per_cell * len(cells))
    return [(seeds[k * len(cells) + c],) + cells[c] for k in range(per_cell) for c in range(len(cells))]


# ---------------------------------------------------------------- record (the teacher plays in its simulator)

def record(a):
    sys.path.insert(0, os.path.join(a.az, "agent", "src"))
    import numpy as np
    import torch
    import antezero_sim as az
    from antezero_agent.model import obs_from_numpy
    from antezero_agent.runtime import load_model, make_env
    torch.set_num_threads(a.threads)
    model, cfg, _ = load_model(a.ckpt, "cpu")
    pc = cfg.env.play_candidates
    backs = a.backs.split(",") if a.backs else L.BACKS
    jobs = jobs_for(a.per_cell, backs, [int(s) for s in a.stakes.split(",")])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "teacher_games.jsonl")
    done = {json.loads(l)["seed"] for l in open(path)} if os.path.exists(path) else set()
    todo = [j for j in jobs if j[0] not in done]
    print(f"{len(todo)} games to record ({len(done)} done)", flush=True)
    kw = dict(endless=False, snapshot_prob=0.0, practice_prob=0.0, order_label_rate=0.0)

    def act(env):
        out = model.act(obs_from_numpy(env, "cpu"), deterministic=True)
        return (out["op"].to(torch.int32).cpu().numpy(), out["cards"].cpu().numpy().astype(np.uint32),
                out["dest"].to(torch.int32).cpu().numpy())

    probe = make_env(cfg, seed=0, num_envs=2, **kw) if a.leak else None
    n_leak = len(done)
    with open(path, "a") as f, torch.no_grad():
        for b0 in range(0, len(todo), a.batch):
            batch = todo[b0:b0 + a.batch]
            env = make_env(cfg, seed=0, num_envs=len(batch), **kw)
            starts = []
            for i, (seed, back, stake) in enumerate(batch):
                g = az.Game(seed, back, stake, False)
                env.set_game(i, g)
                starts.append(env.game(i))          # with the env's options (skip, sorts, rearranges)
            env.pop_episodes()
            n = len(batch)
            codes, first, steps, bad = [[] for _ in range(n)], {}, 0, [0] * n
            while len(first) < n and steps <= cfg.env.step_limit + 10:
                op, cards, dest = act(env)
                for i in range(n):
                    if i in first:
                        continue
                    c = env.game(i).factored_to_code(int(op[i]), int(cards[i]), int(dest[i]), pc)
                    if c is None:
                        bad[i] += 1
                    else:
                        codes[i].append(c)
                env.step(op, cards, dest)
                steps += 1
                for ep in env.pop_episodes():
                    first.setdefault(ep[0], ep)
            fails = 0
            for i, (seed, back, stake) in enumerate(batch):
                ep = first.get(i)
                h = starts[i].clone()
                ok = ep is not None and ep[1] == seed
                for c in codes[i]:
                    if h.is_terminal():
                        ok = False; break
                    h.step(c)
                ok = ok and h.is_terminal() and h.run_won == bool(ep[5])
                fails += not ok
                rec = {"seed": seed, "back": back, "stake": stake, "teacher": TEACHER, "codes": codes[i],
                       "won": bool(ep[5]) if ep else None, "ante": int(ep[4]) if ep else None,
                       "blinds": int(ep[6]) if ep else None, "truncated": bool(ep[9]) if ep else None,
                       "no_code": bad[i], "selfcheck": ok,
                       "options": {"allow_skip": starts[i].allow_skip, "null_sorts": starts[i].null_sorts,
                                   "max_rearranges": starts[i].max_rearranges}}
                if probe is not None and n_leak < a.leak:
                    rec["leak"] = leak_probe(starts[i], codes[i], probe, act, pc)
                    n_leak += 1
                f.write(json.dumps(rec) + "\n")
            f.flush()
            w = sum(1 for i in first if first[i][5])
            print(f"batch {b0 // a.batch}: {n} games, {w} won in sim, selfcheck fails {fails}", flush=True)


def leak_probe(start, codes, probe, act, pc):
    """L1: at every round decision, the teacher decides on a copy of the state and on a copy with its hidden state
    redrawn; both through the same 2-slot probe env, so slot history can't differ."""
    g = start.clone()
    same = diff = repro = 0
    examples = []
    for k, code in enumerate(codes):
        if g.is_terminal():
            break
        if g.phase == "round" and code.split()[0] in ("play", "discard", "use"):
            h0, h1 = g.clone(), g.clone()
            h1.resample_hidden(f"leak:{start.seed}:{k}")
            probe.set_game(0, h0)
            probe.set_game(1, h1)
            op, cards, dest = act(probe)
            c0 = probe.game(0).factored_to_code(int(op[0]), int(cards[0]), int(dest[0]), pc)
            c1 = probe.game(1).factored_to_code(int(op[1]), int(cards[1]), int(dest[1]), pc)
            repro += c0 == code
            if c0 == c1:
                same += 1
            else:
                diff += 1
                if len(examples) < 3:
                    examples.append([k, c0, c1])
        g.step(code)
    return {"same": same, "diff": diff, "probe_reproduced": repro, "examples": examples}


# ---------------------------------------------------------------- label (replayed in our harness, as the student sees it)

def parse_desc(desc):
    """Our action text ("play 1,3,5 lead 3", "use_consumable 1 2,3 src 3", "move_joker 1>3") -> parts."""
    t = desc.split()
    kind, i, j, cards, lead, src = t[0], None, None, (), None, None
    rest = t[1:]
    if kind in ("play", "discard"):
        if rest and rest[0] not in ("lead", "src"):
            cards = tuple(int(x) for x in rest[0].split(",")); rest = rest[1:]
    elif rest and rest[0] not in ("lead", "src"):
        x = rest[0]
        if ">" in x:
            i, j = (int(v) for v in x.split(">"))
        else:
            i = int(x)
        rest = rest[1:]
        if rest and rest[0] not in ("lead", "src"):
            cards = tuple(int(v) for v in rest[0].split(",")); rest = rest[1:]
    if "lead" in rest:
        lead = int(rest[rest.index("lead") + 1])
    if "src" in rest:
        src = int(rest[rest.index("src") + 1])
    return kind, i, j, cards, lead, src


def map_code(code, descs):
    """The teacher's code -> index of our legal move, or None. Where a lead card or Death's source is offered, the
    one that changes nothing: the leftmost lead, the rightmost source (the hand is already in the teacher's order)."""
    kind, i, j, cards = L.translate(code)
    cands = []
    for n, d in enumerate(descs):
        k2, i2, j2, c2, lead, src = parse_desc(d)
        if k2 == kind and (i is None or i2 == i) and (j is None or j2 == j) and tuple(c2) == tuple(cards):
            cands.append((n, lead, src))
    if not cands:
        return None, None
    leads = [c for c in cands if c[1] is not None]
    if leads:
        return min(leads, key=lambda c: c[1])[0], [c[1] for c in leads]
    srcs = [c for c in cands if c[2] is not None]
    if srcs:
        return max(srcs, key=lambda c: c[2])[0], None
    return cands[0][0], None


_SIM = None


def sim_for(src):
    global _SIM
    if _SIM is None:
        from sim import Sim
        _SIM = Sim(src, fast=True, mode="ordered-capped", enc=2, build_phi=True, offer_values=True, play_values=True)
        _SIM.env.lua.execute(open(os.path.join(HERE, "view.lua")).read())
    return _SIM


def kind_of(G, sname):
    if sname == "SELECTING_HAND": return "round"
    if sname == "SHOP": return "shop"
    if sname in PACK_STATES: return "pack"
    if sname == "BLIND_SELECT": return "boss" if str(G.GAME.blind_on_deck) == "Boss" else "blind"
    if sname == "ROUND_EVAL": return "cash"
    return None


def label_one(job):
    src, out_dir, rec = job
    import antezero_sim as az
    from model import phase_member
    t0 = time.time()
    seed, back, stake = rec["seed"], rec["back"], rec["stake"]
    out = {"seed": seed, "teacher": TEACHER, "back": back, "stake": stake}
    try:
        sim = sim_for(src)
        env, H = sim.env, sim.H
        G = env.lua.globals().G
        g = az.Game(seed, back, stake, False)
        o = rec["options"]
        g.allow_skip, g.null_sorts, g.max_rearranges = o["allow_skip"], o["null_sorts"], o["max_rearranges"]
        obs = sim.reset(seed, back, stake)
        sv, gv = L.sim_view(json.loads(g.state_json())), L.game_view(env)
        off = gv["deck"][0]["sort_id"] - sv["deck"][0]["sort_id"] if sv["deck"] and gv["deck"] else 0
        d = L.compare(sv, gv, off)
        decisions, st = [], Counter()
        pre_order = None             # hand order (sort_ids) before the reorders since the last decision
        s1 = Counter()
        for k, code in enumerate(rec["codes"]):
            if d:
                break
            op = code.split()[0]
            if op in ("sort", "move_hand"):
                if pre_order is None:
                    pre_order = [int(c.sort_id) for c in G.hand.cards.values()]
                kind, i, j, cards = L.translate(code)
                lt = env.lua.table_from({"type": kind})
                if i is not None: lt.i = i
                if j is not None: lt.j = j
                if H.step(lt) < 0:
                    raise RuntimeError(f"never settled after {code}")
                H._encoded_actions = None
                st["reorders"] += 1
                g.step(code)
            else:
                sname = H.state_name()
                obs = sim.observe()
                descs = [H.action_desc(n + 1) for n in range(len(obs["act"]))]
                idx, leads = map_code(code, descs)
                if op == "play" and pre_order is not None and leads and len(leads) > 1:
                    hand = [int(c.sort_id) for c in G.hand.cards.values()]
                    lead_ids = {hand[p - 1] for p in leads}          # the scoring cards (offered as leads)
                    now = hand[min(leads) - 1]
                    before = min(lead_ids, key=pre_order.index)
                    s1["changed" if now != before else "same"] += 1
                elif op == "play" and pre_order is not None:
                    s1["order_irrelevant"] += 1
                pre_order = None
                if idx is None:
                    st[f"unmapped:{op}"] += 1
                    L.env_apply(env, L.translate(code))
                    H._encoded_actions = None
                    g.step(code)
                else:
                    r = {"obs": obs, "member": phase_member(sname), "a": int(idx), "labeled": True,
                         "kind": kind_of(G, sname), "desc": descs[idx], "played": int(idx), "code": code}
                    _, _, _, cards, lead, srcp = parse_desc(descs[idx])
                    if (lead is not None and lead != cards[0]) or (srcp is not None and srcp != cards[-1]):
                        # No lead (or source) that changes nothing is offered: a kicker sits left of the first
                        # scoring card. The label is the move with the same scoring order; the game gets the
                        # teacher's exact move (same cards, nothing dragged), so it stays in lockstep.
                        kind2, i2, j2, c2 = L.translate(code)
                        exact = {"type": kind2, "cards": env.lua.table_from(list(c2))}
                        if i2 is not None:
                            exact["i"] = i2
                        H._encoded_actions[idx + 1] = env.lua.table_from(exact)
                        st["exact_substituted"] += 1
                        r["exact_substituted"] = True
                    _, prog, done = sim.step(idx)
                    r["progress"] = prog
                    decisions.append(r)
                    g.step(code)
            sv, gv = L.sim_view(json.loads(g.state_json())), L.game_view(env)
            d = L.compare(sv, gv, off)
            if d and k == len(rec["codes"]) - 1 and env.won and g.is_terminal() and G.SETTINGS.paused and G.OVERLAY_MENU:
                G.FUNCS.exit_overlay_menu(); H.settle()          # the win overlay, as in Z3
                sv, gv = L.sim_view(json.loads(g.state_json())), L.game_view(env)
                d = L.compare(sv, gv, off)
                out["overlay_dismissed"] = True
        if d:
            raise RuntimeError("lockstep diverged at code %d: %s" % (k, json.dumps([list(map(str, x))[:3] for x in d[:6]])))
        if not env.done:
            raise RuntimeError("codes ran out before the game ended")
        prog = sim._prog
        with gzip.open(os.path.join(out_dir, "games", f"{seed}.pkl.gz"), "wb") as f:
            pickle.dump(decisions, f, protocol=4)
        out.update(won=bool(env.won), sim_won=bool(rec["won"]), blinds=int(prog["blinds_beaten"]),
                   ante=int(prog["ante"]), moves=len(rec["codes"]), labeled=len(decisions),
                   decisions=len(decisions) + sum(v for k2, v in st.items() if k2.startswith("unmapped")),
                   reorders=st["reorders"], exact_substituted=st["exact_substituted"], unmapped={k2: v for k2, v in st.items() if k2.startswith("unmapped")},
                   s1=dict(s1))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out["tb"] = traceback.format_exc()[-1200:]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def label(a):
    recs = [json.loads(l) for l in open(os.path.join(a.dir, "teacher_games.jsonl"))]
    os.makedirs(os.path.join(a.dir, "games"), exist_ok=True)
    log = os.path.join(a.dir, "games_w0.jsonl")
    done = {json.loads(l)["seed"] for l in open(log)} if os.path.exists(log) else set()
    jobs = [(a.src, a.dir, r) for r in recs if r["seed"] not in done]
    print(f"{len(jobs)} games to label ({len(done)} done)", flush=True)
    with open(log, "a") as f, Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(label_one, jobs), 1):
            f.write(json.dumps({k: v for k, v in r.items() if k != "tb"}) + "\n"); f.flush()
            print(json.dumps({k: r.get(k) for k in ("seed", "back", "stake", "won", "sim_won", "labeled", "decisions",
                                                    "reorders", "unmapped", "s1", "seconds", "error")})[:400], flush=True)


# ---------------------------------------------------------------- report

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); m = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - m) / d, (c + m) / d)


def report(a):
    tg = [json.loads(l) for l in open(os.path.join(a.dir, "teacher_games.jsonl"))]
    lg_path = os.path.join(a.dir, "games_w0.jsonl")
    lg = {r["seed"]: r for r in map(json.loads, open(lg_path))} if os.path.exists(lg_path) else {}
    print(f"recorded {len(tg)} games, {len({r['seed'] for r in tg})} distinct seeds; "
          f"selfcheck fails {sum(not r['selfcheck'] for r in tg)}; undecodable actions in {sum(r['no_code'] > 0 for r in tg)}")
    for s in sorted({r["stake"] for r in tg}):
        rows = [r for r in tg if r["stake"] == s]
        k = sum(bool(r["won"]) for r in rows)
        lo, hi = wilson(k, len(rows))
        lab = [lg[r["seed"]] for r in rows if r["seed"] in lg]
        ok = [x for x in lab if not x.get("error")]
        kw = sum(bool(x["won"]) for x in ok)
        print(f"stake {s}: sim wins {k}/{len(rows)} = {100 * k / len(rows):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]; "
              f"labeled in lockstep {len(ok)}/{len(lab)}; wins on the game's code {kw}/{len(ok)}")
    ok = [x for x in lg.values() if not x.get("error")]
    if lg:
        dec = sum(x["decisions"] for x in ok); labd = sum(x["labeled"] for x in ok)
        print(f"M1: {len(ok)}/{len(lg)} games labeled to the end ({100 * len(ok) / len(lg):.1f}%)")
        print(f"M2: {labd}/{dec} decisions labeled ({100 * labd / max(1, dec):.2f}%); reorders applied "
              f"{sum(x['reorders'] for x in ok)}")
        um = Counter()
        for x in ok:
            um.update(x.get("unmapped") or {})
        print(f"   unmapped: {dict(um)}")
        s1 = Counter()
        for x in ok:
            s1.update(x.get("s1") or {})
        ch, sa = s1["changed"], s1["same"]
        print(f"S1: plays after a sort where order matters: {ch + sa}; first scoring card changed by the sort: {ch} "
              f"({100 * ch / max(1, ch + sa):.1f}%); plays after a sort where order doesn't matter: {s1['order_irrelevant']}")
        errs = Counter(x["error"][:90] for x in lg.values() if x.get("error"))
        for e, v in errs.most_common(10):
            print(f"   error x{v}: {e}")
    lk = [r["leak"] for r in tg if r.get("leak")]
    if lk:
        same = sum(x["same"] for x in lk); diff = sum(x["diff"] for x in lk); rep = sum(x["probe_reproduced"] for x in lk)
        print(f"L1: {len(lk)} games, {same + diff} round decisions: same {same} ({100 * same / max(1, same + diff):.2f}%), "
              f"different {diff}; probe reproduced the recorded move {rep}/{same + diff}")
        for x in lk:
            for e in x["examples"][:1]:
                if x["diff"]:
                    print("   e.g.", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "label", "report"])
    ap.add_argument("--az"); ap.add_argument("--ckpt"); ap.add_argument("--src")
    ap.add_argument("--out"); ap.add_argument("--dir")
    ap.add_argument("--per-cell", type=int, default=5); ap.add_argument("--stakes", default="1,8")
    ap.add_argument("--backs", default=""); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--leak", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"record": record, "label": label, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
