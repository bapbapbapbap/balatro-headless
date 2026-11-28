"""Z3: AnteZero's released bot plays in its simulator (its own eval loop); every game is replayed in our harness.

**AnteZero is not ours.** namecallfilter's (https://github.com/namecallfilter/antezero, MIT), pinned at
4fe07f771104a811ca8452efd4831b0dbf097e6e. None of its code is in this repository. This script imports, from a clone
passed as --az: `antezero_agent.runtime.load_model`, `antezero_agent.eval._envs` and `EVAL_SEED` (their eval
protocol), `antezero_agent.model.obs_from_numpy`, `Net.act`; and from `antezero_sim`: `VecEnv.step`, `game(i)`,
`pop_episodes`, `Game.factored_to_code`, `Game.step`.

**Ours** (Claude's, reviewed by chud): this file. Registration: docs/hypotheses-reproduce.md, 2026-11-27 "Z3".

    python3 z3.py record --az <antezero clone> --ckpt antezero-v0.2.0.pt --stakes 1,8 --per-cell 20 --out games.jsonl
    python3 z3.py replay --src <game source> --games games.jsonl --workers N --out replay.jsonl [--hist dir]
    python3 z3.py report --games games.jsonl --replay replay.jsonl
"""
import argparse, json, os, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lockstep as L  # noqa: E402


def record(a):
    sys.path.insert(0, os.path.join(a.az, "agent", "src"))
    import numpy as np
    import torch
    from antezero_agent.eval import EVAL_SEED, _envs
    from antezero_agent.model import obs_from_numpy
    from antezero_agent.runtime import load_model
    torch.set_num_threads(a.threads)
    model, cfg, meta = load_model(a.ckpt, "cpu")
    pc = cfg.env.play_candidates
    stakes = [int(s) for s in a.stakes.split(",")]
    backs = a.backs.split(",") if a.backs else L.BACKS
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["back"], json.loads(l)["stake"]) for l in open(a.out)}
    with open(a.out, "a") as f, torch.no_grad():
        for stake in stakes:
            for back in backs:
                if (back, stake) in done:
                    continue
                # One cell per call of their _envs: the seed range is theirs (EVAL_SEED + batch index).
                for env in _envs(cfg, a.per_cell, EVAL_SEED, [back], [stake], None, False, a.per_cell):
                    n = env.num_envs
                    starts = [env.game(i) for i in range(n)]
                    codes = [[] for _ in range(n)]
                    first, steps, bad = {}, 0, [0] * n
                    while len(first) < n and steps <= cfg.env.step_limit + 10:
                        out = model.act(obs_from_numpy(env, "cpu"), deterministic=True)
                        op = out["op"].to(torch.int32).cpu().numpy()
                        cards = out["cards"].cpu().numpy().astype(np.uint32)
                        dest = out["dest"].to(torch.int32).cpu().numpy()
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
                    for i in range(n):
                        ep = first.get(i)
                        g0 = starts[i]
                        # Self-check in the simulator: the codes from the start state reach the same end.
                        h = g0.clone()
                        ok = True
                        for c in codes[i]:
                            if h.is_terminal():
                                ok = False; break
                            h.step(c)
                        rec = {"seed": g0.seed, "back": back, "stake": stake, "codes": codes[i],
                               "sim_won": bool(ep[5]) if ep else None, "sim_ante": int(ep[4]) if ep else None,
                               "sim_blinds": int(ep[6]) if ep else None, "truncated": bool(ep[9]) if ep else None,
                               "no_code": bad[i], "selfcheck": ok and h.run_won == (bool(ep[5]) if ep else None)
                               and h.ante == (ep[4] if ep else None),
                               "options": {"allow_skip": g0.allow_skip, "null_sorts": g0.null_sorts,
                                           "max_rearranges": g0.max_rearranges, "endless": g0.endless,
                                           "steamodded": g0.steamodded}}
                        f.write(json.dumps(rec) + "\n")
                    f.flush()
                    w = sum(bool(first[i][5]) for i in first)
                    print(f"{back} stake {stake}: {w}/{n} won in sim, "
                          f"selfcheck fails {sum(1 for i in range(n) if not rec_ok(codes, starts, first, i))}",
                          flush=True)


def rec_ok(codes, starts, first, i):
    h = starts[i].clone()
    for c in codes[i]:
        if h.is_terminal():
            return False
        h.step(c)
    ep = first.get(i)
    return ep is not None and h.run_won == bool(ep[5])


def replay_one(job):
    src, rec, hist_dir = job
    import antezero_sim
    env = L.env_for(src)
    seed, back, stake = rec["seed"], rec["back"], rec["stake"]
    out = {"seed": seed, "back": back, "stake": stake, "sim_won": rec["sim_won"]}
    L.ART.clear()
    try:
        g = antezero_sim.Game(seed, back, stake, False)
        o = rec["options"]
        g.allow_skip = o["allow_skip"]; g.null_sorts = o["null_sorts"]; g.max_rearranges = o["max_rearranges"]
        env.reset(seed, back, stake)
        sv, gv = L.sim_view(json.loads(g.state_json())), L.game_view(env)
        off = gv["deck"][0]["sort_id"] - sv["deck"][0]["sort_id"] if sv["deck"] and gv["deck"] else 0
        d = L.compare(sv, gv, off)
        steps = unlisted = 0
        for code in rec["codes"]:
            if d:
                break
            g.step(code)
            if not L.env_apply(env, L.translate(code)):
                unlisted += 1
                out.setdefault("unlisted", []).append([steps, code])
            steps += 1
            sv, gv = L.sim_view(json.loads(g.state_json())), L.game_view(env)
            d = L.compare(sv, gv, off)
        out["match_raw"] = not d
        if d and steps == len(rec["codes"]) and env.won and g.is_terminal():
            # The game pauses on its win overlay with the boss still in place; the simulator's terminal state is
            # the one after the overlay is dismissed (as AnteZero's harness does). Dismiss it, settle, compare again.
            G = env.lua.globals().G
            if G.SETTINGS.paused and G.OVERLAY_MENU:
                G.FUNCS.exit_overlay_menu()
                env.H.settle()
                out["overlay_dismissed"] = True
                out["diffs_raw"] = [list(map(str, x))[:3] for x in d[:6]]
                sv, gv = L.sim_view(json.loads(g.state_json())), L.game_view(env)
                d = L.compare(sv, gv, off)
        out.update(steps=steps, of=len(rec["codes"]), match=not d and steps == len(rec["codes"]),
                   diffs=[list(map(str, x))[:3] for x in d[:12]], game_won=bool(env.won), game_ante=gv.get("ante"),
                   game_state=gv.get("state"), unlisted_n=unlisted, art_n=len(L.ART))
        if d and hist_dir:
            os.makedirs(hist_dir, exist_ok=True)
            with open(os.path.join(hist_dir, f"z3_{seed}_{back}_{stake}.json"), "w") as f:
                json.dump({**rec, "diverged_at": steps, "diffs": out["diffs"]}, f)
    except Exception as e:
        import traceback
        out.update(match=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    return out


def replay(a):
    recs = [json.loads(l) for l in open(a.games)]
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["seed"], json.loads(l)["back"], json.loads(l)["stake"]) for l in open(a.out)}
    jobs = [(a.src, r, a.hist) for r in recs if (r["seed"], r["back"], r["stake"]) not in done]
    print(f"{len(jobs)} games to replay ({len(done)} done)", flush=True)
    with open(a.out, "a") as f, Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(replay_one, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if not r["match"] or n % 20 == 0:
                print(n, r["seed"], r["back"], r["stake"], "match" if r["match"] else "DIFF",
                      f'{r.get("steps")}/{r.get("of")}', "won" if r.get("game_won") else "",
                      r.get("error") or r.get("diffs", [])[:3], flush=True)


def report(a):
    from collections import Counter, defaultdict
    games = {(r["seed"], r["back"], r["stake"]): r for r in map(json.loads, open(a.games))}
    reps = {(r["seed"], r["back"], r["stake"]): r for r in map(json.loads, open(a.replay))} if a.replay else {}
    by = defaultdict(lambda: Counter())
    for k, g in games.items():
        c = by[g["stake"]]
        c["games"] += 1; c["sim_won"] += bool(g["sim_won"]); c["selfcheck_fail"] += not g["selfcheck"]
        c["no_code"] += g["no_code"] > 0
        r = reps.get(k)
        if r:
            c["replayed"] += 1; c["match"] += bool(r["match"])
            c["game_won"] += bool(r["match"] and r.get("game_won"))
            c["sim_won_and_replayed"] += bool(g["sim_won"] and r["match"])
    for s, c in sorted(by.items()):
        n = c["games"]
        print(f"stake {s}: games {n}, sim wins {c['sim_won']} ({100 * c['sim_won'] / n:.1f}%), "
              f"selfcheck fails {c['selfcheck_fail']}, games with an undecodable action {c['no_code']}")
        if c["replayed"]:
            print(f"   replayed {c['replayed']}, move-for-move {c['match']} ({100 * c['match'] / c['replayed']:.1f}%), "
                  f"wins on the game's code {c['game_won']} ({100 * c['game_won'] / c['replayed']:.1f}%)")
    kinds = Counter()
    for r in reps.values():
        if not r["match"]:
            kinds[(r.get("error") or "")[:70] or (r["diffs"][0][0] if r.get("diffs") else "?")] += 1
    for k, v in kinds.most_common(30):
        print(f"  {v:4d}  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "replay", "report"])
    ap.add_argument("--az"); ap.add_argument("--ckpt"); ap.add_argument("--src")
    ap.add_argument("--stakes", default="1,8"); ap.add_argument("--backs", default="")
    ap.add_argument("--per-cell", type=int, default=20); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--games"); ap.add_argument("--replay"); ap.add_argument("--out"); ap.add_argument("--hist")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"record": record, "replay": replay, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
