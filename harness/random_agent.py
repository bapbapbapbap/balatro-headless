"""Uniform-random agent: picks any legal action. Plays seeded runs to the end.

usage: python3 random_agent.py --src DIR [--runs 20] [--back b_red] [--stake 1] [--max-steps 3000]
"""
import argparse
import collections
import random
import time

from env import BalatroEnv


def play(env, seed, rng, back, stake, max_steps):
    env.reset(seed, back, stake)
    counts = collections.Counter()
    while not env.done and env.steps < max_steps:
        acts = env.legal_actions()
        if not acts:
            raise RuntimeError(f"no legal actions in state {env.state()} (seed {seed}, step {env.steps})")
        a = rng.choice(acts)
        counts[a[0]] += 1
        env.step(a)
    o = env.observe()
    return {"seed": seed, "won": env.won, "ante": o["ante"], "round": o["round"], "money": o["money"],
            "steps": env.steps, "frames": env.frames, "state": o["state"],
            "capped": not env.done, "actions": counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mods")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--back", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--rng", type=int, default=0, help="agent's own RNG seed")
    args = ap.parse_args()

    t0 = time.time()
    env = BalatroEnv(args.src, args.mods)
    print(f"boot {time.time() - t0:.1f}s")
    rng = random.Random(args.rng)
    total = collections.Counter()
    results = []
    for r in range(args.runs):
        seed = "".join(rng.choice("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ") for _ in range(8))
        t = time.time()
        res = play(env, seed, rng, args.back, args.stake, args.max_steps)
        dt = time.time() - t
        total.update(res["actions"])
        results.append(res)
        print(f"{seed}  ante {res['ante']}  round {res['round']:2d}  {'WON ' if res['won'] else ''}"
              f"{'CAPPED ' if res['capped'] else ''}{res['state']:12s} {res['steps']:4d} steps  "
              f"{dt:5.1f}s  {dt / max(res['steps'], 1) * 1000:.0f} ms/step")
    antes = collections.Counter(r["ante"] for r in results)
    print("\nfinal ante distribution:", dict(sorted(antes.items())))
    print("action counts:", dict(total.most_common()))


if __name__ == "__main__":
    main()
