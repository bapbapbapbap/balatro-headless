"""Which of Balatro Lab's heuristics carry its win rate: its bot with one switched off at a time, in its own
simulator, on model v6's teacher seeds (docs/hypotheses-v6.md, 2026-11-19: ablation test K, chud's idea).

Each arm is ValueBot (VALUEBOT_WEIGHTS=hand, as the teacher) with class settings changed in a subclass made here;
Balatro Lab's code is not edited. The full bot on the same seeds is test P's run (lab_on_teach600.jsonl), so the
arms are paired with it seed by seed.

  K1 search   ROUND_SEARCH off: no playing a boss round out before a play or discard in doubt (search.py)
  K2 power1   HANDS 1: the build's power (behind every valuation) from 1 sample hand instead of 12
  K3 draws1   DRAWS 1, RERANK 0: a discard scored on 1 possible draw instead of 6 (+24 for the leaders)
  K4 bonuses  the hand-set bonuses in Weights at 0 (money, interest, growing jokers, rarity, Negative, kept
              consumables, cards, rerolls, packs, vouchers, tags, joker classes): purchases valued by the build's
              power alone; thresholds (sell margin, skip safety, the shop's worth) and modifiers kept

    python3 ablate.py --lab <balatro-lab clone> --arms search,power1,draws1,bonuses --games 300 --workers 12 \\
        --out <balatro-work>/ablate_k.jsonl
    python3 ablate.py --summary <balatro-work>/ablate_k.jsonl --full <balatro-work>/lab_on_teach600.jsonl
Re-running resumes (arm and seed already in --out are skipped).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ARMS = ("search", "power1", "draws1", "bonuses")


def bot_class(arm):
    from dataclasses import replace
    from balatro.strategy import ValueBot, Weights
    if arm == "full":
        return ValueBot
    if arm == "search":
        return type("K1", (ValueBot,), {"ROUND_SEARCH": False})
    if arm == "power1":
        return type("K2", (ValueBot,), {"HANDS": 1})
    if arm == "draws1":
        return type("K3", (ValueBot,), {"DRAWS": 1, "RERANK": 0})
    if arm == "bonuses":
        w = Weights()
        zero = replace(w, money=0.0, interest=0.0, grows=0.0, pays=0.0, rules=0.0, rarity=(0.0, 0.0, 0.0, 0.0),
                       negative=0.0, keep=0.0, card=0.0, reroll=0.0,
                       packs=dict.fromkeys(w.packs, 0.0), vouchers=dict.fromkeys(w.vouchers, 0.0),
                       joker=dict.fromkeys(w.joker, 0.0), tags=dict.fromkeys(w.tags, 0.0),
                       classes=dict.fromkeys(w.classes, 0.0))
        return type("K4", (ValueBot,), {"w": zero, "net": None})
    raise ValueError(arm)


def init(lab):
    os.environ.setdefault("VALUEBOT_WEIGHTS", "hand")      # the teacher's weights (as teach.py and eval_on_our_seeds)
    sys.path.insert(0, lab)


def play(job):
    arm, seed = job
    from balatro.run import Run
    t = time.time()
    res = Run(bot_class(arm)(), game_seed=seed, deck="Red Deck", max_ante=8).play()
    return {"arm": arm, "seed": seed, "won": bool(res.won), "ante": res.ante, "blinds": res.blinds_beaten,
            "seconds": round(time.time() - t, 1)}


def load(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def summary(path, full_path):
    rows, full = load(path), {r["seed"]: r for r in load(full_path)}
    print(f"full bot (test P): {sum(r['won'] for r in full.values())}/{len(full)} won" if full else "full bot: no file")
    for arm in ARMS:
        a = {r["seed"]: r for r in rows if r["arm"] == arm}
        if not a:
            continue
        seeds = [s for s in a if s in full]
        w, wf = sum(a[s]["won"] for s in seeds), sum(full[s]["won"] for s in seeds)
        lost = sum(1 for s in seeds if full[s]["won"] and not a[s]["won"])
        gained = sum(1 for s in seeds if a[s]["won"] and not full[s]["won"])
        n = lost + gained
        z = (lost - gained) / math.sqrt(n) if n else 0.0
        d = (wf - w) / len(seeds) if seeds else 0.0
        se = math.sqrt(n) / len(seeds) if seeds else 0.0
        print(f"{arm:8s} {len(seeds)} paired seeds: won {w} vs full {wf}; costs {100 * d:+.1f} points "
              f"(±{196 * se:.1f}); lost {lost}, gained {gained} (McNemar z {z:.2f}); "
              f"{sum(a[s]['seconds'] for s in seeds) / max(len(seeds), 1):.0f} s/game")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--games", type=int, default=300, help="teacher seeds 0..games-1, per arm")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="ablate_k.jsonl")
    ap.add_argument("--summary")
    ap.add_argument("--full", help="with --summary: the full bot's games on the same seeds (test P's jsonl)")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary, a.full or "")
        return 0
    from teach import teach_seeds
    seeds = teach_seeds(a.games, a.first)
    arms = [x for x in a.arms.split(",") if x]
    for x in arms:
        assert x in ARMS + ("full",), x
    done = {(r["arm"], r["seed"]) for r in load(a.out)}
    # seed-major order, so a run cut short has every arm on the same first seeds
    todo = [(arm, s) for s in seeds for arm in arms if (arm, s) not in done]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=init, initargs=(a.lab,)) as pool, open(a.out, "a") as out:
        for k, r in enumerate(pool.imap(play, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if k % 24 == 0 or k == len(todo):
                print(f"{time.strftime('%H:%M:%S')}  {k}/{len(todo)} games, {(time.time() - t0) / 60:.0f} min",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
