"""Reads the teacher's-gap measurements (docs/hypotheses-v6.md, 2026-11-19) and prints what they say.

    python3 gap_summary.py <balatro-work folder>

Reads teach_v6/games_w*.jsonl (the bridge, before F1), bell_f1c/games_w*.jsonl (F1c) and lab_on_teach600.jsonl
(test P). Prints F1c's counts, the 1,800-game teacher rate with the F1c reruns in place of the originals, and test
P's paired table on teacher seeds 0-599: their simulator vs our game through the bridge (before F1, and with F1c's
reruns where the seed was rerun). Missing files are reported, not fatal.
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def rows(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["seed"]] = r
    return out


def rate(w, n):
    p = w / n if n else 0.0
    return f"{w}/{n} = {p:.1%} (±{1.96 * math.sqrt(p * (1 - p) / n):.1%})" if n else "-"


def main():
    work = sys.argv[1]
    base = rows(os.path.join(work, "teach_v6", "games_w*.jsonl"))
    f1c = rows(os.path.join(work, "bell_f1c", "games_w*.jsonl"))
    lab = rows(os.path.join(work, "lab_on_teach600.jsonl"))
    bell = [s.strip() for s in open(os.path.join(HERE, "results", "2026-11-19", "bell_seeds.txt")) if s.strip()]
    um = lambda r: sum(r.get("unmapped", {}).values())  # noqa: E731
    print(f"bridge before F1: {rate(sum(r['won'] for r in base.values()), len(base))} over {len(base)} teacher games")

    if f1c:
        b0 = [base[s] for s in bell if s in base and s in f1c]
        b1 = [f1c[s] for s in bell if s in base and s in f1c]
        print(f"F1c: {len(f1c)}/{len(bell)} Bell games rerun. Before: {rate(sum(r['won'] for r in b0), len(b0))}; "
              f"after F1 + F1b: {rate(sum(r['won'] for r in b1), len(b1))}")
        print(f"  unmapped moves after: {sum(um(r) for r in b1)} in {sum(1 for r in b1 if um(r))} games "
              f"(F1c-a: 0 in all; F1c-b: >= 225/244 won)")
        flips = sum(1 for a, b in zip(b0, b1) if not a["won"] and b["won"])
        back = sum(1 for a, b in zip(b0, b1) if a["won"] and not b["won"])
        print(f"  lost -> won {flips}, won -> lost {back}")
        merged = {s: (f1c[s] if s in f1c else r) for s, r in base.items()}
        print(f"  teacher through the bridge after F1 (1,800 games, reruns in place): "
              f"{rate(sum(r['won'] for r in merged.values()), len(merged))}")
    else:
        merged = base
        print("F1c: no results yet (bell_f1c/)")

    if lab:
        from teach import teach_seeds
        seeds = [s for s in teach_seeds(600, 0) if s in lab and s in base]
        L = sum(lab[s]["won"] for s in seeds)
        for name, ours in (("before F1", base), ("after F1 (F1c reruns in place)", merged)):
            O = sum(ours[s]["won"] for s in seeds)
            both = sum(1 for s in seeds if lab[s]["won"] and ours[s]["won"])
            only_lab = [s for s in seeds if lab[s]["won"] and not ours[s]["won"]]
            only_ours = [s for s in seeds if ours[s]["won"] and not lab[s]["won"]]
            n = len(only_lab) + len(only_ours)
            z = (len(only_lab) - len(only_ours)) / math.sqrt(n) if n else 0.0
            print(f"test P, {len(seeds)} paired seeds, our game {name}: their simulator {rate(L, len(seeds))}, "
                  f"our game {rate(O, len(seeds))}; both won {both}; won only in theirs {len(only_lab)}, only in ours "
                  f"{len(only_ours)} (McNemar z {z:.2f})")
        bl = [s for s in seeds if s in set(bell) and not base[s]["won"]]
        print(f"  P3: Bell games lost before F1 among these seeds: {len(bl)}; won in their simulator: "
              f"{sum(lab[s]['won'] for s in bl)}")
        print(f"  seeds won only in theirs (after F1), for teach.py --check: "
              f"{' '.join(s for s in seeds if lab[s]['won'] and not merged[s]['won'])}")
    else:
        print("test P: no results yet (lab_on_teach600.jsonl)")


if __name__ == "__main__":
    main()
