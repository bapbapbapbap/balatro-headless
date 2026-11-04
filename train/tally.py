"""Running tally of seeds proven beatable, naneinf seeds proven, and the highest stake beaten,
written everywhere it is shown.

    python3 tally.py [--wins ../wins] [--readme ../README.md]

Counts the published, replayed routes in wins/routes.jsonl (one per seed) and in
naneinf/proven.jsonl (one per seed, each replayed to a hand with non-finite chips), so the numbers
shown are always the numbers anyone can check. Writes:
  wins/badge.json          shields.io endpoints for the badges at the top of README.md:
  wins/badge_naneinf.json  seeds proven beatable, naneinf seeds proven, highest stake beaten
  wins/badge_stake.json
  README.md          the text between <!-- tally:start --> and <!-- tally:end -->
  wins/README.md     the same block
Run it after publish_wins.py (tools/balatro.ps1 publish does both).
"""
import argparse
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Which model a route came from, by the run folder its wins.jsonl was in. The A/B arms
# (low/high, off/on) played from v1's weights.
MODEL = {"v1": "0.1", "low": "0.1", "high": "0.1", "off": "0.1", "on": "0.1", "v2": "0.2", "v2b": "0.3"}
DECK = {"b_red": "Red Deck"}
STAKE = {1: "White Stake", 2: "Red Stake", 3: "Green Stake", 4: "Black Stake", 5: "Blue Stake",
         6: "Purple Stake", 7: "Orange Stake", 8: "Gold Stake"}
BLOCK = re.compile(r"(<!-- tally:start -->\n).*?(<!-- tally:end -->)", re.S)


def count(routes_path):
    by_model, by_setup, total, top_stake = collections.Counter(), collections.Counter(), 0, 0
    for line in open(routes_path):
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1
        top_stake = max(top_stake, r["stake"])
        run = r.get("source", "").split("/")[0]
        by_model[MODEL.get(run, run or "unknown")] += 1
        by_setup[(DECK.get(r["back"], r["back"]), STAKE.get(r["stake"], f"stake {r['stake']}"))] += 1
    return total, by_model, by_setup, top_stake


def count_lines(path):
    """Routes in a .jsonl file; a missing file counts 0."""
    if not os.path.exists(path):
        return 0
    return sum(1 for line in open(path) if line.strip())


def near_misses(path):
    """Final-boss near misses (wins/near_misses.jsonl): runs that reached the ante-8 boss and lost it.
    Returns (count, median share of the boss's target scored, how many scored 90% or more)."""
    if not os.path.exists(path):
        return 0, 0.0, 0
    fr = []
    for line in open(path):
        if line.strip():
            e = json.loads(line).get("recount", {}).get("end", {})
            fr.append(e["chips"] / e["target"] if e.get("target") else 0.0)
    fr.sort()
    med = (fr[len(fr) // 2] if len(fr) % 2 else (fr[len(fr) // 2 - 1] + fr[len(fr) // 2]) / 2) if fr else 0.0
    return len(fr), med, sum(f >= 0.9 for f in fr)


def stake_text(top_stake):
    if not top_stake:
        return "none yet"
    return f"{top_stake} of 8 ({STAKE.get(top_stake, f'stake {top_stake}')})"


def block(total, by_model, by_setup, top_stake, naneinf, wins_link, naneinf_link, near=(0, 0.0, 0), near_link=""):
    models = ", ".join(f"{by_model[m]:,} by model {m}" for m in sorted(by_model))
    setups = ", ".join(f"{n:,} on {d}, {s}" for (d, s), n in sorted(by_setup.items()))
    return (f"**Seeds proven beatable: {total:,}** ({models}; {setups}). Each one is a route in "
            f"[`{wins_link}`]({wins_link}) that was replayed move for move to the same win before it was "
            "listed. This is the project's production count, updated each time wins are published.\n\n"
            f"**naneinf seeds proven: {naneinf:,}.** A seed counts once its route in "
            f"[`{naneinf_link}`]({naneinf_link}) replays move for move to a played hand with non-finite "
            "chips (see `naneinf/README.md`).\n\n"
            f"**Highest stake beaten: {stake_text(top_stake)}.**\n"
            + (f"\n**Final-boss near misses: {near[0]:,}.** Seeds where the bot reached the Ante 8 boss and "
               f"lost it, kept with their routes in [`{near_link}`]({near_link}): median {near[1]:.0%} of the "
               f"boss's target scored, {near[2]:,} at 90% or more. Until 2026-11-03 these were counted as wins "
               "(see the CHANGELOG, 0.13.2).\n" if near[0] else ""))


def put(path, text):
    s = open(path, encoding="utf-8").read()
    if not BLOCK.search(s):
        raise SystemExit(f"{path}: no <!-- tally:start --> / <!-- tally:end --> markers")
    new = BLOCK.sub(lambda m: m.group(1) + text + m.group(2), s)
    if new != s:
        open(path, "w", encoding="utf-8").write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wins", default=os.path.join(ROOT, "wins"))
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    ap.add_argument("--naneinf", default=os.path.join(ROOT, "naneinf", "proven.jsonl"))
    args = ap.parse_args()
    total, by_model, by_setup, top_stake = count(os.path.join(args.wins, "routes.jsonl"))
    naneinf = count_lines(args.naneinf)
    near = near_misses(os.path.join(args.wins, "near_misses.jsonl"))
    badges = {
        "badge.json": ("seeds proven beatable", f"{total:,}", "brightgreen"),
        "badge_naneinf.json": ("naneinf seeds proven", f"{naneinf:,}", "brightgreen" if naneinf else "lightgrey"),
        "badge_stake.json": ("highest stake beaten", stake_text(top_stake), "blue"),
        "badge_near.json": ("final-boss near misses", f"{near[0]:,}", "orange"),
    }
    for name, (label, message, color) in badges.items():
        with open(os.path.join(args.wins, name), "w") as f:
            json.dump({"schemaVersion": 1, "label": label, "message": message, "color": color}, f)
            f.write("\n")
    put(args.readme, block(total, by_model, by_setup, top_stake, naneinf, "wins/", "naneinf/proven.jsonl",
                               near, "wins/near_misses.jsonl"))
    put(os.path.join(args.wins, "README.md"),
        block(total, by_model, by_setup, top_stake, naneinf, "routes.jsonl", "../naneinf/proven.jsonl",
              near, "near_misses.jsonl"))
    print(f"tally: {total:,} seeds proven beatable ({dict(sorted(by_model.items()))}); "
          f"{naneinf:,} naneinf seeds proven; highest stake beaten {stake_text(top_stake)}; "
          f"{near[0]:,} final-boss near misses")


if __name__ == "__main__":
    main()
