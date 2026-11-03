"""Running tally of seeds proven beatable, written everywhere it is shown.

    python3 tally.py [--wins ../wins] [--readme ../README.md]

Counts the published, replayed routes in wins/routes.jsonl (one per seed), so the number shown
is always the number anyone can check. Writes:
  wins/badge.json    shields.io endpoint for the badge at the top of README.md
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
STAKE = {1: "White Stake"}
BLOCK = re.compile(r"(<!-- tally:start -->\n).*?(<!-- tally:end -->)", re.S)


def count(routes_path):
    by_model, by_setup, total = collections.Counter(), collections.Counter(), 0
    for line in open(routes_path):
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1
        run = r.get("source", "").split("/")[0]
        by_model[MODEL.get(run, run or "unknown")] += 1
        by_setup[(DECK.get(r["back"], r["back"]), STAKE.get(r["stake"], f"stake {r['stake']}"))] += 1
    return total, by_model, by_setup


def block(total, by_model, by_setup, wins_link):
    models = ", ".join(f"{by_model[m]:,} by model {m}" for m in sorted(by_model))
    setups = ", ".join(f"{n:,} on {d}, {s}" for (d, s), n in sorted(by_setup.items()))
    return (f"**Seeds proven beatable: {total:,}** ({models}; {setups}). Each one is a route in "
            f"[`{wins_link}`]({wins_link}) that replays to a win on the untouched game. This is the "
            "project's production count; `tools/balatro.ps1 publish` updates it.\n")


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
    args = ap.parse_args()
    total, by_model, by_setup = count(os.path.join(args.wins, "routes.jsonl"))
    with open(os.path.join(args.wins, "badge.json"), "w") as f:
        json.dump({"schemaVersion": 1, "label": "seeds proven beatable", "message": f"{total:,}",
                   "color": "brightgreen"}, f)
        f.write("\n")
    put(args.readme, block(total, by_model, by_setup, "wins/"))
    put(os.path.join(args.wins, "README.md"), block(total, by_model, by_setup, "routes.jsonl"))
    print(f"tally: {total:,} seeds proven beatable ({dict(sorted(by_model.items()))})")


if __name__ == "__main__":
    main()
