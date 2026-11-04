"""Save-progress ledger and the proof that Balatro can be beaten (docs/proof.md).

Reads every published route (wins/routes.jsonl, plus any --extra wins.jsonl files) and records,
for each thing a full save asks for (chud's 100%, docs/hypotheses.md roadmap):
  - each deck at each stake,
  - each challenge,
  - each joker at each stake (a joker's sticker comes from a run won while holding it),
the FIRST seed that achieved it. First means earliest found: model 0.1's line (v1 and its A/B
runs) by training round, then 0.2, then 0.3, then any later run by the time it was found.

What a win earned is the game's own record, not ours. The bot's runs were never recorded by the
game: every run is seeded, and the game keeps no progress for seeded runs, in a profile that was
thrown away. So each route is replayed once (in parallel), with the profile emptied first and the
seeded flag cleared, and the game's profile is read afterwards: which jokers and which deck it
credited with a win at which stake, and which challenge it marked complete. The result is kept in
wins/won_with.jsonl, with the jokers held at the win as a cross-check. A replay that doesn't
reproduce the saved win is reported and left out.

Writes wins/save_progress.json and the generated part of docs/proof.md (between the markers).

usage: python3 tools/save_progress.py --src ~/balatro/local/source [--extra runs/*/wins.jsonl]
                                      [--jobs 12] [--full]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "harness"))

STAKES = {1: "White", 2: "Red", 3: "Green", 4: "Black", 5: "Blue", 6: "Purple", 7: "Orange", 8: "Gold"}
FAMILY = {"v1": 0, "low": 0, "high": 0, "off": 0, "on": 0, "v2": 1, "v2b": 2}   # 0.1's line, 0.2, 0.3


def key(r):
    return (r["seed"], r["back"], r["stake"], r.get("challenge"))


def found_order(r):
    run = (r.get("source") or "").split("/")[0]
    if run in FAMILY:
        return (0, FAMILY[run], r.get("iteration") or 0, r.get("found_utc") or "", r["seed"])
    return (1, 0, 0, r.get("found_utc") or "9999", r["seed"])


def _replay(job):
    src, full, r = job
    from sim import Sim
    sim = _replay.sims.get((r.get("actions", "subsets"), r.get("enc", 1)))
    if sim is None:
        sim = Sim(src, fast=not full, mode=r.get("actions", "subsets"), enc=r.get("enc", 1), count_progress=True)
        _replay.sims[(r.get("actions", "subsets"), r.get("enc", 1))] = sim
    sim.env.clear_progress()
    prog, n, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"], r.get("challenge"))
    ok = not problems and n == r["n"] and final == r["final"] and prog["won"]
    if not ok:
        return key(r), None, (problems[:1] if problems else ["final state or outcome differs"])
    return key(r), {"record": sim.env.progress_record(), "jokers_held": sim.env.jokers_held()}, []


_replay.sims = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--routes", default=os.path.join(ROOT, "wins", "routes.jsonl"))
    ap.add_argument("--extra", nargs="*", default=[], help="more wins.jsonl files (verified wins only)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--full", action="store_true", help="replay on the untouched game (slower)")
    a = ap.parse_args()

    routes = {}
    for f in [a.routes] + [p for x in a.extra for p in glob.glob(x)]:
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r["progress"]["won"] or r.get("verified") is False:
                continue
            k = key(r)
            if k not in routes or found_order(r) < found_order(routes[k]):
                routes[k] = r

    cache_path = os.path.join(ROOT, "wins", "won_with.jsonl")
    held = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            if line.strip():
                c = json.loads(line)
                if "record" in c:
                    held[(c["seed"], c["back"], c["stake"], c.get("challenge"))] = c
    todo = [r for k, r in routes.items() if k not in held]
    bad = []
    if todo:
        print(f"replaying {len(todo)} wins to read what the game credits each one ({a.jobs} processes)", flush=True)
        with mp.Pool(a.jobs) as pool, open(cache_path, "a") as out:
            for i, (k, got, problems) in enumerate(pool.imap_unordered(_replay, [(a.src, a.full, r) for r in todo], chunksize=4)):
                if got is None:
                    bad.append((k, problems))
                    continue
                c = {"seed": k[0], "back": k[1], "stake": k[2], "challenge": k[3], **got}
                held[k] = c
                out.write(json.dumps(c) + "\n")
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)

    from env import BalatroEnv
    env = BalatroEnv(a.src)
    decks, jokers, challenges = env.deck_keys(), env.joker_keys(), env.challenge_ids()
    names = {k: str(env.lua.globals().G.P_CENTERS[k].name) for k in decks + jokers}

    first = {"deck": {}, "joker": {}, "challenge": {}}         # item -> stake -> route
    unrecorded, disagree = [], []
    for k, r in sorted(routes.items(), key=lambda kv: found_order(kv[1])):
        if k not in held:
            continue
        rec = held[k]["record"]
        for c in rec["challenges"]:
            first["challenge"].setdefault(c, {}).setdefault(1, r)
        for d, s in rec["decks"].items():
            first["deck"].setdefault(d, {}).setdefault(int(s), r)
        for j, s in rec["jokers"].items():
            first["joker"].setdefault(j, {}).setdefault(int(s), r)
        if not (rec["challenges"] or rec["decks"] or rec["jokers"]):
            unrecorded.append(k)                  # a win the game credited with nothing
        elif not r.get("challenge") and set(rec["jokers"]) != set(held[k]["jokers_held"]):
            disagree.append(k)                    # credited jokers differ from those held at the win

    def best(item_map):
        return max(item_map) if item_map else 0
    gold_decks = [d for d in decks if 8 in first["deck"].get(d, {})]
    gold_jokers = [j for j in jokers if 8 in first["joker"].get(j, {})]
    done_ch = [c for c in challenges if c in first["challenge"]]
    total = len(decks) + len(jokers) + len(challenges)
    done = len(gold_decks) + len(gold_jokers) + len(done_ch)
    steps = (sum(best(first["deck"].get(d, {})) for d in decks) + sum(best(first["joker"].get(j, {})) for j in jokers)
             + 8 * len(done_ch))
    summary = {
        "complete": done, "of": total, "percent": round(100 * done / max(total, 1), 2),
        "stake_steps": steps, "stake_steps_of": 8 * total, "stake_steps_percent": round(100 * steps / max(8 * total, 1), 2),
        "decks_gold": len(gold_decks), "decks": len(decks), "jokers_gold": len(gold_jokers), "jokers": len(jokers),
        "challenges_won": len(done_ch), "challenges": len(challenges),
        "wins_counted": len(held), "wins_not_reproduced": len(bad),
        "wins_credited_nothing": len(unrecorded), "wins_where_credited_jokers_differ_from_held": len(disagree),
        "first": {kind: {item: {str(s): {"seed": r["seed"], "back": r["back"], "stake": r["stake"],
                                         "challenge": r.get("challenge"), "source": r.get("source"),
                                         "iteration": r.get("iteration")} for s, r in sorted(m.items())}
                         for item, m in sorted(v.items())} for kind, v in first.items()},
    }
    json.dump(summary, open(os.path.join(ROOT, "wins", "save_progress.json"), "w"), indent=1)

    L = [f"**Save progress: {done} of {total} ({summary['percent']}%)** — Gold Stake won on "
         f"{len(gold_decks)} of {len(decks)} decks, with {len(gold_jokers)} of {len(jokers)} jokers; "
         f"{len(done_ch)} of {len(challenges)} challenges. Counting each stake as a step: "
         f"{steps} of {8 * total} ({summary['stake_steps_percent']}%). From {len(held)} published wins"
         + (f"; {len(bad)} didn't reproduce and aren't counted." if bad else "."), "",
         "### Decks: first seed won at each stake", "",
         "| Deck | " + " | ".join(STAKES[s] for s in range(1, 9)) + " |", "|---|" + "---|" * 8]
    for d in decks:
        m = first["deck"].get(d, {})
        L.append(f"| {names[d]} | " + " | ".join(f"`{m[s]['seed']}`" if s in m else "—" for s in range(1, 9)) + " |")
    L += ["", "### Challenges: first seed won", "", "| Challenge | First seed |", "|---|---|"]
    for c in challenges:
        r = first["challenge"].get(c, {}).get(1)
        L.append(f"| {c} | " + (f"`{r['seed']}`" if r else "—") + " |")
    L += ["", "### Jokers: highest stake won while holding it, and the first seed at that stake", "",
          "| Joker | Highest stake | First seed there |", "|---|---|---|"]
    for j in jokers:
        m = first["joker"].get(j, {})
        b = best(m)
        L.append(f"| {names[j]} | {STAKES[b] if b else '—'} | " + (f"`{m[b]['seed']}` ({STAKES[b]} Stake, "
                 f"{names.get(m[b]['back'], m[b]['back'])})" if b else "—") + " |")
    proof = os.path.join(ROOT, "docs", "proof.md")
    text = open(proof).read()
    head, rest = text.split("<!-- proof:start -->")
    _, tail = rest.split("<!-- proof:end -->")
    open(proof, "w").write(head + "<!-- proof:start -->\n" + "\n".join(L) + "\n<!-- proof:end -->" + tail)
    for k, p in bad[:10]:
        print(f"  not reproduced: {k} {p}")
    if unrecorded:
        print(f"  the game credited nothing to {len(unrecorded)} wins (e.g. {unrecorded[0]}): check that the "
              f"profile fields read in boot.lua progress_record match the pinned source")
    if disagree:
        print(f"  {len(disagree)} wins: the game credited different jokers than were held (e.g. {disagree[0]})")
    print(json.dumps({x: summary[x] for x in summary if x != "first"}))


if __name__ == "__main__":
    main()
