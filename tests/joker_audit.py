"""Per-joker scoring audit: the sim's score for a play (harness/build_phi.lua H.play_scores, the same score()
phi_build uses) against the chips the game itself scores when that play is made (evaluate_play, captured as
tests/v5_play_check.py does). Won routes only reach jokers the bot wins with; this puts every joker on a board.

Each trial: a fresh run (deck from --backs, White Stake), the first blind selected, then the joker under test is
created and added by the game's own create_card / add_to_deck / emplace, with a random edition some of the time
and 0-2 random companion jokers some of the time (copiers and held-card jokers need partners). Some hand cards get a
random enhancement, seal or edition. The board settles (Card:update runs), the sim values every play, one play is
made (the best by S half the time, a random one otherwise), and floor(S) is compared with the game's chips.

Plays are labeled, not dropped, when the check's own registration leaves them out (tests/v5_play_check.py
LEFT_OUT_JOKERS: jokers acting before scoring or by chance; Lucky Cards played). The report lists, per joker,
trials, exact, labeled, and the first misses.

  python3 tests/joker_audit.py <game source dir> [--trials 12] [--workers 2] [--only j_blackboard,...] [--out f.jsonl]
"""
import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, HERE)
from sim import Sim  # noqa: E402
from v5_play_check import CAPTURE, LEFT_OUT_JOKERS  # noqa: E402

BACKS = ["b_red", "b_blue", "b_yellow", "b_plasma", "b_checkered", "b_abandoned"]
ENH = ["m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky"]
SEALS = ["Gold", "Red", "Blue", "Purple"]
EDS = ["foil", "holo", "polychrome"]

SETUP = r"""local H, key, companions, jed, cards = ...
local function add(k, e)
  local c = create_card('Joker', G.jokers, nil, nil, nil, nil, k, 'audit')
  if e ~= '' then c:set_edition({ [e] = true }, true, true) end
  c:add_to_deck(); G.jokers:emplace(c)
end
for k in string.gmatch(companions, '[^,]+') do add(k, '') end
add(key, jed)
for spec in string.gmatch(cards, '[^;]+') do
  local i, what, v = spec:match('^(%d+):(%a+)=(.+)$')
  local c = G.hand.cards[tonumber(i)]
  if c then
    if what == 'enh' then c:set_ability(G.P_CENTERS[v], nil, true)
    elseif what == 'seal' then c:set_seal(v, true, true)
    elseif what == 'ed' then c:set_edition({ [v] = true }, true, true) end
  end
end
return H.settle()"""

NAMES = r"""(function() local t = {} for _, j in ipairs(G.jokers.cards) do t[#t + 1] = j.ability.name end
return table.concat(t, '|') end)()"""


def joker_keys(sim):
    s = sim.env.lua.eval("(function() local t = {} for _, c in ipairs(G.P_CENTER_POOLS.Joker) do t[#t + 1] = c.key end "
                         "return table.concat(t, ',') end)()")
    return s.split(",")


def first(sim, prefix):
    sim.observe()
    for i in range(len(sim.H._encoded_actions)):
        if sim.H.action_desc(i + 1).startswith(prefix):
            return i
    return None


def trial(sim, key, t, all_keys):
    rng = random.Random(f"{key}/{t}")
    seed = "".join(rng.choice("ABCDEFGHIJKLMNPQRSTUVWXYZ123456789") for _ in range(8))
    back = BACKS[t % len(BACKS)]
    sim.reset(seed, back, 1)
    i = first(sim, "select_blind")
    if i is None:
        return {"key": key, "t": t, "error": "no select_blind"}
    sim.step(i)
    comps = rng.sample([k for k in all_keys if k != key], rng.choice([0, 0, 1, 2])) if t % 3 else []
    jed = rng.choice(EDS) if rng.random() < 0.25 else ""
    specs = []
    n_hand = int(sim.env.lua.eval("#G.hand.cards"))
    for c in range(1, n_hand + 1):
        r = rng.random()
        if r < 0.15:
            specs.append(f"{c}:enh={rng.choice(ENH)}")
        elif r < 0.22:
            specs.append(f"{c}:seal={rng.choice(SEALS)}")
        elif r < 0.27:
            specs.append(f"{c}:ed={rng.choice(EDS)}")
    if sim.env.lua.execute(SETUP, sim.H, key, ",".join(comps), jed, ";".join(specs)) < 0:
        return {"key": key, "t": t, "error": "never settled after setup"}
    sim.observe()
    S = sim.H.play_scores()
    plays = [(i, S[i + 1]) for i in range(len(sim.H._encoded_actions))
             if sim.H.action_desc(i + 1).startswith("play") and S[i + 1] is not None]
    if not plays:
        return {"key": key, "t": t, "error": "no scored play"}
    pick = max(plays, key=lambda p: p[1]) if rng.random() < 0.5 else rng.choice(plays)
    desc = sim.H.action_desc(pick[0] + 1)
    names = set(sim.env.lua.eval(NAMES).split("|"))
    label = sorted(names & LEFT_OUT_JOKERS)
    cards = [int(x) for x in desc.split()[1].split(",")] if len(desc.split()) > 1 else []
    if any(sim.env.lua.eval(f"G.hand.cards[{c}] and G.hand.cards[{c}].ability.name or ''") == "Lucky Card" for c in cards):
        label.append("Lucky Card")
    sim.env.lua.execute(CAPTURE, sim.H)
    sim.H._cap = None
    sim.step(pick[0])
    got = sim.H._cap or 0
    return {"key": key, "t": t, "back": back, "seed": seed, "companions": comps, "edition": jed, "cards": specs,
            "play": desc, "S": math.floor(pick[1]), "game": got, "exact": math.floor(pick[1]) == got,
            "label": label}


_SIM = None


def work(job):
    global _SIM
    src, key, t, all_keys = job
    if _SIM is None:
        _SIM = Sim(src, mode="ordered-capped", enc=2, build_phi=True, offer_values=True, play_values=True)
    try:
        return trial(_SIM, key, t, all_keys)
    except Exception as e:  # a crash is a finding too; a fresh sim for the next job
        _SIM = None
        return {"key": key, "t": t, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="joker_audit.jsonl")
    a = ap.parse_args()
    probe = Sim(a.src, mode="ordered-capped", enc=2)
    probe.reset("AAAAAAAA", "b_red", 1)
    keys = joker_keys(probe)
    want = [k for k in a.only.split(",") if k] or keys
    jobs = [(a.src, k, t, keys) for k in want for t in range(a.trials)]
    rows = []
    with Pool(a.workers) as p, open(a.out, "w") as f:
        for r in p.imap_unordered(work, jobs, chunksize=4):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    bad = 0
    print(f"{len(want)} jokers x {a.trials} trials")
    for k in want:
        rs = by.get(k, [])
        err = [r for r in rs if "error" in r]
        ok = [r for r in rs if "error" not in r]
        inside = [r for r in ok if not r["label"]]
        miss = [r for r in inside if not r["exact"]]
        lab_miss = [r for r in ok if r["label"] and not r["exact"]]
        if miss or err:
            bad += 1
        flag = "MISS" if miss else ("ERR " if err else "ok  ")
        line = f"{flag} {k:24s} checked {len(inside):3d} exact {len(inside) - len(miss):3d}"
        if lab_miss or len(ok) > len(inside):
            line += f" | labeled {len(ok) - len(inside)} (off {len(lab_miss)})"
        if err:
            line += f" | errors {len(err)}: {err[0]['error'][:80]}"
        print(line)
        for r in miss[:2]:
            print(f"       {r['back']} {r['seed']} +{r['companions']} {r['edition']} {r['play']}: S {r['S']}, game {r['game']}"
                  f"  cards {r['cards']}")
    print(f"jokers with an unlabeled miss or an error: {bad} of {len(want)}")


if __name__ == "__main__":
    main()
