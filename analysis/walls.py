"""Where v4b's games end, and why: economy or scaling (diagnostic, 2026-11-13; question chud's, script Claude's).

Extends analysis/v4b_drop.py. Replays saved eval games on the sim (no model, no torch). Per game it records
every blind (ante, blind, target, jokers, money, hand levels, plays, score) as v4b_drop does, plus one record
per shop visit taken just before `end_shop`: money held, interest cap (vouchers raise it), joker count and slots,
rerolls and buys made that visit, the reroll cost, and every card still on the shelf with its set and cost.
Also vouchers redeemed. Each step is checked against the saved move text; a game that differs is marked and
stops there. Read-only on the run: writes only --out.

usage (WSL, venv active, from the repo):
  python3 analysis/walls.py --src ~/balatro/local/source \
      --games /mnt/c/Users/Jacobi/balatro-work/runs/v4b/games --rounds 1840,1860,1880 \
      --out /mnt/c/Users/Jacobi/balatro-work/walls.jsonl --workers 4
  python3 analysis/walls_report.py /mnt/c/Users/Jacobi/balatro-work/walls.jsonl
Cost: 1,800 games of v4b's r1840-1880 evals took 2,652 s on 4 workers while v4b trained.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(os.getcwd(), "train"))   # run from the repo with the script elsewhere
from sim import Sim  # noqa: E402

PROBE = r"""
local H = ...
local function keys(area)
  local out = {}
  if area and area.cards then
    for _, c in ipairs(area.cards) do out[#out + 1] = c.config.center.key end
  end
  return table.concat(out, ",")
end
function H.dbg_state()
  local g = G.GAME
  local b = g.blind
  local bk = (b and b.config and b.config.blind and b.config.blind.key) or ""
  local lim = (G.jokers and G.jokers.config and G.jokers.config.card_limit) or 0
  return keys(G.jokers), g.dollars or 0, g.chips or 0, (b and b.chips) or 0, bk,
         g.last_hand_played or "", g.round_resets.ante, H.state_name(),
         G.playing_cards and #G.playing_cards or 0, lim
end
function H.dbg_card(area, i)
  local a = G[area]
  if not (a and a.cards and a.cards[i]) then return "", "", 0 end
  local c = a.cards[i]
  return c.config.center.key, (c.ability and c.ability.set) or "", c.sell_cost or 0
end
function H.dbg_levels()
  local out = {}
  for name, h in pairs(G.GAME.hands) do
    if h.level and h.level > 1 then out[#out + 1] = name .. ":" .. h.level end
  end
  table.sort(out)
  return table.concat(out, ",")
end
function H.dbg_shop()
  local g = G.GAME
  local offers = {}
  if G.shop_jokers and G.shop_jokers.cards then
    for _, c in ipairs(G.shop_jokers.cards) do
      offers[#offers + 1] = c.config.center.key .. ":" .. ((c.ability and c.ability.set) or "") .. ":" .. (c.cost or 0)
    end
  end
  local vouchers = {}
  for k, v in pairs(g.used_vouchers or {}) do if v then vouchers[#vouchers + 1] = k end end
  table.sort(vouchers)
  local rc = (g.current_round and g.current_round.reroll_cost) or 0
  return table.concat(offers, ";"), g.interest_cap or 25, rc, table.concat(vouchers, ",")
end
"""

_sim = None


def _init(src):
    global _sim
    _sim = Sim(src, None, mode="ordered-capped", enc=2, build_phi=True, offer_values=True)
    _sim.env.lua.execute(PROBE, _sim.H)


def state(H):
    jk, dollars, chips, target, bk, last, ante, st, deck, lim = H.dbg_state()
    return {"jokers": [k for k in jk.split(",") if k], "dollars": float(dollars), "chips": float(chips),
            "target": float(target), "blind": bk, "last_hand": last, "ante": int(ante), "state": st,
            "deck": int(deck), "joker_slots": int(lim)}


def shop(H):
    offers, cap, rc, vouchers = H.dbg_shop()
    shelf = []
    for o in [x for x in offers.split(";") if x]:
        key, kind, cost = o.rsplit(":", 2)
        shelf.append([key, kind, float(cost)])
    return shelf, float(cap), float(rc), [v for v in vouchers.split(",") if v]


def replay(job):
    rnd, g = job
    sim, H = _sim, _sim.H
    sim.reset(g["seed"], g["back"], g["stake"])
    rec = {"round": rnd, "seed": g["seed"], "return": g["return"], "won": g["progress"]["won"],
           "ante_end": g["progress"]["ante"], "mismatch": None, "blinds": [], "sells": [], "buys": [],
           "shops": [], "end": None}
    cur = None
    visit = {"rerolls": 0, "buys": 0, "packs": 0, "redeems": 0, "sells": 0, "dollars_in": None}
    done = False
    for k, (a, d) in enumerate(zip(g["acts"], g["desc"])):
        if done:
            rec["mismatch"] = f"ended early at {k}"
            break
        o = sim.observe()
        if a >= len(o["act"]) or H.action_desc(a + 1) != d:
            rec["mismatch"] = f"step {k}: saved '{d}', sim '{H.action_desc(a + 1) if a < len(o['act']) else '-'}'"
            break
        verb = d.split()[0]
        before = state(H) if verb in ("play", "sell_joker", "buy", "buy_and_use", "end_shop") else None
        if verb == "sell_joker":
            key, _, cost = H.dbg_card("jokers", int(d.split()[1]))
            rec["sells"].append({"ante": before["ante"], "state": before["state"], "key": key, "cost": float(cost),
                                 "jokers_before": len(before["jokers"]), "dollars": before["dollars"]})
            visit["sells"] += 1
        elif verb in ("buy", "buy_and_use"):
            key, kind, _ = H.dbg_card("shop_jokers", int(d.split()[1]))
            rec["buys"].append({"ante": before["ante"], "key": key, "set": kind, "verb": verb,
                                "dollars": before["dollars"], "jokers_before": len(before["jokers"])})
            visit["buys"] += 1
        elif verb == "reroll":
            visit["rerolls"] += 1
        elif verb == "open_pack":
            visit["packs"] += 1
        elif verb == "redeem":
            visit["redeems"] += 1
        elif verb == "end_shop":
            shelf, cap, rc, vouchers = shop(H)
            rec["shops"].append({"ante": before["ante"], "dollars": before["dollars"], "dollars_in": visit["dollars_in"],
                                 "cap": cap, "reroll_cost": rc, "jokers": len(before["jokers"]),
                                 "joker_slots": before["joker_slots"], "shelf": shelf, "vouchers": vouchers,
                                 **{x: visit[x] for x in ("rerolls", "buys", "packs", "redeems", "sells")}})
        _, _, done = sim.step(a)
        if verb == "cash_out":
            visit = {"rerolls": 0, "buys": 0, "packs": 0, "redeems": 0, "sells": 0, "dollars_in": state(H)["dollars"]}
        elif verb == "select_blind":
            s = state(H)
            cur = {"ante": s["ante"], "blind": s["blind"], "target": s["target"], "jokers": s["jokers"],
                   "joker_slots": s["joker_slots"], "dollars": s["dollars"], "deck": s["deck"],
                   "levels": H.dbg_levels(), "plays": [], "discards": 0, "score": 0.0}
            rec["blinds"].append(cur)
        elif verb == "play" and cur is not None:
            s = state(H)
            cur["plays"].append([s["last_hand"], s["chips"] - before["chips"]])
            cur["score"] = s["chips"]
        elif verb == "discard" and cur is not None:
            cur["discards"] += 1
    s = state(H)
    _, cap, _, vouchers = shop(H)
    rec["end"] = {"blind": s["blind"], "chips": s["chips"], "target": s["target"], "ante": s["ante"],
                  "jokers": s["jokers"], "dollars": s["dollars"], "state": s["state"], "cap": cap,
                  "vouchers": vouchers, "levels": H.dbg_levels()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--games", required=True, help="the run's games/ folder")
    ap.add_argument("--rounds", default="1840,1860,1880")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for r in [int(x) for x in a.rounds.split(",")]:
        with gzip.open(os.path.join(a.games, f"r{r:05d}_eval.jsonl.gz"), "rt") as f:
            games = [json.loads(line) for line in f]
        if a.max_games:
            games = games[:a.max_games]
        jobs += [(r, g) for g in games]
    t0, n, bad = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.src,)) as pool, open(a.out, "w") as out:
        for rec in pool.imap_unordered(replay, jobs, chunksize=4):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            bad += rec["mismatch"] is not None
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} games, {bad} mismatched, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {n} games, {bad} mismatched, {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
