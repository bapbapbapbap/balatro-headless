"""Replays saved games and logs strategy-relevant events at every step.

Each game is replayed with the same legal-action list the bot saw (H.legal_actions, same order as
the encoder), and checked move-by-move against the saved text. Output: one JSON line per game.

usage: python3 replay_events.py --src DIR --out events.jsonl [--jobs 2] files...
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys

TRAIN = os.environ.get("BH_TRAIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
sys.path.insert(0, TRAIN)

LUA = r"""
local H = ...
local function keyof(c) return c and c.config and c.config.center and c.config.center.key end
local function jokers()
  local out = {}
  for _, j in ipairs(G.jokers.cards) do out[#out + 1] = keyof(j) end
  return out
end
local function blind()
  local b = G.GAME.blind
  if not b or not b.config or not b.config.blind then return nil end
  return { key = b.config.blind.key, name = b.name, boss = b.boss and true or false, disabled = b.disabled and true or false }
end
local function levels()
  local out = {}
  for k, v in pairs(G.GAME.hands) do out[k] = v.level end
  return out
end
local function planet_hand(c)
  local cn = c and c.ability and c.ability.consumeable
  return c and c.ability and c.ability.set == "Planet" and cn and cn.hand_type or nil
end

function H.pre(i)
  local a = H._encoded_actions[i]
  local t = a.type
  local ev = { t = t, state = H.state_name(), ante = G.GAME.round_resets.ante, money = G.GAME.dollars }
  if t == "play" or t == "discard" then
    local cards = {}
    for _, hi in ipairs(a.cards or {}) do cards[#cards + 1] = G.hand.cards[hi] end
    local text, _, _, scoring = G.FUNCS.get_poker_hand_info(cards)
    ev.hand = text; ev.n = #cards
    local deb = 0
    for _, c in ipairs(cards) do if c.debuff then deb = deb + 1 end end
    ev.debuffed = deb
    ev.scoring = scoring and #scoring or 0
    if t == "play" then
      ev.level = G.GAME.hands[text] and G.GAME.hands[text].level or 0
      ev.chips_before = G.GAME.chips
      ev.target = G.GAME.blind and G.GAME.blind.chips or 0
      ev.blind = blind()
      ev.most_played = G.GAME.current_round.most_played_poker_hand
      ev.played_this_round = G.GAME.hands[text] and G.GAME.hands[text].played_this_round or 0
      ev.hands_left = G.GAME.current_round.hands_left
      ev.jokers = jokers()
      local best, besttext = -1, nil
      for _, b in ipairs(H._encoded_actions) do
        if b.type == "play" then
          local cs = {}
          for _, hi in ipairs(b.cards or {}) do cs[#cs + 1] = G.hand.cards[hi] end
          local tx = G.FUNCS.get_poker_hand_info(cs)
          local h = G.GAME.hands[tx]
          if h and h.chips * h.mult > best then best = h.chips * h.mult; besttext = tx end
        end
      end
      ev.best_hud_hand = besttext
    end
  elseif t == "buy" or t == "buy_and_use" then
    local c = G.shop_jokers.cards[a.i]
    ev.key = keyof(c); ev.set = c.ability.set; ev.cost = c.cost; ev.planet = planet_hand(c)
    ev.jokers = jokers()
  elseif t == "redeem" then ev.key = keyof(G.shop_vouchers.cards[a.i])
  elseif t == "open_pack" then ev.key = keyof(G.shop_booster.cards[a.i])
  elseif t == "pick" then
    local c = G.pack_cards.cards[a.i]
    ev.key = keyof(c); ev.set = c.ability.set; ev.planet = planet_hand(c)
    local opts = {}
    for _, o in ipairs(G.pack_cards.cards) do opts[#opts + 1] = { key = keyof(o), set = o.ability.set, planet = planet_hand(o) } end
    ev.options = opts
    ev.jokers = jokers()
    ev.levels = levels()
  elseif t == "use_consumable" or t == "sell_consumable" then
    local c = G.consumeables.cards[a.i]; ev.key = keyof(c); ev.planet = planet_hand(c)
  elseif t == "sell_joker" then ev.key = keyof(G.jokers.cards[a.i])
  elseif t == "select_blind" or t == "skip_blind" then
    ev.on_deck = G.GAME.blind_on_deck
    ev.boss_choice = G.GAME.round_resets.blind_choices.Boss
  elseif t == "reroll_boss" then ev.boss_choice = G.GAME.round_resets.blind_choices.Boss
  end
  return ev
end

function H.post()
  return { chips = G.GAME.chips, money = G.GAME.dollars, state = H.state_name(), ante = G.GAME.round_resets.ante }
end

function H.final()
  local hp = {}
  for k, v in pairs(G.GAME.hands) do hp[k] = { level = v.level, played = v.played } end
  return { jokers = jokers(), hands = hp, money = G.GAME.dollars, won = G.GAME.won and true or false,
           ante = G.GAME.round_resets.ante, vouchers = (function() local o = {}; for k, v in pairs(G.GAME.used_vouchers or {}) do if v then o[#o + 1] = k end end; table.sort(o); return o end)() }
end
"""


def to_py(x):
    if hasattr(x, "items") and not isinstance(x, (str, bytes, dict)):
        d = {k: to_py(v) for k, v in x.items()}
        if d and all(isinstance(k, int) for k in d):
            return [d[i] for i in sorted(d)]
        return d
    return x


_sim = {}


def get_sim(src, mode, enc):
    k = (mode, enc)
    if k not in _sim:
        from sim import Sim
        s = Sim(src, fast=True, mode=mode, enc=enc)
        s.env.lua.execute(LUA, s.H)
        _sim[k] = s
    return _sim[k]


def replay(args):
    src, r = args
    s = get_sim(src, r.get("actions", "ordered"), r.get("enc", 2))
    H = s.H
    s.reset(r["seed"], r["back"], r["stake"])
    evs, problem = [], None
    for k, a in enumerate(r["acts"]):
        s.observe()
        d = H.action_desc(a + 1)
        if d != r["desc"][k]:
            problem = f"step {k}: saved '{r['desc'][k]}', sim '{d}'"
            break
        t = d.split()[0]
        if t in ("move_joker", "move_hand", "toggle", "sort_rank", "sort_suit"):
            s.step(a)
            evs.append({"t": t})
            continue
        pre = to_py(H.pre(a + 1))
        _, prog, done = s.step(a)
        post = to_py(H.post())
        if pre["t"] == "play":
            pre["gained"] = post["chips"] - pre["chips_before"]
            pre["cleared"] = post["state"] != "SELECTING_HAND" or post["chips"] >= pre["target"]
        if pre["t"] in ("buy", "buy_and_use", "redeem", "open_pack", "reroll", "reroll_boss", "sell_joker", "sell_consumable"):
            pre["money_after"] = post["money"]
        evs.append(pre)
        if done:
            break
    return {"seed": r["seed"], "mode": r.get("mode"), "retry": r.get("retry"), "iteration": r.get("iteration"),
            "won_saved": r["progress"]["won"], "blinds": r["progress"]["blinds_beaten"], "ante": r["progress"]["ante"],
            "n": r["n"], "problem": problem, "events": evs, "final": to_py(H.final())}


def records(paths):
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--shard", default="0/1", help="I/N: only records with index %% N == I (after --skip-eval)")
    a = ap.parse_args()
    recs = [r for r in records(a.files) if not (a.skip_eval and r.get("mode") == "eval")]
    si, sn = map(int, a.shard.split("/"))
    recs = [r for k, r in enumerate(recs) if k % sn == si]
    if a.limit:
        recs = recs[:a.limit]
    done = 0
    with open(a.out, "w") as f, mp.Pool(a.jobs) as pool:
        for res in pool.imap_unordered(replay, [(a.src, r) for r in recs], chunksize=1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(recs)}", flush=True)
    print(f"done {done}")


if __name__ == "__main__":
    main()
