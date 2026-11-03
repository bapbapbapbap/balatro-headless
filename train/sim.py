"""The sim as the learner sees it: encoded observations and actions, plus progress counters
for rewards. One Sim = one Lua runtime (create it inside the process that uses it)."""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "..", "harness")
sys.path.insert(0, HARNESS)
from env import BalatroEnv  # noqa: E402

F = 32
ACT_WIDTH = {1: 9, 2: 18}     # action row width per encoding version (encode.lua)

PROGRESS = r"""
local H = ...
function H.progress()
  local b = G.GAME.blind
  -- Boss test as the game does it in end_round (state_events.lua): the blind just played was
  -- neither Small nor Big. Blind.boss is already cleared by the time the cash-out screen shows.
  local rb = G.GAME.round_resets.blind
  local boss = rb ~= nil and rb ~= G.P_BLINDS.bl_small and rb ~= G.P_BLINDS.bl_big
  return H.state_name(), boss, G.GAME.won and true or false,
         G.GAME.round_resets.ante, H.is_terminal(), G.GAME.chips or 0, (b and b.chips) or 0
end
-- The i-th encoded action (1-based) as readable text, e.g. "play 1,3,5", "buy 2", "move_joker 1>3".
-- Saved with every game next to the action index, so a replay can check each step is the same move.
function H.action_desc(i)
  local a = H._encoded_actions and H._encoded_actions[i]
  if not a then return "?" end
  local s = a.type
  if a.i then s = s .. " " .. a.i end
  if a.j then s = s .. ">" .. a.j end
  if a.cards and a.cards[1] then s = s .. " " .. table.concat(a.cards, ",") end
  if a.lead then s = s .. " lead " .. a.lead end
  if a.src then s = s .. " src " .. a.src end
  return s
end
"""


def parse(buf, act_width=9):
    v = np.array(buf.split(), dtype=np.float32)
    nt, na = int(v[0]), int(v[1])
    tok = v[2:2 + nt * (F + 2)].reshape(nt, F + 2)
    act = v[2 + nt * (F + 2):].reshape(na, act_width)
    return {"kind": tok[:, 0].astype(np.int8), "vocab": tok[:, 1].astype(np.int16),
            "feat": tok[:, 2:].astype(np.float16), "act": act.astype(np.int16)}


class Sim:
    def __init__(self, src, mods=None, fast=True, mode="subsets", enc=1):
        """mode: "subsets" (v1), "ordered" (v2: lead card, Death direction, boss reroll), or
        "ordered-capped" (ordered, plus env.lua's cap on consecutive joker drags).
        enc: encoding version, 1 (v1) or 2 (v2: fixed joker slots, hand table, scoring cards)."""
        lua_mode = "ordered" if mode == "ordered-capped" else mode
        self.env = BalatroEnv(src, mods=mods, fast=fast, mode=lua_mode)
        self.H = self.env.H
        self.H.drag_cap = mode == "ordered-capped"
        self.H.enc = enc
        self.enc, self.mode, self.act_width = enc, mode, ACT_WIDTH[enc]
        lua = self.env.lua
        lua.execute(open(os.path.join(HARNESS, "encode.lua")).read(), self.H)
        lua.execute(PROGRESS, self.H)
        self.vocab = self.H.vocab().split("\n")

    def reset(self, seed, back, stake):
        self.env.reset(seed, back, stake)
        self.steps = 0
        self._prog = {"blinds_beaten": 0, "bosses_beaten": 0, "won": False, "ante": 1, "phi": 0.0}
        self._last_state = self.H.state_name()
        self.route, self.desc = [], []      # every action taken, as encoded index and as text
        return self.observe()

    def observe(self):
        return parse(self.H.encode(), self.act_width)

    def step(self, index):
        """index: 0-based into the last observed action list."""
        self.route.append(int(index))
        self.desc.append(self.H.action_desc(index + 1))
        if self.H.step_index(index + 1) < 0:
            raise RuntimeError("game never settled")
        self.steps += 1
        state, boss, won, ante, done, chips, target = self.H.progress()
        p = dict(self._prog)
        if state == "ROUND_EVAL" and self._last_state != "ROUND_EVAL":
            p["blinds_beaten"] += 1
            p["bosses_beaten"] += 1 if boss else 0
        if won and not p["won"]:
            p["won"] = True
            if self._last_state != "ROUND_EVAL" and state != "ROUND_EVAL":
                p["blinds_beaten"] += 1
                p["bosses_beaten"] += 1
        p["ante"] = ante
        # Potential for shaping (see rewards.py): progress toward the blind being played, from what
        # the HUD shows (chips scored, blind target). Zero outside a blind and at the end.
        p["phi"] = min(chips / target, 1.0) if (state == "SELECTING_HAND" and target and target > 0 and not done) else 0.0
        self._last_state = state
        prev, self._prog = self._prog, p
        return prev, p, bool(done)

    @property
    def progress(self):
        return dict(self._prog)

    def final_hash(self):
        """Hash of everything the player sees at this point (the encoded observation and actions)."""
        return hashlib.sha1(self.H.encode().encode()).hexdigest()[:16]

    def replay(self, seed, back, stake, acts, desc=None):
        """Plays a saved game again, exactly as play_games does (observe before every action).
        Returns (progress, steps, final_hash, problems); problems is empty when every step was the
        same move as saved."""
        o = self.reset(seed, back, stake)
        problems, done = [], False
        for k, a in enumerate(acts):
            if done:
                problems.append(f"game ended at step {k}, {len(acts) - k} saved actions left")
                break
            if a >= len(o["act"]):
                problems.append(f"step {k}: action {a} not offered ({len(o['act'])} legal)")
                break
            if desc is not None and self.H.action_desc(a + 1) != desc[k]:
                problems.append(f"step {k}: saved '{desc[k]}', sim offers '{self.H.action_desc(a + 1)}'")
                break
            _, _, done = self.step(a)
            if not done:
                o = self.observe()
        return self.progress, self.steps, self.final_hash(), problems
