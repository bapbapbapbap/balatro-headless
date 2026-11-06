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
ACT_WIDTH = {1: 9, 2: 18, 3: 18}     # action row width per encoding version (encode.lua)

PROGRESS = r"""
local H = ...
-- Outcomes are the game's own, not the harness's reading of it (2026-11-03):
--   won:     the game ran win_game(), its win routine (stickers, deck win, the win screen). Hooked
--            here once per runtime. G.GAME.won alone is not a win: end_round sets it at the end of
--            the ante-8 boss round on a loss too.
--   beaten:  the game's own marks, G.GAME.round_resets.blind_states[Small|Big|Boss] = 'Defeated',
--            set by end_round in the event that opens the cash-out screen ('Skipped' for a skip).
--   final_boss: G.GAME.won, the game's flag that the ante-8 boss round ended (won or lost).
if not H.win_hooked then
  H.win_hooked = true
  local game_win = win_game
  win_game = function(...)
    H.game_won = true
    return game_win(...)
  end
end
H.game_won = false
function H.blind_marks()
  local s = (G.GAME and G.GAME.round_resets and G.GAME.round_resets.blind_states) or {}
  return s.Small, s.Big, s.Boss
end
function H.progress()
  local b = G.GAME.blind
  -- legacy boss test (end_round's own: neither Small nor Big), kept only for the cross-check
  local rb = G.GAME.round_resets.blind
  local boss = rb ~= nil and rb ~= G.P_BLINDS.bl_small and rb ~= G.P_BLINDS.bl_big
  local legacy_won = (G.GAME.won and G.STATE ~= G.STATES.GAME_OVER) and true or false
  local s = G.GAME.round_resets.blind_states or {}
  return H.state_name(), boss, legacy_won,
         G.GAME.round_resets.ante, H.is_terminal(), G.GAME.chips or 0, (b and b.chips) or 0,
         G.GAME.won and true or false, H.game_won and true or false, s.Small, s.Big, s.Boss
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
    def __init__(self, src, mods=None, fast=True, mode="subsets", enc=1, save_mode=False, count_progress=False):
        """mode: "subsets" (v1), "ordered" (v2: lead card, Death direction, boss reroll), or
        "ordered-capped" (ordered, plus env.lua's cap on consecutive joker drags).
        enc: encoding version, 1 (v1) or 2 (v2: fixed joker slots, hand table, scoring cards)."""
        lua_mode = "ordered" if mode == "ordered-capped" else mode
        self.env = BalatroEnv(src, mods=mods, fast=fast, mode=lua_mode, save_mode=save_mode,
                              count_progress=count_progress)
        self.H = self.env.H
        self.H.drag_cap = mode == "ordered-capped"
        self.H.enc = enc
        lua = self.env.lua
        lua.execute(open(os.path.join(HARNESS, "encode.lua")).read(), self.H)
        lua.execute(PROGRESS, self.H)
        # action row width from encode.lua itself (it sets H.ACT_WIDTH for the encoding loaded)
        self.enc, self.mode, self.act_width = enc, mode, int(self.H.ACT_WIDTH)
        assert self.act_width == ACT_WIDTH[enc], (enc, self.act_width)
        self.vocab = self.H.vocab().split("\n")
        self.compare, self.mismatches = False, []     # tests/outcome_check.py: cross-check the old counters

    def reset(self, seed, back, stake, challenge=None):
        self.env.reset(seed, back, stake, challenge)
        self.steps = 0
        self._prog = {"blinds_beaten": 0, "bosses_beaten": 0, "won": False, "ante": 1, "phi": 0.0,
                      "final_boss": False, "near_win": False, "blinds_passed": 0}
        self.H.game_won = False
        self._marks = dict(zip(("Small", "Big", "Boss"), self.H.blind_marks()))
        self._marks_prev = dict(self._marks)
        self._legacy = {"blinds_beaten": 0, "bosses_beaten": 0, "won": False}
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
        (state, boss, legacy_won, ante, done, chips, target, final_boss, game_won,
         m_small, m_big, m_boss) = self.H.progress()
        p = dict(self._prog)
        marks = {"Small": m_small, "Big": m_big, "Boss": m_boss}
        newly = [k for k, v in marks.items() if v == "Defeated" and self._marks.get(k) != "Defeated"]
        self._marks = marks
        p["blinds_beaten"] += len(newly)
        # Blinds passed, beaten or skipped: the game-time clock for --gamma-blind (rewards.step_discount).
        p["blinds_passed"] += sum(1 for k, v in marks.items() if v in ("Defeated", "Skipped")
                                  and self._marks_prev.get(k) not in ("Defeated", "Skipped"))
        self._marks_prev = marks
        p["bosses_beaten"] += 1 if "Boss" in newly else 0
        won = bool(game_won)
        if won:
            p["won"] = True
        if self.compare:
            # the counters as they were before 2026-11-03 (corrected win test), for tests/outcome_check.py
            L = self._legacy
            if state == "ROUND_EVAL" and self._last_state != "ROUND_EVAL":
                L["blinds_beaten"] += 1
                L["bosses_beaten"] += 1 if boss else 0
            if legacy_won and not L["won"]:
                L["won"] = True
                if self._last_state != "ROUND_EVAL" and state != "ROUND_EVAL":
                    L["blinds_beaten"] += 1
                    L["bosses_beaten"] += 1
            if any(L[k] != p[k] for k in L):
                self.mismatches.append({"step": self.steps, "state": state, "game": {k: p[k] for k in L}, "legacy": dict(L)})
        if final_boss and not p.get("final_boss"):
            # The ante-8 boss round ended. Not won = a near win: reached the last fight and lost it.
            # Logged in every game so the win rate can be read against the reach-the-boss rate.
            p["final_boss"] = True
            if not won:
                p["near_win"] = True
                p["boss_chips"], p["boss_target"] = float(chips), float(target)
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

    def replay(self, seed, back, stake, acts, desc=None, challenge=None):
        """Plays a saved game again, exactly as play_games does (observe before every action).
        Returns (progress, steps, final_hash, problems); problems is empty when every step was the
        same move as saved."""
        o = self.reset(seed, back, stake, challenge)
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
