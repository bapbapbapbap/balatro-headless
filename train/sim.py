"""The sim as the learner sees it: encoded observations and actions, plus progress counters
for rewards. One Sim = one Lua runtime (create it inside the process that uses it)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "..", "harness")
sys.path.insert(0, HARNESS)
from env import BalatroEnv  # noqa: E402

F, ACT = 32, 9

PROGRESS = r"""
local H = ...
function H.progress()
  local b = G.GAME.blind
  return H.state_name(), (b and b.boss) and true or false, G.GAME.won and true or false,
         G.GAME.round_resets.ante, H.is_terminal(), G.GAME.chips or 0, (b and b.chips) or 0
end
"""


def parse(buf):
    v = np.array(buf.split(), dtype=np.float32)
    nt, na = int(v[0]), int(v[1])
    tok = v[2:2 + nt * (F + 2)].reshape(nt, F + 2)
    act = v[2 + nt * (F + 2):].reshape(na, ACT)
    return {"kind": tok[:, 0].astype(np.int8), "vocab": tok[:, 1].astype(np.int16),
            "feat": tok[:, 2:].astype(np.float16), "act": act.astype(np.int16)}


class Sim:
    def __init__(self, src, mods=None, fast=True):
        self.env = BalatroEnv(src, mods=mods, fast=fast, mode="subsets")
        self.H = self.env.H
        lua = self.env.lua
        lua.execute(open(os.path.join(HARNESS, "encode.lua")).read(), self.H)
        lua.execute(PROGRESS, self.H)
        self.vocab = self.H.vocab().split("\n")

    def reset(self, seed, back, stake):
        self.env.reset(seed, back, stake)
        self.steps = 0
        self._prog = {"blinds_beaten": 0, "bosses_beaten": 0, "won": False, "ante": 1, "phi": 0.0}
        self._last_state = self.H.state_name()
        return self.observe()

    def observe(self):
        return parse(self.H.encode())

    def step(self, index):
        """index: 0-based into the last observed action list."""
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
