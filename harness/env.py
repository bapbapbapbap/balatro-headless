"""Gym-style environment over the headless game.

    env = BalatroEnv(src)
    obs = env.reset(seed="ABC123", back="b_red", stake=1)
    while not env.done:
        acts = env.legal_actions()          # list of hashable tuples
        obs = env.step(random.choice(acts))

An action tuple is (type, i, j, cards), plus (lead, src) for the ordered mode's lead-card plays
and Death directions: i is a 1-based position in the relevant area (hand, shop,
pack, jokers, consumables), j the destination of a move, cards the 1-based hand positions to select
(subsets mode only). Legality comes from the game's own button checks (see env.lua).

mode="controls" (default) uses the game's own controls: toggle a hand card, move hand cards and
jokers, switch the hand's sort, then play/discard/use/pick with what's selected.
mode="subsets" folds the selection into the action: one step per play, ~2000 actions per hand.

One BalatroEnv is one Lua runtime. Booting takes a few seconds, so reuse it across runs:
reset() starts a new run in the same runtime.
"""
import os

from run import make_runtime

HERE = os.path.dirname(os.path.abspath(__file__))


def to_py(x):
    """Lua table -> dict/list, recursively."""
    if hasattr(x, "items") and not isinstance(x, (str, bytes, dict)):
        d = {k: to_py(v) for k, v in x.items()}
        if d and all(isinstance(k, int) for k in d):
            return [d[i] for i in sorted(d)]
        return d
    return x


class BalatroEnv:
    def __init__(self, src, mods=None, fast=True, unlock_all=True, mode="controls"):
        self.lua, self.H = make_runtime(src, mods)
        self.H.start_up()
        if fast:
            self.H.strip_visuals()
        if unlock_all:
            self.H.unlock_all()
        self.H.mode = mode   # "controls" (the game's own controls) or "subsets" (see env.lua)
        self.lua.execute(open(os.path.join(HERE, "env.lua")).read(), self.H)
        self._acts = None
        self.frames = 0
        self.steps = 0

    def reset(self, seed, back="b_red", stake=1):
        G = self.lua.globals().G
        if G.STAGE == G.STAGES.RUN:
            G.delete_run(G)
        self.H._drags = 0
        n = self.H.start_run(back, stake, seed)
        if n < 0:
            raise RuntimeError("run never settled at start")
        self._acts = None
        self.frames, self.steps = n, 0
        return self.observe()

    @property
    def done(self):
        return bool(self.H.is_terminal())

    @property
    def won(self):
        return bool(self.lua.globals().G.GAME.won)

    def legal_actions(self):
        raw = self.H.legal_actions()
        self._acts = {}
        out = []
        for i in range(1, len(raw) + 1):
            a = raw[i]
            cards = tuple(a.cards[j] for j in range(1, len(a.cards) + 1)) if a.cards else ()
            key = (a.type, a.i, a.j, cards)
            if a.lead or a.src:                    # ordered mode: lead card / card Death copies
                key = key + (a.lead, a.src)
            self._acts[key] = a
            out.append(key)
        return out

    def step(self, action):
        if self._acts is None:
            self.legal_actions()
        a = self._acts.get(action)
        if a is None:
            raise ValueError(f"illegal action {action}")
        n = self.H.step(a)
        self._acts = None
        if n < 0:
            raise RuntimeError(f"game never settled after {action} (state {self.H.state_name()})")
        self.frames += n
        self.steps += 1
        return self.observe()

    def observe(self):
        return to_py(self.H.observe())

    def state(self):
        return self.H.state_name()
