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
    def __init__(self, src, mods=None, fast=True, unlock_all=True, mode="controls", save_mode=False,
                 count_progress=False):
        """save_mode: a real save. The profile starts as the game makes a new one (nothing unlocked
        beyond the defaults), unlocks and stickers accrue as the game awards them, and runs count
        for progress (see boot.lua start_run). Published routes were played all-unlocked and won't
        replay in save mode."""
        self.lua, self.H = make_runtime(src, mods)
        self.H.start_up()
        if fast:
            self.H.strip_visuals()
        if unlock_all and not save_mode:
            self.H.unlock_all()
        # count_progress: runs count toward the profile, as in save mode, but the profile keeps
        # everything unlocked (to read what the game credits to a published, all-unlocked route).
        self.H.save_mode = bool(save_mode or count_progress)
        self.H.mode = mode   # "controls" (the game's own controls) or "subsets" (see env.lua)
        self.lua.execute(open(os.path.join(HERE, "env.lua")).read(), self.H)
        self._acts = None
        self.frames = 0
        self.steps = 0

    def reset(self, seed, back="b_red", stake=1, challenge=None):
        """challenge: a challenge id (see challenge_ids()); the run then uses the challenge's own deck,
        rules and stake 1, as the game's challenge menu starts it."""
        G = self.lua.globals().G
        if G.STAGE == G.STAGES.RUN:
            G.delete_run(G)
        self.H._drags = 0
        n = self.H.start_run(back, stake, seed, challenge)
        if n < 0:
            raise RuntimeError("run never settled at start")
        self._acts = None
        self.frames, self.steps = n, 0
        return self.observe()

    def challenge_ids(self):
        return list(to_py(self.H.challenge_ids()) or [])

    def joker_keys(self):
        return list(to_py(self.H.joker_keys()) or [])

    def clear_progress(self):
        self.H.clear_progress()

    def progress_record(self):
        r = to_py(self.H.progress_record()) or {}
        return {"jokers": dict(r.get("jokers") or {}), "decks": dict(r.get("decks") or {}),
                "challenges": list(r.get("challenges") or [])}

    def save_options(self):
        r = to_py(self.H.save_options()) or {}
        return {"decks": dict(r.get("decks") or {}), "challenges": list(r.get("challenges") or [])}

    def jokers_held(self):
        return list(to_py(self.H.jokers_held()) or [])

    def deck_keys(self):
        return list(to_py(self.H.deck_keys()) or [])

    @property
    def done(self):
        return bool(self.H.is_terminal())

    @property
    def won(self):
        # The game sets G.GAME.won at the end of the ante-8 boss round even when the boss wasn't
        # beaten (end_round, state_events.lua); a loss then goes to GAME_OVER. See train/sim.py.
        G = self.lua.globals().G
        return bool(G.GAME.won) and G.STATE != G.STATES.GAME_OVER

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
