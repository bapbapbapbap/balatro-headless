"""Rewards: what the bot is trying to maximize. This is the ought, and it's the user's choice;
nothing else in the trainer prefers one outcome over another. Each is a function of the change
between two consecutive progress snapshots (see sim.Sim.progress). Recorded in every checkpoint.
"""


def blinds(prev, cur):
    """+1 per blind beaten. Placeholder for the smoke test."""
    return float(cur["blinds_beaten"] - prev["blinds_beaten"])


def antes(prev, cur):
    """+1 per ante cleared (beating a boss blind)."""
    return float(cur["bosses_beaten"] - prev["bosses_beaten"])


def win(prev, cur):
    """+1 for beating the ante 8 boss, nothing else."""
    return float(cur["won"] and not prev["won"])


def shaping(prev, cur, gamma):
    """Potential-based shaping (Ng, Harada & Russell, 1999): gamma*phi(s') - phi(s), with
    phi = fraction of the current blind's chips scored (0 outside a blind and at the end).
    It adds a learning signal for progress inside a blind and takes it back in full when the
    blind ends, so over a game it sums to ~0 and provably leaves the optimal policy for the chosen
    reward unchanged. It is not part of the ought: it changes how fast the bot can learn, not what
    it is trying to do. Applied on top of any reward; turn off with --shaping 0."""
    return gamma * cur["phi"] - prev["phi"]


REWARDS = {"blinds": blinds, "antes": antes, "win": win}
