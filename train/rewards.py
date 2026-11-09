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
    it is trying to do. Applied on top of any reward; turn off with --shaping 0. gamma is the step's own discount
    (step_discount), which keeps the term potential-based when the discount varies by step."""
    return gamma * cur["phi"] - prev["phi"]


def build_shaping(prev, cur, gamma):
    """Model v4's second potential-based term: gamma*phi_build(s') - phi_build(s), phi_build = how strong the
    build is (harness/build_phi.lua; docs/hypotheses.md, v4's definitions). A function of the visible state
    only, 0 at the end of a game, so like shaping() it leaves the optimal policy for the chosen reward
    unchanged (Ng et al. 1999). 0 when the sim doesn't compute it."""
    return gamma * cur.get("phi_build", 0.0) - prev.get("phi_build", 0.0)


def step_discount(prev, cur, args):
    """(gamma, lambda) for the step from prev to cur. Per action (--gamma, --lam) by default, as every
    run before v3. With --gamma-blind set (model v3, docs/hypotheses.md 2026-11-05 ~21:19 CST), the
    clock is game time: the step discounts by --gamma-blind (and lambda by --lam-blind) once per blind
    passed, beaten or skipped, and every other step (plays, discards, shops, packs, joker drags) is
    undiscounted. Not part of the ought: it changes how credit reaches earlier decisions, not what the
    bot is rewarded for."""
    gb = getattr(args, "gamma_blind", None)
    if gb is None:
        return args.gamma, args.lam
    n = cur.get("blinds_passed", 0) - prev.get("blinds_passed", 0)
    return gb ** n, getattr(args, "lam_blind", 0.95) ** n


REWARDS = {"blinds": blinds, "antes": antes, "win": win}
