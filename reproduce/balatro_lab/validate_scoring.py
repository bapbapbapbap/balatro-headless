"""Does Balatro Lab score hands the way the game does?

Ours (Claude's, reviewed by chud). Calls Balatro Lab (KvashaIhor/balatro-lab, MIT, commit 3e09d47e):
balatro.bots.estimate, balatro.bots.chancy and balatro.run.Run.play_hand, through state.py.

Plays seeds in the harness (the game's own code). Before every hand it builds Balatro Lab's Run from the game
state (state.py) and asks Balatro Lab what the hand will score with no chance coming off; then it plays the hand
and reads what the game scored. Hands where a chance can change the score are kept apart.

The policy only has to visit many states: play the hand Balatro Lab scores highest (never discard), buy any
affordable joker, skip packs and consumables, never skip blinds.

    python3 validate_scoring.py --lab <balatro-lab clone> --src <game source> --seeds 50 --out scoring.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "harness"))
sys.path.insert(0, HERE)

from env import BalatroEnv  # noqa: E402  (the harness loads lupa, vendor/ first: LuaJIT 2.0.5)

LUAJIT = sys.modules["lupa.luajit20"].LuaRuntime().eval("jit.version")
if LUAJIT != "LuaJIT 2.0.5" and os.environ.get("ALLOW_STOCK_LUAJIT") != "1":
    # Another LuaJIT orders pairs() differently: To Do List's and the Orbital Tag's hands then differ from the game's
    raise SystemExit(f"{LUAJIT}, not the game's LuaJIT 2.0.5: build or link harness/vendor (README), "
                     "or set ALLOW_STOCK_LUAJIT=1 to accept those two differences")
from state import lab_run, load_lab  # noqa: E402

CHANCY_JOKERS = {"Misprint", "Bloodstone", "Space Joker", "Oops! All 6s"}
CHANCY_BOSSES = {"The Hook"}   # discards 2 random held cards before scoring


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rng", type=int, default=90210)   # Balatro Lab's own evaluation seed set
    ap.add_argument("--out", default="scoring.jsonl")
    a = ap.parse_args()
    load_lab(a.lab)
    from balatro.bots import estimate, chancy
    from balatro.hands import evaluate

    chars = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
    r = random.Random(a.rng)
    seeds = ["".join(r.choice(chars) for _ in range(8)) for _ in range(a.seeds)]
    env = BalatroEnv(a.src, mode="subsets")
    G = None
    pick = random.Random(0)
    with open(a.out, "w") as out:
        for seed in seeds:
            env.reset(seed=seed, back="b_red", stake=1)
            G = env.lua.globals().G
            steps = 0
            while not env.done and steps < 3000:
                steps += 1
                acts = env.legal_actions()
                types = {x[0] for x in acts}
                if "play" in types:
                    try:
                        run = lab_run(G)
                        plays = [x for x in acts if x[0] == "play"]
                        best, best_v = plays[0], -1
                        for x in plays:
                            played = [run.hand[i - 1] for i in x[3]]
                            held = [c for k, c in enumerate(run.hand) if k + 1 not in x[3]]
                            v = estimate(run, played, held, sure=True)
                            if v > best_v:
                                best, best_v, best_played = x, v, played
                        sim_run = copy.deepcopy(run)
                        sim_run.round_score = 0
                        sim_run.blind.chips = 10 ** 18          # so the copy never ends the round and draws nothing
                        sim = sim_run.play_hand([i - 1 for i in best[3]])
                        rec = {"seed": seed, "ante": run.ante, "hand": evaluate(best_played, run.rules).name,
                               "predicted": best_v, "sim": sim, "jokers": [j.name for j in run.jokers],
                               "boss": run.blind.boss.name if run.blind.boss else None,
                               "chancy": bool(chancy(run, best_played)
                                              or any(j.name in CHANCY_JOKERS for j in run.jokers)
                                              or (run.blind.boss is not None and run.blind.boss.name in CHANCY_BOSSES)),
                               "unmapped": run.unmapped}
                    except Exception:
                        best, rec = next(x for x in acts if x[0] == "play"), {"seed": seed, "error": traceback.format_exc(limit=3)}
                    need = float(G.GAME.blind.chips) - float(G.GAME.chips)
                    hands_left = int(G.GAME.current_round.hands_left)
                    discards = [x for x in acts if x[0] == "discard"]
                    if "error" not in rec and discards and best_v * hands_left < need:
                        keep = set(best[3])                       # discard up to 5 cards outside the best play,
                        rest = sorted((i for i in range(1, len(run.hand) + 1) if i not in keep),   # lowest first
                                      key=lambda i: run.hand[i - 1].rank)[:5]
                        d = next((x for x in discards if set(x[3]) == set(rest)), None)
                        if rest and d:
                            env.step(d)
                            continue
                    before = float(G.GAME.chips)
                    env.step(best)
                    rec["actual"] = float(G.GAME.chips) - before
                    out.write(json.dumps(rec) + "\n")
                    continue
                buys = [x for x in acts if x[0] == "buy" and G.shop_jokers.cards[x[1]].ability.set == "Joker"]
                for t in ("select_blind", "cash_out"):
                    if t in types:
                        env.step(next(x for x in acts if x[0] == t))
                        break
                else:
                    if buys:
                        env.step(pick.choice(buys))
                    elif "skip_pack" in types:
                        env.step(next(x for x in acts if x[0] == "skip_pack"))
                    elif "end_shop" in types:
                        env.step(next(x for x in acts if x[0] == "end_shop"))
                    else:
                        env.step(pick.choice(acts))
            print(f"{seed}: ante {int(G.GAME.round_resets.ante)}, {steps} steps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
