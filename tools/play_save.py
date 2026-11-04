"""Plays one Balatro save from a new profile, run after run, and writes every run (wins and losses,
in order) to one file in the route format of wins/routes.jsonl. That file is the proof of 100%:
`train/verify_games.py --save FILE` replays it in order on a new profile and checks the game's own
record at the end (docs/proof.md).

Each run is played by the trainer's own game loop (train.py play_games, eval sampling, so a run is
reproducible from its seed). Next run: a challenge every 5th run while an unlocked one is still
unbeaten; otherwise the unlocked deck with the lowest highest-stake-won, at the next stake
(Claude's schedule; any choice works, as long as it was selectable then). A relaunch replays the
file first, so the profile is rebuilt exactly, then continues.

usage: python3 tools/play_save.py --src ~/balatro/local/source --weights W.pt --save saves/save1.jsonl
                                  [--runs 100]
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
import torch  # noqa: E402
from model import MoE, Net  # noqa: E402
from rewards import REWARDS  # noqa: E402
from sim import Sim  # noqa: E402
from train import load_weights_into, play_games, rand_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True, help="training checkpoint or released weights (plain or MoE)")
    ap.add_argument("--save", required=True)
    ap.add_argument("--runs", type=int, default=100, help="runs to play this launch")
    a = ap.parse_args()

    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {})
    enc, actions = arch.get("enc", 1), arch.get("actions", "subsets")
    sim = Sim(a.src, mode=actions, enc=enc, save_mode=True)
    model = (MoE if arch.get("moe") else Net)(len(sim.vocab), d=arch.get("d", 96), layers=arch.get("layers", 2),
                                              heads=arch.get("heads", 4), enc=enc)
    load_weights_into(model, ck["model"] if "model" in ck else ck["state_dict"], ck["vocab"], sim.vocab)
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}

    played = []
    if os.path.exists(a.save):
        played = [json.loads(l) for l in open(a.save) if l.strip()]
        for i, r in enumerate(played):                       # rebuild the profile exactly
            prog, n, final, problems = sim.replay(r["seed"], r["back"], r["stake"], r["acts"], r["desc"], r.get("challenge"))
            if problems or n != r["n"] or final != r["final"]:
                raise SystemExit(f"run {i + 1} of {a.save} doesn't replay ({problems[:1]}); not continuing")
        print(f"rebuilt the profile from {len(played)} runs")
    os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
    rng = random.Random(f"save-{os.path.basename(a.save)}-{len(played)}")
    ns = argparse.Namespace(seed=0, pin_hand=None, pin_rate=1.0, moe=bool(arch.get("moe")), moe_handoff=4,
                            moe_random=0.0, moe_pin=["Pair=0.5"], moe_train_early=0, explore_hands=0.0,
                            max_steps=1500, shaping=1.0, gamma=0.995, stop_ante=0, verify_wins=False,
                            backs=[], stakes=[], challenge_share=1.01)
    for _ in range(a.runs):
        opts, rec = sim.env.save_options(), sim.env.progress_record()
        open_ch = [c for c in opts["challenges"] if c not in rec["challenges"]]
        if open_ch and len(played) % 5 == 4:
            ns.back, ns.stake, ns.challenges = "b_challenge", 1, [open_ch[0]]
        else:
            deck = min(sorted(opts["decks"]), key=lambda d: rec["decks"].get(d, 0))
            ns.back, ns.stake, ns.challenges = deck, opts["decks"][deck], []
        seed = rand_seed(rng)
        ep = play_games(sim, model, REWARDS["blinds"], rng, ns, 0, (state, [seed], "eval", 0, 0), set())[0]
        r = {"seed": ep["seed"], "back": ep["back"], "stake": ep["stake"], "challenge": ep["challenge"],
             "actions": actions, "enc": enc, "acts": ep["acts"], "desc": ep["desc"], "final": ep["final"],
             "n": ep["n"], "progress": ep["progress"], "mode": "save", "run": len(played) + 1,
             "jokers_held": ep.get("jokers_held"), "found_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "weights": os.path.basename(a.weights)}
        with open(a.save, "a") as f:
            f.write(json.dumps(r) + "\n")
        played.append(r)
        print(f"run {r['run']}: {r['challenge'] or r['back']} stake {r['stake']} seed {seed}: "
              f"{'WON' if r['progress']['won'] else 'lost'} at ante {r['progress']['ante']}", flush=True)


if __name__ == "__main__":
    main()
