"""train.py's command-line options: every setting a run can have. A run's own values are in its config
(configs/<run>.toml) and, as it ran, in runs/<run>/provenance.json."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rewards import REWARDS  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="game source (or <modded>/game with --mods)")
    ap.add_argument("--mods", help="Mods dir with Steamodded, for the modded game")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--out", help="output dir (default: train/runs/<run>)")
    ap.add_argument("--reward", choices=sorted(REWARDS), default="blinds")
    ap.add_argument("--shaping", type=float, default=1.0, help="potential-based shaping weight (0 = off)")
    ap.add_argument("--back", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--games-per-worker", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10**9)
    ap.add_argument("--hours", type=float, default=8.0, help="stop after this much wall time")
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many training games in total (0 = no limit)")
    ap.add_argument("--max-wins", type=int, default=0,
                    help="stop after this many won training games in total, i.e. runs that beat the ante 8 boss (0 = no limit)")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-at-games", default="",
                    help="comma-separated training-game counts: save a checkpoint games_<N>.pt at the first round "
                         "that reaches each (rounds are ~100 games, so within one round of N)")
    ap.add_argument("--stop-after-pins0", type=int, default=0,
                    help="MoE: stop cleanly once the pins-zero eval is done and every --save-at-games checkpoint is saved")
    ap.add_argument("--eval-at-pins-zero", type=int, default=1,
                    help="MoE: run an eval (and save pins0.pt) at the first round after every pin has faded to 0")
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--enc", type=int, choices=[1, 2, 3], default=1,
                    help="encoding: 1 = v1; 2 = fixed joker slots, hand table, scoring cards, HUD score; "
                         "3 = v2 plus deck and challenge tokens (v2 weights load into it)")
    ap.add_argument("--actions", choices=["subsets", "ordered", "ordered-capped"], default="subsets",
                    help="action list: subsets (v1) or ordered (lead card, Death direction, boss reroll)")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--gamma-blind", type=float, default=None,
                    help="model v3: discount once per blind passed (beaten or skipped), every other step "
                         "undiscounted; replaces --gamma/--lam (see rewards.step_discount). Unset = per action")
    ap.add_argument("--lam-blind", type=float, default=0.95,
                    help="with --gamma-blind: GAE lambda applied once per blind passed (1 inside a blind)")
    ap.add_argument("--lam-hand", type=float, default=1.0,
                    help="model v4b (with --phase-split): the table network's GAE lambda is also multiplied by this "
                         "at every play or discard (a hand clock); 1 = off, as in v4")
    ap.add_argument("--stop-sells", type=float, default=0.0,
                    help="stop at the first eval averaging more than this many joker sells per game (0 = off; v4b: 1.5)")
    ap.add_argument("--eval-temp", type=float, default=1.0,
                    help="sampling temperature for eval games only (1 = the policy as trained)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--explore-hands", type=float, default=0.0,
                    help="training games: share of decisions where a poker hand type on offer is picked uniformly "
                         "before the play (see explore_sample); 0 = off")
    ap.add_argument("--pin-hand", default=None,
                    help="strategy expert: poker hand type to pin, as the game names it (e.g. 'Full House'); "
                         "see apply_pin. Off by default")
    ap.add_argument("--pin-rate", type=float, default=1.0,
                    help="share of decisions with that hand on offer where the pin binds (1.0 = every time)")
    ap.add_argument("--backs", nargs="*", default=[],
                    help="decks to mix, one per game, picked from the game's seed (default: --back only)")
    ap.add_argument("--stakes", nargs="*", type=int, default=[],
                    help="stakes to mix, one per game, picked from the seed (default: --stake only)")
    ap.add_argument("--challenges", nargs="*", default=[],
                    help="challenge ids to mix in (tests/list_game.py prints them); stake 1, the challenge's deck")
    ap.add_argument("--challenge-share", type=float, default=0.25,
                    help="share of games that are challenges, when --challenges is given")
    ap.add_argument("--stop-ante", type=int, default=0,
                    help="early-game model: a game ends once this ante is reached (the boss before it beaten), "
                         "counted as 'reached', not as a win; 0 = off")
    ap.add_argument("--best-by", choices=["blinds", "reached", "wins"], default="blinds",
                    help="what picks best.pt from the evals: blinds per game, or share of games reaching --stop-ante")
    ap.add_argument("--moe", action="store_true",
                    help="joint training of the early-game model and the 8 hand experts (model.MoE; see play_games)")
    ap.add_argument("--moe-init", help="MoE: plain Net file the trunk and every head start from (e.g. weights/balatro-bot-0.2.pt)")
    ap.add_argument("--moe-early", help="MoE: a solo early-game run's file; trunk and early head start from it instead")
    ap.add_argument("--moe-experts", nargs="*", default=[],
                    help="MoE: 'Hand=FILE' for experts trained solo first (optional; others start from --moe-init)")
    ap.add_argument("--moe-handoff", type=int, default=4, help="MoE: the ante at which the expert takes over")
    ap.add_argument("--moe-random", type=float, default=0.3,
                    help="MoE: share of training-game handoffs sent to an expert picked at random (load balancing)")
    ap.add_argument("--moe-pin", nargs="*", default=["Pair=0.5"],
                    help="MoE: experts' pin rates before annealing, 'Hand=rate' (others 1.0)")
    ap.add_argument("--moe-early-pin", default=None,
                    help="MoE: 'Hand=rate' pin on the early model: where a play of that hand or a higher one is on "
                         "offer, only those plays are allowed with this rate, fading to 0 with the expert pins "
                         "(chud 2026-11-04: Full House, which it passes over for Two Pair)")
    ap.add_argument("--moe-min-plays", nargs="*", default=[],
                    help="MoE: 'Hand=N': the router picks that hand's expert only after N plays of it in antes 1-3")
    ap.add_argument("--moe-min-games", type=int, default=5000,
                    help="MoE: an expert's pin doesn't finish fading until the expert has had this many training games")
    ap.add_argument("--moe-anneal-games", type=int, default=100000,
                    help="MoE: pins fall linearly to 0 over this many training games of the joint run")
    ap.add_argument("--moe-train-early", type=int, default=1,
                    help="MoE: 1 = the early head trains (on returns up to the handoff with --moe-cut); 0 = its "
                         "head is left as loaded (the shared trunk still trains)")
    ap.add_argument("--moe-cut", type=int, default=1,
                    help="MoE: 1 = the early head's returns end at the handoff (it is judged on reaching ante 4); "
                         "0 = it learns from the whole run's return")
    ap.add_argument("--phase-split", action="store_true",
                    help="model v4: two networks, table (SELECTING_HAND) and shop (everything else), each with its "
                         "own critic, credit routed per network (train/credit.py)")
    ap.add_argument("--build-shaping", type=float, default=0.0,
                    help="v4: weight of the build potential's shaping term, gamma*phi_build' - phi_build "
                         "(harness/build_phi.lua; 0 = off)")
    ap.add_argument("--offer-values", action="store_true",
                    help="v4: each shop offer's action row carries phi_build after taking it minus now")
    ap.add_argument("--curriculum", type=float, default=0.0,
                    help="v4: share of training games that start at the first shop of ante 4, 5 or 6 of a saved game, "
                         "replayed from its seed (out of every stat)")
    ap.add_argument("--curriculum-published", type=int, default=300, help="v4: published wins used as curriculum starts")
    ap.add_argument("--curriculum-keep", type=int, default=600, help="v4: this run's own starts kept per ante")
    ap.add_argument("--shop-ent", default=None,
                    help="v4: the shop network's entropy coefficient by training decisions (both networks), e.g. "
                         "'0.02,0.006@1.5M,0.001@4M' (linear between; default: --ent)")
    ap.add_argument("--init-weights",
                    help="start from a released weights file (weights/*.pt) with a fresh optimizer, at round 0")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.9,
                    help="cap PyTorch's GPU memory at this share of the card (0 = no cap). Past the card's size WSL "
                         "spills to shared system memory and updates slow down; under the cap the allocator reuses its cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--reset-lr", action="store_true",
                    help="on resume, go back to --lr instead of the (possibly halved) rate in the checkpoint")
    # safety
    ap.add_argument("--max-kl", type=float, default=0.2, help="reject an update that moves the policy further than this")
    ap.add_argument("--overlap", type=int, default=1,
                    help="1 = workers play the next round while the learner updates (games use weights one update old)")
    ap.add_argument("--target-kl", type=float, default=0.04,
                    help="stop an update early once its KL passes 1.5x this (0 = off); pre-SIL v1 ran at ~0.035")
    ap.add_argument("--max-strikes", type=int, default=3, help="consecutive rejected updates before stopping")
    ap.add_argument("--round-timeout", type=float, default=1200, help="seconds before a round counts as hung")
    ap.add_argument("--min-mem-gb", type=float, default=1.0, help="stop cleanly below this much free memory")
    ap.add_argument("--worker-mem-gb", type=float, default=0.45, help="memory budget per worker, for sizing")
    ap.add_argument("--min-disk-gb", type=float, default=2.0)
    ap.add_argument("--stall-evals", type=int, default=0,
                    help="stop after this many evals in a row without a new best eval (0 = off); when set, "
                         "it replaces --stall-hours (which watches the noisier training average)")
    ap.add_argument("--stall-hours", type=float, default=3.0,
                    help="stop if the 10-round average training score hasn't improved in this long")
    ap.add_argument("--inject", help="test the safety checks: nan:<it>,crash:<it>")
    # behavior cloning before RL: imitate the moves in saved games (e.g. v1's, converted to v2)
    ap.add_argument("--bc", nargs="*", default=[], help="saved games (jsonl) to imitate before RL starts")
    ap.add_argument("--bc-epochs", type=float, default=2.0, help="passes over the --bc games")
    ap.add_argument("--bc-chunk", type=int, default=0, help="games replayed per step (default: 4 per worker)")
    # saved games and self-imitation
    ap.add_argument("--save-games", choices=["all", "wins", "none"], default="all",
                    help="write every game exactly (seed + actions) to <out>/games/; wins also go to wins.jsonl")
    ap.add_argument("--sil-buffer", type=int, default=0,
                    help="self-imitation: keep the N highest-return training games (0 = off; off by default until an A/B "
                         "test at full size, see CHANGELOG 0.6.0; 64 is the tested setting)")
    ap.add_argument("--sil-games", type=int, default=8, help="saved games replayed into each update")
    ap.add_argument("--sil-weight", type=float, default=0.2,
                    help="1.0 pushed KL past the guard and entropy 2.35 -> 1.9 within 10 rounds on v1 (rounds 307-320)")
    ap.add_argument("--sil-value", type=float, default=0.01, help="weight of the critic's self-imitation loss")
    ap.add_argument("--sil-steps", type=int, default=8, help="self-imitation minibatches per update")
    ap.add_argument("--sil-seed", nargs="*", default=[],
                    help="wins.jsonl files to fill the self-imitation buffer from at start (training wins on this "
                         "deck and stake only; eval-seed wins are left out so the eval stays clean)")
    ap.add_argument("--sil-clip", type=float, default=1.0,
                    help="cap on a saved move's pull, in PPO advantage standard deviations")
    # second drive
    ap.add_argument("--mirror-dir", help="also append every win to <dir>/wins.jsonl (e.g. an external drive)")
    ap.add_argument("--spill-dir", help="when the main drive runs low, write saved games and checkpoints here")
    ap.add_argument("--spill-at-gb", type=float, default=50.0, help="free space on the main drive that triggers spilling")
    # seed retries: the solver side. Lost seeds are kept by the latest ante they reached (1-8) and
    # replayed with fresh sampling. Each round's games are split across 9 arms (fresh seeds, and
    # retries of seeds whose best was ante 1..8) by Thompson sampling on wins per game, so effort
    # goes where it actually turns into beaten seeds.
    ap.add_argument("--retry", type=int, default=1, help="1 = split games across the arms, 0 = fresh seeds only")
    ap.add_argument("--fresh-min", type=float, default=0.5, help="share of each round always on fresh seeds")
    ap.add_argument("--arm-prior", type=float, default=100.0,
                    help="prior losses per arm, Beta(1, prior): how many games before an arm's record dominates")
    ap.add_argument("--retry-max", type=int, default=30, help="attempts per seed before it is dropped")
    ap.add_argument("--retry-pool", type=int, default=500, help="most seeds waiting per ante")
    return ap
