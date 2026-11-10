# Hypotheses: Model v3

Sections moved verbatim on 2026-11-09 from `docs/hypotheses.md`, in their original order. The registration
timestamps are the commits that added them to that file; the file as it stood, whole, is [here](https://github.com/bapbapbapbap/balatro-headless/blob/e3778d071534439425b2eca5eba2fff1b26002f1/docs/hypotheses.md).
New sections about this run go at the end of this file, dated, and get a row in [the index](hypotheses.md).

## Added 2026-11-05 ~21:19 CST: model v3, registered before any of its code exists (section J)

Target set by chud (21:16): **4% of fresh seeds won.** Registered before the code for v3 is written
and before anything is trained.

### Diagnosis (the bet)
**The run is mostly decided by the build, and the trainer can't credit the build.**
- From the same ante-4 position, the six experts land within 0.24 blinds of each other, and picking
  the best expert per seed after the fact would win only 86 of 1,368 positions (6.3%). How a run ends
  is mostly fixed before the handoff, by what was bought in antes 1–4.
- The bot builds badly: one hand (Two Pair ~65%) whatever it holds, money held far past the interest
  cap ($122 median at ante 8), no Full House, ~20% of plays under some bosses score nothing, and 352
  of 600 eval deaths in antes 4–6, where blind targets outgrow a build that didn't scale.
- **Why the trainer can't credit it.** `gamma` = 0.995 is applied per *action*. *(Corrected 22:15 CST,
  before any code ran a game: the first version said a won game is ~700 actions and the win reaches
  an ante-2 shop at ~5%. 700 was v1's first win. moe2's published wins take 196–327 actions, so the
  win reaches an ante-2 shop at ~0.995^200 ≈ 37%. The discount alone is a weaker cause than first
  stated; the λ horizon below is the main one.)* Free moves (joker drags,
  card toggles) use up that horizon while nothing happens in the game. `lam` = 0.95 per action gives
  GAE an effective horizon of 1/(1 − 0.995·0.95) ≈ 18 actions, so any credit to a shop decision has to
  come from the critic, and a critic that has never seen a good build can't value one.

### What changes (v3)
Only the operationalization of the same ought. Nothing encodes strategy.
1. **Discount on game time, not actions.** The discount applies once per blind passed (beaten or
   skipped), `--gamma-blind 0.95`; every other step is undiscounted. Shops and packs sit between
   blinds and are undiscounted too. The shaping term uses the same per-step discount, so it stays
   potential-based (Ng et al. 1999 holds with a per-step discount) and still leaves the optimal policy
   unchanged.
2. **λ on the same clock.** λ = 1 inside a blind, `--lam-blind 0.95` at each blind passed.
3. **Eval temperature** (`--eval-temp`), so sampling can be measured. Training is unchanged by it.

Held from 1.0 so the result is attributable to 1–2: a single network at 1.0's size (`--d 192
--layers 4 --heads 6`, 1.6M parameters), `--enc 2`, `--actions ordered-capped`, reward **`blinds`**
(the ought is unchanged, and the ante-8 boss is the 24th blind), shaping on, `--explore-hands 0.05`,
seed retries on, Red Deck, White Stake. **From random weights; no behavior cloning.** lr 1e-4 with KL
early stopping (as v1/v2). `best.pt` is chosen by eval blinds on the 600 eval seeds, as before.

### The test
- **Seeds:** 2,000 drawn from tag `v3-test`, excluding the eval 600 and the `paired-v1` 2,000. None are
  seen in training or checkpoint selection.
- **Players:** v3's `best.pt` vs `pins0.pt` (moe2, the strongest weights measured; MoE as trained,
  handoff at ante 4). Each plays each seed once, sampled at temperature 1 with the eval's per-seed
  RNG, the same convention for both. Win = the ante-8 boss beaten (the corrected test).
- **Power:** paired, 2,000 seeds detects 2% vs 4% (see the paired run's power table above).

### Predictions (Claude)
- **V1. v3 wins at least 4.0% of the 2,000 seeds, and more than `pins0.pt` (one-sided exact
  McNemar on the seeds where exactly one wins, p < 0.05). (60%)** chud's target. *Falsified* if under
  4.0% or not significant.
- **V2. v3 beats more blinds per game than `pins0.pt` (paired, p < 0.05). (75%)**
- **V3. The mechanism: the gain comes from the build.** v3's share of games ending in antes 4–6 is
  lower than `pins0.pt`'s, and its median money held at the start of ante 6 is lower (it spends).
  **(60%)** If V1 passes and V3 fails, the diagnosis was wrong even though the target was met.
- Reported, not predicted: the same comparison at `--eval-temp 0.5` for both models, and 1.0 on the
  same seeds.

### What this can and can't attribute
v3 differs from `pins0.pt` in more than 1–2: it starts from random weights, not 0.2's, and it's a
single network, not a MoE. A pass shows v3 is stronger. It shows *the clock and λ* caused it only if a
control arm is also run: the same v3 settings with per-action `--gamma 0.995 --lam 0.95`, from the
same random seed, for the same number of games. Without it, the claim is limited to "v3 is stronger".

### Amended 2026-11-05 ~21:24 CST (before any code): the control arm, as a stopping rule (chud agreed)
The control is run only until the two arms separate, then stopped, and the winner gets the machine.
Reason (the goal is the strongest bot): if v3 passes, the control changes no decision; if v3 fails,
it is the only way to tell "the clock is wrong" from "starting from scratch cost us".
- **Arms:** `v3` (`--gamma-blind 0.95 --lam-blind 0.95`) and `control` (`--gamma 0.995 --lam 0.95`
  per action), everything else identical, same training seed, side by side.
- **Measurement at each check:** both arms' latest `best.pt` play the same 2,000 seeds from tag
  `v3-ab` (outside the eval 600, `paired-v1` and `v3-test`), sampled, per-seed RNG; paired difference
  in blinds per game. Sizing: one policy's blinds per seed have SD 5.9 (`paired_moe2.jsonl`); two
  networks trained from scratch share only the seed, correlation 0–0.3, so the paired SD is
  7.0–8.3. To detect 0.5 blinds per game with power 0.8 takes (2.8σ/0.5)² = 1,550–2,150 seeds:
  2,000. (The eval 600 resolve only ~0.8–0.95.) A gap under 0.5 counts as a tie.
- **When:** the first check once both arms average ≥ 9 blinds per game on their eval (ante 4 reached on
  average: the build wall is where the clock should matter); then every 10,000 training games per
  arm; at most 3 checks.
- **Rule:** O'Brien-Fleming boundaries for 3 looks (overall two-sided α 0.05): stop at the first
  check where |z| > 3.47, 2.45, 2.00 (checks 1, 2, 3) and keep the arm ahead. If no boundary is crossed
  by check 3, it's a tie and **v3 is kept** (as registered).
- **What it can't say:** one training run per arm compares these two networks, not the two methods;
  run-to-run luck is unmeasured (that would need ≥ 2 runs per arm). The write-up won't claim more.
- V1–V3 are scored on whichever arm is kept, against `pins0.pt`, on `v3-test`. If the control is kept,
  V1 is still scored, and V3's mechanism claim is scored as not supported.

### Before training
The new discount needs the same checks as every trainer change: per-step discounts unit-tested
against a hand-computed game, the shaping term's telescoping checked on real games, saved games
replaying exactly, and the pre-flight in `run_v2.sh`.
