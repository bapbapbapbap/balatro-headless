# v4b at its stop (round 280): the replays behind the diagnosis

Scored and read in [`docs/hypotheses-v4b.md`](../../../docs/hypotheses-v4b.md), sections of 2026-11-11.
Everything here was produced on chud's laptop from run `v4b`'s saved games and checkpoints (which are not in
the repo) on 2026-11-10 and 11; the files are copies from his work folder, unedited. The `.txt` files are each
script's printed output; the `.jsonl.gz` files are its `--out`.

| File | What | Made by |
|---|---|---|
| `v4b_sells.txt` | the table network's credit around joker sells, rounds 160 and 260 | `analysis/v4_sells.py` on v4b (command not kept) |
| `v4b_drop.txt`, `v4b_drop.jsonl.gz` | the 600 eval games of rounds 240, 260, 280 replayed: jokers, money, plays, buys, sells, how each ended (1,800 games, 0 mismatched) | `analysis/v4b_drop.py --rounds 240,260,280` |
| `v4b_shop_credit.txt`, `.jsonl.gz` | the shop network's credit at sell-legal states, checkpoint 160. The run was stopped during checkpoint 240: its 240 rows (10,816) are partial, and nothing reads them | `analysis/v4b_shop_credit.py --its 160,240,260,280 --window 5` |
| `v4b_shop_credit_b.txt`, `.jsonl.gz` | the same, checkpoints 240, 260, 280 | the same script, rerun for those three |
| `v4b_shop_credit_c.txt`, `.jsonl.gz` | the same, checkpoints 200, 220 | `--its 200,220 --window 5` |
| `v4b_sell_followup.txt` | what the selling cost in the eval games; whether the critic's rise after a sell is money; td by joker sold | `analysis/v4b_sell_followup.py` on the files above |

`v4b_sell_followup.txt` is reproducible from this folder alone (Python 3 with numpy):

```
cd analysis/results/v4b_stop
python3 ../../v4b_sell_followup.py --drop v4b_drop.jsonl.gz \
    --credit v4b_shop_credit_c.jsonl.gz v4b_shop_credit_b.jsonl.gz v4b_shop_credit.jsonl.gz
```

The other files need the run's games and checkpoints to regenerate. SHA-256 of the uncompressed `.jsonl` files
as they are in the work folder:

```
d7695b7b3057245c82978237728fa3d7a3a259a8e77e9ee5cdb007f561a70973  v4b_drop.jsonl
7b154abf715b7ebff315784e5a5efb13ee85b27fce267cbdedcb8e36602c16e4  v4b_shop_credit.jsonl
0d25a32483ab6d55d333420675eb4418fa73e621c6fef0c18b122a61a1ba6dcb  v4b_shop_credit_b.jsonl
9ce7c41fe3ab6d4b8a3e0284672f6ed9e9bb1b4589e86ec06a1f0b3f5c0ca04f  v4b_shop_credit_c.jsonl
```

Limits: training games are 5-round windows ending at each checkpoint (rounds 221-235 and 261-275 are not
sampled); curriculum games are skipped; the credit files hold only shop decisions where a sell was legal (at
least one joker held); a "free-slot sell" in the eval games counts slots as 5.
