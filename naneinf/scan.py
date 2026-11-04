"""Scans seeds for naneinf openings on the pinned game (1.0.1o, unmodded, all unlocked), with the
game's own code: nothing here models Balatro's RNG, so every hit is exactly what the game deals.

Two lines, both from probe.py:
- skip (every seed, cheap: no hands played): skip the Small and Big Blinds of ante 1, open every
  pack the tags give (Charm, Double + Charm, Ethereal, Buffoon, ...), take The Soul whenever a pack
  offers it, and record the legendary it gives. Buffoon packs' jokers are recorded too.
- shop (with --shop, ~10x slower): play the Small Blind with a plain best-HUD-hand player, then the
  first shop, each of its packs, and the rerolls the money allows (probe.line_play_small).

A seed is a hit when the skip line gives a legendary, or either line shows Baron or Mime.
Hits go to OUT (jsonl, one seed per line, with what each line saw); progress every 30 s.

  BALATRO_SRC=~/balatro/local/source python3 scan.py --check            # known seeds first
  BALATRO_SRC=~/balatro/local/source python3 scan.py --n 200000 --workers 2 --out hits.jsonl

--check runs the skip line on three community Ghost seeds where probe.py found Perkeo and fails
unless all three give it. Seeds are random 8-character codes in the game's alphabet (no 0 or O),
drawn from --rng-seed, so a scan can be repeated or split across machines (--rng-seed per machine).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import P, line_play_small, flag  # noqa: E402

ALPHABET = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
LEGENDARY = ("j_perkeo", "j_triboulet", "j_chicot", "j_yorick", "j_caino")
KNOWN = [("E4EI3UQK", "j_perkeo"), ("5MG2J3UY", "j_perkeo"), ("4BTVF4X1", "j_perkeo")]   # Ghost Deck


def take_one(p):
    """The open pack: record it, take every Soul it offers, then leave it. Returns a note."""
    note = {"pack": p.keys("pack_cards"), "soul_gave": []}
    for _ in range(3):
        cur = p.keys("pack_cards")
        if "c_soul" not in cur or not p.state().endswith("PACK"):
            break
        i = cur.index("c_soul") + 1
        before = p.keys("jokers")
        if not p.do(lambda a, i=i: a.type == "pick" and a.i == i):
            break
        after = p.keys("jokers")
        note["soul_gave"] += after[len(before):] if after[:len(before)] == before else [k for k in after if k not in before]
    if p.state().endswith("PACK"):
        if not p.do(lambda a: a.type == "skip_pack"):
            p.do(lambda a: a.type == "pick")
    return note


def skip_line(p, seed, back):
    p.reset(seed, back)
    out = {"tags": [], "packs": [], "legendaries": []}
    for where in ("small", "big"):
        tag = p.tag_on_deck()
        if not p.do(lambda a: a.type == "skip_blind"):
            break
        out["tags"].append(tag)
        for _ in range(4):                       # Double Tag can queue a second pack behind the first
            if not p.state().endswith("PACK"):
                break
            note = take_one(p)
            note["after_skip"] = where
            out["packs"].append(note)
            out["legendaries"] += note["soul_gave"]
    return out


def is_hit(r):
    if any(l.split(" ")[0] in LEGENDARY for l in r["skip"].get("legendaries", [])):
        return True
    return bool({"j_baron", "j_mime"} & set(r["seen"]))


def worker(wid, n, rng_seed, back, shop, q):
    p = P()
    rng = random.Random(f"{rng_seed}-{wid}")
    done = errors = 0
    for _ in range(n):
        seed = "".join(rng.choice(ALPHABET) for _ in range(8))
        r = {"seed": seed, "back": back}
        try:
            r["skip"] = skip_line(p, seed, back)
            if shop:
                r["shop"] = line_play_small(p, seed, back)
        except Exception as e:                   # a line the seed doesn't allow is data, not a crash
            r.setdefault("skip", {})
            r["error"] = str(e)[-300:]
            errors += 1
        r["seen"] = flag({k: r[k] for k in ("skip", "shop") if k in r})
        done += 1
        if is_hit(r):
            q.put(("hit", r))
        if done % 200 == 0:
            q.put(("count", (wid, done, errors)))
    q.put(("count", (wid, done, errors)))
    q.put(("done", wid))


def check(back):
    p = P()
    ok = True
    for seed, want in KNOWN:
        r = skip_line(p, seed, back)
        got = [l for l in r["legendaries"]]
        good = any(l.split(" ")[0] == want for l in got)
        ok &= good
        print(f"{seed}: tags {r['tags']}, legendaries {got} -> {'PASS' if good else 'FAIL'}", flush=True)
    print("CHECK PASS" if ok else "CHECK FAIL", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--back", default="b_ghost")
    ap.add_argument("--n", type=int, default=100000, help="seeds in total")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rng-seed", default="scan1")
    ap.add_argument("--shop", action="store_true", help="also play the Small Blind and look at the first shop")
    ap.add_argument("--out", default="hits.jsonl")
    ap.add_argument("--check", action="store_true", help="only check the known Perkeo seeds")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.back) else 1)
    if a.back == "b_ghost" and not check(a.back):
        sys.exit("known seeds failed; not scanning")
    q = mp.Queue()
    per = [a.n // a.workers + (1 if i < a.n % a.workers else 0) for i in range(a.workers)]
    procs = [mp.Process(target=worker, args=(i, per[i], a.rng_seed, a.back, a.shop, q), daemon=True)
             for i in range(a.workers)]
    for pr in procs:
        pr.start()
    counts, errs, hits, live = {}, {}, {"legendary": 0, "perkeo": 0, "baron": 0, "mime": 0, "baron+mime": 0}, a.workers
    t0 = last = time.time()
    with open(a.out, "a") as f:
        while live:
            try:
                kind, v = q.get(timeout=5)
            except Exception:
                kind = None
            if kind == "hit":
                v["found_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                f.write(json.dumps(v) + "\n")
                f.flush()
                leg = [l for l in v["skip"].get("legendaries", []) if l.split(" ")[0] in LEGENDARY]
                hits["legendary"] += bool(leg)
                hits["perkeo"] += any(l.startswith("j_perkeo") for l in leg)
                b, m = "j_baron" in v["seen"], "j_mime" in v["seen"]
                hits["baron"] += b
                hits["mime"] += m
                hits["baron+mime"] += b and m
                if leg or (b and m):
                    print(f"  HIT {v['seed']}: {', '.join(leg) or ''} {'Baron+Mime' if b and m else ''}", flush=True)
            elif kind == "count":
                counts[v[0]], errs[v[0]] = v[1], v[2]
            elif kind == "done":
                live -= 1
            if time.time() - last > 30 or not live:
                last = time.time()
                n = sum(counts.values())
                rate = n / max(1e-9, time.time() - t0) * 3600
                print(f"{time.strftime('%H:%M:%S')}  {n:,} of {a.n:,} seeds  {rate:,.0f}/h  errors {sum(errs.values())}  "
                      f"hits: " + "  ".join(f"{k} {v}" for k, v in hits.items()), flush=True)
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
