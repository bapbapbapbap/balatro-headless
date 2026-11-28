"""Z1 / Z2: AnteZero's simulator in lockstep with our harness (the game's own Lua), compared by our own view.

**AnteZero is not ours.** It is namecallfilter's (https://github.com/namecallfilter/antezero, MIT), pinned at
4fe07f771104a811ca8452efd4831b0dbf097e6e. None of its code is in this repository: this script imports its Python
bindings (`antezero_sim`, built from that clone with maturin) and calls `Game(seed, back, stake, endless)`,
`legal_codes()`, `step(code)`, `state_json()`, `clone()`, `chips`, `phase`, `is_terminal()`.

**Ours** (Claude's, reviewed by chud): everything here and `view.lua`. Registration: docs/hypotheses-reproduce.md,
"2026-11-27 ... tests Z1-Z2 registered before their code".

    python3 lockstep.py z1 --src <game source> --per-cell 200 --workers N --out z1.jsonl
    python3 lockstep.py z2 --src <game source> --per-cell 10 --workers N --out z2.jsonl --hist <dir>
    python3 lockstep.py report z2.jsonl
"""
import argparse, json, os, random, sys, traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "harness"))

BACKS = ["b_red", "b_blue", "b_yellow", "b_green", "b_black", "b_magic", "b_nebula", "b_ghost", "b_abandoned",
         "b_checkered", "b_zodiac", "b_painted", "b_anaglyph", "b_plasma", "b_erratic"]
STAKES = list(range(1, 9))
SEED_CHARS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"   # the game's seed alphabet (no 0, no O)
STATE_OF_PHASE = None


def seeds_for(tag, n):
    r = random.Random(tag)
    return ["".join(r.choice(SEED_CHARS) for _ in range(8)) for _ in range(n)]


# ---------------------------------------------------------------- views

CARD_FIELDS = ("card", "key", "edition", "seal", "debuff", "facing", "cost", "sell_cost")


def harness_card(c):
    return {k: c.get(k) for k in CARD_FIELDS} | {"sort_id": c.get("sort_id"), "ability": c.get("ability") or {}}


def sim_card(c):
    return {k: c.get(k) for k in CARD_FIELDS} | {"sort_id": c.get("sort_id"), "ability": c.get("ability") or {}}


def norm_edition(e):
    if isinstance(e, dict):
        return e.get("type") or next((k for k, v in e.items() if v is True), None)
    return e


def sim_view(d):
    b = d.get("blind") or {}
    return {
        "state": d.get("state"), "ante": d.get("ante"), "round": d.get("round"), "dollars": d.get("dollars"),
        "chips": d.get("chips"), "hands_left": d.get("hands_left"), "discards_left": d.get("discards_left"),
        "reroll_cost": d.get("reroll_cost"), "interest_cap": d.get("interest_cap"), "skips": d.get("skips"),
        "probabilities": d.get("probabilities"), "won": bool(d.get("won")),
        "joker_limit": d.get("joker_limit"), "hand_limit": d.get("hand_limit"),
        "consumable_limit": d.get("consumable_limit"),
        "blind": {"name": b.get("name"), "chips": b.get("chips"), "mult": b.get("mult"),
                  "disabled": bool(b.get("disabled"))},
        "blind_on_deck": d.get("blind_on_deck"), "blind_choices": d.get("blind_choices"),
        "blind_tags": d.get("blind_tags"), "blind_states": d.get("blind_states"),
        "hand": [sim_card(c) for c in d.get("hand") or []], "deck": [sim_card(c) for c in d.get("deck") or []],
        "discard": [sim_card(c) for c in d.get("discard") or []],
        "jokers": [sim_card(c) for c in d.get("jokers") or []],
        "consumables": [sim_card(c) for c in d.get("consumables") or []],
        "shop_jokers": [sim_card(c) for c in d.get("shop_jokers") or []],
        "shop_vouchers": [sim_card(c) for c in d.get("shop_vouchers") or []],
        "shop_booster": [sim_card(c) for c in d.get("shop_booster") or []],
        "pack_cards": [sim_card(c) for c in d.get("pack_cards") or []],
        "hands": d.get("hands") or {},
        "tags": [t.get("key") if isinstance(t, dict) else t for t in d.get("tag_info") or []],
        "used_vouchers": sorted(k for k, v in (d.get("used_vouchers") or {}).items() if v)
        if isinstance(d.get("used_vouchers"), dict) else sorted(d.get("used_vouchers") or []),
        "rng": {k: v for k, v in (d.get("rng") or {}).items() if isinstance(v, (int, float))},
    }


def harness_view(v):
    v = dict(v)
    for k in ("hand", "deck", "discard", "jokers", "consumables", "shop_jokers", "shop_vouchers", "shop_booster",
              "pack_cards"):
        v[k] = [harness_card(c) for c in v.get(k) or []]
    v["blind"] = v.get("blind") or {"name": None, "chips": None, "mult": None, "disabled": False}
    return v


# Booster centers that differ only in art and are picked by unseeded math.random in the game (first-shop Buffoon
# pack, Charm and Meteor tags): equal for this comparison; each occurrence is counted in the record ("art").
ART_PAIRS = [{"p_buffoon_normal_1", "p_buffoon_normal_2"}, {"p_arcana_mega_1", "p_arcana_mega_2"},
             {"p_celestial_mega_1", "p_celestial_mega_2"}]
ART = []

PACK_STATES = {"TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK"}


def num_eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or (a != a and b != b)
    if isinstance(a, str) and isinstance(b, (int, float)) or isinstance(b, str) and isinstance(a, (int, float)):
        try:
            return float(a) == float(b)
        except ValueError:
            return False
    return a == b


def diff_ability(path, sim, game, out):
    """Every key the simulator dumps must equal the game's value (keys only the game has are not checked)."""
    if isinstance(sim, dict):
        if not isinstance(game, dict):
            out.append((path, sim, game)); return
        for k, sv in sim.items():
            gv = game.get(k)
            if gv is None and isinstance(game, dict) and k.isdigit():
                gv = game.get(int(k))
            if sv is None and gv is None:
                continue
            if sv is False and gv is None:     # the game leaves unset flags nil
                continue
            diff_ability(f"{path}.{k}", sv, gv, out)
    elif isinstance(sim, list):
        if not isinstance(game, list) or len(sim) != len(game):
            out.append((path, sim, game)); return
        for i, (a, b) in enumerate(zip(sim, game)):
            diff_ability(f"{path}[{i}]", a, b, out)
    elif not num_eq(sim, game):
        out.append((path, sim, game))


def compare(sv, gv, sort_off):
    """Differences between the simulator's view and the game's: [(field, sim, game)]."""
    out = []
    st = gv.get("state")
    skip = set()
    if st != "SHOP":
        skip |= {"shop_jokers", "shop_vouchers", "shop_booster"}
    if st not in PACK_STATES:
        skip.add("pack_cards")
    for k, s in sv.items():
        if k in skip:
            continue
        g = gv.get(k)
        if k in ("hand", "deck", "discard", "jokers", "consumables", "shop_jokers", "shop_vouchers", "shop_booster",
                 "pack_cards"):
            if len(s) != len(g):
                out.append((f"{k}.len", len(s), len(g)))
                out.append((k, [c.get("card") or c.get("key") for c in s], [c.get("card") or c.get("key") for c in g]))
                continue
            for i, (a, b) in enumerate(zip(s, g)):
                for f in CARD_FIELDS:
                    av, bv = a.get(f), b.get(f)
                    if f == "edition":
                        av, bv = norm_edition(av), norm_edition(bv)
                    if f in ("debuff",):
                        av, bv = bool(av), bool(bv)
                    if f == "key" and av != bv and {av, bv} in ART_PAIRS:
                        ART.append((f"{k}[{i}]", av, bv, gv.get("state"), gv.get("round")))
                        continue
                    if not (av is None and bv is None) and not num_eq(av, bv):
                        out.append((f"{k}[{i}].{f}", av, bv))
                if a.get("sort_id") is not None and b.get("sort_id") is not None and \
                        a["sort_id"] + sort_off != b["sort_id"]:
                    out.append((f"{k}[{i}].sort_id", a["sort_id"] + sort_off, b["sort_id"]))
                diff_ability(f"{k}[{i}].ability", a.get("ability") or {}, b.get("ability") or {}, out)
        elif k == "hands":
            for name, h in s.items():
                gh = g.get(name) or {}
                for f in ("level", "chips", "mult", "played", "played_this_round", "visible"):
                    if not num_eq(h.get(f), gh.get(f)):
                        out.append((f"hands.{name}.{f}", h.get(f), gh.get(f)))
        elif k == "rng":
            for name in sorted(set(s) | set(g)):
                if not num_eq(s.get(name), g.get(name)):
                    out.append((f"rng.{name}", s.get(name), g.get(name)))
        elif k == "blind":
            for f in ("name", "chips", "mult", "disabled"):
                if not num_eq(s.get(f), g.get(f)):
                    out.append((f"blind.{f}", s.get(f), g.get(f)))
        elif isinstance(s, dict) or isinstance(g, dict):
            if (s or {}) != (g or {}):
                out.append((k, s, g))
        elif isinstance(s, list):
            if list(s) != list(g or []):
                out.append((k, s, g))
        elif not num_eq(s, g):
            out.append((k, s, g))
    return out


# ---------------------------------------------------------------- the two engines

_ENV = None


def env_for(src):
    global _ENV
    if _ENV is None:
        from env import BalatroEnv
        _ENV = BalatroEnv(src, mode="subsets")
        _ENV.lua.execute(open(os.path.join(HERE, "view.lua")).read())
    return _ENV


def game_view(env):
    return harness_view(json.loads(env.lua.globals().AZ_VIEW()))


def translate(code):
    """A simulator action code -> (our env action type, i, j, cards), 1-based as our env."""
    p = code.split()
    op, args = p[0], [int(x) for x in p[1:]]

    def cards(m):
        return tuple(i + 1 for i in range(32) if m >> i & 1)
    if op == "select": return ("select_blind", None, None, ())
    if op == "skip": return ("skip_blind", None, None, ())
    if op == "reroll_boss": return ("reroll_boss", None, None, ())
    if op in ("play", "discard"): return (op, None, None, cards(args[0]))
    if op == "use": return ("use_consumable", args[0] + 1, None, cards(args[1]))
    if op == "sell_joker": return ("sell_joker", args[0] + 1, None, ())
    if op == "sell_consumable": return ("sell_consumable", args[0] + 1, None, ())
    if op == "buy": return ("buy", args[0] + 1, None, ())
    if op == "buy_use": return ("buy_and_use", args[0] + 1, None, cards(args[1]))
    if op == "redeem": return ("redeem", args[0] + 1, None, ())
    if op == "open": return ("open_pack", args[0] + 1, None, ())
    if op == "reroll": return ("reroll", None, None, ())
    if op == "end_shop": return ("end_shop", None, None, ())
    if op == "cash_out": return ("cash_out", None, None, ())
    if op == "pick": return ("pick", args[0] + 1, None, cards(args[1]))
    if op == "skip_pack": return ("skip_pack", None, None, ())
    if op == "move_joker": return ("move_joker", args[0] + 1, args[1] + 1, ())
    if op == "move_hand": return ("move_hand", args[0] + 1, args[1] + 1, ())
    if op == "sort": return ("sort_suit" if args[0] == 1 else "sort_rank", None, None, ())
    raise ValueError(code)


def env_apply(env, act):
    """Apply in our env: through its legal set when the action is there, else directly (flagged)."""
    legal = env.legal_actions()
    match = [k for k in legal if k[0] == act[0] and (act[1] is None or k[1] == act[1])
             and (act[2] is None or k[2] == act[2]) and tuple(k[3]) == act[3]]
    if match:
        env.step(match[0])
        return True
    lt = env.lua.table_from({"type": act[0]})
    if act[1] is not None: lt.i = act[1]
    if act[2] is not None: lt.j = act[2]
    if act[3]: lt.cards = env.lua.table_from(list(act[3]))
    n = env.H.step(lt)
    env._acts = None
    if n < 0:
        raise RuntimeError(f"game never settled after {act}")
    return False


# ---------------------------------------------------------------- policy (weighted random over the sim's codes)

def choose(g, r):
    codes = g.legal_codes()
    ops = {}
    for c in codes:
        ops.setdefault(c.split()[0], []).append(c)
    if "play" in ops:
        if r.random() < 0.75:      # the play that scores most now, by the simulator (a clone)
            best, bv = None, -1.0
            for c in ops["play"]:
                h = g.clone(); h.step(c)
                v = h.chips if h.phase == g.phase else 1e300
                if v > bv: best, bv = c, v
            d = json.loads(g.state_json())
            need = (g.blind_chips - g.chips) / max(1, d.get("hands_left") or 1)
            if bv - g.chips < need and "discard" in ops and r.random() < 0.8:
                keep = int(best.split()[1])     # discard up to 5 cards outside the best play, rightmost first
                n = len(d.get("hand") or [])
                rest = [i for i in range(n) if not keep >> i & 1][-5:]
                code = "discard " + str(sum(1 << i for i in rest))
                if rest and code in ops["discard"]:
                    return code
            return best
        pool = ops.get("discard", []) if r.random() < 0.5 and "discard" in ops else ops["play"]
        if "use" in ops and r.random() < 0.3: pool = ops["use"]
        return r.choice(pool)
    for op, w in (("use", 0.3), ("buy", 0.5), ("redeem", 0.5), ("open", 0.4), ("sell_joker", 0.03),
                  ("sell_consumable", 0.05), ("reroll", 0.15), ("buy_use", 0.2), ("pick", 0.9),
                  ("reroll_boss", 0.1)):
        if op in ops and r.random() < w:
            return r.choice(ops[op])
    for op in ("select", "cash_out", "end_shop", "skip_pack"):
        if op in ops:
            if op == "select" and "skip" in ops and r.random() < 0.15:
                return "skip"
            return ops[op][0]
    return r.choice(codes)


# ---------------------------------------------------------------- jobs

def run_one(job):
    kind, src, seed, back, stake, max_steps, hist_dir = job
    import antezero_sim
    env = env_for(src)
    rec = {"kind": kind, "seed": seed, "back": back, "stake": stake}
    ART.clear()
    try:
        g = antezero_sim.Game(seed, back, stake, False)
        env.reset(seed, back, stake)
        sv, gv = sim_view(json.loads(g.state_json())), game_view(env)
        off = 0
        if sv["deck"] and gv["deck"]:
            off = gv["deck"][0]["sort_id"] - sv["deck"][0]["sort_id"]
        rec["sort_off"] = off
        d = compare(sv, gv, off)
        hist = []
        steps = unlisted = 0
        if not d and kind == "z2":
            r = random.Random(f"{seed}{back}{stake}")
            while not g.is_terminal() and steps < max_steps:
                code = choose(g, r)
                hist.append(code)
                act = translate(code)
                g.step(code)
                if not env_apply(env, act):
                    unlisted += 1
                    rec.setdefault("unlisted", []).append([steps, code])
                steps += 1
                sv, gv = sim_view(json.loads(g.state_json())), game_view(env)
                d = compare(sv, gv, off)
                if d:
                    break
                if g.is_terminal() != env.done:
                    d = [("terminal", g.is_terminal(), env.done)]
                    break
        rec.update(steps=steps, match=not d, diffs=[list(map(str, x))[:3] for x in d[:12]],
                   state=gv.get("state"), ante=gv.get("ante"), won=bool(env.won), terminal=g.is_terminal(),
                   unlisted_n=unlisted, art=[list(map(str, x)) for x in ART[:5]], art_n=len(ART))
        if d and hist_dir:
            os.makedirs(hist_dir, exist_ok=True)
            with open(os.path.join(hist_dir, f"{kind}_{seed}_{back}_{stake}.json"), "w") as f:
                json.dump({"seed": seed, "back": back, "stake": stake, "codes": hist, "diffs": rec["diffs"]}, f)
    except Exception as e:
        rec.update(match=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["z1", "z2", "report"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--src")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--hist")
    ap.add_argument("--backs", default=",".join(BACKS))
    ap.add_argument("--stakes", default="1,2,3,4,5,6,7,8")
    a = ap.parse_args()
    if a.kind == "report":
        return report(a.file)
    tag = "az-z1" if a.kind == "z1" else "az-z2"
    backs, stakes = a.backs.split(","), [int(s) for s in a.stakes.split(",")]
    seeds = seeds_for(tag, a.per_cell * len(BACKS) * len(STAKES))
    jobs, i = [], 0
    for back in BACKS:
        for stake in STAKES:
            for _ in range(a.per_cell):
                if back in backs and stake in stakes:
                    jobs.append((a.kind, a.src, seeds[i], back, stake, a.max_steps, a.hist))
                i += 1
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["seed"], r["back"], r["stake"]))
    jobs = [j for j in jobs if (j[2], j[3], j[4]) not in done]
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    with open(a.out, "a") as f, Pool(a.workers) as pool:
        for n, rec in enumerate(pool.imap_unordered(run_one, jobs), 1):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if not rec["match"] or n % 25 == 0:
                print(n, rec["seed"], rec["back"], rec["stake"], "match" if rec["match"] else "DIFF",
                      rec.get("steps"), rec.get("error") or rec.get("diffs", [])[:3], flush=True)


def report(path):
    rows = [json.loads(l) for l in open(path)]
    m = sum(r["match"] for r in rows)
    print(f"{m}/{len(rows)} games matched ({100 * m / max(1, len(rows)):.1f}%)")
    steps = sum(r.get("steps", 0) for r in rows)
    print(f"decisions compared: {steps}; unlisted sim actions: {sum(r.get('unlisted_n', 0) for r in rows)}")
    from collections import Counter
    c = Counter()
    for r in rows:
        if not r["match"]:
            key = r.get("error", "")[:60] or (r["diffs"][0][0].split("[")[0].split(".")[0] if r["diffs"] else "?")
            c[key] += 1
    for k, v in c.most_common():
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
