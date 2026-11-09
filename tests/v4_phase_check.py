"""Pre-flight for model v4's phase split (docs/hypotheses.md, v4 and its 2026-11-09 amendment, change 2).
Needs torch and the real game. Plays games with random v4 weights through the trainer's own code
(train.play_games, --phase-split) and checks:

  1. Routing: every decision went to the network its game state names (SELECTING_HAND -> table, every
     other state -> shop), re-read from a fresh replay of the saved route; plays and discards only ever
     reach the table network; both networks get decisions.
  2. Saved games replay exactly (seed + moves -> the same final state), as every route must.
  3. One network per decision: PhaseNet's output for a decision equals its own network's output alone,
     whatever else is in the batch; a loss on table decisions moves no shop weight (and the reverse).
  4. PPO with routed credit runs on these games: advantages are train/credit.py's, normalized per
     network; per-network entropy and decision counts are reported; the shop's entropy coefficient
     follows --shop-ent.
  5. A checkpoint's config rebuilds the same network (model.make_model), as play.py and
     compare_models.py load it.

Random weights die in the first blind, so the games that pass through every phase are published wins
(wins/routes.jsonl), replayed through v4's networks as training games (each decision routed, valued and
scored by the trainer's own functions) next to games the random weights play themselves.

  python3 v4_phase_check.py <game source dir> [--games 6] [--routes 4]
"""
import json
import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import train as T  # noqa: E402
from credit import routed_gae  # noqa: E402
from model import PHASE_NAMES, PHASE_SHOP, PHASE_TABLE, PhaseNet, make_model, phase_member  # noqa: E402
from sim import Sim  # noqa: E402

PLAY, DISCARD = 1, 2


def args_for(src):
    """A small v4 configuration: the routing doesn't depend on size."""
    ap_defaults = dict(src=src, mods=None, actions="ordered-capped", enc=2, d=32, layers=1, heads=2,
                       phase_split=True, moe=False, shop_ent="0.02,0.006@100,0.001@200", ent=0.01,
                       reward="blinds", shaping=1.0, gamma=0.995, lam=0.95, gamma_blind=0.95, lam_blind=0.95,
                       back="b_red", stake=1, backs=[], stakes=[], challenges=[], challenge_share=0.25,
                       max_steps=1500, stop_ante=0, explore_hands=0.0, pin_hand=None, pin_rate=1.0,
                       seed=7, eval_temp=1.0, verify_wins=True, clip=0.2, epochs=1, minibatch=64, vf=0.5,
                       grad_clip=0.5, target_kl=0.04, sil_steps=8, moe_cut=0)
    return Namespace(**ap_defaults)


def games_from_routes(sim, model, routes, args):
    """Published wins as v4 training games: the saved moves, each decision routed and valued by v4's
    networks with its log-probability, reward and discounts as play_games records them."""
    reward_fn, out = T.REWARDS[args.reward], []
    for r in routes:
        o = sim.reset(r["seed"], r.get("back", "b_red"), r.get("stake", 1), r.get("challenge"))
        steps, disc, members = [], [], []
        for k, a in enumerate(r["acts"]):
            o["member"] = phase_member(sim.H.state_name())
            members.append(o["member"])
            with torch.no_grad():
                lg, v = model(*T.collate([o]))
            lp = float(torch.log_softmax(lg[0], -1)[a])
            prev, cur, done = sim.step(a)
            g_t, l_t = T.step_discount(prev, cur, args)
            rw = reward_fn(prev, cur)
            steps.append((o, a, lp, float(v[0]), rw + args.shaping * T.shaping(prev, cur, g_t), rw))
            disc.append((g_t, l_t))
            if not done:
                o = sim.observe()
        assert sim.progress["won"], f"{r['seed']}: published win did not win on replay"
        out.append({"seed": r["seed"], "back": r.get("back", "b_red"), "stake": r.get("stake", 1),
                    "challenge": r.get("challenge"), "steps": steps, "disc": disc, "members": members,
                    "n": sim.steps, "acts": list(sim.route), "desc": list(sim.desc), "final": sim.final_hash(),
                    "progress": sim.progress})
    return out


def check_routing(sim, eps):
    counts = {nm: {} for nm in PHASE_NAMES}
    for ep in eps:
        assert len(ep["members"]) == ep["n"] == len(ep["steps"]), "one member per decision"
        assert [s[0]["member"] for s in ep["steps"]] == ep["members"]
        o = sim.reset(ep["seed"], ep["back"], ep["stake"], ep["challenge"])
        for k, a in enumerate(ep["acts"]):
            st = sim.H.state_name()
            m = ep["members"][k]
            assert m == phase_member(st), f"{ep['seed']} step {k}: state {st} went to {PHASE_NAMES[m]}"
            assert sim.H.action_desc(a + 1) == ep["desc"][k], f"{ep['seed']} step {k}: replay differs"
            t = int(o["act"][a][0])
            if t in (PLAY, DISCARD):
                assert m == PHASE_TABLE, f"{ep['seed']} step {k}: a play/discard reached the shop network"
            counts[PHASE_NAMES[m]][st] = counts[PHASE_NAMES[m]].get(st, 0) + 1
            _, _, done = sim.step(a)
            if not done:
                o = sim.observe()
        assert sim.final_hash() == ep["final"], f"{ep['seed']}: final state differs on replay"
    assert counts["table"] and counts["shop"], counts
    print("routing by game state: PASS  " + "; ".join(
        f"{nm}: " + ", ".join(f"{s} {n}" for s, n in sorted(c.items())) for nm, c in counts.items()))
    print(f"{len(eps)} saved games replay exactly: PASS")


def check_isolation(model, eps):
    obs = [s[0] for ep in eps for s in ep["steps"]][:96]
    mixed = T.collate(obs)
    lg, v = model(*mixed)
    for m in (PHASE_TABLE, PHASE_SHOP):
        i = (mixed[6] == m).nonzero(as_tuple=True)[0]
        lg_m, v_m = model.nets[m](*(x[i] for x in mixed[:6]))
        fin = torch.isfinite(lg_m)
        assert torch.equal(fin, torch.isfinite(lg[i]))
        assert torch.allclose(lg[i][fin], lg_m[fin], atol=1e-5) and torch.allclose(v[i], v_m, atol=1e-5)
    print("each decision is its own network's output, whatever the batch: PASS")
    for m in (PHASE_TABLE, PHASE_SHOP):
        model.zero_grad(set_to_none=True)
        only = [o for o in obs if o["member"] == m]
        lg, v = model(*T.collate(only))
        (torch.log_softmax(lg, -1).nan_to_num(neginf=0).sum() + v.sum()).backward()
        other = model.nets[1 - m]
        assert all(p.grad is None or float(p.grad.abs().max()) == 0 for p in other.parameters()), \
            f"{PHASE_NAMES[m]} decisions moved the {PHASE_NAMES[1 - m]} network"
        assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in model.nets[m].parameters())
    model.zero_grad(set_to_none=True)
    print("a decision trains only its own network: PASS")


def check_ppo(model, eps, args):
    # the advantages ppo_update uses are credit.routed_gae's, over each game
    for ep in eps:
        r = [s[4] for s in ep["steps"]]
        v = [s[3] for s in ep["steps"]]
        G, L = [x[0] for x in ep["disc"]], [x[1] for x in ep["disc"]]
        a = routed_gae(r, v, G, L, ep["members"])
        assert np.isfinite(a).all()
        if len(set(ep["members"])) == 1:
            assert np.allclose(a, T.gae(r, v, G, L), atol=1e-5)
    for games, want in ((0, 0.02), (50, 0.013), (100, 0.006), (150, 0.0035), (200, 0.001), (10**6, 0.001)):
        args.games_done = games
        got = T.phase_ent(args)
        assert got[0] == args.ent and abs(got[1] - want) < 1e-9, (games, got, want)
    assert T.schedule_at(T.parse_schedule("0.02,0.006@100,0.001@100"), 100) == 0.001    # a repeated count is a step
    args.games_done = 50
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in model.parameters()]
    st = T.ppo_update(model, opt, eps, args, "cpu")
    n = {nm: sum(m == k for ep in eps for m in ep["members"]) for k, nm in enumerate(PHASE_NAMES)}
    assert st["decisions_table"] == n["table"] and st["decisions_shop"] == n["shop"], (st, n)
    assert abs(st["ent_coef_shop"] - 0.013) < 1e-6 and abs(st["ent_coef_table"] - args.ent) < 1e-6
    assert all(np.isfinite(x) for x in st.values())
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters())), "the update changed nothing"
    print(f"PPO with routed credit and per-network entropy: PASS  (table {n['table']} decisions, entropy "
          f"{st['entropy_table']:.2f}; shop {n['shop']}, entropy {st['entropy_shop']:.2f}, coef {st['ent_coef_shop']:.4f})")


def check_rebuild(model, args, n_vocab):
    m2 = make_model(Namespace(**vars(args)), n_vocab)
    assert isinstance(m2, PhaseNet)
    m2.load_state_dict(model.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), m2.state_dict().values()))
    print("a checkpoint's config rebuilds the same two networks: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--routes", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    args = args_for(a.src)
    sim = Sim(a.src, mode=args.actions, enc=args.enc)
    model = make_model(args, len(sim.vocab))
    model.eval()
    seeds = [f"V4PHASE{i}" for i in range(a.games)]
    eps = T.play_games(sim, model, T.REWARDS[args.reward], __import__("random").Random(0), args, 0,
                       (model.state_dict(), seeds, "train", 0, 0), set())
    print(f"played {len(eps)} games with random v4 weights: {sum(e['n'] for e in eps)} decisions, "
          f"antes reached {[e['progress']['ante'] for e in eps]}")
    lines = open(os.path.join(HERE, "..", "wins", "routes.jsonl")).read().splitlines()
    caps = [json.loads(l) for l in lines if '"ordered-capped"' in l]
    pick = [caps[i] for i in np.linspace(0, len(caps) - 1, a.routes).astype(int)]
    pick = [r for r in pick if r.get("actions") == "ordered-capped"]          # the action list v4 plays on
    wins = games_from_routes(sim, model, pick, args)
    print(f"replayed {len(wins)} published wins through v4's networks: {sum(e['n'] for e in wins)} decisions")
    eps = eps + wins
    check_routing(sim, eps)
    check_isolation(model, eps)
    check_ppo(model, eps, args)
    check_rebuild(model, args, len(sim.vocab))
    print("ALL PASS")


if __name__ == "__main__":
    main()
