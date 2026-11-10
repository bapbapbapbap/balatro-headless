"""Mixture of experts (docs/hypotheses-moe.md): loading members and weights into a MoE."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import EXPERT_HANDS, MoE  # noqa: E402


def load_weights_into(net, sd, ck_vocab, vocab):
    """Loads a state dict saved with ck_vocab. The current vocabulary may extend it at the end
    (encoding 3 appends challenge names to encoding 2's); the new words' embeddings keep their
    fresh initialization."""

    ck_vocab, vocab = list(ck_vocab), list(vocab)
    if ck_vocab != vocab:
        assert vocab[:len(ck_vocab)] == ck_vocab, "weights' vocabulary differs from this game's (not a prefix)"
        sd, cur = dict(sd), net.state_dict()
        for k, v in sd.items():
            if k.endswith("vocab_emb.weight") and v.shape[0] < cur[k].shape[0]:
                w = cur[k].clone()
                w[:v.shape[0]] = v
                sd[k] = w
    net.load_state_dict(sd)


def load_moe(model, args, vocab, device):
    """Fills a MoE: the trunk and the early head (member 0) from --moe-early if given, else from
    --moe-init; each expert head from its --moe-experts file if given, else from --moe-init's head
    (or the early file's, without --moe-init). Any plain Net file works: a training checkpoint or
    released weights, encoding 2 or 3 (the vocabulary may grow)."""
    def read(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        have = ck.get("arch", {}).get("enc", 1)
        assert enc_compatible(have, args.enc), f"{path}: enc={have}, this run has enc={args.enc}"
        ckv = list(ck["vocab"])
        assert list(vocab[:len(ckv)]) == ckv, f"{path}: vocabulary differs from this game's (not a prefix)"
        return MoE.split_net_state(ck["model"] if "model" in ck else ck["state_dict"])
    base = args.moe_early or args.moe_init
    assert base, "--moe needs --moe-init (e.g. weights/balatro-bot-0.2.pt) or --moe-early"
    trunk, head0 = read(base)
    cur = model.state_dict()
    for k, v in trunk.items():
        if k.endswith("vocab_emb.weight") and v.shape[0] < cur[k].shape[0]:
            w = cur[k].clone()
            w[:v.shape[0]] = v
            v = w
        cur[k] = v
    model.load_state_dict(cur)
    model.heads[0].load_state_dict(head0)
    init_head = read(args.moe_init)[1] if args.moe_init else head0
    given = dict(x.rsplit("=", 1) for x in args.moe_experts)
    for i, h in enumerate(EXPERT_HANDS):
        model.heads[1 + i].load_state_dict(read(given[h])[1] if h in given else init_head)
    return {"trunk": os.path.basename(os.path.dirname(base)) + "/" + os.path.basename(base),
            "experts_from_solo_runs": sorted(h for h in EXPERT_HANDS if h in given)}


def enc_compatible(have, want):
    """A network trained on encoding 2 can continue on encoding 3 (same network, more tokens)."""
    return have == want or (have, want) == (2, 3)
