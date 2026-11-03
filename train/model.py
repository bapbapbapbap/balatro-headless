"""Model v1/v2: a transformer over what the player sees, scoring each legal action.

Nothing here encodes strategy. Inputs are the tokens and actions built by harness/encode.lua;
see docs/model_v1.md.
"""
import torch
import torch.nn as nn

N_KINDS = 16       # token kinds (encode.lua KIND), 0 = padding
N_ACT_TYPES = 19   # action types (encode.lua ACTION, reroll_boss = 18 in v2), 0 = padding
F = 32             # features per token
ACT = 9            # v1 row: type, a, b, c1..c5, hand_type
ACT_WIDTH = {1: 9, 2: 18}
# Fields that are embedding ids or numbers (padded with 0); every other field is a token index (-1 = none).
ZERO_FIELDS = {1: (0, 8), 2: (0, 8, 16, 17)}


class Net(nn.Module):
    def __init__(self, n_vocab, d=96, layers=2, heads=4, enc=1):
        super().__init__()
        self.d, self.enc = d, enc
        self.feat = nn.Linear(F, d)
        self.kind_emb = nn.Embedding(N_KINDS, d)
        self.vocab_emb = nn.Embedding(n_vocab, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.type_emb = nn.Embedding(N_ACT_TYPES if enc == 2 else 18, d)   # v1 checkpoints have 18
        self.proj_a = nn.Linear(d, d)
        self.proj_b = nn.Linear(d, d)
        self.proj_c = nn.Linear(d, d)
        self.policy = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        if enc == 2:
            self.proj_s = nn.Linear(d, d)      # scoring cards
            self.proj_l = nn.Linear(d, d)      # lead card / card Death copies
            self.proj_h = nn.Linear(d, d)      # the hand type's token (level, chips, mult, increments)
            self.num = nn.Linear(2, d)         # HUD chips x mult, scoring cards' chips (slog)

    @staticmethod
    def _gather(h, idx):
        """h (B,T,d), idx (B,A) token indices with -1 for none -> (B,A,d), zeros where -1."""
        valid = (idx >= 0).unsqueeze(-1)
        g = torch.gather(h, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        return g * valid

    def forward(self, kind, vocab, feat, tok_mask, act, act_mask):
        """kind, vocab (B,T) long; feat (B,T,F) float; tok_mask (B,T) bool (True = real token);
        act (B,A,9) long; act_mask (B,A) bool (True = legal action).
        Returns logits (B,A) (illegal = -inf) and value (B,)."""
        x = self.feat(feat) + self.kind_emb(kind) + self.vocab_emb(vocab)
        h = self.norm(self.encoder(x, src_key_padding_mask=~tok_mask))
        m = tok_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1) + h[:, 0]          # mean + the GLOBAL token

        a = self.type_emb(act[..., 0])
        a = a + self.proj_a(self._gather(h, act[..., 1])) + self.proj_b(self._gather(h, act[..., 2]))
        cs = act[..., 3:8]                                                   # (B,A,5)
        B, A, _ = cs.shape
        cg = self._gather(h, cs.reshape(B, A * 5)).reshape(B, A, 5, self.d)
        n = (cs >= 0).sum(-1, keepdim=True).clamp(min=1).float()
        a = a + self.proj_c(cg.sum(2) / n)
        ht = act[..., 8]
        a = a + self.vocab_emb(ht) * (ht > 0).unsqueeze(-1)
        if self.enc == 2:
            ss = act[..., 9:14]
            sg = self._gather(h, ss.reshape(B, A * 5)).reshape(B, A, 5, self.d)
            ns = (ss >= 0).sum(-1, keepdim=True).clamp(min=1).float()
            a = a + self.proj_s(sg.sum(2) / ns)
            a = a + self.proj_l(self._gather(h, act[..., 14])) + self.proj_h(self._gather(h, act[..., 15]))
            a = a + self.num(act[..., 16:18].float() / 10000.0)

        logits = self.policy(torch.cat([a, pooled.unsqueeze(1).expand(-1, A, -1)], -1)).squeeze(-1)
        logits = logits.masked_fill(~act_mask, float("-inf"))
        return logits, self.value(pooled).squeeze(-1)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------- mixture of experts (docs/hypotheses.md section I, chud's design) ----------------
# The hand types that get an expert, in a fixed order: member 1 + i is the expert for EXPERT_HANDS[i].
EXPERT_HANDS = ["Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House",
                "Four of a Kind", "Straight Flush"]


class MoE(nn.Module):
    """Member 0 is the early-game model; members 1..8 are the hand experts. Each decision is made
    by exactly one member, named per observation (collate's 7th tensor, `member`). Each member is a
    Net with its own weights; a member's parameters get gradient only from its own decisions.

    Buffers travel with the weights to the workers:
      base_rate (8,): average plays of each expert's hand in the early game (antes 1-3) by the
        early-game model, over recent training games that reached the handoff. The router's
        preference is a hand's plays in this run relative to that rate.
      pin_scale (1,): multiplier on every expert's pin rate (annealed from 1 to 0)."""

    def __init__(self, n_vocab, d=96, layers=2, heads=4, enc=1):
        super().__init__()
        self.members = nn.ModuleList([Net(n_vocab, d, layers, heads, enc) for _ in range(1 + len(EXPERT_HANDS))])
        self.register_buffer("base_rate", torch.zeros(len(EXPERT_HANDS)))
        self.register_buffer("pin_scale", torch.ones(1))

    def forward(self, kind, vocab, feat, tok_mask, act, act_mask, member):
        B, A = act.shape[0], act.shape[1]
        logits = torch.full((B, A), float("-inf"), device=act.device)
        value = torch.zeros(B, device=act.device)
        for m in torch.unique(member).tolist():
            i = (member == m).nonzero(as_tuple=True)[0]
            lg, v = self.members[m](kind[i], vocab[i], feat[i], tok_mask[i], act[i], act_mask[i])
            logits = logits.index_put((i,), lg)
            value = value.index_put((i,), v)
        return logits, value
