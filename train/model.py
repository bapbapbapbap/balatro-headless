"""Model v1: a small transformer over what the player sees, scoring each legal action.

Nothing here encodes strategy. Inputs are the tokens and actions built by harness/encode.lua;
see docs/model_v1.md.
"""
import torch
import torch.nn as nn

N_KINDS = 16       # token kinds (encode.lua KIND), 0 = padding
N_ACT_TYPES = 18   # action types (encode.lua ACTION), 0 = padding
F = 32             # features per token
ACT = 9            # type, a, b, c1..c5, hand_type


class Net(nn.Module):
    def __init__(self, n_vocab, d=96, layers=2, heads=4):
        super().__init__()
        self.d = d
        self.feat = nn.Linear(F, d)
        self.kind_emb = nn.Embedding(N_KINDS, d)
        self.vocab_emb = nn.Embedding(n_vocab, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.type_emb = nn.Embedding(N_ACT_TYPES, d)
        self.proj_a = nn.Linear(d, d)
        self.proj_b = nn.Linear(d, d)
        self.proj_c = nn.Linear(d, d)
        self.policy = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

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

        logits = self.policy(torch.cat([a, pooled.unsqueeze(1).expand(-1, A, -1)], -1)).squeeze(-1)
        logits = logits.masked_fill(~act_mask, float("-inf"))
        return logits, self.value(pooled).squeeze(-1)


def param_count(model):
    return sum(p.numel() for p in model.parameters())
