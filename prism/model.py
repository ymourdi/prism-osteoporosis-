"""PRISM head: gated-attention pooling of image tokens, views and modalities, FiLM conditioning on
clinical variables, and three output heads (CORN ordinal, T-score, privileged DXA measurements).

This module reproduces, as a standalone import, the model defined in `notebooks/PRISM_LUMOS.ipynb`
(section 9). The notebook remains the reference implementation used to produce the results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 256          # internal dimension d
N_HU = 3               # opportunistic CT densitometry descriptors (median HU, 25th percentile HU, bone fraction)


class GatedAttnPool(nn.Module):
    """Gated attention pooling (Ilse et al., ICML 2018) with an optional boolean mask (True = keep)."""

    def __init__(self, d, h=128):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)

    def forward(self, x, mask=None):
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1)
        return (w.unsqueeze(-1) * x).sum(1), w


def mlp(i, o, p=0.2):
    return nn.Sequential(nn.Linear(i, o), nn.GELU(), nn.Dropout(p), nn.Linear(o, o))


class PRISM(nn.Module):
    """PRISM head operating on frozen-encoder tokens.

    Args:
        d_tok: dimension of the radiograph tokens (384 after whitened PCA).
        n_tok: number of spatial tokens per image (16 x 10 = 160).
        n_slices: number of CT slices (16).
        n_priv: number of privileged DXA targets (44), used during training only.
        n_clin: number of clinical inputs (5, or 6 with the implant flag).
        d_ct: dimension of the CT slice tokens (defaults to d_tok).
        n_view_types: AP / LAT, whole image or spine crop (4).
        clin_enc: 'mlp' (5 -> 256 -> 256) or 'linear' (5 -> 256).
        film: whether the clinical representation modulates the radiographic one (FiLM).
    """

    def __init__(self, d_tok, n_tok, n_slices, n_priv, n_clin, d=D_MODEL, n_hu=N_HU, p=0.2, d_ct=None,
                 n_view_types=4, clin_enc='mlp', film=True):
        super().__init__()
        d_ct = d_ct or d_tok
        self.x_proj = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d), nn.GELU(), nn.Dropout(p))
        self.x_pos = nn.Parameter(torch.zeros(1, n_tok, d))
        self.x_pool = GatedAttnPool(d)
        self.view_emb = nn.Embedding(n_view_types, d)
        self.v_pool = GatedAttnPool(d)
        self.ct_proj = nn.Sequential(nn.LayerNorm(d_ct), nn.Linear(d_ct, d), nn.GELU(), nn.Dropout(p))
        self.ct_pos = nn.Parameter(torch.zeros(1, n_slices, d))
        self.ct_pool = GatedAttnPool(d)
        self.hu_mlp = mlp(n_hu, d, p)
        self.clin_mlp = mlp(n_clin, d, p) if clin_enc == 'mlp' else nn.Linear(n_clin, d)
        self.film = nn.Linear(d, 2 * d) if film else None
        self.mod_emb = nn.Parameter(torch.zeros(1, 3, d))
        self.m_pool = GatedAttnPool(d)
        self.trunk = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p))
        self.head_corn = nn.Linear(d, 2)
        self.head_t = nn.Linear(d, 1)
        self.head_priv = nn.Linear(d, n_priv)
        nn.init.normal_(self.x_pos, std=0.02)
        nn.init.normal_(self.ct_pos, std=0.02)
        nn.init.normal_(self.mod_emb, std=0.02)

    def encode(self, b, tok_drop=0.0):
        """Encodes a batch dict with keys xtok (B,V,N,d_tok), tokmask (B,V,N), xmask (B,V), view (B,V),
        clin (B,n_clin), cttok (B,S,d_ct) and hu (B,n_hu)."""
        B, V, N, _ = b['xtok'].shape
        t = self.x_proj(b['xtok']).reshape(B * V, N, -1) + self.x_pos
        m = b['tokmask'].reshape(B * V, N)
        if tok_drop > 0:
            m = m & (torch.rand(B * V, N, device=t.device) > tok_drop)
        m = m | ~m.any(1, keepdim=True)                                 # never mask every token
        ex, a_sp = self.x_pool(t, m)
        ex = ex.reshape(B, V, -1) + self.view_emb(b['view'])
        xm = b['xmask']
        ex, a_v = self.v_pool(ex, xm | ~xm.any(1, keepdim=True))
        ec = self.clin_mlp(b['clin'])
        if self.film is not None:                                       # FiLM: clinical variables modulate the image
            g, be = self.film(ec).chunk(2, -1)
            ex = ex * (1 + g) + be
        ct = self.ct_proj(b['cttok']) + self.ct_pos
        ect, a_ct = self.ct_pool(ct)
        ect = ect + self.hu_mlp(b['hu'])
        return dict(ex=ex, ect=ect, ec=ec, a_sp=a_sp.reshape(B, V, N), a_v=a_v, a_ct=a_ct)

    def fuse(self, e, mods_mask):
        """Fuses the modality representations; mods_mask (B,3) marks available [radiograph, CT, clinical]."""
        M = torch.stack([e['ex'], e['ect'], e['ec']], 1) + self.mod_emb
        h, a_m = self.m_pool(M, mods_mask)
        h = self.trunk(torch.cat([h, e['ec']], -1))
        return dict(corn=self.head_corn(h), t=self.head_t(h).squeeze(-1), priv=self.head_priv(h), a_m=a_m)


def corn_probs(logits):
    """Class probabilities from the two CORN logits: P(y > normal) and P(y > osteopenia | y > normal)."""
    s1 = torch.sigmoid(logits[:, 0])
    s2 = s1 * torch.sigmoid(logits[:, 1])
    return torch.stack([1 - s1, s1 - s2, s2], 1).clamp(1e-6, 1)


def corn_loss(logits, y):
    """CORN loss (Shi, Cao and Raschka, 2023): the second task is trained on patients above 'normal' only."""
    l = F.binary_cross_entropy_with_logits(logits[:, 0], (y > 0).float())
    sel = y > 0
    if sel.any():
        l = l + F.binary_cross_entropy_with_logits(logits[sel, 1], (y[sel] > 1).float())
    return l
