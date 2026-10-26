"""Smoke tests of the prism package on random inputs (CPU, a few seconds). Run with: python -m pytest tests/"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from prism.model import PRISM, corn_probs, corn_loss  # noqa: E402
from prism.losses import prism_loss  # noqa: E402
from prism.metrics import metrics3, net_benefit, fit_cutpoints  # noqa: E402
from prism.stats import holm, delong_test, paired_qwk_pvalue  # noqa: E402

B, V, N, D_TOK, S, N_PRIV, N_CLIN = 6, 4, 160, 384, 16, 44, 5


def batch(seed=0, with_ct=True):
    g = torch.Generator().manual_seed(seed)
    b = dict(xtok=torch.randn(B, V, N, D_TOK, generator=g), tokmask=torch.ones(B, V, N, dtype=torch.bool),
             xmask=torch.tensor([[1, 1, 0, 0]] * B, dtype=torch.bool), view=torch.tensor([[0, 1, 2, 3]] * B),
             clin=torch.randn(B, N_CLIN, generator=g), cttok=torch.randn(B, S, D_TOK, generator=g), hu=torch.randn(B, 3, generator=g),
             y=torch.tensor([0, 1, 2, 0, 1, 2]), t=torch.randn(B, generator=g),
             priv=torch.randn(B, N_PRIV, generator=g), priv_m=torch.ones(B, N_PRIV, dtype=torch.bool))
    mods = torch.ones(B, 3, dtype=torch.bool)
    if not with_ct:
        mods[:, 1] = False
    return b, mods


def test_trainable_parameters_match_paper():
    m = PRISM(D_TOK, N, S, N_PRIV, N_CLIN)
    n = sum(p.numel() for p in m.parameters())
    assert abs(n / 1e6 - 0.92) < 0.01, n            # 0.92 M trainable parameters reported in the paper


def test_forward_shapes_and_probabilities():
    m = PRISM(D_TOK, N, S, N_PRIV, N_CLIN).eval()
    b, mods = batch()
    e = m.encode(b)
    o = m.fuse(e, mods)
    assert o['corn'].shape == (B, 2) and o['t'].shape == (B,) and o['priv'].shape == (B, N_PRIV)
    assert e['a_sp'].shape == (B, V, N)
    p = corn_probs(o['corn'])
    assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-4)
    # masked views receive no attention
    assert torch.all(e['a_v'][:, 2:] < 1e-3)


def test_missing_ct_handled():
    m = PRISM(D_TOK, N, S, N_PRIV, N_CLIN).eval()
    b, mods = batch(with_ct=False)
    o = m.fuse(m.encode(b), mods)
    assert torch.all(o['a_m'][:, 1] < 1e-3)          # CT modality masked in the fusion


@pytest.mark.parametrize('clin_enc,film', [('mlp', True), ('linear', True), ('mlp', False), ('linear', False)])
def test_loss_backward(clin_enc, film):
    m = PRISM(D_TOK, N, S, N_PRIV, N_CLIN, clin_enc=clin_enc, film=film).train()
    b, mods = batch()
    e = m.encode(b, tok_drop=0.1)
    loss = prism_loss(m, e, m.fuse(e, mods), mods, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.head_corn.weight.grad is not None


def test_corn_loss_ordinal():
    y = torch.tensor([0, 1, 2])
    good = torch.tensor([[-5., 0.], [5., -5.], [5., 5.]])
    bad = -good
    assert corn_loss(good, y) < corn_loss(bad, y)


def test_metrics_and_stats():
    rs = np.random.RandomState(0)
    y = rs.randint(0, 3, 300)
    p = np.eye(3)[y] * 0.6 + rs.dirichlet([1, 1, 1], 300) * 0.4
    m = metrics3(y, p)
    assert m['qwk'] > 0.5 and 0.5 < m['auc_osteoporosis'] <= 1
    assert delong_test(y == 2, p[:, 2], p[:, 2])[2] == pytest.approx(1.0)
    assert paired_qwk_pvalue(y, p, p, n_boot=50) == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert net_benefit((y == 2).astype(int), np.zeros(300), 0.2) == 0
    c1, c2 = fit_cutpoints(p[:, 1] + 2 * p[:, 2], y)
    assert c1 < c2
