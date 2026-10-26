"""Statistical tests used in the paper (same implementation and seeds as in the notebook)."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


def delong_test(y_true, s1, s2):
    """DeLong test for two correlated AUCs (same patients). Returns (auc1, auc2, two-sided p-value)."""
    y_true = np.asarray(y_true).astype(bool)
    pos, neg = np.where(y_true)[0], np.where(~y_true)[0]
    m, n = len(pos), len(neg)

    def comps(s):
        X, Yn = s[pos], s[neg]
        psi = (X[:, None] > Yn[None, :]) + 0.5 * (X[:, None] == Yn[None, :])
        return psi.mean(), psi.mean(1), psi.mean(0)

    a1, v10_1, v01_1 = comps(np.asarray(s1))
    a2, v10_2, v01_2 = comps(np.asarray(s2))
    S10 = np.cov(np.vstack([v10_1, v10_2]))
    S01 = np.cov(np.vstack([v01_1, v01_2]))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(max(var, 1e-12))
    return a1, a2, 2 * (1 - norm.cdf(abs(z)))


def bootstrap_ci(y, prob, n_boot=1000, seed=0):
    """Percentile bootstrap 95% CIs (patients resampled) for QWK, osteoporosis AUC and macro F1."""
    rs = np.random.RandomState(seed)
    q, a, f = [], [], []
    pred = prob.argmax(1)
    for _ in range(n_boot):
        i = rs.randint(0, len(y), len(y))
        q.append(cohen_kappa_score(y[i], pred[i], weights='quadratic'))
        a.append(roc_auc_score(y[i] == 2, prob[i, 2]))
        f.append(f1_score(y[i], pred[i], average='macro'))
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    return dict(qwk_ci=ci(q), auc_osteo_ci=ci(a), f1_ci=ci(f))


def paired_qwk_pvalue(y, pa, pb, n_boot=1000, seed=0):
    """Two-sided paired bootstrap p-value for the QWK difference between two models: p = 2 min[P(D <= 0), P(D >= 0)]."""
    rs = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        i = rs.randint(0, len(y), len(y))
        d.append(cohen_kappa_score(y[i], pa[i].argmax(1), weights='quadratic')
                 - cohen_kappa_score(y[i], pb[i].argmax(1), weights='quadratic'))
    d = np.asarray(d)
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(pvals):
    """Holm-Bonferroni adjustment; NaN entries are left untouched."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    run = 0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * p[ok][i]))
        adj[i] = run
    out[ok] = adj
    return out
