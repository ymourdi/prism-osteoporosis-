"""Evaluation metrics used in the paper (same definitions as in the notebook)."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, recall_score,
                             roc_auc_score, roc_curve)


def metrics3(y, prob, t_true=None, t_pred=None, pred=None):
    """Three-class metrics; the predicted class is the most probable one unless `pred` is given."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = prob.argmax(1) if pred is None else np.asarray(pred)
    m = dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
             f1_macro=f1_score(y, pred, average='macro'), qwk=cohen_kappa_score(y, pred, weights='quadratic'),
             auc_macro=roc_auc_score(y, prob, multi_class='ovr', average='macro'),
             auc_osteoporosis=roc_auc_score(y == 2, prob[:, 2]),
             sens_osteoporosis=recall_score(y == 2, pred == 2), spec_osteoporosis=recall_score(y != 2, pred != 2))
    if t_pred is not None:
        m['t_mae'] = float(np.mean(np.abs(np.asarray(t_true) - np.asarray(t_pred))))
        m['t_spearman'] = float(spearmanr(t_true, t_pred).correlation)
    return m


def expected_calibration_error(y_bin, p, n_bins=10):
    """Mean absolute difference between observed frequency and mean predicted probability over quantile bins
    (as reported in the paper, unweighted across bins)."""
    from sklearn.calibration import calibration_curve
    frac, mean_pred = calibration_curve(y_bin, p, n_bins=n_bins, strategy='quantile')
    return float(np.mean(np.abs(frac - mean_pred)))


def net_benefit(y_bin, p, thr):
    """Decision-curve net benefit: TP/n - FP/n * thr / (1 - thr)."""
    y_bin = np.asarray(y_bin).astype(int)
    pred = np.asarray(p) >= thr
    n = len(y_bin)
    tp = (pred & (y_bin == 1)).sum()
    fp = (pred & (y_bin == 0)).sum()
    return tp / n - fp / n * thr / (1 - thr)


def operating_points(y_bin, p, targets=(0.80, 0.85, 0.90, 0.95)):
    """Thresholds on the osteoporosis probability reaching each target sensitivity (chosen on the same data)."""
    y_bin = np.asarray(y_bin).astype(bool)
    rows = []
    for target in targets:
        fpr, tpr, thr = roc_curve(y_bin, p)
        i = int(np.argmax(tpr >= target))
        pred = p >= thr[i]
        tp, fp = (pred & y_bin).sum(), (pred & ~y_bin).sum()
        rows.append(dict(target_sensitivity=target, threshold=thr[i], sensitivity=tpr[i], specificity=1 - fpr[i],
                         ppv=tp / max(pred.sum(), 1), npv=((~pred) & ~y_bin).sum() / max((~pred).sum(), 1),
                         referred=int(pred.sum())))
    return rows


def fit_cutpoints(score, y):
    """Ordinal cut-points on the expected score s = p1 + 2 p2 maximizing QWK (inner validation set)."""
    grid = np.linspace(0.2, 1.8, 81)
    best = (-1, 0.5, 1.5)
    for c1 in grid:
        for c2 in grid[grid > c1]:
            q = cohen_kappa_score(y, np.digitize(score, [c1, c2]), weights='quadratic')
            if q > best[0]:
                best = (q, c1, c2)
    return best[1:]
