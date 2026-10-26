"""Training objective of PRISM (Eq. 2-4 of the paper), as in `compute_loss` of the notebook.

L = L_CORN + lambda_T * SmoothL1(t_hat, t) + lambda_priv * L_priv + lambda_dist * L_dist,
with lambda_T = lambda_priv = lambda_dist = 0.5 (fixed a priori, not tuned).
"""
import torch
import torch.nn.functional as F

from .model import corn_loss, corn_probs

LAMBDA_T, LAMBDA_PRIV, LAMBDA_DISTILL = 0.5, 0.5, 0.5


def privileged_loss(priv_pred, priv_true, priv_mask):
    """Masked MSE on the standardized DXA measurements (learning using privileged information).
    priv_mask is 1 where the measurement is available."""
    pm = priv_mask.float()
    return (((priv_pred - priv_true) ** 2) * pm).sum() / pm.sum().clamp_min(1)


def distillation_loss(out_full, out_no_ct, has_ct):
    """Self-distillation: predictions with CT masked (student) match those with all modalities (teacher,
    stop-gradient), on the patients that have a CT scan."""
    if not has_ct.any():
        return out_full['t'].new_zeros(())
    pt = corn_probs(out_full['corn'][has_ct]).detach()
    ps = corn_probs(out_no_ct['corn'][has_ct])
    kl = (pt * (pt.log() - ps.log())).sum(1).mean()
    return kl + F.mse_loss(out_no_ct['t'][has_ct], out_full['t'][has_ct].detach())


def prism_loss(model, e, out, mods_mask, batch, use_priv=True, use_distill=True, train=True,
               lambda_t=LAMBDA_T, lambda_priv=LAMBDA_PRIV, lambda_distill=LAMBDA_DISTILL):
    """Total loss for one batch, given the encoded representations `e = model.encode(batch)` and the fused
    outputs `out = model.fuse(e, mods_mask)`."""
    loss = corn_loss(out['corn'], batch['y']) + lambda_t * F.smooth_l1_loss(out['t'], batch['t'])
    if use_priv:
        loss = loss + lambda_priv * privileged_loss(out['priv'], batch['priv'], batch['priv_m'])
    if use_distill and train:
        has = mods_mask[:, 1]
        if has.any():
            mm2 = mods_mask.clone()
            mm2[:, 1] = False
            loss = loss + lambda_distill * distillation_loss(out, model.fuse(e, mm2), has)
    return loss
