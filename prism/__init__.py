"""PRISM: opportunistic osteoporosis screening from lumbar radiographs (LUMOS benchmark).

The reference pipeline is `notebooks/PRISM_LUMOS.ipynb`. This package exposes the same model,
loss, metrics and statistical tests as importable modules.

The model and losses need PyTorch and are imported explicitly:
    from prism.model import PRISM, corn_probs
    from prism.losses import prism_loss
Metrics and statistics only need NumPy, SciPy and scikit-learn.
"""
from .metrics import metrics3, expected_calibration_error, net_benefit, operating_points, fit_cutpoints
from .stats import delong_test, bootstrap_ci, paired_qwk_pvalue, holm

__version__ = '1.0.0'
