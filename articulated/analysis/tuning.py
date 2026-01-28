"""Tuning curve analysis for neural representations.

Tasks:
1. Compute tuning curves of neurons w.r.t. configuration variables
2. Identify neurons selective to specific joint angles
3. Compare to known grid cell patterns from spatial navigation
"""

import numpy as np


def compute_tuning_curves(
    hidden_states: np.ndarray,
    configuration_variables: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """Compute tuning curves of neurons with respect to configuration variables."""
    raise NotImplementedError("Not implemented")
