"""Dimensionality reduction tools for representation analysis.

Tasks:
1. Apply PCA to hidden states
2. Apply UMAP for nonlinear embedding
3. Analyze how latent dimensions relate to configuration space
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_pca(
    hidden_states: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, PCA, dict]:
    """Compute PCA on hidden states.

    Args:
        hidden_states: Array of shape (n_samples, hidden_size).
        n_components: Number of principal components.

    Returns:
        Tuple of (projected_states, pca_model, stats).
        projected_states: Shape (n_samples, n_components).
        stats: Dictionary with explained variance info.
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(hidden_states)

    stats = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components_95": np.argmax(
            np.cumsum(pca.explained_variance_ratio_) >= 0.95
        ) + 1,
    }

    return projected, pca, stats