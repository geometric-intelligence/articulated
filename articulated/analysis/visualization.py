"""Visualization utilities for representation analysis.

Tasks:
1. Visualize latent trajectories in PCA space
2. Plot tuning curves
3. Create publication-quality figures
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_latent_trajectory(
    latent_trajectory: np.ndarray,
    configurations: Optional[np.ndarray] = None,
    color_by: str = "time",
    ax: Optional[plt.Axes] = None,
    title: str = "Latent Trajectory",
) -> plt.Figure:
    """Plot a trajectory in latent space.

    Args:
        latent_trajectory: Array of shape (n_timesteps, 2 or 3).
        configurations: Optional config values for coloring.
        color_by: How to color points ('time', 'config', or 'none').
        ax: Optional matplotlib axes.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        if latent_trajectory.shape[1] == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    n_timesteps = latent_trajectory.shape[0]

    # Determine colors
    if color_by == "time":
        colors = np.arange(n_timesteps)
        cmap = "viridis"
    elif color_by == "config" and configurations is not None:
        colors = configurations[:, 0]  # Color by first config dimension
        cmap = "coolwarm"
    else:
        colors = "blue"
        cmap = None

    # Plot
    if latent_trajectory.shape[1] == 3:
        scatter = ax.scatter(
            latent_trajectory[:, 0],
            latent_trajectory[:, 1],
            latent_trajectory[:, 2],
            c=colors,
            cmap=cmap,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        scatter = ax.scatter(
            latent_trajectory[:, 0],
            latent_trajectory[:, 1],
            c=colors,
            cmap=cmap,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    ax.set_title(title)

    if cmap:
        plt.colorbar(scatter, ax=ax, label=color_by.capitalize())

    return fig


def plot_tuning_curves(
    tuning_curves: np.ndarray,
    neuron_indices: list[int],
    config_dim: int = 0,
    config_labels: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot tuning curves for selected neurons.

    Args:
        tuning_curves: Array of shape (hidden_size, n_config_dims, n_bins).
        neuron_indices: Which neurons to plot.
        config_dim: Which configuration dimension to show.
        config_labels: Labels for configuration bins.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    n_bins = tuning_curves.shape[2]
    x = np.arange(n_bins)

    for neuron_idx in neuron_indices:
        curve = tuning_curves[neuron_idx, config_dim]
        ax.plot(x, curve, label=f"Neuron {neuron_idx}", alpha=0.8)

    ax.set_xlabel("Configuration bin")
    ax.set_ylabel("Mean activity")
    ax.set_title(f"Tuning curves (config dim {config_dim})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig


def plot_2d_tuning_map(
    hidden_states: np.ndarray,
    configurations: np.ndarray,
    neuron_idx: int,
    config_dims: tuple[int, int] = (0, 1),
    n_bins: int = 20,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot 2D tuning map for a single neuron.

    Shows how a neuron's activity varies over two configuration dimensions.

    Args:
        hidden_states: Array of shape (n_samples, hidden_size).
        configurations: Array of shape (n_samples, n_config_dims).
        neuron_idx: Which neuron to plot.
        config_dims: Which two config dimensions to use.
        n_bins: Number of bins per dimension.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    dim1, dim2 = config_dims
    x = configurations[:, dim1]
    y = configurations[:, dim2]
    z = hidden_states[:, neuron_idx]

    # Create 2D histogram
    tuning_map = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))

    x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), n_bins + 1)

    x_idx = np.clip(np.digitize(x, x_bins[:-1]) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(y, y_bins[:-1]) - 1, 0, n_bins - 1)

    for i in range(len(z)):
        tuning_map[y_idx[i], x_idx[i]] += z[i]
        counts[y_idx[i], x_idx[i]] += 1

    # Average
    with np.errstate(invalid="ignore"):
        tuning_map = tuning_map / counts
    tuning_map = np.nan_to_num(tuning_map)

    # Plot
    im = ax.imshow(
        tuning_map,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="hot",
    )
    ax.set_xlabel(f"Config dim {dim1}")
    ax.set_ylabel(f"Config dim {dim2}")
    ax.set_title(f"Neuron {neuron_idx} tuning map")
    plt.colorbar(im, ax=ax, label="Activity")

    return fig
