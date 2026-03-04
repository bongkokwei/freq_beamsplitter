"""
Visualisation utilities for ring scattering matrices.
Depends on: ring_scattering.py

Usage (standalone demo):
    python plot_scattering.py

Or import in your own script:
    from plot_scattering import plot_matrix, plot_matrix_grid
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Single matrix plot
# ---------------------------------------------------------------------------

def plot_matrix(
    M: np.ndarray,
    title: str = "Scattering matrix",
    ax_amp=None,
    ax_phase=None,
    amp_vmax: float = 1.0,
) -> plt.Figure:
    """
    Plot amplitude and phase of a single scattering matrix M.

    Parameters
    ----------
    M : complex ndarray, shape (N, N)
    title : str
        Figure/suptitle label.
    ax_amp, ax_phase : matplotlib Axes, optional
        Pass existing axes to embed in a larger figure (e.g. via plot_matrix_grid).
        If None, a new figure is created.
    amp_vmax : float
        Colour scale maximum for |M|. Default 1.0 (unitary range).

    Returns
    -------
    fig : matplotlib Figure (None if axes were provided externally)
    """
    N = M.shape[0]
    labels = [str(m) for m in range(-(N // 2), N // 2 + 1)]

    own_fig = ax_amp is None
    if own_fig:
        fig, (ax_amp, ax_phase) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(title, fontsize=12)
    else:
        fig = None
        ax_amp.set_title(title, fontsize=10)

    # Amplitude
    im0 = ax_amp.imshow(np.abs(M), vmin=0, vmax=amp_vmax, cmap="viridis")
    plt.colorbar(im0, ax=ax_amp, label="|M|")
    ax_amp.set_xlabel("Input mode m'")
    ax_amp.set_ylabel("Output mode m")
    ax_amp.set_xticks(range(N)); ax_amp.set_xticklabels(labels)
    ax_amp.set_yticks(range(N)); ax_amp.set_yticklabels(labels)
    if own_fig:
        ax_amp.set_title("|M|")

    # Phase
    im1 = ax_phase.imshow(np.angle(M) / np.pi, vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar(im1, ax=ax_phase, label="phase / π")
    ax_phase.set_xlabel("Input mode m'")
    ax_phase.set_ylabel("Output mode m")
    ax_phase.set_xticks(range(N)); ax_phase.set_xticklabels(labels)
    ax_phase.set_yticks(range(N)); ax_phase.set_yticklabels(labels)
    if own_fig:
        ax_phase.set_title("∠M / π")

    if own_fig:
        plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-panel grid
# ---------------------------------------------------------------------------

def plot_matrix_grid(
    matrices: list,
    titles: list,
    amp_vmax: float = 1.0,
    savepath: str = None,
) -> plt.Figure:
    """
    Plot several scattering matrices side-by-side in a single figure.
    Each matrix gets two columns: |M| and ∠M.

    Parameters
    ----------
    matrices : list of complex ndarrays
    titles   : list of str, same length as matrices
    amp_vmax : float
        Shared colour scale max for all |M| panels.
    savepath : str, optional
        If given, saves the figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(matrices)
    assert len(titles) == n, "matrices and titles must have the same length"

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    N = matrices[0].shape[0]
    labels = [str(m) for m in range(-(N // 2), N // 2 + 1)]

    for col, (M, title) in enumerate(zip(matrices, titles)):
        ax_amp   = axes[0, col]
        ax_phase = axes[1, col]

        im0 = ax_amp.imshow(np.abs(M), vmin=0, vmax=amp_vmax, cmap="viridis")
        plt.colorbar(im0, ax=ax_amp, label="|M|")
        ax_amp.set_title(title, fontsize=10)
        ax_amp.set_xlabel("Input mode m'")
        ax_amp.set_ylabel("Output mode m")
        ax_amp.set_xticks(range(N)); ax_amp.set_xticklabels(labels)
        ax_amp.set_yticks(range(N)); ax_amp.set_yticklabels(labels)

        im1 = ax_phase.imshow(np.angle(M) / np.pi, vmin=-1, vmax=1, cmap="RdBu")
        plt.colorbar(im1, ax=ax_phase, label="phase / π")
        ax_phase.set_title(f"∠M / π", fontsize=10)
        ax_phase.set_xlabel("Input mode m'")
        ax_phase.set_ylabel("Output mode m")
        ax_phase.set_xticks(range(N)); ax_phase.set_xticklabels(labels)
        ax_phase.set_yticks(range(N)); ax_phase.set_yticklabels(labels)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")

    return fig


# ---------------------------------------------------------------------------
# Standalone demo — mirrors ring_scattering.py demo cases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from freq_beamsplitter.core import (
        scattering_matrix,
        cascaded_scattering_matrix,
        unitarity_error,
    )

    N_sb = 2
    N_f  = 4
    rng  = np.random.default_rng(42)
    kappa = rng.random(N_f) * 0.5 + 1j * rng.random(N_f) * 0.5

    M_ideal = scattering_matrix(kappa, N_sb)
    M_lossy = scattering_matrix(kappa, N_sb, gamma_i=0.05)
    M_det   = scattering_matrix(kappa, N_sb, delta_omega=0.5)

    N_r = 4
    kappa_list = [rng.random(N_f) * 0.5 + 1j * rng.random(N_f) * 0.5 for _ in range(N_r)]
    M_casc = cascaded_scattering_matrix(kappa_list, N_sb)

    # Individual plots
    fig1 = plot_matrix(M_ideal, title="Lossless, on-resonance")
    fig2 = plot_matrix(M_lossy, title="γ_i = 0.05 γ_e")
    fig3 = plot_matrix(M_det,   title="Δω = 0.5 γ_e")
    fig4 = plot_matrix(M_casc,  title="4 cascaded rings")

    # Combined grid
    fig_grid = plot_matrix_grid(
        [M_ideal, M_lossy, M_det, M_casc],
        ["Lossless", "γ_i = 0.05 γ_e", "Δω = 0.5 γ_e", "4 rings (cascaded)"],
        savepath="/mnt/user-data/outputs/M_grid.png",
    )

    plt.show()
