"""
freq_beamsplitter
=================
Simulator for EO-modulated ring resonator scattering matrices.

Based on:
    Buddhiraju et al., Nature Communications (2021)
    "Arbitrary linear transformations for photons in the frequency synthetic dimension"

Quick start
-----------
    from freq_beamsplitter import scattering_matrix, plot_matrix

    import numpy as np
    kappa = np.array([0.3+0.1j, 0.2-0.15j])
    M = scattering_matrix(kappa, N_sb=2)
    plot_matrix(M, title="Single ring")
"""

from .core import (
    build_K,
    scattering_matrix,
    cascaded_scattering_matrix,
    unitarity_error,
    fidelity,
)

from .visualise import (
    plot_matrix,
    plot_matrix_grid,
)

from .optimise import (
    haar_unitary,
    inverse_design,
    InverseDesignResult,
)

__all__ = [
    # core
    "build_K",
    "scattering_matrix",
    "cascaded_scattering_matrix",
    "unitarity_error",
    "fidelity",
    # visualise
    "plot_matrix",
    "plot_matrix_grid",
    # optimise
    "haar_unitary",
    "inverse_design",
    "InverseDesignResult",
]

__version__ = "0.1.0"
__author__  = "Kok-Wei"
