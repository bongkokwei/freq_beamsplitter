"""
freq_beamsplitter.heralded_design
==================================
Inverse design for a *heralded* 2x2 gate on a chosen (i, j) subspace.

Cost function:
    L = 1 - F_block(M_ij, U_2)

Leakage out of {i, j} is NOT penalised.  Instead, the heralding success
probability P_ij = sum |M[a,b]|^2 over a, b in {i, j} is reported as a
separate experimental figure of merit.

This is the right framing when:
  - the platform structurally cannot deliver U_2 (+) I (single ring,
    banded K -> mode mixing leaks across the lattice).
  - post-selection on outputs in {i, j} is operationally available.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize

from .core import cascaded_scattering_matrix

# ---------------------------------------------------------------------------
# Pack/unpack (matches optimise.py convention)
# ---------------------------------------------------------------------------


def _pack(kappa_list: list) -> np.ndarray:
    flat = np.concatenate([k for k in kappa_list])
    return np.concatenate([flat.real, flat.imag])


def _unpack(x: np.ndarray, N_r: int, N_f: int) -> list:
    total = N_r * N_f
    real_part = x[:total].reshape(N_r, N_f)
    imag_part = x[total:].reshape(N_r, N_f)
    return [real_part[r] + 1j * imag_part[r] for r in range(N_r)]


# ---------------------------------------------------------------------------
# Block metrics
# ---------------------------------------------------------------------------


def block_fidelity(M: np.ndarray, U2: np.ndarray, i: int, j: int) -> float:
    """
    Global-phase invariant fidelity between M[{i,j}, {i,j}] and U2.

        F = |<U2, M_ij>| / (||U2||_F ||M_ij||_F)

    F = 1 means M_ij is proportional to U2 (gate shape exact, magnitude TBD).
    The *magnitude* is reported separately via heralding_probability().
    """
    M_block = M[np.ix_([i, j], [i, j])]
    norm = np.linalg.norm(U2) * np.linalg.norm(M_block)
    if norm == 0:
        return 0.0
    return float(np.abs(np.vdot(U2, M_block)) / norm)


def heralding_probability(M: np.ndarray, i: int, j: int) -> float:
    """
    Subspace retention summed over {i, j} inputs:

        P_ij = sum_{a, b in {i,j}} |M[a, b]|^2
        ranges over [0, 2]; equals 2 if no leakage.

    Operationally: when you inject a frequency-bin qubit into {i, j} and
    post-select on outputs in {i, j}, the average success probability per
    input mode is P_ij / 2.
    """
    M_block = M[np.ix_([i, j], [i, j])]
    return float(np.sum(np.abs(M_block) ** 2))


# ---------------------------------------------------------------------------
# Loss + finite-difference gradient
# ---------------------------------------------------------------------------


def _loss_and_grad(
    x,
    U2,
    i,
    j,
    N_sb,
    N_r,
    N_f,
    gamma_e,
    gamma_i,
    delta_omega_list,
    eps=1e-6,
):
    """L = 1 - F_block.  Centred finite differences."""

    def loss_fn(xv):
        kl = _unpack(xv, N_r, N_f)
        M = cascaded_scattering_matrix(
            kl,
            N_sb,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            delta_omega_list=delta_omega_list,
        )
        return 1.0 - block_fidelity(M, U2, i, j)

    f0 = loss_fn(x)
    grad = np.zeros_like(x)
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        grad[k] = (loss_fn(xp) - loss_fn(xm)) / (2 * eps)
    return f0, grad


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class HeraldedDesignResult:
    kappa_list: list
    block_fidelity: float
    heralding_prob: float  # P_ij in [0, 2]
    success_per_input: float  # P_ij / 2  (heralding success per mode)
    loss_history: list = field(default_factory=list)
    n_restarts: int = 0
    converged: bool = False

    def __repr__(self):
        return (
            f"HeraldedDesignResult("
            f"F_block={self.block_fidelity:.6f}, "
            f"P_ij={self.heralding_prob:.4f}/2, "
            f"P_success={self.success_per_input:.4f}, "
            f"converged={self.converged}, restarts={self.n_restarts})"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def inverse_design_heralded(
    U2: np.ndarray,
    i: int,
    j: int,
    N_sb: int,
    N_r: int,
    N_f: int,
    gamma_e: float = 1.0,
    gamma_i: float = 0.0,
    delta_omega_list: float = 0.0,
    n_restarts: int = 30,
    kappa_scale: float = 0.5,
    fidelity_tol: float = 1 - 1e-4,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> HeraldedDesignResult:
    """
    Heralded-gate inverse design: maximise F(M_ij, U_2) only.
    Heralding probability P_ij is reported but NOT optimised.

    Parameters
    ----------
    U2 : (2, 2) complex ndarray   target unitary on modes (i, j)
    i, j : int                    subspace mode indices (0-based)
    N_sb : int                    sidebands; N = 2*N_sb + 1
    N_r, N_f : int                rings, modulation tones per ring
    n_restarts : int              multi-start L-BFGS-B (non-convex)
    fidelity_tol : float          early stop: F_block >= fidelity_tol
                                  (no leakage stop -- P_ij is a free output)

    Returns
    -------
    HeraldedDesignResult
    """
    N = 2 * N_sb + 1
    assert U2.shape == (2, 2)
    assert i != j
    assert 0 <= i < N and 0 <= j < N

    rng = np.random.default_rng(seed)
    n_params = 2 * N_r * N_f

    best_loss = np.inf
    best_kappa = None
    best_F = 0.0
    best_P = 0.0
    history = []
    converged = False
    trial = 0

    for trial in range(n_restarts):
        x0 = rng.standard_normal(n_params) * kappa_scale

        result = minimize(
            _loss_and_grad,
            x0,
            args=(U2, i, j, N_sb, N_r, N_f, gamma_e, gamma_i, delta_omega_list),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
        )

        kl = _unpack(result.x, N_r, N_f)
        M = cascaded_scattering_matrix(
            kl,
            N_sb,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            delta_omega_list=delta_omega_list,
        )
        F = block_fidelity(M, U2, i, j)
        P = heralding_probability(M, i, j)
        history.append({"trial": trial, "loss": result.fun, "F": F, "P": P})

        if verbose:
            print(
                f"  restart {trial+1:2d}/{n_restarts}: "
                f"loss={result.fun:.4e}  F_block={F:.6f}  P_ij={P:.4f}/2"
            )

        if result.fun < best_loss:
            best_loss = result.fun
            best_kappa = kl
            best_F = F
            best_P = P

        if F >= fidelity_tol:
            converged = True
            if verbose:
                print(f"  early stop: F_block >= {fidelity_tol}")
            break

    return HeraldedDesignResult(
        kappa_list=best_kappa,
        block_fidelity=best_F,
        heralding_prob=best_P,
        success_per_input=best_P / 2.0,
        loss_history=history,
        n_restarts=trial + 1,
        converged=converged,
    )
