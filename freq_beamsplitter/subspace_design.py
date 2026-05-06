"""
freq_beamsplitter.subspace_design
==================================
Inverse design where the *target* is a 2x2 unitary acting on a chosen
pair of modes (i, j), with the rest of the matrix unconstrained EXCEPT
for a no-leakage condition: photons injected into modes i or j must
stay in the {i, j} subspace.

Cost function
-------------
    L = (1 - F_block(M_ij, U_2)) + lambda_leak * leakage(M)

where
    M_ij        = 2x2 submatrix M[ix(i,j), ix(i,j)]
    F_block     = |<U_2, M_ij>| / (||U_2||_F ||M_ij||_F)
    leakage(M)  = sum of |M[k, i]|^2 + |M[k, j]|^2  for k not in {i, j}
                  (and analogously for the rows -- ensures unitarity
                   on the subspace, no in/out of {i,j})

Compared to full-matrix inverse_design, this relaxes ~N^2 - (4N-4)
constraints (off-block, off-diagonal elements are free).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize

from .core import cascaded_scattering_matrix

# ---------------------------------------------------------------------------
# Pack/unpack (same convention as optimise.py)
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
# Cost components
# ---------------------------------------------------------------------------


def _block_fidelity(M: np.ndarray, U2: np.ndarray, i: int, j: int) -> float:
    """Fidelity between M[{i,j}, {i,j}] and U2, invariant to global phase."""
    M_block = M[np.ix_([i, j], [i, j])]
    inner = np.vdot(U2, M_block)  # tr(U2^dagger M_block)
    norm = np.linalg.norm(U2) * np.linalg.norm(M_block)
    if norm == 0:
        return 0.0
    return float(np.abs(inner) / norm)


def _leakage(M: np.ndarray, i: int, j: int) -> float:
    """
    Sum |M[k, i]|^2 + |M[k, j]|^2 + |M[i, k]|^2 + |M[j, k]|^2 for k not in {i,j}.

    Zero leakage means injecting into mode i or j gives output only in {i,j},
    AND the columns of M for k not in {i,j} don't dump into modes i, j.
    """
    N = M.shape[0]
    k_other = [k for k in range(N) if k not in (i, j)]
    col_leak = np.sum(np.abs(M[np.ix_(k_other, [i, j])]) ** 2)  # rows other, cols i,j
    row_leak = np.sum(np.abs(M[np.ix_([i, j], k_other)]) ** 2)  # rows i,j, cols other
    return float(col_leak + row_leak)


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
    lambda_leak,
    eps=1e-6,
):
    """
    L = (1 - F_block) + lambda_leak * leakage
    Centred finite differences for the gradient.
    """

    def loss_fn(xv):
        kl = _unpack(xv, N_r, N_f)
        M = cascaded_scattering_matrix(
            kl,
            N_sb,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            delta_omega_list=delta_omega_list,
        )
        F = _block_fidelity(M, U2, i, j)
        L = _leakage(M, i, j)
        return (1.0 - F) + lambda_leak * L

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
class SubspaceDesignResult:
    kappa_list: list
    block_fidelity: float
    leakage: float
    loss_history: list = field(default_factory=list)
    n_restarts: int = 0
    converged: bool = False

    def __repr__(self):
        return (
            f"SubspaceDesignResult("
            f"block_fidelity={self.block_fidelity:.6f}, "
            f"leakage={self.leakage:.2e}, "
            f"converged={self.converged}, "
            f"restarts={self.n_restarts})"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def inverse_design_subspace(
    U2: np.ndarray,
    i: int,
    j: int,
    N_sb: int,
    N_r: int,
    N_f: int,
    gamma_e: float = 1.0,
    gamma_i: float = 0.0,
    delta_omega_list: float = 0.0,
    lambda_leak: float = 10.0,
    n_restarts: int = 10,
    kappa_scale: float = 0.5,
    fidelity_tol: float = 1 - 1e-4,
    leakage_tol: float = 1e-3,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> SubspaceDesignResult:
    """
    Inverse-design kappa_l for a 2x2 unitary on the (i, j) subspace,
    leaving the rest of the scattering matrix unconstrained except for
    no leakage in/out of {i, j}.

    Parameters
    ----------
    U2 : (2,2) complex ndarray   target unitary on modes (i, j)
    i, j : int                   subspace mode indices (0-based)
    N_sb : int                   sidebands; N = 2*N_sb + 1
    N_r, N_f : int               rings, modulation tones per ring
    lambda_leak : float          weight on the no-leakage penalty
                                 (10.0 is a reasonable starting point;
                                 increase if leakage is too high,
                                 decrease if block fidelity stalls)
    n_restarts : int             multi-start L-BFGS-B (non-convex landscape)
    fidelity_tol, leakage_tol : float
                                 early-stopping criteria. Both must be met.

    Returns
    -------
    SubspaceDesignResult
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
    best_L = np.inf
    history = []
    converged = False

    for trial in range(n_restarts):
        x0 = rng.standard_normal(n_params) * kappa_scale

        result = minimize(
            _loss_and_grad,
            x0,
            args=(
                U2,
                i,
                j,
                N_sb,
                N_r,
                N_f,
                gamma_e,
                gamma_i,
                delta_omega_list,
                lambda_leak,
            ),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
        )

        # Evaluate split metrics at the optimum
        kl = _unpack(result.x, N_r, N_f)
        M = cascaded_scattering_matrix(
            kl,
            N_sb,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            delta_omega_list=delta_omega_list,
        )
        F = _block_fidelity(M, U2, i, j)
        L = _leakage(M, i, j)
        history.append({"trial": trial, "loss": result.fun, "F": F, "leak": L})

        if verbose:
            print(
                f"  restart {trial+1:2d}/{n_restarts}: "
                f"loss={result.fun:.4e}  F_block={F:.6f}  leak={L:.2e}"
            )

        if result.fun < best_loss:
            best_loss = result.fun
            best_kappa = kl
            best_F = F
            best_L = L

        if F >= fidelity_tol and L <= leakage_tol:
            converged = True
            if verbose:
                print(f"  early stop: F_block >= {fidelity_tol}, leak <= {leakage_tol}")
            break

    return SubspaceDesignResult(
        kappa_list=best_kappa,
        block_fidelity=best_F,
        leakage=best_L,
        loss_history=history,
        n_restarts=trial + 1,
        converged=converged,
    )
