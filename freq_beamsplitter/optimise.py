"""
freq_beamsplitter.optimise
==========================
Inverse design: find κ_l parameters for each ring that maximise fidelity
to a target unitary matrix.

Method follows Buddhiraju et al. (2021), Sec. "Implementation of linear
transformations": gradient-based optimisation (L-BFGS) of the fidelity

    F(U, V) = |⟨U, V⟩| / (‖U‖_F ‖V‖_F)

where V = M_total(κ) is the cascaded scattering matrix.

Because the parameters κ_l are complex, we split each into (Re, Im) and
pass a real parameter vector to scipy's L-BFGS-B.  Gradients are
estimated via forward-mode finite differences (ε = 1e-6).

Degrees-of-freedom reminder
---------------------------
A (2N_sb+1)² unitary has (2N_sb+1)² real DOF.
Each ring contributes 2·N_f real DOF (Re+Im per tone).
Sufficient condition:  N_r · N_f  ≥  (2N_sb+1)² / 2
  → for N_sb=2 (5×5): need N_r·N_f ≥ 12.5, e.g. N_r=4, N_f=4 (16 DOF/ring → 32 total)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize

from .core import cascaded_scattering_matrix, fidelity


# ---------------------------------------------------------------------------
# Haar-random unitary
# ---------------------------------------------------------------------------

def haar_unitary(N: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample a Haar-random unitary matrix of size N×N.

    Uses the standard QR decomposition method (Mezzadri 2006):
      1. Draw Z ~ Ginibre (complex Gaussian iid entries).
      2. QR decompose: Z = Q R.
      3. Correct phases: Q ← Q · diag(R_ii / |R_ii|).

    Parameters
    ----------
    N    : int   Matrix dimension.
    seed : int, optional   RNG seed for reproducibility.

    Returns
    -------
    U : complex ndarray, shape (N, N), unitary to machine precision.
    """
    rng = np.random.default_rng(seed)
    Z   = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    # Phase correction — makes distribution uniform (Haar measure)
    phase = np.diag(R) / np.abs(np.diag(R))
    return Q * phase[np.newaxis, :]


# ---------------------------------------------------------------------------
# Packing / unpacking of complex κ vectors
# ---------------------------------------------------------------------------

def _pack(kappa_list: list) -> np.ndarray:
    """Flatten list of complex arrays → real 1-D vector [Re..., Im...]."""
    flat = np.concatenate([k for k in kappa_list])
    return np.concatenate([flat.real, flat.imag])


def _unpack(x: np.ndarray, N_r: int, N_f: int) -> list:
    """Inverse of _pack."""
    total = N_r * N_f
    real_part = x[:total].reshape(N_r, N_f)
    imag_part = x[total:].reshape(N_r, N_f)
    return [real_part[i] + 1j * imag_part[i] for i in range(N_r)]


# ---------------------------------------------------------------------------
# Objective and gradient
# ---------------------------------------------------------------------------

def _loss_and_grad(x, U_target, N_sb, N_r, N_f,
                   gamma_e, gamma_i, delta_omega_list, eps=1e-6):
    """
    Returns (loss, grad) where loss = 1 - F(U_target, M(x)).
    Gradient via centred finite differences.
    """
    def loss_fn(xv):
        kl = _unpack(xv, N_r, N_f)
        M  = cascaded_scattering_matrix(kl, N_sb,
                                        gamma_e=gamma_e,
                                        gamma_i=gamma_i,
                                        delta_omega_list=delta_omega_list)
        return 1.0 - fidelity(U_target, M)

    f0   = loss_fn(x)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm       = x.copy(), x.copy()
        xp[i]       += eps
        xm[i]       -= eps
        grad[i]      = (loss_fn(xp) - loss_fn(xm)) / (2 * eps)

    return f0, grad


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InverseDesignResult:
    """Container returned by :func:`inverse_design`."""
    kappa_list   : list               # optimised κ_l per ring
    fidelity     : float              # achieved F(U_target, M_total)
    loss_history : list = field(default_factory=list)
    n_restarts   : int  = 0
    converged    : bool = False

    def __repr__(self):
        return (f"InverseDesignResult("
                f"fidelity={self.fidelity:.6f}, "
                f"converged={self.converged}, "
                f"restarts={self.n_restarts})")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def inverse_design(
    U_target         : np.ndarray,
    N_sb             : int,
    N_r              : int,
    N_f              : int,
    gamma_e          : float = 1.0,
    gamma_i          : float = 0.0,
    delta_omega_list : float = 0.0,
    n_restarts       : int   = 5,
    kappa_scale      : float = 0.5,
    fidelity_tol     : float = 1 - 1e-4,
    seed             : Optional[int] = None,
    verbose          : bool  = True,
) -> InverseDesignResult:
    """
    Optimise κ_l parameters to match target unitary U_target.

    Uses L-BFGS-B (scipy) with finite-difference gradients, matching the
    method of Buddhiraju et al. (2021).  Multiple random restarts are run;
    the best result is returned.

    Parameters
    ----------
    U_target         : complex ndarray, shape (2*N_sb+1, 2*N_sb+1)
        Target unitary matrix (e.g. from :func:`haar_unitary`).
    N_sb             : int   Number of sidebands (matrix size = 2*N_sb+1).
    N_r              : int   Number of rings in cascade.
    N_f              : int   Number of modulation tones per ring.
    gamma_e          : float External coupling rate (units of γ).
    gamma_i          : float Intrinsic loss rate.
    delta_omega_list : float or list   Per-ring detuning.
    n_restarts       : int   Random restarts.  Best result is returned.
    kappa_scale      : float Std-dev of initial κ_l samples.
    fidelity_tol     : float Stop early if fidelity exceeds this.
    seed             : int, optional   Base RNG seed.
    verbose          : bool  Print per-restart progress.

    Returns
    -------
    InverseDesignResult
    """
    rng    = np.random.default_rng(seed)
    best   = None
    hist   = []

    dof_available = 2 * N_r * N_f
    dof_required  = (2 * N_sb + 1) ** 2
    if verbose:
        print(f"Target: {2*N_sb+1}×{2*N_sb+1} unitary  "
              f"({dof_required} real DOF)")
        print(f"System: N_r={N_r} rings × N_f={N_f} tones  "
              f"→  {dof_available} real DOF  "
              f"({'≥' if dof_available >= dof_required else '<'} required)\n")

    for r in range(n_restarts):
        # Random initialisation of complex κ_l
        kappa_init = [
            rng.normal(0, kappa_scale, N_f) + 1j * rng.normal(0, kappa_scale, N_f)
            for _ in range(N_r)
        ]
        x0 = _pack(kappa_init)

        restart_hist = []

        def callback(xk):
            kl = _unpack(xk, N_r, N_f)
            M  = cascaded_scattering_matrix(kl, N_sb,
                                            gamma_e=gamma_e,
                                            gamma_i=gamma_i,
                                            delta_omega_list=delta_omega_list)
            f  = fidelity(U_target, M)
            restart_hist.append(f)

        res = minimize(
            fun=_loss_and_grad,
            x0=x0,
            args=(U_target, N_sb, N_r, N_f, gamma_e, gamma_i, delta_omega_list),
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options=dict(maxiter=2000, ftol=1e-15, gtol=1e-8),
        )

        kl_opt = _unpack(res.x, N_r, N_f)
        M_opt  = cascaded_scattering_matrix(kl_opt, N_sb,
                                            gamma_e=gamma_e,
                                            gamma_i=gamma_i,
                                            delta_omega_list=delta_omega_list)
        F      = fidelity(U_target, M_opt)
        hist.append(restart_hist)

        if verbose:
            print(f"  Restart {r+1}/{n_restarts}:  F = {F:.6f}")

        if best is None or F > best.fidelity:
            best = InverseDesignResult(
                kappa_list=kl_opt,
                fidelity=F,
                loss_history=restart_hist,
                n_restarts=r + 1,
                converged=(F >= fidelity_tol),
            )

        if best.fidelity >= fidelity_tol:
            if verbose:
                print(f"\n  ✓ Fidelity tolerance reached at restart {r+1}.")
            break

    if verbose:
        print(f"\nBest fidelity: {best.fidelity:.6f}")

    return best
