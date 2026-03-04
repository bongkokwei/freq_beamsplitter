"""
Single-ring scattering matrix simulator
Based on: Buddhiraju et al., Nature Communications (2021)
  "Arbitrary linear transformations for photons in the frequency synthetic dimension"

Full Eq. (5):
    M = I + i sqrt(2 Γ_e) [Δω - i(Γ_e + Γ_i) - K]^{-1} sqrt(2 Γ_e)

where:
    Γ_e  = diag(γ^e_m)   external coupling rates (per mode)
    Γ_i  = diag(γ^i_m)   intrinsic loss rates    (per mode)
    Δω   = scalar detuning of input comb from ring resonances
    K    = Hermitian Toeplitz coupling matrix built from κ_l

Modes indexed m = {-N_sb, ..., 0, ..., +N_sb}, size N = 2*N_sb + 1.

κ_l is complex: magnitude sets coupling strength, phase sets modulation phase.
For real refractive-index modulation: κ_{-l} = κ_l* (enforced by Hermitian K).
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Core: build K matrix
# ---------------------------------------------------------------------------

def build_K(kappa: np.ndarray, N_sb: int) -> np.ndarray:
    """
    Build the (2*N_sb+1) x (2*N_sb+1) Hermitian coupling matrix K.

    Parameters
    ----------
    kappa : complex array, shape (N_f,)
        Modulation coupling constants κ_1, κ_2, ..., κ_{N_f}.
        κ_l couples mode m to modes m±l.
        For real refractive-index modulation κ_{-l} = κ_l* is enforced automatically.
    N_sb : int
        Number of sidebands. Matrix size = 2*N_sb + 1.

    Returns
    -------
    K : complex ndarray, shape (N, N), Hermitian
    """
    N = 2 * N_sb + 1
    K = np.zeros((N, N), dtype=complex)

    for l, kl in enumerate(kappa, start=1):
        for m in range(N):
            if m + l < N:
                K[m, m + l] += kl          # κ_l  couples m → m+l
                K[m + l, m] += np.conj(kl) # κ_l* couples m+l → m (Hermitian)

    return K


# ---------------------------------------------------------------------------
# Core: scattering matrix for a single ring
# ---------------------------------------------------------------------------

def scattering_matrix(
    kappa: np.ndarray,
    N_sb: int,
    gamma_e: Union[float, np.ndarray] = 1.0,
    gamma_i: Union[float, np.ndarray] = 0.0,
    delta_omega: float = 0.0,
) -> np.ndarray:
    """
    Compute the (2*N_sb+1) x (2*N_sb+1) scattering matrix for a single
    dynamically modulated ring coupled to an external waveguide.

    Implements Eq. (5) of Buddhiraju et al. (2021):

        M = I + i sqrt(2 Γ_e) [Δω - i(Γ_e + Γ_i) - K]^{-1} sqrt(2 Γ_e)

    Parameters
    ----------
    kappa : complex array, shape (N_f,)
        Modulation coupling constants κ_1 ... κ_{N_f} (units of γ_e).
    N_sb : int
        Number of sidebands. Matrix size N = 2*N_sb + 1.
    gamma_e : float or array of shape (N,)
        External (waveguide) coupling rate(s) γ^e_m.
        Scalar → uniform coupling across all modes.
        Array  → per-mode coupling (e.g., apodised or dispersive coupling).
    gamma_i : float or array of shape (N,)
        Intrinsic loss rate(s) γ^i_m.
        Scalar → uniform loss.
        Array  → per-mode loss.
        Set to 0 for lossless (unitary) case.
    delta_omega : float
        Scalar detuning Δω of the input comb from ring resonances (units of γ_e).
        Set to 0 for resonant excitation.

    Returns
    -------
    M : complex ndarray, shape (N, N)
        Scattering matrix. Unitary if gamma_i == 0.
    """
    N = 2 * N_sb + 1

    # Build diagonal rate matrices
    gamma_e_vec = np.broadcast_to(gamma_e, N).astype(complex).copy()
    gamma_i_vec = np.broadcast_to(gamma_i, N).astype(complex).copy()

    Gamma_e  = np.diag(gamma_e_vec)
    Gamma_i  = np.diag(gamma_i_vec)
    Gamma_tot = Gamma_e + Gamma_i

    sqrt_Gamma_e = np.diag(np.sqrt(gamma_e_vec))

    # Build coupling matrix
    K = build_K(kappa, N_sb)

    # Core resolvent: [Δω·I - i·Γ_tot - K]^{-1}
    resolvent_inv = delta_omega * np.eye(N) - 1j * Gamma_tot - K
    resolvent = np.linalg.inv(resolvent_inv)

    # Scattering matrix: M = I + i sqrt(2Γ_e) · resolvent · sqrt(2Γ_e)
    sqrt2_Gamma_e = np.sqrt(2) * sqrt_Gamma_e
    M = np.eye(N) + 1j * (sqrt2_Gamma_e @ resolvent @ sqrt2_Gamma_e)

    return M


# ---------------------------------------------------------------------------
# Cascaded rings
# ---------------------------------------------------------------------------

def cascaded_scattering_matrix(
    kappa_list: list,
    N_sb: int,
    gamma_e: Union[float, np.ndarray] = 1.0,
    gamma_i: Union[float, np.ndarray] = 0.0,
    delta_omega_list: Union[float, list] = 0.0,
) -> np.ndarray:
    """
    Compute the total scattering matrix for N_r rings in series.

    The total transformation is M_total = M_{N_r} @ ... @ M_2 @ M_1.
    (light passes through ring 1 first)

    Parameters
    ----------
    kappa_list : list of arrays, length N_r
        kappa_list[i] = array of κ_l values for ring i.
    N_sb : int
        Number of sidebands (same for all rings).
    gamma_e, gamma_i : float or array
        Coupling rates (shared across rings unless you pass a list of arrays).
    delta_omega_list : float or list of floats, length N_r
        Per-ring detunings. Scalar → same detuning for all rings.

    Returns
    -------
    M_total : complex ndarray, shape (N, N)
    """
    N_r = len(kappa_list)

    if np.isscalar(delta_omega_list):
        delta_omega_list = [delta_omega_list] * N_r

    N = 2 * N_sb + 1
    M_total = np.eye(N, dtype=complex)

    for i, kappa in enumerate(kappa_list):
        M_i = scattering_matrix(
            kappa, N_sb,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            delta_omega=delta_omega_list[i],
        )
        M_total = M_i @ M_total  # accumulate left-to-right (ring i+1 acts after ring i)

    return M_total


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def unitarity_error(M: np.ndarray) -> float:
    """
    Returns ||M†M - I||_F as a measure of non-unitarity.
    Should be ~machine epsilon for lossless rings.
    """
    N = M.shape[0]
    return np.linalg.norm(M.conj().T @ M - np.eye(N), ord='fro')


def fidelity(U: np.ndarray, V: np.ndarray) -> float:
    """
    Fidelity between target unitary U and achieved transformation V.
    Eq. (6) of Buddhiraju et al.:
        F(U, V) = |<U, V>| / (||U||_F ||V||_F)
    Tolerates a global phase.
    """
    inner = np.sum(np.conj(U) * V)
    return np.abs(inner) / (np.linalg.norm(U, 'fro') * np.linalg.norm(V, 'fro'))


