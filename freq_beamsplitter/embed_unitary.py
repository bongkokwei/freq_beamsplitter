"""
Embed a 2x2 unitary into an NxN system acting on modes i and j.

[U(N)]_kl = a  if k=i, l=i
             b  if k=i, l=j
             c  if k=j, l=i
             d  if k=j, l=j
             1  if k=l, k not in {i,j}
             0  otherwise
"""

import numpy as np


def embed_unitary_2x2(U2: np.ndarray, i: int, j: int, N: int) -> np.ndarray:
    """
    Embed a 2x2 unitary U2 acting on modes i and j into an NxN unitary.

    Parameters
    ----------
    U2 : complex ndarray, shape (2, 2)
    i  : int  — first mode index  (0-based)
    j  : int  — second mode index (0-based)
    N  : int  — total system dimension

    Returns
    -------
    U_full : complex ndarray, shape (N, N)
    """
    assert U2.shape == (2, 2), "U2 must be 2x2"
    assert i != j, "i and j must be distinct"
    assert 0 <= i < N and 0 <= j < N, f"Indices must be in [0, {N-1}]"

    U_full = np.eye(N, dtype=complex)
    U_full[i, i] = U2[0, 0]
    U_full[i, j] = U2[0, 1]
    U_full[j, i] = U2[1, 0]
    U_full[j, j] = U2[1, 1]
    return U_full


def beamsplitter(theta: float, phi: float = 0.0) -> np.ndarray:
    """
    Standard 2x2 beamsplitter unitary (Clements convention):

        U = [[ cos(theta),          -exp(i*phi) * sin(theta) ],
             [ sin(theta),           exp(i*phi) * cos(theta) ]]

    Parameters
    ----------
    theta : float   Mixing angle,  theta=0 -> identity, theta=pi/4 -> 50:50
    phi   : float   Phase shift on the cross term (default 0)
    """
    return np.array(
        [
            [np.cos(theta), -np.exp(1j * phi) * np.sin(theta)],
            [np.sin(theta), np.exp(1j * phi) * np.cos(theta)],
        ],
        dtype=complex,
    )
