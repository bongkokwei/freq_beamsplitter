"""Run: python tests/test_aux_boundary.py"""
import numpy as np
from freq_beamsplitter.core import scattering_matrix
from freq_beamsplitter.subspace_design import _block_fidelity, _leakage

# Aux FSR = 5 x main FSR -> barriers every 5th site. N = 11 lattice,
# barriers at 3 and 8, even segment {4,5,6,7} in the middle.
N, seg = 11, [4, 5, 6, 7]
d = np.zeros(N, complex)
d[[3, 8]] = 1e4  # strong barrier (g >> kappa)

k1 = np.array([0.4 + 0.2j])  # nearest-neighbour only
M = scattering_matrix(k1, N, delta_omega=d)

# Scalar and per-mode-constant detuning agree (old behaviour kept)
assert np.allclose(scattering_matrix(k1, N, delta_omega=0.3),
                   scattering_matrix(k1, N, delta_omega=np.full(N, 0.3)))

# kappa_1 only: barrier isolates the segment -> no leakage
assert _leakage(M, seg[:2], seg[2:]) < 1e-6

# kappa_2 hops over the barrier -> leakage appears
M2 = scattering_matrix(np.array([0.4, 0.3]), N, delta_omega=d)
assert _leakage(M2, seg[:2], seg[2:]) > 1e-2

# 4x4 block fidelity against itself is 1
assert np.isclose(_block_fidelity(M, M[np.ix_(seg, seg)], seg[:2], seg[2:]), 1.0)
print("ok")
