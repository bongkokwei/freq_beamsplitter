"""
examples/single_ring_subspace_2x2.py
=====================================
Single-ring inverse design of a 2x2 unitary on a chosen (i, j) subspace,
with the rest of the scattering matrix unconstrained except for the
no-leakage condition.

Cost: L = (1 - F_block) + lambda_leak * leakage.
"""

import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import (
    scattering_matrix,
    unitarity_error,
    plot_matrix_grid,
    embed_unitary_2x2,
    beamsplitter,
)

# NOTE: add this to freq_beamsplitter/__init__.py once subspace_design.py
# is dropped into the package directory:
#   from .subspace_design import inverse_design_subspace, SubspaceDesignResult
from freq_beamsplitter.subspace_design import inverse_design_subspace

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

N_sb = 3  # N = 7 modes
N_f = 6  # modulation tones (single ring DOF = 2*N_f = 12)
i, j = 3, 4  # subspace mode indices

# 2x2 target -- pick one
U2 = beamsplitter(theta=np.pi / 4, phi=0.0)  # 50:50 BS
# U2 = np.array([[1, 0], [0, -1]], dtype=complex)      # sigma_z
# U2 = np.array([[0, -1j], [1j, 0]], dtype=complex)    # sigma_y
# U2 = np.array([[0, 1], [1, 0]], dtype=complex)  # sigma_x

GAMMA_I = 0.0
LAMBDA_LEAK = 1.0  # leakage penalty weight
N_RESTARTS = 30
SEED = 0

# ── 2. Build embedded "target view" for plotting ─────────────────────────────

N = 2 * N_sb + 1
U_target_view = embed_unitary_2x2(U2=U2, i=i, j=j, N=N)  # for visualisation only

print("=" * 60)
print("Single-ring SUBSPACE inverse design")
print("=" * 60)
print(f"\nSystem       : N = {N}, single ring, N_f = {N_f}")
print(f"Subspace     : modes ({i}, {j})")
print(f"Block target U2:")
print(np.array2string(U2, precision=3, suppress_small=True))
print(f"\nDOF available    : 2*N_f = {2*N_f}")
print(f"Constraints (~)  : 4 (block) + 4(N-2) (no leakage) = {4 + 4*(N-2)}")
print(f"lambda_leak      : {LAMBDA_LEAK}")
print()

# ── 3. Run inverse design ────────────────────────────────────────────────────

result = inverse_design_subspace(
    U2=U2,
    i=i,
    j=j,
    N_sb=N_sb,
    N_r=1,
    N_f=N_f,
    gamma_e=1.0,
    gamma_i=GAMMA_I,
    delta_omega_list=0.0,
    lambda_leak=LAMBDA_LEAK,
    n_restarts=N_RESTARTS,
    kappa_scale=0.5,
    fidelity_tol=1 - 1e-4,
    leakage_tol=1e-3,
    seed=SEED,
    verbose=True,
)

# ── 4. Reconstruct ───────────────────────────────────────────────────────────

kappa = result.kappa_list[0]
M_opt = scattering_matrix(kappa, N_sb=N_sb, gamma_e=1.0, gamma_i=GAMMA_I)
M_block = M_opt[np.ix_([i, j], [i, j])]

print(f"\nResults")
print(f"  Block fidelity F(U2, M_ij)  : {result.block_fidelity:.6f}")
print(f"  Leakage out of subspace     : {result.leakage:.4e}")
print(f"  Unitarity error |M+M-I|     : {unitarity_error(M_opt):.2e}")
print(f"  Converged                   : {result.converged}")

print(f"\nM_ij (the 2x2 we built):")
print(np.array2string(M_block, precision=3, suppress_small=True))

print(f"\nOptimised kappa_l / gamma_e (polar):")
for l, kl in enumerate(kappa, start=1):
    print(f"  kappa_{l}  =  {np.abs(kl):.4f} * exp(i {np.angle(kl)/np.pi:+.4f} pi)")

# ── 5. Visualise ─────────────────────────────────────────────────────────────

fig = plot_matrix_grid(
    [U_target_view, M_opt],
    [
        f"Target view: 2x2 on ({i},{j}), rest = identity",
        f"Optimised  F_block = {result.block_fidelity:.4f}  leak = {result.leakage:.1e}",
    ],
)
plt.tight_layout()
plt.savefig("figures/single_ring_subspace_2x2.png", dpi=120)
plt.show()
