"""
examples/single_ring_heralded_2x2.py
=====================================
Heralded 2x2 inverse design on a single ring.

Cost: maximise F(M_ij, U_2) only.  Subspace retention P_ij is reported as
a separate experimental figure of merit (the heralding success
probability).
"""

import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import (
    scattering_matrix,
    plot_matrix_grid,
    embed_unitary_2x2,
    beamsplitter,
    inverse_design_heralded,
)

from freq_beamsplitter.heralded_design import inverse_design_heralded

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

N_sb = 3  # N = 7 modes
N_f = 6  # modulation tones (single ring DOF = 12)
i, j = 3, 4  # central pair (m = 0, +1) -- best from sweep

# pick a target
U2 = beamsplitter(theta=np.pi / 4, phi=0.0)  # 50:50 BS
U2 = np.array([[1, 0], [0, -1]], dtype=complex)  # sigma_z
U2 = np.array([[0, -1j], [1j, 0]], dtype=complex)  # sigma_y
U2 = np.array([[0, 1], [1, 0]], dtype=complex)  # sigma_x

GAMMA_I = 0.0
N_RESTARTS = 30
SEED = 0


# ── 2. Run ───────────────────────────────────────────────────────────────────

N = 2 * N_sb + 1
print("=" * 60)
print("Single-ring HERALDED 2x2 inverse design")
print("=" * 60)
print(f"\nSystem    : N = {N}, single ring, N_f = {N_f}")
print(f"Subspace  : modes ({i}, {j})  (m = {i-N_sb}, {j-N_sb})")
print(f"Target U2 :")
print(np.array2string(U2, precision=3, suppress_small=True))
print()

result = inverse_design_heralded(
    U2=U2,
    i=i,
    j=j,
    N_sb=N_sb,
    N_r=1,
    N_f=N_f,
    gamma_e=1.0,
    gamma_i=GAMMA_I,
    delta_omega_list=0.0,
    n_restarts=N_RESTARTS,
    kappa_scale=0.5,
    fidelity_tol=1 - 1e-4,
    seed=SEED,
    verbose=True,
)


# ── 3. Reconstruct ───────────────────────────────────────────────────────────

kappa = result.kappa_list[0]
M_opt = scattering_matrix(kappa, N_sb=N_sb, gamma_e=1.0, gamma_i=GAMMA_I)
M_block = M_opt[np.ix_([i, j], [i, j])]

# Renormalise the block to its largest singular value -- this is the
# *effective* unitary you measure after heralding.
s = np.linalg.svd(M_block, compute_uv=False)
M_block_normed = M_block / s[0]

print(f"\nResults")
print(f"  Block fidelity F(U2, M_ij)   : {result.block_fidelity:.6f}")
print(f"  Heralding weight P_ij        : {result.heralding_prob:.4f} / 2")
print(f"  Success per input mode       : {result.success_per_input:.4f}")
print(f"  Converged                    : {result.converged}")

print(f"\nM_ij raw (gate * heralding amplitude):")
print(np.array2string(M_block, precision=3, suppress_small=True))

print(f"\nM_ij normalised by s_0 (= heralded unitary):")
print(np.array2string(M_block_normed, precision=3, suppress_small=True))

print(f"\nTarget U2:")
print(np.array2string(U2, precision=3, suppress_small=True))

print(f"\nOptimised kappa_l / gamma_e (polar):")
for l, kl in enumerate(kappa, start=1):
    print(f"  kappa_{l}  =  {np.abs(kl):.4f} * exp(i {np.angle(kl)/np.pi:+.4f} pi)")


# ── 4. Visualise ─────────────────────────────────────────────────────────────

U_target_view = embed_unitary_2x2(U2=U2, i=i, j=j, N=N)

fig = plot_matrix_grid(
    [U_target_view, M_opt],
    [
        f"Target view: 2x2 on ({i},{j}), rest = identity",
        (
            f"Heralded result\n"
            f"F = {result.block_fidelity:.4f}, "
            f"P_succ = {result.success_per_input:.3f}"
        ),
    ],
)
plt.tight_layout()
plt.savefig("figures/single_ring_heralded_2x2.png", dpi=120)
plt.show()
