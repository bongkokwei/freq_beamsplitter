"""
examples/single_ring_embedded_2x2.py
====================================
Inverse design of a single ring (N_r = 1) targeting a 2x2 rotation
embedded into an N x N identity, acting on user-chosen modes (i, j).

System
------
    N_r  = 1          (single ring)
    N    = 2*N_sb + 1 (matrix dimension; user picks via N_sb)
    N_f  = number of modulation tones (user picks)

DOF check
---------
    Available : 2 * N_f real DOF (single ring).
    Target    : sparse 2x2 block + identity elsewhere -- much easier than
                a full Haar unitary, but coverage is not guaranteed.
                Buddhiraju Supp. Fig. 3 shows N_r = 1 generally underperforms.

Usage
-----
Edit the CONFIG block below to choose:
    - N_sb (system size)
    - N_f  (modulation tones)
    - i, j (which modes the 2x2 block acts on)
    - theta, phi (rotation parameters)
"""

import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import (
    inverse_design,
    scattering_matrix,
    fidelity,
    unitarity_error,
    plot_matrix_grid,
    embed_unitary_2x2,
    beamsplitter,
)

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

N_sb = 3  # sidebands  -> N = 2*N_sb + 1 = 5 modes
N_f = 6  # modulation tones (single ring DOF = 2*N_f)
i, j = 4, 3  # mode indices the 2x2 block acts on (0-based)
theta = np.pi / 2  # mixing angle (pi/2 = 0:100 beamsplitter)
phi = np.pi / 2  # cross-term phase

GAMMA_I = 0.0  # intrinsic loss (in units of gamma_e)
N_RESTARTS = 40  # multi-restart L-BFGS-B (non-convex landscape)
SEED = 0

# ── 2. Build target unitary ──────────────────────────────────────────────────

N = 2 * N_sb + 1
U2 = beamsplitter(theta=theta, phi=phi)
U_target = embed_unitary_2x2(U2=U2, i=i, j=j, N=N)

print("=" * 60)
print("Single-ring inverse design -- 2x2 block embedded in identity")
print("=" * 60)
print(f"\nSystem size  : {N} x {N}  (N_sb = {N_sb})")
print(f"Modulation   : N_f = {N_f} tones, single ring (N_r = 1)")
print(f"DOF          : available 2*N_f = {2*N_f},  target ~4 (block) + identity rest")
print(f"Block action : modes (i, j) = ({i}, {j})")
print(f"2x2 unitary  : theta = {theta:.4f} rad,  phi = {phi:.4f} rad\n")

print("U_target |amplitude|:")
print(np.array2string(np.abs(U_target), precision=3, suppress_small=True))
print(f"\nUnitarity check on U_target: {unitarity_error(U_target):.2e}\n")

# ── 3. Inverse design ────────────────────────────────────────────────────────

result = inverse_design(
    U_target,
    N_sb=N_sb,
    N_r=1,  # SINGLE RING
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

# ── 4. Reconstruct ───────────────────────────────────────────────────────────

# Single ring: result.kappa_list has length 1 -- pull out the lone kappa vector.
kappa = result.kappa_list[0]
M_opt = scattering_matrix(kappa, N_sb=N_sb, gamma_e=1.0, gamma_i=GAMMA_I)

print(f"\nResults")
print(f"  Fidelity F(U, M)        : {result.fidelity:.6f}")
print(f"  Unitarity error |M+M-I| : {unitarity_error(M_opt):.2e}")
print(f"  Converged               : {result.converged}")

print(f"\nOptimised kappa_l / gamma_e (polar):")
for l, kl in enumerate(kappa, start=1):
    mag = np.abs(kl)
    phase = np.angle(kl) / np.pi
    print(f"  kappa_{l}  =  {mag:.4f} * exp(i {phase:+.4f} pi)")

# ── 5. Visualise ─────────────────────────────────────────────────────────────

fig = plot_matrix_grid(
    [U_target, M_opt],
    [f"Target: 2x2 on modes ({i},{j})", f"Optimised  F = {result.fidelity:.4f}"],
)
plt.tight_layout()
plt.savefig("figures/single_ring_embedded_2x2.png", dpi=120)
plt.show()
