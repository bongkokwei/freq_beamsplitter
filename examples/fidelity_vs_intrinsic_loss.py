"""
fidelity_vs_gamma_i.py
======================
Sweep intrinsic loss γ_i / γ_e and plot the best achievable fidelity
for a fixed Haar-random target unitary using inverse_design.

Physics reminder:
    γ_e is normalised to 1 in the codebase, so γ_i here is the
    fractional parasitic loss relative to the external coupling rate.
    At γ_i = 0 the ring is lossless and M is unitary.
"""

import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import haar_unitary, inverse_design

# ── System parameters ─────────────────────────────────────────────────────────
N_sb = 2  # sidebands → 5×5 matrix
N_f = 2 * N_sb  # modulation tones per ring
N_r = N_sb + 2  # rings in cascade
SEED = 42  # reproducible Haar sample

# ── γ_i sweep ─────────────────────────────────────────────────────────────────
gamma_i_values = np.linspace(0.0, 1.0, 15)  # adjust resolution / range as needed

# ── Target unitary (fixed across sweep) ───────────────────────────────────────
U_target = haar_unitary(2 * N_sb + 1, seed=SEED)

# ── Run sweep ─────────────────────────────────────────────────────────────────
fidelities = []

for gi in gamma_i_values:
    print(f"\n{'='*50}")
    print(f"γ_i / γ_e = {gi:.3f}")
    print(f"{'='*50}")

    result = inverse_design(
        U_target,
        N_sb=N_sb,
        N_r=N_r,
        N_f=N_f,
        gamma_e=1.0,
        gamma_i=gi,
        delta_omega_list=0.0,
        n_restarts=5,
        kappa_scale=0.5,
        fidelity_tol=1 - 1e-4,
        seed=SEED,
        verbose=True,
    )
    fidelities.append(result.fidelity)
    print(f"  → Best fidelity: {result.fidelity:.6f}")

fidelities = np.array(fidelities)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: fidelity
ax1.plot(gamma_i_values, fidelities, "o-", color="tab:blue", linewidth=1.5)
ax1.set_xlabel(r"$\gamma_i / \gamma_e$", fontsize=13)
ax1.set_ylabel(r"Fidelity $F(U, M)$", fontsize=13)
ax1.set_title("Fidelity vs intrinsic loss", fontsize=14)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

# Right: log(1 - F) — easier to see degradation at high fidelity
infidelity = 1.0 - fidelities
infidelity = np.clip(infidelity, 1e-16, None)  # avoid log(0)
ax2.semilogy(gamma_i_values, infidelity, "s-", color="tab:red", linewidth=1.5)
ax2.set_xlabel(r"$\gamma_i / \gamma_e$", fontsize=13)
ax2.set_ylabel(r"$1 - F$", fontsize=13)
ax2.set_title("Infidelity vs intrinsic loss", fontsize=14)
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(
    f"$N_{{sb}}={N_sb}$,  $N_r={N_r}$,  $N_f={N_f}$  "
    f"(Haar-random {2*N_sb+1}×{2*N_sb+1} target)",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
plt.savefig("figures/fidelity_vs_gamma_i.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Saved: figures/fidelity_vs_gamma_i.png")
