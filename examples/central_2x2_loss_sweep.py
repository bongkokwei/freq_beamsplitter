"""
examples/central_2x2_loss_sweep.py
==================================
Heralded (post-selected) 2x2 beamsplitter embedded near the centre of a
7x7 single-ring scattering matrix.  Design the kappas for a *lossless*
ring (gamma_i / gamma_e = 0), then freeze them and sweep the intrinsic
loss gamma_i / gamma_e from 0 to 10.

    N_sb = 3  ->  N = 7 modes, m = -3..+3
    N_r  = 1, N_f = 3
    (i, j) = (3, 4)  ->  m = 0, +1
    U2   = beamsplitter(theta = pi/4)  (50:50)

Post-selected fidelity: F(U2, M_ij), with M_ij = M[{i,j}, {i,j}].
Heralding probability P_ij / 2 = sum |M_ij|^2 / 2 is reported alongside.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from freq_beamsplitter import (
    scattering_matrix,
    beamsplitter,
    embed_unitary_2x2,
    inverse_design_heralded,
)
from freq_beamsplitter.heralded_design import block_fidelity, heralding_probability

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

N_sb = 3  # N = 7 modes
N_f = 3  # modulation tones (single ring DOF = 6)
i, j = 3, 4  # central pair (m = 0, +1)
U2 = beamsplitter(theta=np.pi / 4, phi=0.0)

GAMMA_E = 1.0
GAMMA_I_DESIGN = 0.0  # gamma_i / gamma_e = 0  ->  lossless design
gamma_i_sweep = np.linspace(0.0, 10.0, 101)  # fidelity curve
gamma_i_matrices = np.arange(0, 11)  # one |M| panel per integer loss
CHIPS = {"Existing chip": (7.2, "tab:orange"), "Redesign chip": (0.25, "tab:green")}

N_RESTARTS = 30
SEED = 0

N = 2 * N_sb + 1
mode_labels = [str(m) for m in range(-N_sb, N_sb + 1)]


# ── 2. Lossless heralded design ──────────────────────────────────────────────

print("=" * 60)
print("Central 2x2 (pi/4 BS) in 7x7  ->  lossless design  ->  loss sweep")
print("=" * 60)
print(f"N = {N}, single ring, N_f = {N_f},  subspace m = ({i-N_sb}, {j-N_sb})\n")

result = inverse_design_heralded(
    U2=U2,
    i=i,
    j=j,
    N_sb=N_sb,
    N_r=1,
    N_f=N_f,
    gamma_e=GAMMA_E,
    gamma_i=GAMMA_I_DESIGN,
    delta_omega_list=0.0,
    n_restarts=N_RESTARTS,
    kappa_scale=0.5,
    fidelity_tol=1 - 1e-4,
    seed=SEED,
    verbose=True,
)
kappa = result.kappa_list[0]

print(f"\nLossless design: {result}")
print(f"\nOptimised kappa_l / gamma_e (polar):")
for l, kl in enumerate(kappa, start=1):
    print(f"  kappa_{l}  =  {np.abs(kl):.4f} * exp(i {np.angle(kl)/np.pi:+.4f} pi)")


# ── 3. Freeze kappas, sweep intrinsic loss ───────────────────────────────────


def M_at(gi):
    return scattering_matrix(kappa, N_sb=N_sb, gamma_e=GAMMA_E, gamma_i=gi)


F_post = np.array([block_fidelity(M_at(gi), U2, i, j) for gi in gamma_i_sweep])
P_success = np.array(
    [heralding_probability(M_at(gi), i, j) / 2 for gi in gamma_i_sweep]
)

print(f"\n{'gamma_i/gamma_e':>16} {'F_post':>9} {'P_success':>10}")
for gi, f, p in list(zip(gamma_i_sweep, F_post, P_success))[::10]:
    print(f"{gi:>16.2f} {f:>9.5f} {p:>10.2e}")


# ── 4. Plot ──────────────────────────────────────────────────────────────────

plt.style.use("dark_background")
X_LABEL = r"$\gamma_i / \gamma_e$"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight", transparent=True)


# 4a. Post-selected fidelity vs loss
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.axvline(GAMMA_E, color="white", ls="--", lw=1, label="Critical coupling")
ax.plot(gamma_i_sweep, F_post, "-", color="tab:blue", label=r"$F(U_2, M_{ij})$")
for name, (gi, color) in CHIPS.items():
    F_chip = block_fidelity(M_at(gi), U2, i, j)
    print(f"{name}: gamma_i/gamma_e = {gi},  F_post = {F_chip:.4f}")
    ax.axvline(gi, color=color, ls=":", lw=1)
    ax.plot(
        gi,
        F_chip,
        "o",
        ms=7,
        color=color,
        zorder=5,
        label=rf"{name}: $\gamma_i/\gamma_e = {gi}$, F = {F_chip:.3f}",
    )
ax.set_xlabel(X_LABEL)
ax.set_ylabel("Post-selected fidelity")
ax.set_ylim(0, 1.05)
ax.set_xlim(gamma_i_sweep[0], gamma_i_sweep[-1])
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, loc="center right")
ax.set_title(rf"$\pi/4$ BS on m = ({i-N_sb}, {j-N_sb}),  $N_f = {N_f}$")
save(fig, "central_2x2_fidelity_vs_loss")

# 4b. |M| at each integer loss, 2x2 block outlined
U_view = embed_unitary_2x2(U2=U2, i=i, j=j, N=N)
panels = [("Target view", U_view)] + [
    (
        rf"$\gamma_i/\gamma_e = {gi:g}$, F = {block_fidelity(M_at(gi), U2, i, j):.3f}",
        M_at(gi),
    )
    for gi in gamma_i_matrices
]

n_cols = 4
n_rows = int(np.ceil(len(panels) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows))
for ax, (title, M) in zip(axes.flat, panels):
    im = ax.imshow(np.abs(M), vmin=0, vmax=1, cmap="viridis")
    ax.add_patch(
        Rectangle(
            (min(i, j) - 0.5, min(i, j) - 0.5),
            2,
            2,
            fill=False,
            edgecolor="tab:red",
            lw=2.5,
        )
    )
    ax.set_xticks(range(N), mode_labels)
    ax.set_yticks(range(N), mode_labels)
    ax.set_title(title, fontsize=10)
for ax in axes.flat[len(panels) :]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel("Output m")
for ax in axes[-1, :]:
    ax.set_xlabel("Input m'")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="|M|")
fig.savefig(
    "figures/central_2x2_matrices_vs_loss.png",
    dpi=150,
    bbox_inches="tight",
    transparent=True,
)

plt.show()
