"""
examples/random_NxN_loss_sweep.py
=================================
Inverse-design a Haar-random D x D unitary on a *lossless* ring cascade,
then freeze the kappas and sweep the intrinsic loss gamma_i / gamma_e
from 0 to 10 to see how the fixed design degrades.

Usage:
    python examples/random_NxN_loss_sweep.py [D]      (default D = 5)

Sizing (N_sb, N_r, N_f) for a D x D target:
    The ring always acts on an odd number of modes N = 2*N_sb + 1, so the
    smallest system that holds D modes is N_sb = ceil((D - 1) / 2).
      * D odd  ->  N = D, the target fills the whole lattice:  U_target = U_D.
      * D even ->  N = D + 1, the target sits on modes m = -N_sb..N_sb-1 and
                   the spare mode m = +N_sb is asked to be identity:
                   U_target = U_D (+) 1.

    N_f : tone l couples m -> m + l, and |l| <= N - 1 is the largest
          separation inside the lattice, so N_f = N - 1 = 2*N_sb.
    N_r : the full N x N target has N^2 real DOF and each ring brings
          2*N_f, so N_r = ceil(N^2 / (2*N_f)).

    e.g. D = 4 -> N_sb = 2, N = 5, N_f = 4, N_r = 4  (32 DOF >= 25 required)
         D = 5 -> same system, no spare mode.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from freq_beamsplitter import (
    haar_unitary,
    inverse_design,
    cascaded_scattering_matrix,
    unitarity_error,
    fidelity,
)

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

D = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # target dimension
SEED = 1
N_RESTARTS = 10

GAMMA_E = 1.0
GAMMA_I_DESIGN = 0.0  # gamma_i / gamma_e = 0  ->  lossless design
gamma_i_sweep = np.linspace(0.0, 10.0, 51)
F_OVERCOUPLED = 0.9  # show |M| where the overcoupled fidelity drops to this


# ── 2. Size the system ───────────────────────────────────────────────────────

N_sb = int(np.ceil((D - 1) / 2))
N = 2 * N_sb + 1
N_f = 2 * N_sb
N_r = int(np.ceil(N**2 / (2 * N_f)))
padded = N > D  # D even -> one spare mode held at identity

modes = list(range(D))  # indices 0..D-1  ->  m = -N_sb .. D-1-N_sb
tag = f"random_{D}x{D}"  # figure filename prefix

# LaTeX labels for the target and its block
U_D = rf"U_{{{D}}}"
U_full = rf"{U_D} \oplus 1" if padded else U_D

print("=" * 60)
print(f"Random {D}x{D} unitary  ->  lossless design  ->  loss sweep")
print("=" * 60)
print(f"N_sb = {N_sb}  (N = {N} modes),  N_f = {N_f},  N_r = {N_r}")
print(f"DOF : required {N**2},  available {2 * N_r * N_f}")
print(f"Target block on m = {[k - N_sb for k in modes]}", end="")
print(f",  spare mode m = {N_sb} -> identity\n" if padded else "\n")


# ── 3. Target: Haar-random U_D, embedded as U_D (+) 1 when D is even ──────

U = haar_unitary(D, seed=SEED)
U_target = np.eye(N, dtype=complex)
U_target[np.ix_(modes, modes)] = U


# ── 4. Lossless inverse design ───────────────────────────────────────────────

result = inverse_design(
    U_target,
    N_sb=N_sb,
    N_r=N_r,
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
kappa_list = result.kappa_list


def M_at(gi):
    return cascaded_scattering_matrix(kappa_list, N_sb, gamma_e=GAMMA_E, gamma_i=gi)


def block(M):
    return M[np.ix_(modes, modes)]


M_lossless = M_at(GAMMA_I_DESIGN)
print(f"\nLossless design")
print(f"  F(U_target, M)        : {result.fidelity:.6f}")
print(f"  Unitarity error       : {unitarity_error(M_lossless):.2e}")

print(f"\nOptimised kappa_l / gamma_e (polar):")
for r, kappa in enumerate(kappa_list, start=1):
    terms = "  ".join(
        f"{np.abs(kl):.3f}*e^(i{np.angle(kl)/np.pi:+.3f}pi)" for kl in kappa
    )
    print(f"  Ring {r}: {terms}")


# ── 5. Freeze kappas, sweep intrinsic loss ───────────────────────────────────

F_full, F_block, eta = [], [], []
for gi in gamma_i_sweep:
    M = M_at(gi)
    F_full.append(fidelity(U_target, M))
    F_block.append(fidelity(U, block(M)))
    eta.append(np.linalg.norm(block(M), "fro") ** 2 / D)  # mean power kept per input

F_full, F_block, eta = map(np.array, (F_full, F_block, eta))

blk = f"F_{D}x{D}"
print(f"\n{'gamma_i/gamma_e':>16} {'F_full':>9} {blk:>9} {'eta':>9}")
for gi, ff, fb, e in list(zip(gamma_i_sweep, F_full, F_block, eta))[::5]:
    print(f"{gi:>16.2f} {ff:>9.5f} {fb:>9.5f} {e:>9.2e}")


# Overcoupled operating point: gamma_i in (0, gamma_e) where F_full = F_OVERCOUPLED
f_over = lambda gi: fidelity(U_target, M_at(gi)) - F_OVERCOUPLED
if f_over(0.0) * f_over(GAMMA_E) < 0:
    gi_over = brentq(f_over, 0.0, GAMMA_E)
else:  # design never reached / never left F_OVERCOUPLED in (0, gamma_e)
    gi_over = GAMMA_E
    print(f"\nF_full does not cross {F_OVERCOUPLED} in (0, gamma_e); using critical point")
M_over = M_at(gi_over)
F_over = fidelity(U_target, M_over)
eta_over = np.linalg.norm(block(M_over), "fro") ** 2 / D
print(f"\nOvercoupled point: gamma_i/gamma_e = {gi_over:.4f}")
print(
    f"  F_full = {F_over:.4f},  {blk} = "
    f"{fidelity(U, block(M_over)):.4f},  eta = {eta_over:.3f}"
)
print(np.array2string(M_over, precision=3, suppress_small=True))


# ── 6. Plot (one figure each, dark theme, transparent background) ─────────

plt.style.use("dark_background")
X_LABEL = r"$\gamma_i / \gamma_e$"
mode_labels = [str(m) for m in range(-N_sb, N_sb + 1)]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight", transparent=True)


# 6a. Fidelity vs loss
fig, ax = plt.subplots(figsize=(6, 4.5))
# gamma_i < gamma_e: overcoupled | gamma_i = gamma_e: critical | > : undercoupled
ax.axvspan(0, GAMMA_E, color="tab:blue", alpha=0.15, lw=0)
ax.axvspan(GAMMA_E, gamma_i_sweep[-1], color="tab:orange", alpha=0.10, lw=0)
ax.axvline(GAMMA_E, color="white", ls="--", lw=1, label="Critical coupling")
ax.text(GAMMA_E / 2, 0.03, "Over-\ncoupled", ha="center", va="bottom", fontsize=9)
ax.text(
    (GAMMA_E + gamma_i_sweep[-1]) / 2,
    0.03,
    "Undercoupled",
    ha="center",
    va="bottom",
    fontsize=9,
)
ax.plot(gamma_i_sweep, F_full, "o-", ms=3, label=rf"Full {N}x{N}: $F({U_full}, M)$")
if padded:  # with no spare mode the block is the full matrix
    ax.plot(
        gamma_i_sweep,
        F_block,
        "s-",
        ms=3,
        label=rf"{D}x{D} block: $F({U_D}, M_{{{D}}})$",
    )
ax.plot(
    gi_over,
    F_over,
    "*",
    ms=12,
    color="white",
    zorder=5,
    label=rf"$\gamma_i/\gamma_e = {gi_over:.2f}$, F = {F_over:.2f}",
)
ax.set_xlabel(X_LABEL)
ax.set_ylabel("Fidelity")
ax.set_ylim(0, 1.05)
ax.set_xlim(gamma_i_sweep[0], gamma_i_sweep[-1])
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
save(fig, f"{tag}_fidelity_vs_loss")

# 6b. Transmission vs loss
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.semilogy(gamma_i_sweep, eta, "^-", ms=3, color="tab:red")
ax.set_xlabel(X_LABEL)
ax.set_ylabel(rf"$\eta = \|M_{{{D}}}\|_F^2 / {D}$")
ax.grid(True, alpha=0.3, which="both")
save(fig, f"{tag}_transmission_vs_loss")

# 6c. |M| for target, lossless, overcoupled point, maximum loss
M_max_loss = M_at(gamma_i_sweep[-1])
for name, mat, title in [
    ("target", U_target, rf"Target $|{U_full}|$"),
    ("lossless", M_lossless, rf"$|M|$, lossless, F = {F_full[0]:.4f}"),
    (
        "overcoupled",
        M_over,
        rf"$|M|$, $\gamma_i/\gamma_e = {gi_over:.2f}$, F = {F_over:.2f}",
    ),
    (
        "max_loss",
        M_max_loss,
        rf"$|M|$, $\gamma_i/\gamma_e = {gamma_i_sweep[-1]:.0f}$, F = {F_full[-1]:.4f}",
    ),
]:
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(np.abs(mat), vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N), mode_labels)
    ax.set_yticks(range(N), mode_labels)
    ax.set_xlabel("Input m'")
    ax.set_ylabel("Output m")
    ax.set_title(title)
    save(fig, f"{tag}_matrix_{name}")

plt.show()
