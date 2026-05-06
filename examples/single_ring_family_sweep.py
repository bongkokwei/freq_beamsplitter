"""
examples/single_ring_family_sweep.py
=====================================
Monte Carlo sweep over kappa-space for a single ring.  For each random
kappa, compute the full scattering matrix M, then extract the 2x2 block
M_ij = M[{i,j}, {i,j}] for every adjacent mode pair (i, j).

For each block we record three quantities:
  - success probability  P_ij = |M_ii|^2 + |M_ji|^2 + |M_ij|^2 + |M_jj|^2  (averaged)
                         (i.e. how much weight stays in {i,j} when injecting
                          into {i,j}, normalised so 2.0 = no leakage)
  - subspace unitarity  U_ij = ||M_ij^dagger M_ij - I||_F   after renormalisation
                         (how close the post-selected block is to a unitary)
  - parameterisation     theta, phi extracted from M_ij (after renormalisation)
                         using the SU(2) decomposition; tells us *which* gate.

Output: a multi-panel scatter plot per pair, plus a per-pair summary table.
"""

import numpy as np
import matplotlib.pyplot as plt
from freq_beamsplitter import scattering_matrix

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

N_sb = 3  # N = 7 modes
N_f = 6  # modulation tones
N_SAMPLES = 5000  # Monte Carlo samples
KAPPA_SCALE = 0.5  # std-dev of complex Gaussian on each kappa_l
GAMMA_I = 0.0
SEED = 42


# ── 2. Helpers ───────────────────────────────────────────────────────────────


def block_metrics(M, i, j):
    """
    Extract physically meaningful descriptors of the 2x2 block on modes (i, j).

    Returns
    -------
    P_ij    : float in [0, 2]
              Sum of |.|^2 over the 2x2 block. P=2 means injecting
              into i or j stays entirely in {i,j} (no leakage).
              P<2 means leakage out of the subspace.
    U_err   : float
              ||M_block^dagger M_block - I||_F after rescaling so largest
              singular value = 1. Measures how close the block is to a
              (sub-)unitary -- relevant for heralded-gate fidelity.
    F_to_BS : float in [0, 1]
              Block fidelity to a 50:50 beamsplitter (theta=pi/4, phi=0).
              Gives a single scalar to colour scatter plots by.
    F_to_X  : float in [0, 1]
              Block fidelity to sigma_x ([[0,1],[1,0]]).
    F_to_Z  : float in [0, 1]
              Block fidelity to sigma_z ([[1,0],[0,-1]]).
    """
    M_block = M[np.ix_([i, j], [i, j])]
    P = float(np.sum(np.abs(M_block) ** 2))

    # Unitarity of the block (after rescaling so it could plausibly be unitary)
    s = np.linalg.svd(M_block, compute_uv=False)
    if s[0] > 1e-12:
        M_normed = M_block / s[0]
        U_err = float(np.linalg.norm(M_normed.conj().T @ M_normed - np.eye(2)))
    else:
        U_err = np.inf

    # Block fidelities to landmark gates (global-phase invariant)
    def fid(target):
        n = np.linalg.norm(target) * np.linalg.norm(M_block)
        return 0.0 if n == 0 else float(np.abs(np.vdot(target, M_block)) / n)

    BS_5050 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    return P, U_err, fid(BS_5050), fid(sigma_x), fid(sigma_z)


# ── 3. Sweep ─────────────────────────────────────────────────────────────────

N = 2 * N_sb + 1
adjacent_pairs = [(k, k + 1) for k in range(N - 1)]
n_pairs = len(adjacent_pairs)

rng = np.random.default_rng(SEED)
data = {
    pair: {"P": [], "U_err": [], "F_BS": [], "F_X": [], "F_Z": []}
    for pair in adjacent_pairs
}

print(f"Sweeping {N_SAMPLES} kappa samples, single ring, N_f={N_f}, N={N}")
print(f"Adjacent pairs: {adjacent_pairs}")

for s in range(N_SAMPLES):
    kappa = (rng.standard_normal(N_f) + 1j * rng.standard_normal(N_f)) * KAPPA_SCALE
    M = scattering_matrix(kappa, N_sb=N_sb, gamma_e=1.0, gamma_i=GAMMA_I)
    for pair in adjacent_pairs:
        P, Uerr, F_BS, F_X, F_Z = block_metrics(M, *pair)
        data[pair]["P"].append(P)
        data[pair]["U_err"].append(Uerr)
        data[pair]["F_BS"].append(F_BS)
        data[pair]["F_X"].append(F_X)
        data[pair]["F_Z"].append(F_Z)

# ── 4. Per-pair summary table ────────────────────────────────────────────────

print("\nPer-pair summary (best-of-sweep, all metrics):")
print(
    f"  {'Pair':>8}  {'P_max':>8}  {'F_BS_max':>9}  " f"{'F_X_max':>9}  {'F_Z_max':>9}"
)
for pair in adjacent_pairs:
    d = data[pair]
    print(
        f"  {str(pair):>8}  "
        f"{max(d['P']):>8.4f}  "
        f"{max(d['F_BS']):>9.4f}  "
        f"{max(d['F_X']):>9.4f}  "
        f"{max(d['F_Z']):>9.4f}"
    )

# Also: best simultaneous P and F (the "good gate" region: high P AND high F)
print("\nBest simultaneous P_ij >= 1.8 (low leakage) AND high F to landmark:")
print(f"  {'Pair':>8}  {'F_BS':>8}  {'F_X':>8}  {'F_Z':>8}")
for pair in adjacent_pairs:
    d = data[pair]
    P = np.asarray(d["P"])
    mask = P >= 1.8
    if mask.sum() == 0:
        print(f"  {str(pair):>8}     no samples with P >= 1.8")
        continue
    F_BS_best = max(np.asarray(d["F_BS"])[mask])
    F_X_best = max(np.asarray(d["F_X"])[mask])
    F_Z_best = max(np.asarray(d["F_Z"])[mask])
    print(f"  {str(pair):>8}  {F_BS_best:>8.4f}  {F_X_best:>8.4f}  {F_Z_best:>8.4f}")


# ── 5. Plot: 2D scatter (P_ij vs F_BS) per pair ──────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.ravel()
fig.suptitle(
    f"Single-ring family sweep -- {N_SAMPLES} samples, N_f={N_f}\n"
    f"x: subspace weight P_ij (2 = no leakage)   "
    f"y: block fidelity to 50:50 BS",
    fontsize=12,
)

for ax, pair in zip(axes, adjacent_pairs):
    d = data[pair]
    sc = ax.scatter(
        d["P"],
        d["F_BS"],
        c=d["U_err"],
        cmap="viridis_r",
        s=4,
        alpha=0.5,
        vmin=0,
        vmax=1.5,
    )
    ax.set_title(f"modes {pair}  (m={pair[0]-N_sb}, {pair[1]-N_sb})", fontsize=10)
    ax.axhline(0.95, ls="--", c="red", lw=0.6, alpha=0.5)
    ax.axvline(1.8, ls="--", c="red", lw=0.6, alpha=0.5)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

# Hide unused panels (we have 6 pairs and 6 panels for N=7, so this is fine,
# but if N changes the grid may need adjusting)
for ax in axes[n_pairs:]:
    ax.set_visible(False)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(r"subspace unitarity error $\|M_{ij}^\dagger M_{ij}/s_0^2 - I\|_F$")

for ax in axes[3:6]:
    ax.set_xlabel(r"$P_{ij}$ (subspace weight)")
for ax in (axes[0], axes[3]):
    ax.set_ylabel(r"$F(M_{ij}, U_{\rm BS})$")

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.savefig("figures/family_sweep_BS.png", dpi=120)


# ── 6. Plot: histograms of P_ij per pair ─────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(9, 5))
for pair in adjacent_pairs:
    P = np.asarray(data[pair]["P"])
    ax2.hist(
        P,
        bins=60,
        range=(0, 2),
        histtype="step",
        label=f"modes {pair}",
        lw=1.5,
        alpha=0.8,
    )
ax2.set_xlabel(r"$P_{ij}$ -- subspace weight (2 = no leakage)")
ax2.set_ylabel("count")
ax2.set_title(
    f"Subspace-weight distribution per adjacent pair  "
    f"(single ring, N_f={N_f}, {N_SAMPLES} samples)"
)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/family_sweep_histograms.png", dpi=120)

plt.show()
