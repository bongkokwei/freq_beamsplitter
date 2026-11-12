import argparse
import numpy as np
from freq_beamsplitter import scattering_matrix
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Single ring scattering matrix visualiser"
    )
    parser.add_argument(
        "--N", type=int, default=7, help="Number of frequency modes (default: 7)"
    )
    parser.add_argument(
        "--N-tones", type=int, default=2, help="Number of modulation tones (default: 2)"
    )
    parser.add_argument(
        "--mode-num", type=int, default=0, help="Input mode index m (default: 0)"
    )
    args = parser.parse_args()

    N = args.N
    c = N // 2  # centre index: sideband label m = k - c
    N_tones = args.N_tones
    mode_num = args.mode_num

    assert -c <= mode_num < N - c, f"mode_num must be in [{-c}, {N - c - 1}]"
    assert N_tones <= N - 1, f"N_tones must be <= N - 1 = {N - 1}"

    rng = np.random.default_rng(42)
    kappa = rng.standard_normal(N_tones) + 1j * rng.standard_normal(N_tones)

    M = scattering_matrix(kappa, N, gamma_i=0.0)

    s_in = np.zeros(N, dtype=complex)
    s_in[c + mode_num] = 1.0
    s_out = M @ s_in

    modes = np.arange(-c, N - c)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Single ring — N={N}, {N_tones} tone(s), input m={mode_num}, γ_i = 0"
    )

    im0 = axes[0].imshow(np.abs(M), vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("|M|")
    axes[0].set_xlabel("Input mode m'")
    axes[0].set_ylabel("Output mode m")
    axes[0].set_xticks(range(N))
    axes[0].set_xticklabels(modes)
    axes[0].set_yticks(range(N))
    axes[0].set_yticklabels(modes)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.angle(M) / np.pi, vmin=-1, vmax=1, cmap="RdBu")
    axes[1].set_title("∠M / π")
    axes[1].set_xlabel("Input mode m'")
    axes[1].set_ylabel("Output mode m")
    axes[1].set_xticks(range(N))
    axes[1].set_xticklabels(modes)
    axes[1].set_yticks(range(N))
    axes[1].set_yticklabels(modes)
    plt.colorbar(im1, ax=axes[1])

    axes[2].bar(modes, np.abs(s_out) ** 2)
    axes[2].set_xlabel("Output mode m")
    axes[2].set_ylabel("Power $|s^-_m|^2$")
    axes[2].set_title(
        f"Output — input m={mode_num}  (total = {np.sum(np.abs(s_out)**2):.4f})"
    )
    axes[2].set_xticks(modes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
