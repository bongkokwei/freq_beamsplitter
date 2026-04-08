import numpy as np
from freq_beamsplitter import scattering_matrix, plot_matrix
import matplotlib.pyplot as plt

N_sb = 3
N_f = 2 * N_sb
rng = np.random.default_rng(42)
kappa = rng.standard_normal(N_f) + 1j * rng.standard_normal(N_f)

M = scattering_matrix(kappa, N_sb, gamma_i=0.0)

plot_matrix(M, title="Single ring — |κ| random, γ_i = 0")
plt.show()
