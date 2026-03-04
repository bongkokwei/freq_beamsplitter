# freq-beamsplitter

Simulator for EO-modulated ring resonator scattering matrices in the frequency synthetic dimension.

**Reference:** Buddhiraju et al., *Nature Communications* (2021) — "Arbitrary linear transformations for photons in the frequency synthetic dimension"

---

## Install

```bash
pip install -e .
```

---

## Core equation

The scattering matrix for a single ring (Eq. 5 of Buddhiraju et al.):

$$M = I + i\sqrt{2\Gamma_e}\left[\Delta\omega \cdot I - i(\Gamma_e + \Gamma_i) - K\right]^{-1}\sqrt{2\Gamma_e}$$

where $K$ is a Hermitian Toeplitz matrix built from the EO modulation coupling constants $\kappa_l$.

---

## Quick start

```python
import numpy as np
from freq_beamsplitter import scattering_matrix, cascaded_scattering_matrix, plot_matrix_grid

# Single ring, 5 modes (N_sb=2), 4 modulation tones
kappa = np.array([0.3+0.1j, 0.2-0.15j, 0.1+0.05j, 0.05+0.2j])
M = scattering_matrix(kappa, N_sb=2)

# 4 cascaded rings
kappa_list = [np.random.randn(4) + 1j*np.random.randn(4) for _ in range(4)]
M_total = cascaded_scattering_matrix(kappa_list, N_sb=2)

# Visualise
plot_matrix_grid([M, M_total], ["Single ring", "4-ring cascade"])
```

---

## Package layout

```
freq_beamsplitter/
├── __init__.py       ← public API
├── core.py           ← scattering matrix physics (build_K, scattering_matrix, ...)
└── visualise.py      ← matplotlib plotting utilities
examples/
└── demo.py           ← reproduces Buddhiraju et al. demo cases
```

---

## API reference

### `core`

| Function | Description |
|---|---|
| `build_K(kappa, N_sb)` | Build Hermitian Toeplitz coupling matrix $K$ |
| `scattering_matrix(kappa, N_sb, ...)` | Single-ring scattering matrix (Eq. 5) |
| `cascaded_scattering_matrix(kappa_list, N_sb, ...)` | Product of N_r ring matrices |
| `unitarity_error(M)` | $\|M^\dagger M - I\|_F$ — deviation from unitarity |
| `fidelity(U, V)` | Normalised inner product $\|{U,V}\| / (\|U\|_F\|V\|_F)$ |

### `visualise`

| Function | Description |
|---|---|
| `plot_matrix(M, title, ...)` | Amplitude + phase panels for one matrix |
| `plot_matrix_grid(matrices, titles, ...)` | Multi-matrix comparison figure |
