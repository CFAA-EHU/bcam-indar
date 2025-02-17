#!/usr/bin/env python
# Example of the usage of Tustin class to compute the eigenvalues of a delay system.

# %%
import numpy as np
import matplotlib.pyplot as plt

from bcam.resonance import matrices

# %%
# Define delay system.
rng = np.random.default_rng(seed=288475)
B0 = 20 + rng.normal(0, 2, (4, 4)) + 1j * rng.normal(0, 5, (4, 4))
B1 = -5j + rng.normal(0, 10, (4, 3)) + 1j * rng.normal(0, 2, (4, 3))
B1 = np.hstack((B1, np.zeros((4, 1))), dtype=B0.dtype)

tustin = matrices.Tustin(degree=3, method='bspline')
# Compute eigenvalues using QR algorithm.
M = 150
eigs = tustin.eigvals(B0, B1, M)

# Plot eigenvalues.
fig, ax = plt.subplots()
ax.plot(eigs.real, eigs.imag, 'o')
ax.set_xlim(-5, 10)
ax.set_ylim(-210, 210)
ax.set_xlabel('Re')
ax.set_ylabel('Im')

plt.show()
