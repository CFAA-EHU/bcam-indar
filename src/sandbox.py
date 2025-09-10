#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical, derivatives

# %%
# Rudimentary plot of objective function in 2D slice of input space.
dof, n_out, n_in = 4, 3, 2
rng = np.random.default_rng()
freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
x0 = rng.normal(size=(n_out, dof))
z0 = 2e-2*rng.normal(size=(n_out, dof))
z0[:n_out, :n_out] = z0[:n_out, :n_out] - z0[:n_out, :n_out].T
p0 = mechanical.reshape_modes_output(x0, z0)
coords = np.arange(n_out)
modes0 = mechanical.PartialModesMap(freqs, coords)
modes0 = modes0(x0, z0)
amps0 = mechanical.mode_to_amps(modes0, n_out, n_in)

ns, fs = 210, 100
modes = mechanical.Modes(freqs, amps0, coords, fs, ns)

xi = rng.normal(size=(n_out, dof))
zi = 2e-2*rng.normal(size=(n_out, dof))
zi[:n_out, :n_out] = zi[:n_out, :n_out] - zi[:n_out, :n_out].T
pi = mechanical.reshape_modes_output(xi, zi)

xf = rng.normal(size=(n_out, dof))
zf = 2e-2*rng.normal(size=(n_out, dof))
zf[:n_out, :n_out] = zf[:n_out, :n_out] - zf[:n_out, :n_out].T
pf = mechanical.reshape_modes_output(xf, zf)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.linspace(-1, 1, 100)
Y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(X, Y)
xa = rng.normal(size=(n_out, dof))
pa = mechanical.reshape_modes_output(xa, np.zeros_like(xa))
Z = [[modes._fun(pa + l1*pi + l2*pf) for l1, l2 in zip(X[i, :], Y[i, :])]
     for i in range(X.shape[0])]
Z = np.array(Z)
ax.plot_surface(X, Y, Z)
# Label the axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Objective')

plt.show()

# %%

# dof, n_out, n_in = 4, 3, 2
# rng = np.random.default_rng()
# freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
# X = 0.1*rng.normal(size=(n_out, dof))
# amps0 = _mode_to_amps(X, n_out, n_in)
# amps = amps0 + 1e-4*(rng.normal(size=amps0.shape) + 1j*rng.normal(size=amps0.shape))
# ns, fs = 210, 100

# modes = _ModesProp(
#     freqs, fs, ns, n_out=n_out, n_in=n_in)
# res = modes.fit(amps)
# print('Original amps:\n', amps0)
# print('Noisy amps:\n', amps)
# print('===================')
# print('Original:\n', X)
# print('Fitted:\n', res.x.reshape(n_out, dof))

# print(res.success, res.message)

# # %%
# DoF = 4
# mech, modal = mechanical.randomSystem(
#     DoF,
#     mass_range=(1, 2),
#     damping_range=(0.02, 0.05),
#     freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
#     damping_type='nop',
#     seed=None)

# psi = modal['mode_shapes']
# freqs = modal['frequencies']

# # %%
# ns, fs = 210, 100
# n_out, n_in = 3, 2

# amps = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
# amps = amps * (1/np.imag(freqs)).reshape(1, 1, DoF)
# amps = amps[:n_out, :n_in]

# t = np.arange(ns) / fs
# kernel = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
# kernel = amps * kernel.reshape(1, 1, DoF)
# kernel = kernel.reshape(n_out, n_in, 1, DoF)
# kernel = kernel * np.exp(freqs[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
# kernel = np.imag(np.sum(kernel, axis=-1))
