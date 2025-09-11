#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical

# %%
dof, n_out, n_in = 4, 3, 2
rng = np.random.default_rng()
freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
coords = np.arange(n_out)
modes_m = np.nan
while isinstance(modes_m, float):
    xm = rng.normal(size=(n_out, dof))
    zm = 5e-2*rng.normal(size=(n_out, dof))
    zm[:n_out, :n_out] = zm[:n_out, :n_out] - zm[:n_out, :n_out].T
    modes_m = mechanical.PartialModesMap(freqs, coords)(xm, zm)
amps_m = mechanical.mode_to_amps(modes_m, n_out, n_in)

ns, fs = 210, 100
# Fit as proportional as initial guess.
modes = mechanical.ModesProp(
        freqs, amps_m, fs, ns)
res = modes.fit()

modes = mechanical.Modes(freqs, coords, amps_m, fs, ns)
x0 = np.pad(res.x, (0, n_out*dof - n_out*(n_out+1)//2))
res = modes.fit(x0, options={'verbose': 2})
xf, zf = mechanical.reshape_modes_input(res.x, dof, n_out)
modes_fit = mechanical.PartialModesMap(freqs, coords)(xf, zf)
# The result is unique up to a sign flip in each mode.
modes_fit *= np.sign(np.real(modes_m[0, :]/modes_fit[0, :]))[np.newaxis, :]

# # %%
# # Rudimentary plot of objective function in 2D slice of input space.
# xi = rng.normal(size=(n_out, dof))
# zi = 2e-2*rng.normal(size=(n_out, dof))
# zi[:n_out, :n_out] = zi[:n_out, :n_out] - zi[:n_out, :n_out].T
# pi = mechanical.reshape_modes_output(xi, zi)

# xf = rng.normal(size=(n_out, dof))
# zf = 2e-2*rng.normal(size=(n_out, dof))
# zf[:n_out, :n_out] = zf[:n_out, :n_out] - zf[:n_out, :n_out].T
# pf = mechanical.reshape_modes_output(xf, zf)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X = np.linspace(-1, 1, 100)
# Y = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(X, Y)
# xa = rng.normal(size=(n_out, dof))
# pa = mechanical.reshape_modes_output(xa, np.zeros_like(xa))
# Z = [[modes._fun(pa + l1*pi + l2*pf) for l1, l2 in zip(X[i, :], Y[i, :])]
#      for i in range(X.shape[0])]
# Z = np.array(Z)
# ax.plot_surface(X, Y, Z)
# # Label the axes
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Objective')

# plt.show()

# # %%
# DoF = 4
# mech, modal = mechanical.randomSystem(
#     DoF,
#     mass_range=(1, 2),
#     damping_range=(0.02, 0.05),
#     freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
#     damping_type='nop',
#     seed=None)

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
