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
modes = mechanical.ModesProp(
        freqs, fs, ns, n_out=n_out, n_in=n_in)
res = modes.fit(amps_m)
print('Original:\n', modes_m)
print('Fitted:\n', res.x.reshape(n_out, dof))

modes = mechanical.Modes(freqs, coords, amps_m, fs, ns)
x0 = np.pad(res.x, (0, n_out*dof - n_out*(n_out+1)//2))
res_c = modes.fit(x0)
xf, zf = mechanical.reshape_modes_input(res_c.x, dof, n_out)

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
