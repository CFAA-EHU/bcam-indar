#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical

# %%
rng = np.random.default_rng()
n = 4
x = rng.normal(size=(n, n))
lu, piv = scipy.linalg.lu_factor(x)
print(piv)

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


# %%
DoF = 4
mech, modal = mechanical.randomSystem(
    DoF,
    mass_range=(1, 2),
    damping_range=(0.02, 0.05),
    freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
    damping_type='nop',
    seed=None)

psi = modal['mode_shapes']
freqs = modal['frequencies']

# %%
ns, fs = 210, 100
n_out, n_in = 3, 2

# amps = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
# amps = amps * (1/np.imag(freqs)).reshape(1, 1, DoF)
# amps = amps[:n_out, :n_in]

# t = np.arange(ns) / fs
# kernel = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
# kernel = amps * kernel.reshape(1, 1, DoF)
# kernel = kernel.reshape(n_out, n_in, 1, DoF)
# kernel = kernel * np.exp(freqs[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
# kernel = np.imag(np.sum(kernel, axis=-1))

# %%
m = _metric_amps(freqs, fs, ns)
def fun(x):
    X, Z = _reshape_mode_shapes_input(x, DoF, n_out)
    psi, _ = _partial_mode_shapes_map(X, Z, freqs, coords=coords)
    if psi is np.nan:
        return 1e10
    amps = _mode_to_amps(psi, n_out, n_in)

    dif = amps - amps0
    dif = np.concatenate(
        [np.real(dif), np.imag(dif)], axis=-1)
    return np.einsum('kl,ijk,ijl', m, dif, dif)

# x = rng.normal(size=(2*n_out*DoF - n_out*(n_out+1)//2))
x0 = _amps_to_modes(amps0)
x0 = _reshape_mode_shapes_output(x0, np.zeros_like(x0))
print(fun(x0))

# %%
from scipy.optimize import basinhopping

r = basinhopping(fun, x0)

# %%
psi_min = _reshape_mode_shapes_input(r.x, DoF, n_out)
psi_min = _partial_mode_shapes_map(psi_min[0], psi_min[1], freqs, coords=coords)[0]
