#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical

# %%
DoF = 4
mech, modal = mechanical.randomSystem(
    DoF,
    mass_range=(1, 2),
    damping_range=(0.02, 0.05),
    freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
    damping_type='non-proportional',
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
def mode_to_amps(mode_shape, n_out, n_in):
    amps = mode_shape[:n_out, np.newaxis] * mode_shape[np.newaxis, :n_in]
    amps = amps * (1/np.imag(freqs)).reshape(1, 1, -1)
    return amps

def guess(amps, freqs):
    n_out, n_in, dof = amps.shape
    amps_ = amps * np.imag(freqs).reshape(1, 1, -1)
    psi = np.zeros((n_out, dof), dtype=amps.dtype)
    psi[:n_in] = np.sqrt(amps_[np.arange(n_in), np.arange(n_in)])

    dom_idxs = np.argmax(np.abs(psi[:n_in]), axis=0)
    dom = psi[dom_idxs, np.arange(dof)]
    psi = amps_[:, dom_idxs, np.arange(dof)]
    psi /= dom[np.newaxis, :]
    return np.real(psi)

rng = np.random.default_rng()

# Reference mode shape.
X0 = 0.1*rng.normal(size=(n_out, DoF))
Z0 = 0.01*rng.normal(size=(n_out, DoF))
Z0 = np.triu(Z0, k=1)
Z0[:n_out, :n_out] = Z0[:n_out, :n_out] - Z0[:n_out, :n_out].T
psi0, cns0 = mechanical.partial_mode_shapes_map(X0, Z0, freqs)
amps0 = mode_to_amps(psi0, n_out, n_in)

# Test mode shape
X = 0.1*rng.normal(size=(n_out, DoF))
Z = 0.01*rng.normal(size=(n_out, DoF))
Z = np.triu(Z, k=1)
Z[:n_out, :n_out] = Z[:n_out, :n_out] - Z[:n_out, :n_out].T
psi, cns = mechanical.partial_mode_shapes_map(X, Z, freqs)
amps = mode_to_amps(psi, n_out, n_in)

psi_guess = guess(amps, freqs)

from scipy.optimize import basinhopping

def fun(x):
    dif = amps - amps0
    dif = np.concatenate(
        [np.real(dif), np.imag(dif)], axis=-1)

    m = mechanical.test_metric_amps(freqs, fs, ns)
    D = np.einsum('kl,ijk,ijl', m, dif, dif)
