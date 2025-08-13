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
z = modal['frequencies']

# %%
def _factor(freqs, fs):
    return 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))

ns, fs = 210, 100
amps = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
amps = amps * (1/np.imag(z)).reshape(1, 1, DoF)

n_out, n_in = 3, 2
amps = amps[:n_out, :n_in]
t = np.arange(ns) / fs
factor = _factor(z, fs)
kernel = amps * factor.reshape(1, 1, DoF)
kernel = kernel.reshape(n_out, n_in, 1, DoF)
kernel = kernel * np.exp(z[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
kernel = np.imag(np.sum(kernel, axis=-1))
