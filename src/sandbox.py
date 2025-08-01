#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical

# %%
DoF = 3
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
amplitudes = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
amplitudes = amplitudes * (1/np.imag(z)).reshape(1, 1, DoF)

t = np.arange(ns) / fs
factor = _factor(z, fs)
r = amplitudes * factor.reshape(1, 1, DoF)
r = r.reshape(DoF, DoF, 1, DoF)
r = r * np.exp(z[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
kernel = np.imag(np.sum(r, axis=-1))
del r

rng = np.random.default_rng(None)
k_noise = kernel + rng.normal(0, 0.03, kernel.shape)

# %%
i, j = 1, 1

fig, axs = plt.subplots()
axs.plot(t, kernel[i, j, :], label='Kernel')
axs.plot(t, k_noise[i, j, :], label='Noisy Kernel')
axs.legend()
axs.set_xlabel('Time')
axs.set_ylabel('Velocity')

plt.show()

# %%
k_hat = np.fft.rfft(kernel, axis=-1)
k_hat_noise = np.fft.rfft(k_noise, axis=-1)
fr = np.fft.rfftfreq(kernel.shape[-1], 1/fs)

i, j = 1, 1
fig, axs = plt.subplots()
axs.plot(fr, np.abs(k_hat[i, j, :]), label='Kernel')
axs.plot(fr, np.abs(k_hat_noise[i, j, :]), label='Noisy Kernel')
axs.legend()
axs.set_xlabel('Frequency')

plt.show()

# %%
def _factor(freqs, fs):
    return 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))

ns, fs = 210, 100
amplitudes = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
amplitudes = amplitudes * (1/np.imag(z)).reshape(1, 1, DoF)

t = np.arange(ns) / fs
factor = _factor(z, fs)
r = amplitudes * factor.reshape(1, 1, DoF)
r = r.reshape(DoF, DoF, 1, DoF)
r = r * np.exp(z[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
kernel = np.imag(np.sum(r, axis=-1))
del r

# %%
