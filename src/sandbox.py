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

# %%
model = mechanical.Amplitudes(
    freqs=z,
    fs=fs,
    ns=ns,
    n_out=n_out,
    n_in=n_in,
    a_type='mechanical')
amps_fit = model.fit(kernel)

error = np.linalg.norm(amps_fit - amps, axis=-1) / np.linalg.norm(amps, axis=-1)
print(np.max(error))

fig, axs = plt.subplots(nrows=2)
axs[0].plot(amps[0, 0].real, 'o-', label='original')
axs[0].plot(amps_fit[0, 0].real, 'o-', label='fitted')
axs[0].legend()

axs[1].plot(amps[1, 0].real, 'o-', label='original')
axs[1].plot(amps_fit[1, 0].real, 'o-', label='fitted')
axs[1].legend()

plt.show()

# # %%
# model = mechanical.Amplitudes(
#     freqs=z,
#     fs=fs,
#     ns=ns,
#     n_out=n_out,
#     n_in=n_in
# )
# amps_fit = model.fit(data, penalty=0.)

# rng = np.random.default_rng(None)
# k_noise = kernel + rng.normal(0, 0.03, kernel.shape)

# # %%
# i, j = 1, 1

# fig, axs = plt.subplots()
# axs.plot(t, kernel[i, j, :], label='Kernel')
# axs.plot(t, k_noise[i, j, :], label='Noisy Kernel')
# axs.legend()
# axs.set_xlabel('Time')
# axs.set_ylabel('Velocity')

# plt.show()

# # %%
# k_hat = np.fft.rfft(kernel, axis=-1)
# k_hat_noise = np.fft.rfft(k_noise, axis=-1)
# fr = np.fft.rfftfreq(kernel.shape[-1], 1/fs)

# i, j = 1, 1
# fig, axs = plt.subplots()
# axs.plot(fr, np.abs(k_hat[i, j, :]), label='Kernel')
# axs.plot(fr, np.abs(k_hat_noise[i, j, :]), label='Noisy Kernel')
# axs.legend()
# axs.set_xlabel('Frequency')

# plt.show()

# # %%
# def _factor(freqs, fs):
#     return 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))

# ns, fs = 210, 100
# amplitudes = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
# amplitudes = amplitudes * (1/np.imag(z)).reshape(1, 1, DoF)

# t = np.arange(ns) / fs
# factor = _factor(z, fs)
# r = amplitudes * factor.reshape(1, 1, DoF)
# r = r.reshape(DoF, DoF, 1, DoF)
# r = r * np.exp(z[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
# kernel = np.imag(np.sum(r, axis=-1))
# del r

# # %%
