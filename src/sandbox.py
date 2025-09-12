#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical

# %%
seed = 12345
dof = 4
modal = mechanical.randomSystem(
    dof,
    mass_range=(1, 2),
    damping_range=(0.02, 0.05),
    freqs_range=(2*np.pi, 2*np.pi*20),
    damping_type='nop',
    seed=seed)[1]

def kernel(ns, fs, a, freqs):
    dof = len(freqs)
    t = np.arange(ns) / fs
    K = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
    K = a * K.reshape(1, 1, dof)
    K = np.expand_dims(K, axis=2)
    K = K * np.exp(freqs[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, dof)
    K = np.imag(np.sum(K, axis=-1))
    return K

modes_ns = modal['mode_shapes']
freqs = modal['frequencies']
modes = modes_ns * (1/np.imag(freqs))[np.newaxis, :]

# %%
ns, fs = 210, 100
n_out, n_in = 3, 2
rng = np.random.default_rng(2*seed)

amps = mechanical.mode_to_amps(modes, n_out, n_in)
amps = amps[:n_out, :n_in]

data = kernel(ns, fs, amps, freqs)
data_noise = data + rng.normal(scale=1e-3, size=data.shape)

# %%
# Plot kernel and noisy version
fig, axs = plt.subplots(n_out, n_in, figsize=(10, 6))
fig.suptitle('True and Noisy Data')
t = np.arange(ns) / fs
for i in range(n_out):
    for j in range(n_in):
        axs[i, j].plot(t, data[i, j, :], label='True')
        axs[i, j].plot(t, data_noise[i, j, :], label='Noisy', alpha=0.7)
        axs[i, j].set_title(f'Output {i+1}, Input {j+1}')
        axs[i, j].set_xlabel('Time [s]')
        axs[i, j].set_ylabel('Amplitude')
        axs[i, j].legend()
plt.tight_layout()
plt.show()

# %%
model = mechanical.Amplitudes(
    freqs=freqs,
    fs=fs,
    ns=ns,
    n_out=n_out,
    n_in=n_in,
    a_type='mechanical')
amps_fit = model.fit(data_noise, penalty=0)

# %%
# Evaluate fit quality
data_fit = kernel(ns, fs, amps_fit, freqs)

fig, axs = plt.subplots(n_out, n_in, figsize=(10, 6))
fig.suptitle('True and Fitted Data')
t = np.arange(ns) / fs
for i in range(n_out):
    for j in range(n_in):
        axs[i, j].plot(t, data[i, j, :], label='True')
        axs[i, j].plot(t, data_fit[i, j, :], label='Noisy', alpha=0.7)
        axs[i, j].set_title(f'Output {i+1}, Input {j+1}')
        axs[i, j].set_xlabel('Time [s]')
        axs[i, j].set_ylabel('Amplitude')
        axs[i, j].legend()
plt.tight_layout()
plt.show()
del data_fit

# %%
mode_fitter = mechanical.ModesProp(freqs, amps_fit, fs, ns)
modes_fit_R = mode_fitter.fit()

print('Original modes:\n', modes[:n_out])
print('Fitter real modes:\n', modes_fit_R)

# %%
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
