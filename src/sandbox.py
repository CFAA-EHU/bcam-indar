#!/usr/bin/env python

# %%
import logging

import numpy as np
import scipy
import matplotlib.pyplot as plt

from bcam.resonance import mechanical, espira

# %%
def kernel(ns, fs, a, freqs):
    dof = len(freqs)
    t = np.arange(ns) / fs
    K = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
    K = a * K.reshape(1, 1, dof)
    K = np.expand_dims(K, axis=2)
    K = K * np.exp(freqs[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, dof)
    K = np.imag(np.sum(K, axis=-1))
    return K

# %%
seed = None
dof = 4
modal = mechanical.randomSystem(
    dof,
    mass_range=(1, 2),
    damping_range=(0.02, 0.05),
    freqs_range=(2*np.pi, 2*np.pi*20),
    damping_type='nop',
    seed=seed)[1]

freqs = modal['frequencies']
modes_ns = modal['mode_shapes']
modes = modes_ns * (1/np.sqrt(np.imag(freqs)))[np.newaxis, :]

# %%
ns, fs = 210, 100
n_out, n_in = 3, 2
rng = np.random.default_rng(seed)

amps = mechanical.mode_to_amps(modes, n_out, n_in)

data = kernel(ns, fs, amps, freqs)
data_noise = data + rng.normal(scale=4e-2, size=data.shape)

# %%
# Plot kernel and noisy version
fig, axs = plt.subplots(n_out, n_in, figsize=(10, 6))
axs = np.atleast_2d(axs)
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
axs = np.atleast_2d(axs)
fig.suptitle('True and Fitted Data')
t = np.arange(ns) / fs
for i in range(n_out):
    for j in range(n_in):
        axs[i, j].plot(t, data[i, j, :], label='True')
        axs[i, j].plot(t, data_fit[i, j, :], label='Fitted', alpha=0.7)
        axs[i, j].set_title(f'Output {i+1}, Input {j+1}')
        axs[i, j].set_xlabel('Time [s]')
        axs[i, j].set_ylabel('Amplitude')
        axs[i, j].legend()
plt.tight_layout()
plt.show()

# %%
# Fit with real modes.
rescale = np.max(np.abs(amps_fit))
mode_fitter = mechanical.RealModes(freqs, amps_fit/rescale, fs, ns)
options = {'disp': True}
modes_fit_R = np.sqrt(rescale) * mode_fitter.fit(options=options)

with np.printoptions(formatter={'complexfloat': '{:.4e}'.format, 'float': '{:.4e}'.format}):
    print('Original modes:\n', modes[:n_out])
    print('Fitter real modes:\n', modes_fit_R)

# %%
# Fit with complex modes.
coords = np.arange(n_out)
rescale = np.max(np.abs(amps_fit))
mode_fitter = mechanical.ComplexModes(freqs, coords, amps_fit/rescale, fs, ns)
x0 = (modes_fit_R/np.sqrt(rescale), np.zeros_like(modes_fit_R))
modes_fit_C = np.sqrt(rescale) * mode_fitter.fit(
    x0,
    options={'verbose': 2})

with np.printoptions(formatter={'complexfloat': '{:.2e}'.format}):
    print(f'Original modes:\n{modes[:n_out]}')
    print(f'Fitter complex modes:\n{modes_fit_C}')

# %%
# Rudimentary plot of objective function in 2D slice of input space.
p0_, p1_ = mode_fitter._raw_modes_fit
center = modes_fit_R/np.sqrt(rescale)
# Normalize directions.
center = mechanical.reshape_modes_output(center, np.zeros_like(center))
p0_ = mechanical.reshape_modes_output(p0_, np.zeros_like(p0_))
p1_ = mechanical.reshape_modes_output(np.zeros_like(p1_), p1_)
p0_ = p0_ - center
l0_ = np.linalg.norm(p0_)
l1_ = np.linalg.norm(p1_)
p0 = p0_/l0_
p1 = p1_/l1_

mech_logger = logging.getLogger('bcam.resonance._core.mechanical')
mech_logger.setLevel('ERROR')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.linspace(-2, 2, 200)
Y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(X, Y)
xa = rng.normal(size=(n_out, dof))
pa = mechanical.reshape_modes_output(xa, np.zeros_like(xa))
Z = [[mode_fitter._fun(center + l0*p0 + l1*p1) for l0, l1 in zip(X[i, :], Y[i, :])]
     for i in range(X.shape[0])]
Z = np.array(Z)
ax.plot_surface(X, Y, Z)
ax.scatter([0], [0], [mode_fitter._fun(center)], color='r')
ax.scatter([l0_], [l1_], [mode_fitter._fun(center + p0_ + p1_)], color='g')
# Label the axes
ax.set_xlabel('p0')
ax.set_ylabel('p1')
ax.set_zlabel('Objective')

plt.show()

# %%
