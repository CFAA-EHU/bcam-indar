#!/usr/bin/env python

# %%
import numpy as np

from bcam.resonance import mechanical

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


def check_stability(
    dof, n_out, n_in, ns, fs, noise, seed=None):
    parent_rng = np.random.default_rng(seed)
    child = parent_rng.spawn(2)

    modal = mechanical.randomSystem(
        dof,
        mass_range=(1, 2),
        damping_range=(0.02, 0.05),
        freqs_range=(2*np.pi, 2*np.pi*20),
        damping_type='nop',
        seed=child[0])[1]

    freqs = modal['frequencies']
    modes_ns = modal['mode_shapes']
    modes = modes_ns * (1/np.sqrt(np.imag(freqs)))[np.newaxis, :]

    amps = mechanical.mode_to_amps(modes, n_out, n_in)
    data = kernel(ns, fs, amps, freqs)
    data_noise = data + child[1].normal(scale=noise, size=data.shape)

    # Fit amplitudes
    model = mechanical.Amplitudes(
        freqs=freqs,
        fs=fs,
        ns=ns,
        n_out=n_out,
        n_in=n_in,
        a_type='mechanical')
    amps_fit = model.fit(data_noise, penalty=0)

    # Fit with real modes.
    rescale = np.max(np.abs(amps_fit))
    mode_fitter = mechanical.RealModes(freqs, amps_fit/rescale, fs, ns)
    modes_fit_R = np.sqrt(rescale) * mode_fitter.fit(options={'gtol': 1e-3})

    amps_R = mechanical.mode_to_amps(modes_fit_R, n_out, n_in)
    kernel_fit = kernel(ns, fs, amps_R, freqs)
    error = np.sum(
        ((kernel_fit - data)**2)*(1 - np.arange(ns)/ns).reshape(1, 1, ns)
    )
    error = np.sqrt(error/fs)
    print(f'Error (real modes): {error:.2e}')

    # Fit with complex modes.
    coords = np.arange(n_out)
    mode_fitter = mechanical.ComplexModes(freqs, coords, amps_fit/rescale, fs, ns)
    x0 = (modes_fit_R/np.sqrt(rescale), np.zeros_like(modes_fit_R))
    modes_fit_C = np.sqrt(rescale) * mode_fitter.fit(x0, options={'verbose': 0, 'gtol': 1e-3})

    amps_C = mechanical.mode_to_amps(modes_fit_C, n_out, n_in)
    kernel_fit = kernel(ns, fs, amps_C, freqs)
    error = np.sum(
        ((kernel_fit - data)**2)*(1 - np.arange(ns)/ns).reshape(1, 1, ns)
    )
    error = np.sqrt(error/fs)
    print(f'Error (complex modes): {error:.2e}')

# %%
check_stability(
    dof=4,
    n_out=3,
    n_in=2,
    ns=210,
    fs=100,
    noise=0,
    seed=None
)
