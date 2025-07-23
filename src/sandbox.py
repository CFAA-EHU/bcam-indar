#!/usr/bin/env python

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.sparse

from bcam.resonance import mechanical

# %%
DoF = 3
mech, modal = mechanical.randomSystem(
    DoF,
    mass_range=(1, 2),
    damping_range=(0.02, 0.06),
    freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
    damping_type='non-proportional',
    seed=None)

psi = modal['mode_shapes']
z = modal['frequencies']

# %%
def _factor(freqs, fs):
    return fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))

ns, fs = 200, 100
amplitudes = psi.reshape(DoF, 1, DoF) * psi.reshape(1, DoF, DoF)
amplitudes = amplitudes * (1/np.imag(z)).reshape(1, 1, DoF)

t = np.arange(ns) / fs
factor = _factor(z, fs)
r = amplitudes * factor.reshape(1, 1, DoF)
r = r.reshape(DoF, DoF, 1, DoF)
r = r * np.exp(z[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, DoF)
kernel = 2*np.imag(np.sum(r, axis=-1))
del r

rng = np.random.default_rng(None)
k_noise = kernel + rng.normal(0, 0.0, kernel.shape)

# %%
# fig, axs = plt.subplots()
# axs.plot(t, kernel[0, 0, :], label='Kernel')
# axs.plot(t, k_noise[0, 0, :], label='Noisy Kernel')
# axs.legend()
# axs.set_xlabel('Time')
# axs.set_ylabel('Velocity')

# plt.show()

# # %%
# k_hat = np.fft.rfft(kernel, axis=-1)
# fr = np.fft.rfftfreq(kernel.shape[-1], 1/fs)

# fig, axs = plt.subplots()
# axs.plot(fr, np.abs(k_hat[0, 0, :]))
# axs.set_xlabel('Frequency')

# plt.show()

# %%
def _trig_fft(x):
    N = x.shape[-1]
    x_hat = np.fft.rfft(x, axis=-1, norm='ortho')
    c = 2*np.ones(N//2 + 1)
    c[0] = 1
    if N % 2 == 0:
        c[-1] = 1
    x_hat = [
        np.sqrt(c) * np.real(x_hat),
        -np.sqrt(2) * np.imag(x_hat)[..., 1:(N-1)//2+1]
    ]
    # Concatenate both parts along the last axis.
    x_hat = np.concatenate(x_hat, axis=-1)
    return x_hat

def _trig_ifft(x):
    N = x.shape[-1]
    c = 2*np.ones(N//2 + 1)
    c[0] = 1
    if N % 2 == 0:
        c[-1] = 1
    x_inv = x[..., :N//2+1].astype(np.complex128)/np.sqrt(c)
    x_inv[..., 1:(N-1)//2+1] = x_inv[..., 1:(N-1)//2+1] - 1j*x[..., N//2 + 1:]/np.sqrt(2)
    x_inv = np.fft.irfft(x_inv, N, axis=-1, norm='ortho')
    return x_inv

# y = rng.normal(size=(5))
# y_hat = _trig_fft(y)
# y_ = _trig_ifft(y_hat)

def _sinh_m(x):
    eps = np.finfo(x.dtype).eps
    y = np.where(x, x, eps)
    return np.exp(-y/2) * y / (2*np.sinh(y/2))

def _sum_exp_weighted(a, fs: float, ns: int):
    # TODO: add the case a = 0.
    T = ns / fs
    r = 1 + (1 - np.exp(a*T))/ns
    r = r - _sinh_m(a/fs) * np.exp(a/fs) * (np.exp(a*T)-1) / (a*T)
    r = r * _sinh_m(a/fs) / a
    return r

def _local_matrix(freqs, fs: float, ns: int):
    N = len(freqs)
    r = np.zeros((2*N-1, 2*N-1))

    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = np.exp(r/(2*fs)) * _sum_exp_weighted(r, fs, ns)
        r *= np.sinh(x[np.newaxis, :]/(2*fs)) * np.sinh(y[:, np.newaxis]/(2*fs))
        r *= fs**2
        return r

    r1 = _mult(freqs, np.conj(freqs))
    r2 = _mult(freqs, freqs)

    r[:N, :N] = np.real(r1 - r2)
    r[N:, :N] = _trig_fft(np.imag(r1 + r2).T)[..., 1:].T
    r[:N, N:] = r[N:, :N].T
    r[N:, N:] = _trig_fft(_trig_fft(np.real(r1 + r2))[..., 1:].T)[..., 1:]
    r *= 2.

    return r

def _reshape_injection(x, dof: int):
    L = x.shape[-1]
    x_ = np.zeros((dof, dof, 2*dof - 1, L), dtype=x.dtype)
    x = x.reshape(dof*(dof+1)//2, 2*dof - 1, L)
    x_[(np.arange(dof), np.arange(dof))] = x[:dof]
    c = dof
    for i in range(1, dof):
        cn = c + dof - i
        x_[(np.arange(dof-i), i+np.arange(dof-i))] = x[c:cn]/np.sqrt(2)
        x_[(i+np.arange(dof-i), np.arange(dof-i))] = x[c:cn]/np.sqrt(2)
        c = cn
    return x_

def _reshape_projection(x, dof: int):
    L = x.shape[-1]
    x_ = np.zeros((dof*(dof+1)//2, 2*dof - 1, L), dtype=x.dtype)
    x_[:dof] = x[(np.arange(dof), np.arange(dof))]
    c = dof
    for i in range(1, dof):
        cn = c + dof - i
        x_[c:cn] = (x[(np.arange(dof-i), i+np.arange(dof-i))] + x[(i+np.arange(dof-i), np.arange(dof-i))]) / np.sqrt(2)
        c = cn
    return x_.reshape(-1, L)

# x = rng.normal(size=((DoF*(DoF+1)//2) * (2*DoF - 1), 2))
# x_ = _reshape_injection(x, DoF)
# x2 = _reshape_projection(x_, DoF)

class _PreCoeff_to_Kernel(scipy.sparse.linalg.LinearOperator):
    
    def __init__(self, freqs, fs: float, ns: int):
        self.freqs = freqs
        self.fs = fs
        self.ns = ns

        dof = len(self.freqs)
        M = dof*(dof+1)//2 * (2*dof - 1)
        super().__init__(
            dtype=np.float64,
            shape=(M, M)
        )

        self._local_matrix = _local_matrix(freqs, fs, ns)

    def _matmat(self, x):
        dof = len(self.freqs)
        r = _reshape_injection(x, dof)
        M_local = self._local_matrix
        r = np.einsum('lk,ijkm->ijlm', M_local, r)
        return _reshape_projection(r, dof) / dof

    def _adjoint(self):
        return self


class _PreCoeff():

    def __init__(self, freqs, fs: float, ns: int):
        self.freqs = freqs
        self.fs = fs
        self.ns = ns

    @property
    def amplitudes_(self):
        dof = len(self.freqs)
        x = self.raw_coeff_
        r = np.concatenate(
            [np.zeros((*x.shape[:2], 1)), x[..., dof:]], axis=-1)
        r = _trig_ifft(r).astype(np.complex128)
        r = x[..., :dof] + 1j*r
        return r

    def _rhs(self, y):
        freqs = self.freqs
        fs = self.fs
        ns = self.ns
        dof = len(freqs)

        def _prod(t):
            T = ns / fs
            t = t.reshape(1, -1)
            freqs_ = fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
            r_ = np.exp(freqs[:, np.newaxis] * t) * (1 - t/T)
            r_ *= freqs_[:, np.newaxis]
            return r_

        r = np.einsum(
            'ijt,kt->ijk',
            y,
            _prod(np.arange(ns)/fs)
        )
        r = 2*r / (dof*fs)
        r = np.concatenate(
            [np.imag(r), _trig_fft(np.real(r))[..., 1:]],
            axis=-1
        )

        return r

    def fit(self, y, cg_kwargs=None):
        dof = len(self.freqs)
        cg_kwargs = {} if cg_kwargs is None else cg_kwargs
        lhs = _PreCoeff_to_Kernel(self.freqs, self.fs, self.ns)
        rhs = self._rhs(y)
        rhs = _reshape_projection(rhs[..., np.newaxis], dof)
        r, info = scipy.sparse.linalg.cg(lhs, rhs, **cg_kwargs)
        # if info != 0:
        #     logger.warning(f'Conjugate gradient did not converge, info={info}')

        r = _reshape_injection(r[..., np.newaxis], dof)
        self.raw_coeff_ = r[..., 0]
        return self.amplitudes_

# %%
preCoeff = _PreCoeff(z, fs, ns)
amplitudes_fit = preCoeff.fit(k_noise)

# print(
#     np.max(np.linalg.norm(amplitudes + amplitudes_fit, axis=-1)/np.linalg.norm(amplitudes, axis=-1))
# )

i, j = 2, 1
fig, axs = plt.subplots(nrows=2)
axs[0].set_title('Real')
axs[0].plot(amplitudes[i, j, :].real, label='Amplitudes')
axs[0].plot(amplitudes_fit[i, j, :].real, label='Fitted amplitudes')
axs[0].legend()
axs[0].set_xlabel('Time')

axs[1].set_title('Imaginary')
axs[1].plot(amplitudes[i, j, :].imag, label='Amplitudes')
axs[1].plot(amplitudes_fit[i, j, :].imag, label='Fitted amplitudes')
axs[1].legend()
axs[1].set_xlabel('Time')

plt.show()

# %%
