#!/usr/bin/env python3
'''
This is ...
'''

import logging

import numpy as np
import jitcode
from sklearn.base import BaseEstimator
import scipy

from . import derivatives


logger = logging.getLogger(__name__)

# =================================
# Helper Functions
# =================================

def _validate_dims(M, C, K, check_symmetry=True):
    M, C, K = np.asarray(M), np.asarray(C), np.asarray(K)
    
    if M.ndim != 2:
        msg = f'Expected a 2-darray, got {M.ndim}-darray.'
        raise ValueError(msg)
    
    if M.shape[0] != M.shape[1]:
        msg = 'M must be a square array.'
        raise ValueError(msg)

    if (M.shape != C.shape) or (M.shape != K.shape):
        msg = 'Incompatible shapes for M, C, and K.'
        raise ValueError(msg)

    if check_symmetry:
        if not np.allclose(M, M.T):
            msg = 'M must be symmetric.'
            raise ValueError(msg)

        if not np.allclose(K, K.T):
            msg = 'K must be symmetric.'
            raise ValueError(msg)
    
    return M, C, K


# =================================
# Amplitudes
# =================================

def _sum_exp_weighted(a, fs:float, ns:int):
    z = np.exp(a/fs)
    sl = np.abs(z-1)>1e-4
    z[sl] = (z[sl]*((z[sl]**ns)-ns-1)+ns)/((z[sl]-1)**2)
    sl = ~sl
    z[sl] = sum((ns-i)*(z[sl]**i) for i in range(ns))
    return z/ns

def trig_fft(x):
    '''
    Trigonometric expansion of a real signal.
    '''
    N = x.shape[-1]
    if x.size == 0:
        return x
    x_hat = np.fft.rfft(x, axis=-1, norm='ortho')
    c = 2*np.ones(N//2 + 1)
    c[0] = 1
    if N % 2 == 0:
        c[-1] = 1
    x_hat = [
        np.sqrt(c) * np.real(x_hat), # cosine part
        -np.sqrt(2) * np.imag(x_hat)[..., 1:(N-1)//2+1] # sine part
    ]
    # Concatenate both parts along the last axis.
    x_hat = np.concatenate(x_hat, axis=-1)
    return x_hat

def trig_ifft(x):
    '''
    Inverse trigonometric expansion of a real signal.
    '''
    N = x.shape[-1]
    c = 2*np.ones(N//2 + 1)
    c[0] = 1
    if N % 2 == 0:
        c[-1] = 1
    x_inv = x[..., :N//2+1].astype(np.complex128)/np.sqrt(c)
    x_inv[..., 1:(N-1)//2+1] = x_inv[..., 1:(N-1)//2+1] - 1j*x[..., N//2 + 1:]/np.sqrt(2)
    x_inv = np.fft.irfft(x_inv, N, axis=-1, norm='ortho')
    return x_inv

def metric_amps(freqs, fs, ns, response='a'):
    '''
    Compute the metric for amplitude coefficients.

    This matrix is the product A^TA, where A represents the components
    of a exponential sum with three components:
    - resonances: complex frequencies of the modes.
    - c_freqs: complex frequencies not attached to mode shapes.
    - r_freqs: real frequencies not attached to mode shapes.
    '''
    g_freqs = 3*[0]
    c = 0
    for k in ['resonances', 'complex', 'real']:
        tmp = freqs.get(k)
        g_freqs[c] = np.array(tmp) if tmp is not None else np.array([])
        c += 1
    resonances, c_freqs, r_freqs = g_freqs
    dof, n_c, n_r = len(resonances), len(c_freqs), len(r_freqs)

    if response not in ['a', 'v']:
        raise ValueError(f'Unknown response type: {response}')

    # Model resonances block.
    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = np.exp(r/(2*fs)) * _sum_exp_weighted(r, fs, ns)
        r *= (4*fs**2)*np.sinh(x[np.newaxis, :]/(2*fs))*np.sinh(y[:, np.newaxis]/(2*fs))
        if response == 'a':
            r *= x[np.newaxis, :]*y[:, np.newaxis]
        return r

    m1 = _mult(resonances, np.conj(resonances))
    m2 = _mult(resonances, resonances)

    dim_c = 2*dof - 1 if dof > 0 else 0
    m_r_r = np.zeros((dim_c, dim_c))
    m_r_r[:dof, :dof] = np.real(m1 - m2)
    m_r_r[dof:, :dof] = trig_fft(np.imag(m1 + m2).T)[..., 1:].T
    m_r_r[:dof, dof:] = m_r_r[dof:, :dof].T
    m_r_r[dof:, dof:] = trig_fft(trig_fft(np.real(m1 + m2))[..., 1:].T)[..., 1:]
    m_r_r *= 0.5

    # Block (c_freqs, resonances).
    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = np.exp(x[np.newaxis, :]/(2*fs)) * _sum_exp_weighted(r, fs, ns)
        r *= (2*fs)*np.sinh(x[np.newaxis, :]/(2*fs))
        if response == 'a':
            r *= x[np.newaxis, :]
        return r

    m1 = _mult(resonances, c_freqs)
    m2 = _mult(resonances, np.conj(c_freqs))

    m_r_f = np.zeros((2*n_c, dim_c))
    m_r_f[:n_c, :dof] = np.imag(m1 + m2)
    m_r_f[n_c:, :dof] = np.real(m1 - m2)
    m_r_f[:n_c, dof:] = trig_fft(np.real(m1 + m2))[..., 1:]
    m_r_f[n_c:, dof:] = trig_fft(np.imag(-m1 + m2))[..., 1:]
    m_r_f *= 0.5

    # Block (r_freqs, resonances).
    m1 = _mult(resonances, r_freqs)

    m_r_fr = np.zeros((n_r, dim_c))
    m_r_fr[:, :dof] = np.imag(m1)
    m_r_fr[:, dof:] = trig_fft(np.real(m1))[..., 1:]

    # Block (c_freqs, c_freqs).
    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = _sum_exp_weighted(r, fs, ns)
        return r

    m1 = _mult(c_freqs, c_freqs)
    m2 = _mult(c_freqs, np.conj(c_freqs))

    m_f_f = np.zeros((2*n_c, 2*n_c))
    m_f_f[:n_c, :n_c] = np.real(m1 + m2)
    m_f_f[n_c:, :n_c] = -np.imag(m1 - m2)
    m_f_f[:n_c, n_c:] = m_f_f[n_c:, :n_c].T
    m_f_f[n_c:, n_c:] = np.real(-m1 + m2)
    m_f_f *= 0.5

    # Block (r_freqs, c_freqs).
    m1 = _mult(c_freqs, r_freqs)

    m_fr_f = np.zeros((n_r, 2*n_c))
    m_fr_f[:, :n_c] = np.real(m1)
    m_fr_f[:, n_c:] = -np.imag(m1)

    # Block (r_freqs, r_freqs).
    m1 = _mult(r_freqs, r_freqs)

    m_fr_fr = m1

    return np.real(np.block(
        [[m_r_r, m_r_f.T, m_r_fr.T],
         [m_r_f, m_f_f, m_fr_f.T],
         [m_r_fr, m_fr_f, m_fr_fr]]
    ))


def reshape_injection(x, n_out:int, n_in:int):
    '''
    2darray (n_in*(n_in+1)//2 + (n_out-n_in)*n_in, dof) to 3darray (n_out, n_in, dof).

    The returned slice [:n_in, :n_in] is symmetric.
    '''
    assert n_out >= n_in, 'n_out must be greater than or equal to n_in.'
    assert x.ndim == 2, 'Expected a 2D-array.'
    assert x.shape[0] == n_in*(n_in+1)//2 + (n_out-n_in)*n_in, 'Unexpected array shape.'

    L = x.shape[-1]
    x_ = np.zeros((n_out, n_in, L), dtype=x.dtype)
    x_[np.arange(n_in), np.arange(n_in)] = x[:n_in]
    c = n_in
    for i in range(1, n_in):
        cn = c + n_in - i
        x_[np.arange(n_in-i), i+np.arange(n_in-i)] = x[c:cn]/np.sqrt(2)
        x_[i+np.arange(n_in-i), np.arange(n_in-i)] = x[c:cn]/np.sqrt(2)
        c = cn
    if n_out > n_in:
        x_[n_in:, :] = x[c: c+(n_out-n_in)*n_in].reshape(n_out-n_in, n_in, -1)
    return x_

def reshape_projection(x):
    '''
    3darray (n_out, n_in, dof) to 2darray (flatten*, dof).

    When the array is flattened, the symmetry of x[:n_in, :n_in] is taken into account.
    '''
    assert x.ndim == 3, 'Expected a 3D-array.'
    n_out, n_in, L = x.shape
    assert n_out >= n_in, 'n_out must be greater than or equal to n_in.'

    x_ = np.zeros((n_in*(n_in+1)//2 + (n_out-n_in)*n_in, L), dtype=x.dtype)
    x_[:n_in] = x[np.arange(n_in), np.arange(n_in)]
    c = n_in
    for i in range(1, n_in):
        cn = c + n_in - i
        x_[c:cn] = x[np.arange(n_in-i), i+np.arange(n_in-i)]
        x_[c:cn] += x[i+np.arange(n_in-i), np.arange(n_in-i)]
        x_[c:cn] /= np.sqrt(2)
        c = cn
    if n_out > n_in:
        x_[c: c + (n_out - n_in)*n_in] = x[n_in:, :].reshape((n_out - n_in)*n_in, -1)
    return x_


class Amplitudes(BaseEstimator):

    def __init__(
        self,
        *,
        fs:float=1,
        response:str='a',
        penalty:float=0.,
    ):
        self.fs = fs
        self.response = response
        self.penalty = penalty

    @property
    def all_amps_(self):
        if not hasattr(self, 'tensor_modes_'):
            msg = 'Call fit() before accessing all_amps_.'
            raise ValueError(msg)
        return {
            'resonances': self.tensor_modes_,
            'complex': self.complex_amps_,
            'real': self.real_amps_
        }

    def _matrix(self, freqs, ns):
        resonances, c_freqs, r_freqs = list(freqs.values())
        dof = len(resonances)
        n_c, n_r = len(c_freqs), len(r_freqs)

        m = metric_amps(
            freqs, self.fs, ns, response=self.response)

        if ((n_c == 0) & (n_r == 0)) or (dof == 0):
            m += self.penalty * np.eye(m.shape[0])
            return m
        else:
            dim_c = 2*dof - 1 if dof > 0 else 0
            dim = dim_c + 2*(2*n_c + n_r)
            s = dim_c + 2*n_c + n_r
            m_ = np.zeros((dim, dim), dtype=m.dtype)

            # hide (mask) rows and columns in the region [dim_c:s)
            mask = np.ones(m_.shape, dtype=bool)
            mask[dim_c:s, :] = False
            mask[:, dim_c:s] = False
            np.place(m_, mask, m.flatten())

            m_[:s, :s] += m
            return m, m_

    def _rhs(self, y, freqs):
        fs = self.fs
        n_out, n_in, ns = y.shape
        resonances, c_freqs, r_freqs = list(freqs.values())

        # Resonances.
        def prod(t):
            freqs_ = 2*fs*np.exp(resonances/(2*fs))*np.sinh(resonances/(2*fs))
            if self.response == 'a':
                freqs_ *= resonances
            T = ns/fs
            t = t.reshape(1, -1)
            r_ = np.exp(resonances[:, np.newaxis]*t)*(1-t/T)
            r_ *= freqs_[:, np.newaxis]
            return r_

        dof = len(resonances)
        if dof != 0:
            r1 = np.einsum(
                'ijt,kt->ijk',
                y, prod(np.arange(ns)/fs))
            r1 = np.concatenate(
                [np.imag(r1), trig_fft(np.real(r1))[..., 1:]],
                axis=-1)
        else:
            r1 = np.zeros((n_out, n_in, 0), dtype=y.dtype)

        # Complex frequencies.
        def prod(t, freqs):
            T = ns / fs
            t = t.reshape(1, -1)
            r_ = np.exp(freqs[:, np.newaxis]*t)*(1-t/T)
            return r_

        n_c = len(c_freqs)
        if n_c != 0:
            r2 = np.einsum(
                'ijt,kt->ijk',
                y, prod(np.arange(ns)/fs, c_freqs))
            r2 = np.concatenate(
                [np.real(r2), -np.imag(r2)],
                axis=-1)
        else:
            r2 = np.zeros((n_out, n_in, 0), dtype=y.dtype)

        # Real frequencies.
        n_r = len(r_freqs)
        if n_r != 0:
            r3 = np.einsum(
                'ijt,kt->ijk',
                y, prod(np.arange(ns)/fs, r_freqs))
        else:
            r3 = np.zeros((n_out, n_in, 0), dtype=y.dtype)

        r1, r2, r3 = np.real(r1), np.real(r2), np.real(r3)
        if (n_c == 0) and (n_r == 0):
            # Project to space of 'symmetric' matrices.
            return reshape_projection(r1).T
        elif dof == 0:
            r2 = r2.reshape(-1, 2*n_c)
            r3 = r3.reshape(-1, n_r)
            return np.concatenate([r2, r3], axis=1).T
        else:
            r = np.concatenate([r1, r2, r3], axis=-1)
            dim = (2*dof-1)+2*n_c+n_r

            # Diagonal elements or without symmetric pair.
            rd = r[np.arange(n_in), np.arange(n_in)]
            rd = np.concatenate([rd, r[n_in:].reshape(n_in*(n_out-n_in), dim)])
            del r

            # Off-diagonal pairs.
            dim_a = (2*dof-1)+2*(2*n_c+n_r)
            roff = np.zeros((n_in*(n_in-1)//2, dim_a))
            c = 0
            dim_res = 2*dof-1
            for i in range(1, n_in):
                cn = c+n_in-i
                diag_u = (np.arange(n_in-i), i+np.arange(n_in-i))
                diag_l = (i+np.arange(n_in-i), np.arange(n_in-i))
                roff[c:cn, :dim_res] = r1[diag_u]+r1[diag_l]
                roff[c:cn, dim_res:dim_res+2*n_c] = r2[diag_u]
                roff[c:cn, dim_res+2*n_c:dim] = r3[diag_u]
                roff[c:cn, dim:dim+2*n_c] = r2[diag_l]
                roff[c:cn, dim+2*n_c:] = r3[diag_l]
                c = cn

            return rd.T, roff.T

    def fit(self, y, freqs):
        n_out, n_in, ns = y.shape
        _freqs = {}
        for k in ['resonances', 'complex', 'real']:
            tmp = freqs.get(k)
            _freqs[k] = np.array(tmp) if tmp is not None else np.array([])
        self._freqs = _freqs

        dof = len(self._freqs['resonances'])
        dim_c = 2*dof-1 if dof > 0 else 0
        n_c, n_r = len(self._freqs['complex']), len(self._freqs['real'])

        if (n_c == 0) and (n_r == 0):
            r = scipy.linalg.solve(
                self._matrix(self._freqs, ns),
                self._rhs(y, self._freqs),
                assume_a='pos').T

            r1 = reshape_injection(r[:, :dim_c], n_out, n_in)
            r2, r3 = None, None
        elif dof == 0:
            r = scipy.linalg.solve(
                self._matrix(self._freqs, ns),
                self._rhs(y, self._freqs),
                assume_a='pos').T

            r1 = None
            r2 = r[:, :2*n_c].reshape(n_out, n_in, 2*n_c)
            r3 = r[:, 2*n_c:].reshape(n_out, n_in, n_r)
        else:
            md, moff = self._matrix(self._freqs, ns)
            rhs_d, rhs_off = self._rhs(y, self._freqs)

            # Diagonal elements or without symmetric pair.
            rd = scipy.linalg.solve(
                md, rhs_d,
                assume_a='pos').T
            del md, rhs_d

            r1 = np.zeros((n_out, n_in, dim_c), dtype=float)
            r2 = np.zeros((n_out, n_in, 2*n_c), dtype=float)
            r3 = np.zeros((n_out, n_in, n_r), dtype=float)

            # Diagonal elements.
            dim_t = dim_c+2*n_c+n_r
            slices = [slice(None, dim_c), slice(dim_c, dim_c+2*n_c), slice(dim_c+2*n_c, None)]
            for r_, s in zip([r1, r2, r3], slices):
                r_[n_in:] = rd[n_in:].reshape(n_out-n_in, n_in, dim_t)[..., s]
                r_[np.arange(n_in), np.arange(n_in)] = rd[:n_in, s]
            del rd

            # Off-diagonal pairs.
            roff = scipy.linalg.solve(
                moff, rhs_off,
                assume_a='pos').T
            del moff, rhs_off

            c = 0
            dim_g = dim_c+2*n_c+n_r
            for i in range(1, n_in):
                cn = c+n_in-i
                diag_u = (np.arange(n_in-i), i+np.arange(n_in-i))
                diag_l = (i+np.arange(n_in-i), np.arange(n_in-i))
                # Resonances.
                r1[diag_u] = roff[c:cn, :dim_c]
                r1[diag_l] = roff[c:cn, :dim_c]
                # Other frequencies.
                r2[diag_u] = roff[c:cn, dim_c:dim_c+2*n_c]
                r3[diag_u] = roff[c:cn, dim_c+2*n_c:dim_g]
                r2[diag_l] = roff[c:cn, dim_g:dim_g+2*n_c]
                r3[diag_l] = roff[c:cn, dim_g+2*n_c:]
                c = cn
            del roff

        # Recover amplitudes.
        if r1 is None:
            self.tensor_modes_ = np.array([])
        else:
            r = np.concatenate(
                [np.zeros((*r1.shape[:2], 1)), r1[..., dof:]], axis=-1)
            r = trig_ifft(r).astype(np.complex128)
            self.tensor_modes_ = r1[..., :dof] + 1j*r

        if r2 is None:
            self.complex_amps_ = np.array([])
        else:
            self.complex_amps_ = r2[..., :n_c] + 1j*r2[..., n_c:]

        if r3 is None:
            self.real_amps_ = np.array([])
        else:
            self.real_amps_ = r3

        return self

    def predict(self, X):
        resonances, c_freqs, r_freqs = list(self._freqs.values())
        X = np.atleast_1d(X)
        ns = X.shape[0]
        fs = self.fs
        K = 0

        dof = len(resonances)
        if dof > 0:
            K1 = 2*fs*np.exp(resonances/(2*fs)) * np.sinh(resonances/(2*fs))
            if self.response == 'a':
                K1 *= resonances
            K1 = self.tensor_modes_ * K1.reshape(1, 1, dof)
            K1 = np.expand_dims(K1, axis=2)
            K1 = K1 * np.exp(resonances[np.newaxis, :]*X[:, np.newaxis]).reshape(1, 1, ns, dof)
            K1 = np.imag(np.sum(K1, axis=-1))
            K = K1

        freqs = np.concatenate([c_freqs, r_freqs], dtype=complex)
        amps = np.concatenate([self.complex_amps_, self.real_amps_], axis=-1)
        nf = len(freqs)
        if nf > 0:
            K2 = np.expand_dims(amps, axis=2)
            K2 = K2 * np.exp(freqs[np.newaxis, :]*X[:, np.newaxis]).reshape(1, 1, ns, nf)
            K2 = np.real(np.sum(K2, axis=-1))
            K += K2

        return K
    
    def mech_part(self, X):
        resonances = self._freqs['resonances']
        X = np.atleast_1d(X)
        ns = X.shape[0]
        fs = self.fs

        dof = len(resonances)
        if dof > 0:
            K = 2*fs*np.exp(resonances/(2*fs)) * np.sinh(resonances/(2*fs))
            if self.response == 'a':
                K *= resonances
            K = self.tensor_modes_ * K.reshape(1, 1, dof)
            K = np.expand_dims(K, axis=2)
            K = K * np.exp(resonances[np.newaxis, :]*X[:, np.newaxis]).reshape(1, 1, ns, dof)
            K = np.imag(np.sum(K, axis=-1))
        else:
            K = 0

        return K
    
    def residual(self, X):
        _, c_freqs, r_freqs = list(self._freqs.values())
        X = np.atleast_1d(X)
        ns = X.shape[0]

        freqs = np.concatenate([c_freqs, r_freqs], dtype=complex)
        amps = np.concatenate([self.complex_amps_, self.real_amps_], axis=-1)
        nf = len(freqs)
        if nf > 0:
            K = np.expand_dims(amps, axis=2)
            K = K * np.exp(freqs[np.newaxis, :]*X[:, np.newaxis]).reshape(1, 1, ns, nf)
            K = np.real(np.sum(K, axis=-1))
        else:
            K = 0

        return K

# =================================
# Modal Parameters
# =================================

def reshape_modes_input(x, dof:int, n_out:int):
    '''
    Transform 1darray into (x, z) for use in PartialModesMap.
    '''
    assert len(x) == 2*n_out*dof - n_out*(n_out+1)//2, 'Unexpected length for x.'

    X = x[:dof*n_out].reshape(n_out, dof)
    
    Zu = np.zeros((n_out, n_out))
    init = dof*n_out
    for i in range(n_out-1):
        end = init + n_out - i - 1
        Zu[i, i+1:] = x[init:end]
        Zu[i+1:, i] = -x[init:end]
        init = end
    Zc = x[dof*n_out + n_out*(n_out-1)//2:].reshape(n_out, dof - n_out)
    Z = np.concatenate((Zu, Zc), axis=1)

    return X, Z

def reshape_modes_output(X, Z):
    '''
    Transform (x, z) into 1darray compatible with scipy.optimize.

    If z[:n_out, :n_out] is not anti-symmetric, then the upper triangle is used
    to construct an anti-symmetric matrix, and the lower triangle is ignored.
    '''
    n_out, dof = X.shape
    
    x = np.zeros(
        (2*n_out*dof - n_out*(n_out+1)//2,), dtype=X.dtype)
    x[:dof*n_out] = X.flatten()

    Zu = Z[:n_out, :n_out]
    init = dof*n_out
    for i in range(n_out-1):
        end = init + n_out - i - 1
        x[init:end] = Zu[i, i+1:]
        init = end
    x[dof*n_out + n_out*(n_out-1)//2:] = Z[:n_out, n_out:].flatten()

    return x

def basis_iterator(n_out:int, dof:int):
    e = np.zeros(2*n_out*dof - n_out*(n_out+1)//2)
    e[0] = 1
    for _ in range(len(e)):
        yield reshape_modes_input(e, dof, n_out)
        e = np.roll(e, 1)

def mode_to_amps(modes, n_out, n_in):
    return modes[:n_out, np.newaxis] * modes[np.newaxis, :n_in]

def amps_to_modes(amps):
    n_out, n_in, dof = amps.shape
    psi = np.zeros((n_out, dof), dtype=amps.dtype)
    psi[:n_in] = np.sqrt(amps[np.arange(n_in), np.arange(n_in)])

    dom_idxs = np.argmax(np.abs(psi[:n_in]), axis=0)
    dom = psi[dom_idxs, np.arange(dof)]
    psi = amps[:, dom_idxs, np.arange(dof)]
    psi /= dom[np.newaxis, :]
    return psi


class PartialModesMap:
    r'''
    A parameterization of mode shapes.

    Returns mode shapes with real part X with shape (N, dof), where N is the number of observations.
    The mode shapes are :math:`\Psi = X(I + i Z)`.

    The matrix Z is written in the basis :math:`\langle q_0, \ldots, q_{N-1}, e_{i_1}, \ldots \rangle`,
    where :math:`q_i` is an orthogonal basis for the range of X, and
    :math:`e_{i_j}` are elements of the canonical basis from coords, that is, coords = [i_1, i_2, ...].
    The matrix Z[:N, :N] is antisymmetric (the function uses the upper triangular part).

    Attributes
    ----------
    freqs : 1D-array
        Modal frequencies.
    coords : 1D-array
        Subset of canonical basis to complement the kernel of X.
    '''

    rtol = 1e-5
    atol = 1e-8

    def __init__(self, freqs, coords):
        assert freqs.ndim == 1, 'Expected a 1D-array for frequencies.'
        self.freqs = freqs

        assert coords.ndim == 1, 'Expected a 1D-array for coords.'
        assert len(coords) <= len(freqs), 'Invalid length for coords.'
        self.coords = coords.astype(int)

        self._point = None
        self._vector = None
        self._psi = None
        self._dq = None

    @property
    def point(self):
        return self._point
    
    @point.setter
    def point(self, value):
        x, z = value
        assert x.ndim == 2, 'Expected a 2D-array for x.'
        n_out, dof = x.shape
        assert dof == len(self.freqs), 'x.shape[1] should equal dof.'
        assert n_out == len(self.coords), 'x.shape[0] should equal the number of observations.'
        assert x.shape == z.shape, 'Incompatible shapes for x and z.'
        assert np.allclose(z[:n_out, :n_out], -z[:n_out, :n_out].T), 'The matrix z must be skew-symmetric in the observed block.'
        if (self._point is None) or \
            not (np.allclose(x, self._point[0], rtol=self.rtol, atol=self.atol) & \
             np.allclose(z, self._point[1], rtol=self.rtol, atol=self.atol)):
            self._expensive_fun(x, z)
            self._point = value
            self._dq = None

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, value):
        dx, dz = value
        assert dx.ndim == 2, 'Expected a 2D-array for dx.'
        n_out, dof = dx.shape
        assert dof == len(self.freqs), 'dx.shape[1] should equal dof.'
        assert n_out == len(self.coords), 'dx.shape[0] should equal the number of observations.'
        assert dx.shape == dz.shape, 'Incompatible shapes for dx and dz.'
        if (self._vector is None) or \
            (self._dq is None) or \
            not (np.allclose(dx, self._vector[0], rtol=self.rtol, atol=self.atol) & \
            np.allclose(dz, self._vector[1], rtol=self.rtol, atol=self.atol)):
            self._compute_jac_expensive(dx, dz)
            self._vector = value

    def _inv_qe(self, a):
        '''
        Compute a@[q e]^{-1}, where [q e] = [q_0 ... q_{n_out-1} e_{i_1} ...].
        '''
        dof = len(self.freqs)
        n_out = len(self.coords)
        coords_c = np.setdiff1d(
            np.arange(dof), self.coords, assume_unique=True)
        q = self._grass[0]

        a_ = np.zeros((dof, dof), dtype=a.dtype)
        a_[:, coords_c] = a[:, n_out:]
        a_[:, self.coords] = scipy.linalg.solve(
            q[self.coords].T,
            (a[:, :n_out] - a_[:, coords_c]@q[coords_c, :]).T,
            assume_a='upper triangular').T
        return a_

    def _expensive_fun(self, x, z):
        try:
            q, s, s_inv = derivatives.grass(x.T, self.coords)
        except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            logger.warning('q[coords] is singular.')
            self._psi = np.nan
            return
        self._grass = (q, s, s_inv)

        z_ = q@z
        z_ = self._inv_qe(z_)
        self._z_ = z_

        # Check mass positivity.
        dof = len(self.freqs)
        D = np.real(self.freqs), np.imag(self.freqs)
        m = D[0][:, np.newaxis] * z_.T
        m += m.T
        m[np.arange(dof), np.arange(dof)] += D[1]
        m -= (z_ * D[1][np.newaxis, :]) @ z_.T
        try:
            self._chk = scipy.linalg.cholesky(
                m, lower=False, overwrite_a=True)
        except scipy.linalg.LinAlgError:
            self._chk = np.nan
            logger.warning('Mass matrix is not positive-definite.')
            self._psi = np.nan
            return

        self._psi = x + 1j * x @ z_

    def __call__(self, x, z):
        """
        Evaluate the mode shapes at the given points.

        Parameters
        ----------
        x : 2D-array (N, dof)
            Real part of mode shapes.
        z : 2D-array (N, dof)
            Multiplicative factor of complex part of mode shapes.
        """
        self.point = (x, z)
        return self._psi

    def _jac_z_(self, dz, dq):
        _, z = self.point
        n_out, dof = len(self.coords), len(self.freqs)
        q = self._grass[0]

        # d(q@z@[q e]^{-1})
        # b1 = dq@z@[q e]^{-1} + q@dz@[q e]^{-1}.
        b1 = self._inv_qe(dq@z + q@dz)
        # b2 = [q e]d([q e]^{-1}) = -[dq 0][q e]^{-1}
        b2 = self._inv_qe(-np.pad(dq, ((0, 0), (0, dof-n_out))))

        return b1 + self._z_@b2

    def _compute_jac_expensive(self, dx, dz):
        # TODO: if dx, dz is just rescaled, it is not necessary to recompute dq either.
        if (dx == 0).all():
            self._dq = np.zeros_like(self._grass[0])
            self._ds = np.zeros_like(self._grass[1])
        else:
            self._dq, self._ds = derivatives.jac_grass(dx.T, self.coords, self._grass)

        self._dz_ = self._jac_z_(dz, self._dq)

    def jac(self, x, z):
        self.point = (x, z)
        dof = len(self.freqs)
        def dpsi(dx, dz):
            self.vector = (dx, dz)

            a = 1j * self._z_
            a[range(dof), range(dof)] += 1
            a = dx@a
            return a + 1j * x@self._dz_
        return dpsi

    def _hessp_z_(self, pz, pdq, pd2q):
        n_out, dof = len(self.coords), len(self.freqs)
        _, z = self.point
        _, dz = self.vector
        q = self._grass[0]
        dq = self._dq

        t1 = self._inv_qe(pd2q@z + pdq@dz + dq@pz)

        # a = [q e]d([q e]^{-1}) = -[dq 0][q e]^{-1}
        a = self._inv_qe(-np.pad(dq, ((0, 0), (0, dof-n_out))))
        t2 = self._inv_qe(pdq@z + q@pz)@a

        # pa = [q e]d([q e]^{-1})(p) = -[dq(p) 0][q e]^{-1}
        pa = self._inv_qe(-np.pad(pdq, ((0, 0), (0, dof-n_out))))
        t3 = self._inv_qe(dq@z + q@dz)@pa

        # b1 = a@pa
        b1 = a@pa 
        # b2 = d([q e] d[q e]^{-1}(p)) = -d([pdq 0][q e]^{-1})
        # b2 = -[pd2q 0][q e]^{-1} - [pdq 0]d([q e]^{-1})
        b2 = -self._inv_qe(np.pad(pd2q, ((0, 0), (0, dof-n_out))))
        b2_ = -self._inv_qe(np.pad(pdq, ((0, 0), (0, dof-n_out))))@a
        b2 = b2 + b2_
        # b1 + b2 = [q e]d(d[q e]^{-1}(p))
        t4 = self._inv_qe(q@z)@(b1 + b2)

        return t1 + t2 + t3 + t4

    def hessp(self, x, z, px, pz):
        self.point = (x, z)
        self._compute_jac_expensive(px, pz)
        pdq, pds, pdz_ = self._dq, self._ds, self._dz_

        def d2psi_p(dx, dz):
            self.vector = (dx, dz)
            dz_ = self._dz_
            cross = px@dz_ + dx@pdz_

            pd2q, _ = derivatives.hessp_grass(
                pdq, pds, self._dq, self._ds, self.coords, self._grass)
            pd2z_ = self._hessp_z_(pz, pdq, pd2q)
            return 1j*(cross + x@pd2z_)
        return d2psi_p

    def constraints(self, x, z):
        self.point = (x, z)
        q = self._grass[0]
        c_res = -np.mean(np.log(np.diag(q[self.coords])))
        if isinstance(self._chk, float) and np.isnan(self._chk):
            c_pos = 1e50
        else:
            c_pos = -np.mean(np.log(np.diag(self._chk)))
        return np.array([c_res, c_pos])

    def _dmass(self, dz_):
        D = np.real(self.freqs), np.imag(self.freqs)

        dm = D[0][:, np.newaxis] * dz_.T
        dm += dm.T
        dm_ = (self._z_ * D[1][np.newaxis, :]) @ dz_.T
        dm_ += dm_.T
        return dm - dm_

    def jac_constraints(self, x, z):
        self.point = (x, z)
        q = self._grass[0]

        def d_constr(dx, dz):
            self.vector = (dx, dz)
            dq = self._dq
            jac = np.zeros(2)

            jac[0] = -np.mean(np.diag(dq[self.coords])/np.diag(q[self.coords]))

            if isinstance(self._chk, float):
                jac[1] = 0
            else:
                dchk = self._dmass(self._dz_)
                dchk = derivatives.jac_cho(self._chk, dchk)
                jac[1] = -np.mean(np.diag(dchk)/np.diag(self._chk))
            return jac

        return d_constr

    def _hessp_m(self, pz, pdz_, pdq, pd2q):
        d2z_ = self._hessp_z_(pz, pdq, pd2q)
        d2m = self._dmass(d2z_)
        d2m_ = (pdz_ * np.imag(self.freqs)[np.newaxis, :]) @ self._dz_.T
        d2m_ += d2m_.T
        return d2m - d2m_

    def hessp_constraints(self, x, z, px, pz):
        self.point = (x, z)
        self._compute_jac_expensive(px, pz)
        pdq, pds, pdz_ = self._dq, self._ds, self._dz_
        pdchk = self._dmass(pdz_)
        pdchk = derivatives.jac_cho(self._chk, pdchk)

        def local(a, da, pda, pd2a):
            idx = (np.arange(len(a)), np.arange(len(a)))
            r = pd2a[idx] - pda[idx]*da[idx]/a[idx]
            return r/a[idx]

        def hessp_fun(dx, dz):
            self.vector = (dx, dz)
            q = self._grass[0]
            pd2q, _ = derivatives.hessp_grass(
                pdq, pds, self._dq, self._ds, self.coords, self._grass)
            hessp = np.zeros(2)

            hessp[0] = -np.mean(local(
                q[self.coords], self._dq[self.coords], pdq[self.coords], pd2q[self.coords]))

            if isinstance(self._chk, float):
                hessp[1] = 0
            else:
                dchk = self._dmass(self._dz_)
                dchk = derivatives.jac_cho(self._chk, dchk)
                d2chk = self._hessp_m(pz, pdz_, pdq, pd2q)

                d2chk = derivatives.jac_cho(self._chk, d2chk - pdchk.T@dchk - dchk.T@pdchk)
                hessp[1] = -np.mean(local(self._chk, dchk, pdchk, d2chk))
            return hessp

        return hessp_fun


def _metric_amps_modes(resonances, fs, ns, response='a'):
    '''
    Compute the metric for amplitude coefficients.
    '''
    dof = len(resonances)

    if response not in ['a', 'v']:
        raise ValueError(f'Unknown response type: {response}')

    # Model resonances block.
    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = np.exp(r/(2*fs)) * _sum_exp_weighted(r, fs, ns)
        r *= (4*fs**2)*np.sinh(x[np.newaxis, :]/(2*fs))*np.sinh(y[:, np.newaxis]/(2*fs))
        if response == 'a':
            r *= x[np.newaxis, :]*y[:, np.newaxis]
        return r

    m1 = _mult(resonances, np.conj(resonances))
    m2 = _mult(resonances, resonances)

    m_r_r = np.zeros((2*dof, 2*dof))
    m_r_r[:dof, :dof] = np.real(m1 - m2)
    m_r_r[dof:, :dof] = np.imag(m1 + m2)
    m_r_r[:dof, dof:] = m_r_r[dof:, :dof].T
    m_r_r[dof:, dof:] = np.real(m1 + m2)
    m_r_r *= 0.5

    return np.real(m_r_r)

class RealModes:

    def __init__(
        self,
        resonances,
        amps,
        ns:int,
        response:str='a',
        fs:int=1,
    ):
        assert resonances.ndim == 1, 'Expected a 1D-array for frequencies.'
        assert amps.ndim == 3, 'Expected a 3D-array for amplitudes.'
        n_out, n_in, dof = amps.shape
        assert n_out >= n_in, 'n_out must be greater than or equal to n_in.'
        assert dof == len(resonances), 'The last dimension of amplitudes must match the number of frequencies.'
        assert dof >= n_out, 'dof must be greater than or equal to n_out.'

        self.resonances = resonances
        self.amps = amps
        self.fs = fs
        self.ns = ns
        self.response = response

        self._rescale = np.max(np.abs(amps))
        self.modes_fit_ = None
        self.success_ = None
        self.message_ = None

        self._get_metric()

    def _get_metric(self):
        self._metric = _metric_amps_modes(
            self.resonances, self.fs, self.ns, response=self.response)

    def _fun(self, x):
        n_out, n_in, dof = self.amps.shape
        x = x.reshape(n_out, dof)
        amps = mode_to_amps(x, n_out, n_in)

        dif = amps - self.amps/self._rescale
        dif = np.concatenate(
            [np.real(dif), np.imag(dif)], axis=-1)
        trans = np.einsum('ijk,kl->ijl', dif, self._metric)

        # Compute f(x).
        f = np.einsum('ijk,ijk', trans, dif)
        return f

    def _jac(self, x):
        n_out, n_in, dof = self.amps.shape
        x = x.reshape(n_out, dof)
        amps = mode_to_amps(x, n_out, n_in)

        dif = amps - self.amps/self._rescale
        dif = np.concatenate(
            [np.real(dif), np.imag(dif)], axis=-1)
        trans = np.einsum('ijk,kl->ijl', dif, self._metric)

        # Compute df(x).
        trans = trans[..., :dof]
        t1 = np.array(
            [np.sum(trans[q, :n_in] * x[:n_in], axis=0)
             for q in range(n_out)])
        t2 = np.array(
            [np.sum(trans[n_in:n_out, q] * x[n_in:n_out], axis=0)
             for q in range(n_in)])
        df = 2*np.concatenate(
            [2*t1[:n_in] + t2, t1[n_in:]], axis=0)
        return df.flatten()

    def _hessp(self, x, p):
        n_out, n_in, dof = self.amps.shape
        x = x.reshape(n_out, dof)
        p = p.reshape(n_out, dof)

        amps = mode_to_amps(x, n_out, n_in)
        A = amps - self.amps/self._rescale
        A = np.concatenate(
            [np.real(A), np.imag(A)], axis=-1)

        B = p[:, np.newaxis]*x[np.newaxis, :n_in]
        B += x[:, np.newaxis]*p[np.newaxis, :n_in]

        Ap = np.einsum('ijk,kl->ijl', A, self._metric[:, :dof])
        t1 = np.array(
            [np.sum(Ap[q, :n_in] * p[:n_in], axis=0)
             for q in range(n_out)])
        t2 = np.array(
            [np.sum(Ap[n_in:n_out, q] * p[n_in:n_out], axis=0)
             for q in range(n_in)])
        Ap = 2*np.concatenate(
            [2*t1[:n_in] + t2, t1[n_in:]], axis=0)

        Bx = np.einsum('ijk,kl->ijl', B, self._metric[:dof, :dof])
        t1 = np.array(
            [np.sum(Bx[q, :n_in] * x[:n_in], axis=0)
             for q in range(n_out)])
        t2 = np.array(
            [np.sum(Bx[n_in:n_out, q] * x[n_in:n_out], axis=0)
             for q in range(n_in)])
        Bx = 2*np.concatenate(
            [2*t1[:n_in] + t2, t1[n_in:]], axis=0)
        return (Ap + Bx).flatten()

    def fit(self, options_ncg:dict=None):
        n_out, _, dof = self.amps.shape
        options_ncg = {} if options_ncg is None else options_ncg
        if 'gtol' not in options_ncg.keys():
            options_ncg['gtol'] = 1e-2

        x0 = np.real(amps_to_modes(self.amps/self._rescale))
        idx = np.nonzero(x0[0] < 0)[0]
        x0[:, idx] = -x0[:, idx]
        x0 = x0.flatten()
        lb = -2*np.ones_like(x0)
        lb[:dof] = 0
        ub = 2*np.ones_like(x0)
        bounds = scipy.optimize.Bounds(
            lb=lb, ub=ub)
        res = scipy.optimize.dual_annealing(
            self._fun,
            x0=x0,
            bounds=bounds,
            minimizer_kwargs={
                'method': 'trust-ncg',
                'jac': self._jac,
                'hessp': self._hessp,
                'options': options_ncg},
            callback=None)
        self.modes_fit_ = np.sqrt(self._rescale) * res.x.reshape(n_out, dof)
        self.success_ = res.success
        self.message_ = res.message

        return self

    def predict(self, X, response:str=None):
        X = np.atleast_1d(X)
        ns = X.shape[0]
        dof = len(self.resonances)
        response = self.response if response is None else response
        K = np.zeros((ns, self.amps.shape[0]), dtype=float)

        if self.modes_fit_ is None:
            msg = 'Call fit() before accessing modes_fit_.'
            raise ValueError(msg)

        n_out, n_in = self.amps.shape[:2]
        amps_fit = mode_to_amps(self.modes_fit_, n_out, n_in)
        K = 2*self.fs*np.exp(self.resonances/(2*self.fs)) * np.sinh(self.resonances/(2*self.fs))
        if response == 'a':
            K *= self.resonances
        K = amps_fit * K.reshape(1, dof)
        K = np.expand_dims(K, axis=2)
        K = K * np.exp(self.resonances[np.newaxis, :]*X[:, np.newaxis]).reshape(1, 1, ns, dof)
        K = np.imag(np.sum(K, axis=-1))

        return K



class ConstraintModifier:

    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale
        self._base = scipy.interpolate.BSpline.basis_element([0, 0.5, 1])

    def __call__(self, x):
        def f(x):
            y = np.array(x)
            y[y < 0] = 0
            r1 = 2*self._base.antiderivative(nu=2)(y[y < 1]) + 0.5
            r2 = y[y >=1]
            return np.concatenate((r1, r2))

        return self.scale*f((x - self.shift)/self.scale) + self.shift

    def derivative(self, nu=0):
        if nu == 0:
            return self
        elif nu == 1:
            def f(x):
                y = np.array(x)
                y[y < 0] = 0
                y[y > 1] = 1
                return 2*self._base.antiderivative(nu=1)(y)
            def f_(x):
                return f((x - self.shift)/self.scale)
            return f_
        elif nu == 2:
            def f(x):
                y = np.array(x)
                y[y < 0] = 0
                y[y > 1] = 1
                return 2*self._base(y)
            def f_(x):
                return f((x - self.shift)/self.scale)/self.scale
            return f_

class ScalarComposition:

    def __init__(self, scalar, f, df, d2f, dim):
        self.scalar = scalar
        self.f = f
        self.df = df
        self.d2f = d2f
        self.dim = dim

    def __call__(self, x):
        return self.scalar(self.f(x))

    def jac(self, x):
        a = self.scalar.derivative(nu=1)(self.f(x))
        if np.all(a == 0):
            return np.zeros(self.dim)
        else:
            return a[:, np.newaxis] * self.df(x)

    def hessp(self, x, p):
        a = self.scalar.derivative(nu=1)(self.f(x))
        if np.all(a == 0):
            r = np.zeros(self.dim)
        else:
            r = a[:, np.newaxis] * self.d2f(x, p)

        a = self.scalar.derivative(nu=2)(self.f(x))
        if np.all(a == 0):
            return r
        else:
            df = self.df(x)
            r += (a * (df @ p))[:, np.newaxis] * df
            return r

class _Hess_constraint(scipy.sparse.linalg.
LinearOperator):
    
    def __init__(self, dim, hess):
        super().__init__(dtype=np.float64, shape=(dim, dim))
        self.hess = hess

    def _matvec(self, p):
        dim = self.shape[0]
        if p.shape == (dim, 1):
            p = p.flatten()
        return self.hess(p)

    def _adjoint(self):
        return self

class ComplexModes:

    def __init__(
        self,
        resonances,
        coords,
        amps,
        fs:int,
        ns:int,
        response:str='a',
    ):
        self.resonances = resonances
        assert resonances.ndim == 1, 'Expected 1D array for frequencies.'

        self.amps = amps
        assert amps.ndim == 3, 'Expected 3D array for amplitudes.'
        assert amps.shape[2] == len(resonances), 'Incompatible shapes for frequencies and amplitudes.'

        self.fs = fs
        self.ns = ns
        self.response = response
        PartialModesMap.atol = 1e-10
        PartialModesMap.rtol = 1e-8
        self._modes_map = PartialModesMap(resonances, coords)

        self._rescale = np.max(np.abs(amps))
        self._get_metric()

        self._raw_modes_fit = None
        self.success_ = None
        self.message_ = None

    @property
    def modes_fit_(self):
        if self._raw_modes_fit is None:
            msg = 'Call fit() before accessing modes_fit_.'
            raise ValueError(msg)
        return np.sqrt(self._rescale) * self._modes_map(*self._raw_modes_fit)

    def _get_metric(self):
        self._metric = _metric_amps_modes(
            self.resonances, self.fs, self.ns, response=self.response)

    def _fun(self, x):
        n_out, n_in, dof = self.amps.shape
        x_, z_ = reshape_modes_input(x, dof, n_out)

        modes = self._modes_map(x_, z_)
        if isinstance(modes, float):
            return np.inf
        amps = mode_to_amps(modes, n_out, n_in)
        diff = amps - self.amps/self._rescale
        diff = np.concatenate(
                [np.real(diff), np.imag(diff)], axis=-1)
        dist = np.einsum('ijk,kl,ijl->', diff, self._metric, diff)
        return dist

    def _jac_amps(self, p, dp):
        n_out, n_in, _ = self.amps.shape
        r = p[:n_out, np.newaxis] * dp[np.newaxis, :n_in]
        r += dp[:n_out, np.newaxis] * p[np.newaxis, :n_in]
        r = np.concatenate(
            [np.real(r), np.imag(r)], axis=-1)
        return r

    def _jac(self, x):
        n_out, n_in, dof = self.amps.shape
        x_, z_ = reshape_modes_input(x, dof, n_out)

        modes = self._modes_map(x_, z_)
        d_modes = self._modes_map.jac(x_, z_)

        amps = mode_to_amps(modes, n_out, n_in)
        diff = np.concatenate(
            [np.real(amps - self.amps/self._rescale), np.imag(amps - self.amps/self._rescale)], axis=-1)
        del amps
        diff = 2*np.einsum('ijk,kl->ijl', diff, self._metric)

        jac = [np.einsum('ijl,ijl', diff, self._jac_amps(modes, d_modes(dx, dz)))
               for dx, dz in basis_iterator(n_out, dof)]
        return np.array(jac)

    def _hessp_amp(self, modes, pd_modes, d_modes, d2_modes):
        n_in = self.amps.shape[1]

        d2_amps = d2_modes[:, np.newaxis]*modes[np.newaxis, :n_in]
        d2_amps += pd_modes[:, np.newaxis]*d_modes[np.newaxis, :n_in]
        d2_amps += d_modes[:, np.newaxis]*pd_modes[np.newaxis, :n_in]
        d2_amps += modes[:, np.newaxis]*d2_modes[np.newaxis, :n_in]

        d2_amps = np.concatenate(
            [np.real(d2_amps), np.imag(d2_amps)], axis=-1)
        return d2_amps

    def _hessp(self, x, p):
        n_out, n_in, dof = self.amps.shape
        x_, z_ = reshape_modes_input(x, dof, n_out)
        px, pz = reshape_modes_input(p, dof, n_out)

        modes = self._modes_map(x_, z_)
        amps = mode_to_amps(modes, n_out, n_in)
        diff = np.concatenate(
            [np.real(amps - self.amps/self._rescale), np.imag(amps - self.amps/self._rescale)], axis=-1)

        jac_modes = self._modes_map.jac(x_, z_)
        hessp_modes = self._modes_map.hessp(x_, z_, px, pz)
        pd_amps = self._jac_amps(modes, jac_modes(px, pz))

        def hess_f(dx, dz):
            d2_amps = self._hessp_amp(
                modes, jac_modes(px, pz), jac_modes(dx, dz), hessp_modes(dx, dz))
            d_amps = self._jac_amps(modes, jac_modes(dx, dz))

            d2_dist = 2*np.einsum(
                'ijk,kl,ijl->', d2_amps, self._metric, diff)
            d2_dist += 2*np.einsum(
                'ijk,kl,ijl->', pd_amps, self._metric, d_amps)
            return d2_dist

        hessp = [hess_f(dx, dz) for dx, dz in basis_iterator(n_out, dof)]
        return np.array(hessp)

    def _get_constraints(self):
        n_out, _, dof = self.amps.shape

        def constr_fun(x):
            x_, z_ = reshape_modes_input(x, dof, n_out)
            return self._modes_map.constraints(x_, z_)

        def constr_jac(x):
            x_, z_ = reshape_modes_input(x, dof, n_out)
            jac = self._modes_map.jac_constraints(x_, z_)
            jac = [jac(dx, dz) for dx, dz in basis_iterator(n_out, dof)]
            return np.array(jac).T

        def constr_hessp(x, p):
            x_, z_ = reshape_modes_input(x, dof, n_out)
            px, pz = reshape_modes_input(p, dof, n_out)
            hessp = self._modes_map.hessp_constraints(x_, z_, px, pz)
            hessp = [hessp(dx, dz) for dx, dz in basis_iterator(n_out, dof)]
            return np.array(hessp).T

        dim = 2*n_out*dof - n_out*(n_out+1)//2
        shift = self._ref_constr + np.array([1, 1])
        scalar_f = ConstraintModifier(shift, 0.1)
        constraints = ScalarComposition(
            scalar_f, constr_fun, constr_jac, constr_hessp, (2, dim))

        def constr_hess(x, v):
            def hess(p):
                return v @ constraints.hessp(x, p)
            return _Hess_constraint(dim, hess)

        r = scipy.optimize.NonlinearConstraint(
            constraints,
            lb=np.array([-np.inf, -np.inf]),
            ub=np.array([5, 5]),
            jac=constraints.jac,
            hess=constr_hess,
            keep_feasible=True
        )
        return r

    def fit(self, x0, options:dict=None, maxiter:int=1e3):
        options = {} if options is None else options
        if 'gtol' not in options.keys():
            options['gtol'] = 1e-2
        if 'xtol' not in options.keys():
            options['xtol'] = 1e-5

        x0 = tuple(e/np.sqrt(self._rescale) for e in x0)

        self._ref_constr = self._modes_map.constraints(*x0)

        def callback(intermediate_result:scipy.optimize.OptimizeResult):
            if intermediate_result.nit >= maxiter:
                raise StopIteration

        res = scipy.optimize.minimize(
            self._fun,
            x0=reshape_modes_output(*x0),
            method='trust-constr',
            jac=self._jac,
            hessp=self._hessp,
            constraints=self._get_constraints(),
            options=options,
            callback=callback
        )
        n_out, _, dof = self.amps.shape
        self._raw_modes_fit = reshape_modes_input(res.x, dof, n_out)
        self.optRes_ = res

        return self

    def predict(self, X, response:str=None):
        X = np.atleast_1d(X)
        ns = X.shape[0]
        dof = len(self.resonances)
        response = self.response if response is None else response
        K = np.zeros((ns, self.amps.shape[0]), dtype=float)

        if self.modes_fit_ is None:
            msg = 'Call fit() before accessing modes_fit_.'
            raise ValueError(msg)

        n_out, n_in = self.amps.shape[:2]
        amps_fit = mode_to_amps(self.modes_fit_, n_out, n_in)
        K = 2*self.fs*np.exp(self.resonances/(2*self.fs)) * np.sinh(self.resonances/(2*self.fs))
        if response == 'a':
            K *= self.resonances
        K = amps_fit * K.reshape(1, dof)
        K = np.expand_dims(K, axis=2)
        K = K * np.exp(self.resonances[np.newaxis, :]*X[:, np.newaxis]).reshape(1, 1, ns, dof)
        K = np.imag(np.sum(K, axis=-1))

        return K


def modal_to_system(mode_shapes, Z):
    '''Recover system matrices from mode shapes and complex frequencies.

    Parameters
    ----------
    mode_shapes : 2D-array
        Square matrix with mode shapes.
    Z : 1D-array
        Complex frequencies.

    Returns
    -------
    M, C, K : 2D-arrays
    '''
    ms = np.asarray(mode_shapes, dtype=np.complex128)
    Z = np.asarray(Z, dtype=np.complex128)
    if ms.shape[0] != ms.shape[1]:
        msg = 'Expected a square matrix for mode shapes.'
        raise ValueError(msg)
    if ms.shape[0] != Z.shape[0]:
        msg = 'Incompatible shapes for mode shapes and complex frequencies.'
        raise ValueError(msg)

    M = ((ms * (Z / Z.imag)[np.newaxis, :]) @ ms.T).imag
    M = np.linalg.inv(M)

    K = ((ms * (-1 / (Z * Z.imag))[np.newaxis, :]) @ ms.T).imag
    K = np.linalg.inv(K)

    C = ((ms * (Z**2 / Z.imag)[np.newaxis, :]) @ ms.T).imag
    C = -M @ C @ M

    return M, C, K

def system_to_modal(M, C, K):
    '''Recover mode shapes and complex frequencies from system matrices.
    
    Parameters
    ----------
    M, C, K : 2D-arrays
    
    Returns
    -------
    mode_shapes : 2D-array
    Z : 1D-array
    '''
    # Check parameters.
    N = M.shape[0]
    if N != M.shape[1]:
        msg = 'M must be a square matrix.'
        raise ValueError(msg)
    if (M.shape != C.shape) or (M.shape != K.shape):
        msg = 'Incompatible shapes for M, C, and K.'
        raise ValueError(msg)

    # Construct matrix for QEP.
    a = np.zeros((2 * N, 2 * N))
    b = np.zeros((2 * N, 2 * N))
    a[:N, N:] = K
    a[N:, :N] = -K
    a[N:, N:] = -C
    b[:N, :N] = K
    b[N:, N:] = M
    eigvals, eigvecs = scipy.linalg.eig(
        a, b,
        overwrite_a=True, overwrite_b=True)
    Z = eigvals[::2]
    mode_shapes = eigvecs[:N, ::2]
    
    # Order by size of complex frequency.
    idxs = np.argsort(np.abs(Z))
    mode_shapes = mode_shapes[:, idxs]
    Z = Z[idxs]

    # Mass normalization.
    # TODO: optimize.
    d =  np.diag(Z) @ mode_shapes.T @ M @ mode_shapes * np.diag(Z)
    d = d - mode_shapes.T @ K @ mode_shapes
    mu = 2j * Z * np.imag(Z) / np.diag(d)
    mu = np.sqrt(mu)
    mode_shapes = mode_shapes @ np.diag(mu)
    
    return mode_shapes, Z

# =================================
# Systems
# =================================

def randomSystem(
    masses, dampings, resonances,
    damping_type='prop', seed=None):
    '''Generate a random linear vibrating system.

    Parameters
    ----------
    dofs : int
        Number of degrees of freedom.
    masses : tuple, optional
        Range for mass values.
    dampings : tuple, optional
        Range for damping ratios.
    resonances : tuple, optional
        Range for natural frequencies.
    damping_type : str, optional
        Type of damping ('prop' for proportional, 'nop' for non-proportional).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mechanical : dict
        Dictionary containing mass, damping, and stiffness matrices.
    modal : dict
        Dictionary containing mode shapes (mass normalized) and resonances.
    '''
    m = np.asarray(masses)
    zeta = np.asarray(dampings)
    freqs = np.asarray(resonances)
    dofs = len(freqs)
    # Raise an exception if the lengths are different.
    if (len(m) != dofs) or (len(zeta) != dofs):
        msg = 'Incompatible lengths for masses, dampings, and resonances.'
        raise ValueError(msg)

    # Generate rotation matrices.
    if dofs > 1:
        U = scipy.stats.ortho_group(dim=dofs, seed=seed)
        Um = U.rvs()
        Uk = U.rvs()
        if damping_type == 'prop':
            Uc = Uk
        elif damping_type == 'nop':
            Uc = U.rvs()
        else:
            msg = 'Invalid damping type. Choose between `prop` and `nop`.'
            raise ValueError(msg)
    else:
        Um = np.array([[1]])
        Uk, Uc = Um, Um
    
    modes_ = Um @ np.diag(1 / np.sqrt(m))
    invModes_ = np.diag(np.sqrt(m)) @ Um.T
    M = Um @ np.diag(m) @ Um.T
    C = invModes_.T @ Uc @ np.diag(2*zeta*freqs) @ Uc.T @ invModes_
    K = invModes_.T @ Uk @ np.diag(freqs**2) @ Uk.T @ invModes_
    normal_modes = modes_ @ Uk

    if damping_type == 'prop':
        mode_shapes = normal_modes
        Z = freqs*(-zeta + 1j*np.sqrt(1-zeta**2))
    else:
        A = normal_modes.T @ C @ normal_modes
        B = np.diag(freqs**2)
        mode_shapes, Z = system_to_modal(np.eye(dofs), A, B)
        mode_shapes = normal_modes @ mode_shapes

    # Sort by size of complex frequency.
    idxs = np.argsort(np.abs(Z))
    mode_shapes = mode_shapes[:, idxs]
    Z = Z[idxs]

    mechanical = {
        'mass': M,
        'damping': C,
        'stiffness': K
    }
    modal = {
        'frequencies': Z,
        'mode_shapes': mode_shapes
    }
    return mechanical, modal


class Spring:

    def __init__(self, M, C, K, force=None):
        '''Simulate a linear vibrating system.

        Set up an object representing a linear system
        to be used in JiTCODE.
        '''
        self.M, self.C, self.K = _validate_dims(M, C, K)
        self.force = np.zeros(self.M.shape[0]) if force is None else np.array(force, ndmin=1)
        if len(self.force) != self.M.shape[0]:
            msg = f'Expected a force with {self.M.shape[0]}-DoF,\
                got {len(self.force)} instead.'
            raise ValueError(msg)

        Minv = np.linalg.inv(self.M)
        self._A = Minv @ C
        self._B = Minv @ K
        self._force = Minv @ self.force

    def __iter__(self):
        y = jitcode.y
        dofs = self.M.shape[0]
        q = [y(i) for i in range(dofs)]
        p = [y(i + dofs) for i in range(dofs)]
        for i in range(dofs):
            yield p[i]
        for i in range(dofs):
            tmp = -sum(self._B[i, j] * q[j] for j in range(dofs))
            tmp += -sum(self._A[i, j] * p[j] for j in range(dofs))
            tmp += self._force[i]
            yield tmp

    def __len__(self):
        return 2 * self.M.shape[0]
