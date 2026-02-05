#!/usr/bin/env python

import logging
import bisect
import attrs

import numpy as np
import scipy
import pandas as pd
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from . import vectfit3 as vf

logger = logging.getLogger(__name__)


# ========================
# Rational Approximation
# ========================

def rational_function(poles, residues, z):
    '''
    Parameters
    ----------
    poles : (M,) array_like
    residues : (M, L) array_like
    z : scalar or (N, ) array_like

    Return
    ------
    out : (N, L) complex ndarray
        The values of the rational function at the points z.
        If z is scalar, N = 1.
    '''
    poles = np.asarray(poles)
    if poles.ndim != 1:
        msg = f'Expected a 1D array for poles, got an array of dimension {poles.ndim}.'
        raise ValueError(msg)

    z = np.asarray(z)
    if z.ndim == 0:
        z = np.expand_dims(z, axis=0)
    elif z.ndim > 1:
        msg = f'Expected a 1D array for z, got an array of dimension {z.ndim}.'
        raise ValueError(msg)

    residues = np.asarray(residues)
    if residues.ndim == 1:
        residues = np.expand_dims(residues, axis=1)
    elif residues.ndim > 2:
        msg = f'Expected a 1D or 2D array for residues, got an array of dimension {residues.ndim}.'
        raise ValueError(msg)
    
    if poles.shape[0] != residues.shape[0]:
        msg = f'The number of poles and residues must be the same, got {poles.shape[0]} and {residues.shape[0]} instead.'
        raise ValueError(msg)
    
    return (1 / (z[:, np.newaxis] - poles[np.newaxis, :])) @ residues

def pos_imag(x):
    x = np.asarray(x, dtype=complex)
    x[np.imag(x) < 0] = np.conj(x[np.imag(x) < 0])
    return x

@attrs.define
class Poles:
    real = attrs.field(converter=lambda x: np.asarray(x, dtype=float), default=np.array([]))
    imag = attrs.field(converter=pos_imag, default=np.array([]))

def _get_max_order(shape):
    ns, rank = shape
    # Counting complex paramaters in a complex time series
    # and in a rational function, we have that:
    max_order = rank*ns//(rank+1)

    return max_order

def _update_sets(gS, gG, idx):
    gS['index'].append(gG['index'][idx])
    gS['data'].append(gG['data'][idx])
    gG['index'].pop(idx)
    gG['data'].pop(idx)


class AAA(BaseEstimator):

    def __init__(
            self,
            *,
            order:int=None,
            tol:float=0.,
            compute_residues:bool=True):
        self.order = order
        self.tol = tol
        self.compute_residues = compute_residues

    @staticmethod
    def count_freqs(idxs, ns, parity):
        idxs = np.array(idxs)
        idxs = np.sort(idxs)
        n_freqs = 0
        if idxs[0] == 0:
            n_freqs += 1
        if parity == 0 and (idxs[-1] == ns-1):
            n_freqs += 1
        n_freqs = 2*len(idxs) - n_freqs
        return n_freqs

    def set_seed_freqs(self, y, parity, seed_freqs=None):
        '''Set initial frequencies for the algorithm.

        Parameters
        ----------
        freqs : 1darray
            They must be ordered in decreasing order of importance,
            and the indices must be unique.
        '''
        N, rank = y.shape
        if seed_freqs is None:
            # Choose rank + 1 peaks as initial frequencies.
            abs_v = np.linalg.norm(y, axis=1)
            abs_v = np.concatenate((abs_v, abs_v[-2+parity:0:-1]))
            abs_v = np.concatenate((abs_v, abs_v))
            peaks, h = scipy.signal.find_peaks(
                abs_v,
                height=np.max(abs_v)/5,
                distance=np.max((N/(4*(rank+1)), 2))
            )
            idxs = np.argsort(h['peak_heights'])[::-1]
            peaks = peaks[idxs]%(len(abs_v)//2)
            _, idxs = np.unique(peaks, return_index=True)
            peaks = peaks[np.sort(idxs)]
            freqs = peaks[peaks < N]
        else:
            freqs = np.asarray(seed_freqs)

        if freqs.ndim != 1:
            raise ValueError('freqs must be a 1D array.')
        if not (np.all(freqs >= 0) and np.all(freqs < N)):
            raise ValueError('freqs must be in the range [0, N).')
        if len(freqs) != len(np.unique(freqs)):
            raise ValueError('freqs must be unique.')

        freqs = list(freqs)
        freqs.reverse()
        freqs_ = []
        c = 0
        while (c < rank+1) and (len(freqs) > 0):
            p = freqs.pop()
            freqs_.append(p)
            if (p == 0) or (parity == 0 and p == N-1):
                c += 1
            else:
                c += 2
        # Generate additional random indices if c < rank + 1.
        if c < rank + 1:
            logger.warning(
                'Not enough freqs found. Adding random indices.', stacklevel=2)
            diff = np.setdiff1d(np.arange(N), freqs_, assume_unique=True)
            rng = np.random.default_rng()
            rng.shuffle(diff)
            diff = list(diff)
            while c < rank+1:
                p = diff.pop()
                freqs_.append(p)
                if (p == 0) or (parity == 0 and p == N-1):
                    c += 1
                else:
                    c += 2

        return np.array(freqs_)

    def _initialize_sets(self, y, freqs):
        N = y.shape[0]
        gG = {'index': list(range(N)), 'data': list(y)}
        gS = {}

        gS['index'] = [gG['index'][f] for f in freqs]
        gS['data'] = [gG['data'][f] for f in freqs]
        for f in freqs:
            idx = bisect.bisect_left(gG['index'], f)
            gG['index'].pop(idx)
            gG['data'].pop(idx)
        return gS, gG

    def _get_weights(self, gS, gG, parity):
        S_, gS_ = np.array(gS['index']), np.array(gS['data'])
        G_, gG_ = np.array(gG['index']), np.array(gG['data'])
        idxs = np.argsort(S_)
        S_, gS_ = S_[idxs], gS_[idxs]
        n_S, rank = gS_.shape
        n_G = gG_.shape[0]

        N = n_S + n_G
        N_ = 2*(N-1) + parity
        ωN = np.exp(-2j * np.pi / N_)

        # Locate zero and N-1 in S_ and G_.
        e = []
        innS, innG = [0, len(S_)], [0, len(G_)]
        if S_[0] == 0:
            e.append(0)
            innS[0] = 1
        else:
            innG[0] = 1
        if parity == 1:
            pass
        elif S_[-1] == N-1:
            e.append(n_S-1)
            innS[1] = n_S - 1
        else:
            innG[1] = n_G - 1
        innS, innG = np.s_[innS[0]:innS[1]], np.s_[innG[0]:innG[1]]
        e = np.array(e, dtype=np.int64)
        M = 2*len(S_) - len(e)

        # Construct Loewner matrix L.
        Cr = 1/(ωN**(-S_[e])[:, np.newaxis]-ωN**(-G_)[np.newaxis, :])
        Lr = (gS_[e][:, np.newaxis] - gG_[np.newaxis, :]) * Cr[..., np.newaxis]
        C1 = 1/(ωN**(-S_[innS])[:, np.newaxis]-ωN**(-G_)[np.newaxis, :])
        L1 = (gS_[innS][:, np.newaxis] - gG_[np.newaxis, :]) * C1[..., np.newaxis]
        C2 = 1/(ωN**(S_[innS])[:, np.newaxis]-ωN**(-G_)[np.newaxis, :])
        L2 = (np.conj(gS_[innS])[:, np.newaxis] - gG_[np.newaxis, :]) * C2[..., np.newaxis]
        L = np.concatenate((Lr, L1+L2, 1j*(L1-L2)), axis=0)
        del Lr, L1, L2, G_, gG_

        # Separate real and imaginary parts.
        L = np.concatenate(
            (np.real(L), np.imag(L[:, innG])), axis=1, dtype=float
        )
        L = L.reshape(M, -1).T

        # Construct inclusion matrix into the space that satisfies (3.14) of [From ESPRIT to ESPIRA].
        tmpS = np.concatenate(
            (np.real(gS_[e]), 2*np.real(gS_[innS]), -2*np.imag(gS_[innS])),
            axis=0, dtype=float
        )
        In = scipy.linalg.qr(tmpS, pivoting=True)[0]
        In = In[:, rank:]

        # Compute weights to find best rational approximation.
        L = L @ In
        eigval, w = scipy.linalg.svd(
            L,
            overwrite_a=True,
            full_matrices=False)[1:]
        logger.debug(f'Lowest eigenvalue: {eigval[-1]}')
        w = In @ w[-1]
        w_ = np.zeros_like(S_, dtype=complex)
        w_[innS] = w[len(e):len(S_)] + 1j*w[len(S_):]
        w_[e] = w[:len(e)]
        w = w_

        # Compute rational function at G frequencies.
        rp = w[:, np.newaxis] * gS_
        p = Cr.T@rp[e] + C1.T@rp[innS] + C2.T@np.conj(rp[innS])
        q = Cr.T@w[e] + C1.T@w[innS] + C2.T@np.conj(w[innS])
        return p/(q[:, np.newaxis]), w

    def _get_poles(self, barycentric, N, parity):
        N_ = 2*(N-1) + parity
        S, w = barycentric
        S = np.sort(S)
        # No need to sort w as it comes sorted from _get_weights.

        e = np.array([0, len(S)])
        c = 0
        if S[0] == 0:
            e[0] = 1
            c += 1
        if parity == 0 and (S[-1] == N-1):
            e[1] = len(S)-1
            c += 1
        M = 2*len(S) - c - 1
        ωN = np.exp(-2j * np.pi / N_)

        w = np.concatenate((w, np.conj(w[e[0]:e[1]])))
        a = np.zeros((M+2, M+2), dtype=np.complex128)
        a[1:, 0] = 1
        a[0, 1:] = w
        a[1:, 1:] = np.diag(ωN**(np.concatenate((-S, S[e[0]:e[1]]))))

        b = np.eye(M+2, dtype=np.complex128)
        b[0, 0] = 0

        poles = scipy.linalg.eigvals(a, b, overwrite_a=True)
        poles = poles[2:]

        # Separate real and complex conjugated poles.
        poles_u = list(poles[np.imag(poles) >= 0])
        poles_l = list(poles[np.imag(poles) < 0])
        poles_r, poles_c = [], []
        while (len(poles_u) > 0) and (len(poles_l) > 0):
            distances = np.abs([p - np.conj(poles_u[-1]) for p in poles_l])
            idx = np.argmin(distances)
            if distances[idx] < 1e-8:
                poles_c.append(poles_u.pop())
                poles_l.pop(idx)
            else:
                poles_r.append(poles_u.pop())
        poles_r.extend(poles_u)
        poles_r.extend(poles_l)
        poles_r = np.real(poles_r)
        poles_c = np.array(poles_c)
        poles = [poles_r, poles_c]
        dim = 2*len(poles[1]) + len(poles[0])
        # Check that the number of poles is correct.
        if dim != M:
            msg = f'Expected {M} poles, got {dim} instead.'
            raise ValueError(msg)

        return poles

    def _get_residues(self, y, parity):
        poles = [self.r_poles_, self.c_poles_]
        N = y.shape[0]
        N_ = 2*(N-1) + parity
        ωN = np.exp(-2j * np.pi / N_)
        I = np.arange(N)

        # Construct Cauchy matrix.
        Cr = 1/(ωN**(-I[:, np.newaxis])-poles[0][np.newaxis, :])
        Ci1 = 1/(ωN**(-I[:, np.newaxis])-poles[1][np.newaxis, :])
        Ci2 = 1/(ωN**(-I[:, np.newaxis])-np.conj(poles[1])[np.newaxis, :])
        C = np.concatenate(
            (Cr, Ci1 + Ci2, 1j*(Ci1 - Ci2)),
            axis=1
        )
        
        # Separate real and imaginary of matrix and y.
        e = [1, N+parity-1]
        C = np.concatenate(
            (np.real(C), np.imag(C[e[0]:e[1]])),
            axis=0, dtype=float
        )
        y = np.concatenate(
            (np.real(y), np.imag(y[e[0]:e[1]])), axis=0,
            dtype=float
        )

        # Solve for residues.
        r = scipy.linalg.lstsq(
            C, y, overwrite_a=True, overwrite_b=True)[0]
        lr, lc = len(poles[0]), len(poles[1])
        residues = [r[:lr], r[lr:lr+lc] + 1j*r[lr+lc:]]

        return residues

    def _fit(self, y, parity, indices):
        N = y.shape[0]
        gS, gG = self._initialize_sets(y, indices)
        n_freqs = self.count_freqs(gS['index'], N, parity)

        step = 1
        while n_freqs-1 < self.order:
            logger.debug(f'==== step: {step} ====')
            r, w = self._get_weights(gS, gG, parity)
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            _update_sets(gS, gG, idx)

            if (gS['index'][-1] == 0) or (parity == 0 and gS['index'][-1] == N-1):
                n_freqs += 1
            else:
                n_freqs += 2
            step += 1
        self.n_poles_ = n_freqs-1
        logger.debug(f'==== step: {step} ====')
        r, w = self._get_weights(gS, gG, parity)
        idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))

        # Store indices for reuse.
        if len(gS['index'])+1 > len(self.indices_):
            self.indices_ = gS['index'].copy()
            self.indices_.append(gG['index'][idx])
            self.indices_ = np.array(self.indices_)

        # Find poles and residues.
        barycentric = (np.array(gS['index']), w)
        poles = self._get_poles(barycentric, N, parity)

        return poles

    @property
    def poles_(self):
        return np.concatenate([self.r_poles_, self.c_poles_, np.conj(self.c_poles_)])

    @property
    def residues_(self):
        if self.r_residues_ is None:
            return None
        else:
            return np.concatenate(
            [self.r_residues_, self.c_residues_, np.conj(self.c_residues_)],
            axis=0)

    def fit(self, y, parity:bool=None, seed_freqs=None):
        y = np.asarray(y)

        # Determine parity if not given.
        if parity is None:
            eps = np.finfo(y.dtype).eps
            tiny = np.imag(y[-1, :])
            parity = np.max(np.abs(tiny)) > 100*eps
        parity = int(parity)
        N, rank = y.shape
        N_ = 2*(N-1)+parity

        max_order = _get_max_order((N_, rank))
        if self.order is None:
            self.order = rank
        elif rank > self.order:
            msg = f'The minimum order is {rank}, which is greater than the set order {self.order}.'
            raise ValueError(msg)

        if hasattr(self, 'indices_'):
            # Ignores seed_freqs and reuse previous indices.
            indices = self.indices_
            n_freqs = 0
            i = 0
            while n_freqs-1 < self.order:
                if (indices[i] == 0) or (parity == 0 and indices[i] == N-1):
                    n_freqs += 1
                else:
                    n_freqs += 2
                i += 1
            indices = indices[:i]
        else:
            indices = self.set_seed_freqs(y, parity, seed_freqs=seed_freqs)
            self.indices_ = indices

        if self.order > max_order:
            msg = f'The order exceeds the maximum allowed order {max_order}.'
            raise ValueError(msg)

        poles = self._fit(y, parity, indices)

        self.r_poles_ = np.array(poles[0])
        self.c_poles_ = np.array(poles[1])
        if self.compute_residues is True:
            residues = self._get_residues(y, parity)
            self.r_residues_ = residues[0]
            self.c_residues_ = residues[1]
            if self.tol > 0.:
                self._pole_pruning()
                residues = self._get_residues(y, parity)
                self.r_residues_ = residues[0]
                self.c_residues_ = residues[1]
        else:
            self.r_residues_ = None
            self.c_residues_ = None

        return self

    def _pole_pruning(self):
        poles = [self.r_poles_, self.c_poles_]
        residues = [self.r_residues_, self.c_residues_]

        for i in range(2):
            idxs = np.nonzero(
                np.linalg.norm(residues[i], axis=1) > self.tol)[0]
            poles[i] = poles[i][idxs]

        self.r_poles_ = poles[0]
        self.c_poles_ = poles[1]

        return self

    def predict(self, X):
        return rational_function(
            self.poles_,
            self.residues_,
            X
        )

class VF(BaseEstimator):

    def __init__(
        self,
        *,
        order:int=None,
        cond:float=None,
        lapack_driver:str=None
    ):
        self.order = order
        self.cond = cond
        self.lapack_driver = lapack_driver

    def _get_weights(self, y, parity):
        N, rank = y.shape
        ns = 2*(N-1)+parity

        n_poles = self.order
        self.n_poles_ = n_poles
        poles = 0.9*np.exp(1j*np.pi*np.arange(1, n_poles+1)/(n_poles+1))
        poles = Poles(imag=poles)

        u = np.exp(2j*np.pi/ns)
        # Compute Cauchy matrix.
        Cr = 1/(u**np.arange(N)[:, np.newaxis] - poles.real[np.newaxis, :])
        Cr = np.concatenate((Cr, np.ones(shape=(N, 1), dtype=float)), axis=1)
        C1 = 1/(u**np.arange(N)[:, np.newaxis] - poles.imag[np.newaxis, :])
        C2 = 1/(u**np.arange(N)[:, np.newaxis] - np.conj(poles.imag)[np.newaxis, :])
        
        # Add constraint that rational function is symmetric.
        C = np.concatenate(
            (Cr, C1 + C2, 1j*(C1 - C2)),
            dtype=complex, axis=1)

        # Constraint vector.
        const_r = np.sum(Cr[ends], axis=0) + 2*np.real(np.sum(Cr[1:last], axis=0))
        const_c = np.sum(C1[ends], axis=0)
        const_c_r = const_c + np.sum(C1[1:last] + C2[1:last], axis=0)
        const_c_r = 2*np.real(const_c_r)
        const_c_i = const_c + np.sum(C1[1:last] - C2[1:last], axis=0)
        const_c_i = -2*np.imag(const_c_i)
        const = np.concatenate((const_r, const_c_r, const_c_i), axis=1)/ns
        In = scipy.linalg.qr(const.T, pivoting=True)[0]
        In = In[:, 1:]
        del Cr, C1, C2, const

        # Construct right-most columns of system matrix. Shape = (ns, n_poles, rank).
        R = -y[:, np.newaxis, :] * C[..., np.newaxis]
        R = np.einsum('ijk,jl->ilk', R, In)

        # Separate real and imaginary parts.
        ends = [0]
        if parity == 0:
            ends.append(N-1)
        ends = np.array(ends, dtype=np.int64)
        last = N-1+parity
        C = np.concatenate(
            (np.real(C[ends]), np.sqrt(2)*np.real(C[1:last]), np.sqrt(2)*np.imag(C[1:last])),
            axis=0, dtype=float
        )
        R = np.concatenate(
            (np.real(R[ends]), np.sqrt(2)*np.real(R[1:last]), np.sqrt(2)*np.imag(R[1:last])),
            axis=0, dtype=float
        )

        # Data to approximate. Shape = (ns, rank).
        b = np.concatenate(
            (np.real(y[ends]), np.sqrt(2)*np.real(y[1:last]), np.sqrt(2)*np.imag(y[1:last])),
            axis=0, dtype=float
        )

        Rb = np.concatenate((R, b[:, np.newaxis, :]), axis=1)
        Rb = Rb.reshape((ns, -1))
        del R, b

        # Solve LS for small Cauchy matrix.
        normals = scipy.linalg.lstsq(
            C, Rb, cond=self.cond, lapack_driver=self.lapack_driver)[0]
        Rb = Rb - C@normals
        del C, normals
        Rb = Rb.reshape((ns, n_poles, rank))
        R_tilde, b_tilde = Rb[:, :-1, :], Rb[:, -1, :]
        del Rb
        R_tilde = np.concatenate((R_tilde[..., k] for k in range(rank)), axis=0)
        b_tilde = b_tilde.T.reshape((1, -1)).T

        # Solve reduced LS problem.
        t = scipy.linalg.lstsq(
            R_tilde, b_tilde, cond=self.cond, lapack_driver=self.lapack_driver)[0]
        t = In @ t

        n_real = len(poles.real) + 1 # +1 for the constant term.
        n_complex = len(poles.imag)
        w_r = t[:n_real-1]
        d = t[n_real-1]
        w_c = t[n_real:n_real+n_complex] + 1j*t[n_real+n_complex:]

    def fit(self, y, parity:bool=None):
        y = np.asarray(y)
        N, rank = y.shape
        ns = 2*(N-1)+parity

        # Determine parity if not given.
        if parity is None:
            eps = np.finfo(y.dtype).eps
            tiny = np.imag(y[-1, :])
            parity = np.max(np.abs(tiny)) > 100*eps
        parity = int(parity)

        max_order = _get_max_order((ns, rank))
        if self.order is None:
            self.order = rank
        elif self.order > max_order:
            msg = f'The order exceeds the maximum allowed order {max_order}. \
                Redefining order to {max_order}.'
            logging.warning(msg, stacklevel=2)
            self.order = max_order

        return self



# ===============================
# Exponential Sums Decomposition
# ===============================

def exp_sum(poles, amplitudes, t, fs=1):
    '''
    Parameters
    ----------
    poles : (M,) array_like
    amplitudes : (M, L) array_like
    t : scalar or (N, ) array_like
    fs : scalar, default=1
        Sampling frequency.

    Return
    ------
    out : (N, L) complex ndarray
        The values of the rational function at the points z.
        If z is scalar, N = 1.
    '''
    poles = np.asarray(poles)
    if poles.ndim != 1:
        msg = f'Expected a 1D array for poles, got an array of dimension {poles.ndim}.'
        raise ValueError(msg)

    t = np.asarray(t)
    if t.ndim == 0:
        t = np.expand_dims(t, axis=0)
    elif t.ndim > 1:
        msg = f'Expected a 1D array for t, got an array of dimension {t.ndim}.'
        raise ValueError(msg)
    
    amplitudes = np.asarray(amplitudes)
    if amplitudes.ndim == 1:
        amplitudes = np.expand_dims(amplitudes, axis=1)
    elif amplitudes.ndim > 2:
        msg = f'Expected a 1D or 2D array for amplitudes, got an array of dimension {amplitudes.ndim}.'
        raise ValueError(msg)

    return (poles[np.newaxis, :]**(t[:, np.newaxis] * fs)) @ amplitudes


class ExpVF(BaseEstimator):

    def __init__(
            self,
            *,
            order:int=None,
            damping:float=0.,
            fs:float=1,
            tol:float=0.,
            compute_amps:bool=True
        ):
        self.order = order
        self.damping = damping
        self.fs = fs
        self.tol = tol
        self.compute_amps = compute_amps

    @property
    def r_resonances_(self):
        return np.log(self.r_poles_.astype(complex))*self.fs

    @property
    def c_resonances_(self):
        return np.log(self.c_poles_)*self.fs

    @property
    def resonances_(self):
        return np.concatenate(
            [self.r_resonances_, self.c_resonances_, np.conj(self.c_resonances_)],
            dtype=complex)

    @property
    def amps_(self):
        if self.r_amps_ is None:
            return None
        else:
            return np.concatenate(
                [self.r_amps_, self.c_amps_, np.conj(self.c_amps_)],
                axis=0, dtype=complex)

    def _get_amps(self, y, parity):
        N = y.shape[0]
        ns = 2*(N-1)+parity
        x = np.fft.irfft(y, n=ns, axis=0)
        poles = [self.r_poles_, self.c_poles_]

        # Construct Cauchy matrix.
        Vr = poles[0][np.newaxis, :]**(np.arange(ns)[:, np.newaxis])
        Vi = poles[1][np.newaxis, :]**(np.arange(ns)[:, np.newaxis])
        V = np.concatenate(
            (Vr, 2*np.real(Vi), -2*np.imag(Vi)),
            axis=1, dtype=float
        )

        # Solve for amplitudes.
        a = scipy.linalg.lstsq(
            V, x, overwrite_a=True, overwrite_b=True)[0]
        lr, lc = len(poles[0]), len(poles[1])
        amps = [a[:lr], a[lr:lr+lc] + 1j*a[lr+lc:]]

        return amps

    def fit(self, y, parity:bool=None, seed_freqs=None):
        y = np.asarray(y)
        N = y.shape[0]

        # Determine parity if not given.
        if parity is None:
            eps = np.finfo(y.dtype).eps
            tiny = np.imag(y[-1, :])
            parity = int(np.max(np.abs(tiny)) > 100*eps)
        parity = int(parity)
        ns = 2*(N-1)+parity

        if hasattr(self, '_rational'):
            rational = self._rational
        else:
            rational = VF()
            self._rational = rational
        rational.set_params(order=self.order)

        ωN = np.exp(-2j*np.pi/ns)
        rational.fit(
            y*(ωN**np.arange(N))[:, np.newaxis],
            parity=parity,
            seed_freqs=seed_freqs)

        self.r_poles_ = np.copy(rational.r_poles_)
        self.c_poles_ = np.copy(rational.c_poles_)

        poles = [self.r_poles_, self.c_poles_]
         # Remove unstable poles.
        for i in [0, 1]:
            idxs = np.nonzero(np.abs(poles[i]) < 1+1e-10)[0]
            poles[i] = poles[i][idxs]
        self.r_poles_ = poles[0]
        self.c_poles_ = poles[1]
        if self.compute_amps is True:
            amps = self._get_amps(y, parity)
            self.r_amps_ = amps[0]
            self.c_amps_ = amps[1]
        else:
            self.r_amps_ = None
            self.c_amps_ = None

        return self

    def predict(self, X):
        return np.real(exp_sum(
            np.concatenate([self.r_poles_, self.c_poles_, np.conj(self.c_poles_)]),
            np.concatenate([self.r_amps_, self.c_amps_, np.conj(self.c_amps_)], axis=0),
            X
        ))


class SuperResolution(BaseEstimator):

    def __init__(
            self,
            *,
            order:int=None,
            damping:float=0.,
            fs:float=1,
            tol:float=0.,
            compute_amps:bool=True
        ):
        self.order = order
        self.damping = damping
        self.fs = fs
        self.tol = tol
        self.compute_amps = compute_amps

    @property
    def r_resonances_(self):
        return np.log(self.r_poles_.astype(complex))*self.fs

    @property
    def c_resonances_(self):
        return np.log(self.c_poles_)*self.fs

    @property
    def resonances_(self):
        return np.concatenate(
            [self.r_resonances_, self.c_resonances_, np.conj(self.c_resonances_)],
            dtype=complex)

    @property
    def amps_(self):
        if self.r_amps_ is None:
            return None
        else:
            return np.concatenate(
                [self.r_amps_, self.c_amps_, np.conj(self.c_amps_)],
                axis=0, dtype=complex)

    def _get_amps(self, y, parity):
        N = y.shape[0]
        ns = 2*(N-1)+parity
        x = np.fft.irfft(y, n=ns, axis=0)
        poles = [self.r_poles_, self.c_poles_]

        # Construct Cauchy matrix.
        Vr = poles[0][np.newaxis, :]**(np.arange(ns)[:, np.newaxis])
        Vi = poles[1][np.newaxis, :]**(np.arange(ns)[:, np.newaxis])
        V = np.concatenate(
            (Vr, 2*np.real(Vi), -2*np.imag(Vi)),
            axis=1, dtype=float
        )

        # Solve for amplitudes.
        a = scipy.linalg.lstsq(
            V, x, overwrite_a=True, overwrite_b=True)[0]
        lr, lc = len(poles[0]), len(poles[1])
        amps = [a[:lr], a[lr:lr+lc] + 1j*a[lr+lc:]]

        return amps

    def fit(self, y, parity:bool=None, seed_freqs=None):
        y = np.asarray(y)
        N = y.shape[0]

        # Determine parity if not given.
        if parity is None:
            eps = np.finfo(y.dtype).eps
            tiny = np.imag(y[-1, :])
            parity = int(np.max(np.abs(tiny)) > 100*eps)
        parity = int(parity)
        ns = 2*(N-1)+parity

        if hasattr(self, '_rational'):
            rational = self._rational
        else:
            rational = Rational(compute_residues=False)
            self._rational = rational
        rational.set_params(order=self.order)

        ωN = np.exp(-2j*np.pi/ns)
        if self.damping > 0.:
            x = np.fft.irfft(y, n=ns, axis=0)
            x *= np.pow(self.damping, np.arange(ns)/(ns-1))[:, np.newaxis]
            y = np.fft.rfft(x, axis=0)
            del x
        rational.fit(
            y*(ωN**np.arange(N))[:, np.newaxis],
            parity=parity,
            seed_freqs=seed_freqs)

        self.r_poles_ = np.copy(rational.r_poles_)
        self.c_poles_ = np.copy(rational.c_poles_)
        if self.damping > 0.:
            self.r_poles_ *= np.pow(self.damping, -1/(ns-1))
            self.c_poles_ *= np.pow(self.damping, -1/(ns-1))

        poles = [self.r_poles_, self.c_poles_]
         # Remove unstable poles.
        for i in [0, 1]:
            idxs = np.nonzero(np.abs(poles[i]) < 1+1e-10)[0]
            poles[i] = poles[i][idxs]
        self.r_poles_ = poles[0]
        self.c_poles_ = poles[1]
        if self.compute_amps is True:
            amps = self._get_amps(y, parity)
            self.r_amps_ = amps[0]
            self.c_amps_ = amps[1]
            if self.tol > 0.:
                self._pole_pruning()
                amps = self._get_amps()
                self.r_amps_ = amps[0]
                self.c_amps_ = amps[1]
        else:
            self.r_amps_ = None
            self.c_amps_ = None

        return self

    def _pole_pruning(self):
        poles = [self.r_poles_, self.c_poles_]
        amps = [self.r_amps_, self.c_amps_]

        for i in range(2):
            idxs = np.nonzero(
                np.linalg.norm(amps[i], axis=1) > self.tol)[0]
            poles[i] = poles[i][idxs]

        self.r_poles_ = poles[0]
        self.c_poles_ = poles[1]

        return self

    def predict(self, X):
        return np.real(exp_sum(
            np.concatenate([self.r_poles_, self.c_poles_, np.conj(self.c_poles_)]),
            np.concatenate([self.r_amps_, self.c_amps_, np.conj(self.c_amps_)], axis=0),
            X
        ))

# Stabilization algorithm

def _inner_prod(ns:int, x, y=None):
    if y is None:
        r = np.abs(x)**2
    else:
        r = x*np.conj(y)
    idxs = np.nonzero(np.abs(1-r) > 0.1)
    r[idxs] = (1 - r[idxs]**ns) / (ns*(1 - r[idxs]))
    idxs = np.nonzero((np.abs(1-r) > 1e-10) & (np.abs(1-r) <= 0.1))
    eta = np.log(r[idxs])
    r[idxs] = np.sqrt(r[idxs]**(ns-1))
    r[idxs] *= np.sinh(ns*eta/2) / (ns*np.sinh(eta/2))
    idxs = np.nonzero(np.abs(1-r) <= 1e-10)
    r[idxs] = 1.
    return r

def dist(x, y, ns:int):
    floats = isinstance(x, float) and isinstance(y, float)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    abs_x = np.sqrt(np.real(_inner_prod(ns, x)))
    abs_y = np.sqrt(np.real(_inner_prod(ns, y)))
    t1 = (abs_x - abs_y)**2
    t2 = 2*abs_x*abs_y
    t2 *= (1 - np.real(_inner_prod(ns, x, y)) / (abs_x*abs_y))

    if floats:
        return (t1 + t2)[0]
    else:
        return t1 + t2

def _std_new(x, ns):
    x = np.asarray(x)
    mean = np.mean(x)
    return np.sqrt(np.mean(dist(x, mean, ns)**2))

class StablePoles:

    def __init__(
        self,
        model,
        max_order,
        *,
        radius=None,
        min_scale:int=-10
    ):
        self.model = model
        self.max_order = max_order
        self.radius = radius
        self.min_scale = min_scale

    def _add_poles(
        self, clusters, new_set, order, ns):
        radius = self.radius
        radius = dist(0., 0.9, ns) if radius is None else radius

        def mean_weighted(c):
            l = len(c) if len(c) < 4 else 4
            w = 1/(2**np.arange(l))
            poles = list(c.values())[-l:]
            avg = sum([p*w_ for p, w_ in zip(poles[::-1], w)]) / np.sum(w)
            return avg

        mean_clusters = [mean_weighted(c) for c in clusters]
        mean_clusters = np.array(mean_clusters)
        # Compute distance matrix.
        Pa, Pb = np.meshgrid(mean_clusters, new_set, indexing='ij')
        D = dist(Pa, Pb, ns)
        del Pa, Pb

        pop_idxs = []
        while np.min(D) < np.inf:
            # Get index of minimum distance without flattening.
            idx = np.unravel_index(np.argmin(D), D.shape)
            if D[idx] < radius:
                clusters[idx[0]].update({order: new_set[idx[1]]})
                pop_idxs.append(idx[1])
                D[idx[0], :] = np.inf
                D[:, idx[1]] = np.inf
            else:
                D[idx[0], :] = np.inf
        pop_idxs = np.array(pop_idxs)
        add_idxs = np.setdiff1d(
            np.arange(len(new_set)), pop_idxs,
            assume_unique=True)

        clusters.extend([{order: p} for p in new_set[add_idxs]])
        return clusters

    def _level_clustering(
        self, poles, amps, level, n_orders, ns
    ):
        # For each order, find poles within the level.
        poles_scale, idxs_scale = {}, {}
        for order, set_ in amps.items():
            idxs_scale[order] = np.nonzero((set_ > level/8) & (set_ <= level))[0]
            if len(idxs_scale[order]) == 0:
                continue
            poles_scale[order] = poles[order][idxs_scale[order]]

        clusters = []
        if len(poles_scale) == 0:
            return clusters, poles, amps

        # Add all the poles in the maximum order to clusters.
        initial_order = np.max(list(poles_scale.keys()))
        clusters = [{initial_order: p} for p in poles_scale[initial_order]]

        # Detect and add clusters.
        for order, poles_ in poles_scale.items():
            if (order == initial_order) or (len(poles_) == 0):
                continue
            clusters = self._add_poles(
                clusters, poles_, order, ns)
        clusters = [c for c in clusters if len(c) > 0.25*n_orders]

        # Reorder clustered poles by order.
        clusters_ = {order: [] for order in amps.keys()}
        for c in clusters:
            for order, p in c.items():
                clusters_[order].append(p)

        # Remove clustered poles from original sets.
        for order, set_ in poles_scale.items():
            idxs_p = [np.argmin(np.abs(p - set_)) for p in clusters_[order]]
            poles[order] = np.delete(poles[order], idxs_scale[order][idxs_p])
            amps[order] = np.delete(amps[order], idxs_scale[order][idxs_p])

        return clusters, poles, amps

    def _clustering(self, poles, amps, ns):
        min_scale = self.min_scale
        poles = {k: v.copy() for k, v in poles.items() if len(v) > 0}
        amps = {k: v.copy() for k, v in amps.items() if len(v) > 0}
        n_orders = len(poles)

        # Compute the largest amplitude for each order.
        grand_max = np.max([e[0] for e in amps.values()])
        scale = 0
        level = grand_max

        # Find clusters.
        min_scale = -np.inf if min_scale is None else min_scale
        clusters = []
        while (scale > min_scale) and (len(poles) > 0.25*n_orders):
            clusters_, poles, amps = self._level_clustering(
                poles, amps, level, n_orders, ns)
            clusters.extend(clusters_)

            # Remove empty orders.
            to_delete = []
            for order, poles_ in poles.items():
                if len(poles_) == 0:
                    to_delete.append(order)
            for order in to_delete:
                del poles[order], amps[order]

            level /= 2
            scale -= 1

        return clusters

    def fit(self, y, parity, seed_freqs=None):
        N = y.shape[0]
        ns = 2*(N-1) + parity
        self._ns = ns

        # Validate orders.
        max_order = self.max_order
        min_order = y.shape[1]
        max_order_ = _get_max_order((ns, y.shape[1]))
        if (max_order is None) or (max_order > max_order_):
            max_order = max_order_

        real_order = -1
        amps_set, poles_set = {}, {}
        for order in range(min_order, max_order+1):
            if order <= real_order:
                continue
            self.model.set_params(order=order)
            if order == min_order:
                self.model.fit(y, parity, seed_freqs=seed_freqs)
            else:
                self.model.fit(y, parity)
            real_order = self.model._rational.n_poles_

            amps_ = np.concatenate(
                (self.model.r_amps_, self.model.c_amps_),
                dtype=complex,
                axis=0)
            amps_ = np.linalg.norm(amps_, axis=1)
            poles_ = np.concatenate(
                (self.model.r_poles_, self.model.c_poles_), dtype=complex)
            idxs = np.argsort(amps_)[::-1]
            amps_ = amps_[idxs]
            poles_ = poles_[idxs]
            amps_set[real_order] = amps_
            poles_set[real_order] = poles_

        # Reverse order of keys.
        amps_set = {order: amps_set[order] for order in sorted(amps_set.keys(), reverse=True)}
        poles_set = {order: poles_set[order] for order in sorted(poles_set.keys(), reverse=True)}
        self.amps_set_ = amps_set
        self.poles_set_ = poles_set

        # Get clusters.
        clusters = self._clustering(
            poles_set, amps_set, ns=ns)
        # Sort cluster stats. by decreasing number of poles first, then by increasing pole std.
        prop = ['number', 'pole std.']
        stats = {p: [] for p in prop}
        for c in clusters:
            stats['number'].append(len(c))
            stats['pole std.'].append(_std_new(list(c.values()), self._ns))
        idxs = np.lexsort((
            stats['pole std.'],
            -np.array(stats['number'])
        ))
        clusters = np.array(clusters, dtype=object)
        self.clusters_ = list(clusters[idxs])
        self.clusters_amps_ = self._collect_amps()

        return self

    def _collect_amps(self):
        amps = []
        for c in self.clusters_:
            amps_c = {}
            for order, p in c.items():
                idx = np.argmin(np.abs(self.poles_set_[order] - p))
                amps_c[order] = self.amps_set_[order][idx]
            amps.append(amps_c)
        return amps
    
    def clusters_stats(self):
        if not hasattr(self, 'clusters_'):
            raise ValueError('You must run find_clusters() first.')

        columns = ['number', 'pole', 'pole std.', 'amp.', 'amp. std.', 'highest order', 'lowest order']
        stats = {c: [] for c in columns}
        for c, amps_c in zip(self.clusters_, self.clusters_amps_):
            stats['number'].append(len(c))
            orders = list(c.keys())
            stats['highest order'].append(orders[0])
            stats['lowest order'].append(orders[-1])
            stats['pole'].append(c[orders[0]])
            stats['pole std.'].append(_std_new(list(c.values()), self._ns))
            stats['amp.'].append(np.mean(list(amps_c.values())))
            stats['amp. std.'].append(np.std(list(amps_c.values())))

        return pd.DataFrame(stats)

    def plot_poles(self, scale, ax=None):
        if not hasattr(self, 'amps_set_'):
            raise ValueError('You must run find_clusters() first.')

        if ax is None:
            _, ax = plt.subplots(nrows=1)
            show = True
        else:
            ax = ax
            show = False

        # Compute maximum amplitude across all orders.
        grand_max = np.max([e[0] for e in self.amps_set_.values()])
        level = grand_max / (2**scale)

        poles_scale = {}
        for order, set_ in self.amps_set_.items():
            idxs_scale = np.nonzero((set_ > level/8) & (set_ <= level))[0]
            poles_scale[order] = self.poles_set_[order][idxs_scale]

        ax.set_title(f'Level [{level/8:.2f}, {level:.2f}]')
        for i, poles_ in poles_scale.items():
            ax.plot(poles_.real, poles_.imag, 'o', label=f'order {i}')
            # Annotate points with order.
            for p in poles_:
                ax.annotate(
                    f'{i}', xy=(p.real, p.imag),
                    textcoords='offset points', xytext=(0,5),
                    ha='center', fontsize=8)

        # Plot a unit circle.
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='k')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-.05, 1.1])

        # Draw the diameter at y=0 (only across the unit circle)
        ax.plot([-1.0, 1.0], [0.0, 0.0], color='k', lw=0.5)

        ax.set_aspect('equal')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        if show:
            plt.show()

    def plot_clusters(self, clusters=None, ax=None):
        if not hasattr(self, 'clusters_'):
            raise ValueError('You must run find_clusters() first.')

        clusters = clusters if clusters is not None else range(len(self.clusters_))
        if ax is None:
            _, ax = plt.subplots(nrows=1)
            show = True
        else:
            ax = ax
            show = False

        ax.set_title('Clusters')
        for i, c in zip(clusters, [self.clusters_[k] for k in clusters]):
            poles_ = list(c.values())
            poles_ = np.array(poles_)
            ax.plot(poles_.real, poles_.imag, 'o', label=f'cluster {i}')
            # Annotate points with order.
            for order, p in c.items():
                ax.annotate(
                    f'{order}', xy=(p.real, p.imag),
                    textcoords='offset points', xytext=(0,5),
                    ha='center', fontsize=8)

        # Plot a unit circle.
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='k')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-.05, 1.1])

        # Draw the diameter at y=0 (only across the unit circle)
        ax.plot([-1.0, 1.0], [0.0, 0.0], color='k', lw=0.5)

        ax.set_aspect('equal')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        if show:
            plt.show()
