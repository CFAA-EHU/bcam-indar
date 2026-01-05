#!/usr/bin/env python

import logging
import bisect

import numpy as np
from sklearn.base import BaseEstimator
import scipy
import matplotlib.pyplot as plt

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


def _get_max_order(shape):
    N, m = shape
    # Counting complex paramaters in a complex time series
    # and in a rational function, we have that:
    max_order = m*N//(m+1)

    return max_order

def _update_sets(gS, gG, idx):
    gS['index'].append(gG['index'][idx])
    gS['data'].append(gG['data'][idx])
    gG['index'].pop(idx)
    gG['data'].pop(idx)

class RatApp:
    '''Rational Approximation using AAA algorithm.

    This implementation is based on...

    Parameters
    ----------
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    mode : str, default='normal'
        Mode of the algorithm. Either 'normal' or 'symmetric'.
    max_order : int, default=None
        Maximum number of poles to be used in the approximation.
        By default, the maximum possible.

    Attributes
    ----------
    error_ : float
    poles_ : array_like
    residues_ : array_like
    '''

    def __init__(self, x, rank_tol=1e-2):
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        self.x = x
        if x.shape[1] > 1:
            # Check for rank deficiency.
            s = scipy.linalg.svdvals(x, overwrite_a=False)
            rank = np.nonzero(s/s[0] < rank_tol)[0]
            if len(rank) > 0:
                logger.info(f'Rank deficient data.')
                rank = rank[0]
            else:
                rank = len(s)
        else:
            rank = 1
        self._rank = rank
    
    def set_seed_freqs(self, freqs=None):
        '''Set initial frequencies for the algorithm.
        
        Parameters
        ----------
        freqs : 1darray
            They must be ordered in decreasing order of importance,
            and the indices must be unique.
        '''
        N = self.x.shape[0]
        if freqs is None:
            # Choose peaks as default initial frequencies.
            abs_v = np.linalg.norm(self.x, axis=1)
            abs_v = np.concatenate((abs_v, abs_v))
            peaks, h = scipy.signal.find_peaks(
                abs_v,
                height=np.max(abs_v)/5,
                distance=np.max((N/(4*(self._rank+1)), 2))
            )
            idxs = np.argsort(h['peak_heights'])[::-1]
            peaks = peaks[idxs]%N
            _, idxs = np.unique(peaks, return_index=True)
            freqs = peaks[np.sort(idxs)]
        
        assert freqs.ndim == 1, 'freqs must be a 1D array.'
        assert np.all(freqs >= 0) and np.all(freqs < N), 'freqs must be in the range [0, N).'
        assert len(freqs) == len(np.unique(freqs)), 'freqs must be unique.'

        if len(freqs) >= self._rank+1:
            freqs = freqs[:self._rank+1]
        else:
            # Generate additional random indices if c < rank + 1.
            logger.warning(
                'Not enough peaks found. Adding random indices.', stacklevel=2)
            diff = np.setdiff1d(np.arange(N), freqs, assume_unique=True)
            rng = np.random.default_rng()
            rng.shuffle(diff)
            diff = list(diff)
            c = len(freqs)
            freqs_ = []
            while c < self._rank+1:
                freqs_.append(diff.pop())
                c += 1
            freqs = np.concatenate((freqs, freqs_))

        self._freqs = freqs

    def _initialize_sets(self, seed_freqs):
        N = self.x.shape[0]
        gG = {'index': list(range(N)), 'data': list(self.x)}
        gS = {}

        gS['index'] = [gG['index'][f] for f in seed_freqs]
        gS['data'] = [gG['data'][f] for f in seed_freqs]
        for f in seed_freqs:
            idx = bisect.bisect_left(gG['index'], f)
            gG['index'].pop(idx)
            gG['data'].pop(idx)
        return gS, gG

    @staticmethod
    def _fit(N, gS, gG, rank):
        n_freqs = len(gS['data'])
        ωN = np.exp(-2j * np.pi / N)
        C = ωN**(-np.array(gS['index']))[:, np.newaxis] - ωN**(-np.array(gG['index']))[np.newaxis, :]
        L = np.array(gS['data'])[:, np.newaxis] - np.array(gG['data'])[np.newaxis, :]
        L = L / C[..., np.newaxis]
        L = L.reshape(n_freqs, -1).T

        # Construct inclusion matrix into the space that satisfies (3.14) of [From ESPRIT to ESPIRA].
        In = scipy.linalg.qr(
            np.array(gS['data']), pivoting=True)[0]
        In = np.conj(In[:, rank:])
        # Compute weights to find best rational approximation.
        L = L @ In
        eigval, w = scipy.linalg.svd(
            L,
            overwrite_a=True,
            full_matrices=False)[1:]
        logger.debug(f'Lowest eigenvalue: {eigval[-1]}')
        w = In @ np.conj(w[-1])
        r = w[:, np.newaxis] * np.array(gS['data'])
        r = ((1/C.T) @ r) / ((1/C.T) @ w)[:, np.newaxis]

        return r, w

    @staticmethod
    def _normal_form(N, barycentric):
        gS, w = barycentric
        S = np.array(gS['index'])
        gS = np.array(gS['data'])
        M = len(S) - 1
        ωN = np.exp(-2j * np.pi / N)

        a = np.zeros((M+2, M+2), dtype=np.complex128)
        a[1:, 0] = 1
        a[0, 1:] = w
        a[1:, 1:] = np.diag(ωN**(-S))

        b = np.eye(M+2, dtype=np.complex128)
        b[0, 0] = 0

        poles = scipy.linalg.eigvals(a, b, overwrite_a=True)
        poles = poles[2:]

        C = ωN**(-S)[:, np.newaxis] - poles[np.newaxis, :]
        C = 1 / C
        C = np.concatenate((C, np.ones((len(S), 1))), axis=1)
        residues = scipy.linalg.solve(C, gS)
        if np.linalg.norm(residues[-1]) > 1e-8:
            msg = 'The constant term of rational function is not close to zero.'
            logger.error(msg, stacklevel=2)
        residues = residues[:-1]

        return poles, residues
    
    def fit(self, tol=1e-3, max_order=None):
        N, m = self.x.shape
        max_order_ = _get_max_order(max_order, (N, m))

        if not hasattr(self, '_freqs'):
            self.set_seed_freqs()

        if len(self._freqs) >= max_order_ + 1:
            gS, gG = self._initialize_sets(self._freqs[:max_order_+1])
        else:
            gS, gG = self._initialize_sets(self._freqs)
        r, w = self._fit(N, gS, gG, self._rank)

        logger.debug(f'==== step: 0 ====')
        succeed = False
        step = 1
        while not succeed:
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            error = np.linalg.norm(gG['data'] - r, axis=1)[idx]
            logger.debug(f'error: {error}')

            if error < tol:
                succeed = True
            elif len(gS) >= max_order_ + 1:
                msg = 'Convergence failed after the maximum number of poles is reached.\n'
                msg += f'The error is {error}'
                logger.warning(msg, stacklevel=2)
                break
            else:
                logger.debug(f'==== step: {step} ====')
                _update_sets(gS, gG, idx)
                r, w = self._fit(N, gS, gG, self._rank)
            step += 1
        
        self._freqs = np.array(gS['index'])
        barycentric = (gS, w)
        p, res = self._normal_form(N, barycentric)

        return p, res, error

class RatAppSym(BaseEstimator):

    def __init__(
            self,
            *,
            order:int=None):
        self.order = order

    @staticmethod
    def count_freqs(idxs, N, parity):
        idxs = np.array(idxs)
        idxs = np.sort(idxs)
        n_freqs = 0
        if idxs[0] == 0:
            n_freqs += 1
        if parity == 0 and (idxs[-1] == N-1):
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
            abs_v = np.concatenate((abs_v, abs_v[-2+parity::-1]))
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
        if parity == 0 and (S_[-1] == N-1):
            e.append(len(S_)-1)
            innS[1] = len(S_) - 1
        else:
            innG[1] = len(G_) - 1
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

    def _get_residues(self, y, parity, poles):
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
        residues = self._get_residues(y, parity, poles)

        return poles, residues

    @property
    def poles_(self):
        return np.concatenate([self.r_poles_, self.c_poles_, np.conj(self.c_poles_)])

    @property
    def residues_(self):
        return np.concatenate(
            [self.r_residues_, self.c_residues_, np.conj(self.c_residues_)],
            axis=0)

    def fit(self, y, parity, seed_freqs=None, reuse:bool=True):
        N, rank = y.shape
        N_ = 2*(N-1)+parity
        max_order = _get_max_order((N_, rank))
        if self.order is None:
            self.order = rank
        elif rank > self.order:
            msg = f'The minimum order is {rank}, which is greater than the set order {self.order}.'
            raise ValueError(msg)

        if hasattr(self, 'indices_') and reuse:
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

        poles, residues = self._fit(y, parity, indices)

        self.r_poles_ = np.array(poles[0])
        self.c_poles_ = np.array(poles[1])
        self.r_residues_ = np.array(residues[0])
        self.c_residues_ = np.array(residues[1])

        return self

    def pole_pruning(self, tol=1e-6):
        poles = [self.r_poles_, self.c_poles_]
        residues = [self.r_residues_, self.c_residues_]

        for i in range(2):
            idxs = np.nonzero(
                np.linalg.norm(residues[i], axis=1) > np.abs((np.abs(poles[i])-1))*tol)[0]
            poles[i] = poles[i][idxs]
            residues[i] = residues[i][idxs]

        self.r_poles_ = poles[0]
        self.c_poles_ = poles[1]
        self.r_residues_ = residues[0]
        self.c_residues_ = residues[1]

        return self

    def predict(self, X):
        return rational_function(
            self.poles_,
            self.residues_,
            X
        )


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


class Espira:
    '''Approximate signal as a sum of exponentials using ESPIRA algorithm.
    
    Parameters
    ----------
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_order : int, default=None
        Maximum number of poles to be used in the approximation.
        By default, the maximum possible.
    fs : float, default=1
        Sampling frequency.
    mode : str, default='complex'
        Either 'complex' or 'real'.

    Attributes
    ----------
    damping_ : array_like
    normal_frequencies_ : array_like
    amplitudes_ : array_like
    rational_ : RationalApproximation
    '''

    def __init__(self, x, rank_tol=1e-2):
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        y = np.copy(x)
        N = y.shape[0]
        # Add a small damping to stabilize the algorithm.
        false_damping = np.power(2, -2 / N)
        y *= (false_damping**np.arange(N))[:, np.newaxis]
        y = np.fft.fft(y, axis=0)
        ωN = np.exp(-2j * np.pi / N)
        y = (ωN**np.arange(N))[:, np.newaxis] * y
        self._rational = RatApp(y, rank_tol=rank_tol)

    def fit(self, tol=1e-3, max_order=None):
        N = self._rational.x.shape[0]

        poles, residues = self._rational.fit(
            tol=tol, max_order=max_order)[:2]
        amps = residues / (1 - (poles[:, np.newaxis])**N)
        # Remove false damping.
        poles *= np.power(2, 2 / N)

        return amps, poles


class EspiraR(BaseEstimator):

    def __init__(
            self,
            *,
            order:int=None,
            damping:float=0.,
            fs:float=1.,
            copy_y:bool=True):
        self.order = order
        self.damping = damping
        self.fs = fs
        self.copy_y = copy_y

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
        return np.concatenate(
            [self.r_amps, self.c_amps, np.conj(self.c_amps)],
            axis=0,
            dtype=complex)

    def fit(self, y, parity, seed_freqs=None, reuse:bool=True):
        N = y.shape[0]
        ns = 2*(N-1)+parity
        self._ns = ns

        if hasattr(self, '_rational'):
            rational = self._rational
        else:
            rational = RatAppSym()
            self._rational = rational
        rational.set_params(order=self.order)

        ωN = np.exp(-2j*np.pi/ns)
        if self.copy_y:
            y = np.copy(y)
        if self.damping > 0.:
            x = np.fft.irfft(y, n=ns, axis=0)
            x *= np.pow(self.damping, np.arange(ns)/(ns-1))[:, np.newaxis]
            np.fft.rfft(x, axis=0, out=y)
            del x
        y *= (ωN**np.arange(N))[:, np.newaxis]
        rational.fit(y, parity, seed_freqs=seed_freqs, reuse=reuse)

        self.r_poles_ = np.copy(rational.r_poles_)
        self.c_poles_ = np.copy(rational.c_poles_)
        self.r_amps_ = rational.r_residues_/(1-(rational.r_poles_[:, np.newaxis])**ns)
        self.c_amps_ = rational.c_residues_/(1-(rational.c_poles_[:, np.newaxis])**ns)
        if self.damping > 0.:
            self.r_poles_ *= np.pow(rational.damping, -1/(ns-1))
            self.c_poles_ *= np.pow(rational.damping, -1/(ns-1))

        return self

    def pole_pruning(self, tol=1e-6):
        self._rational.pole_pruning(tol=tol)

        poles = [self._rational.r_poles_, self._rational.c_poles_]
        residues = [self._rational.r_residues_, self._rational.c_residues_]

        # Remove unstable poles.
        for i in [0, 1]:
            idxs = np.nonzero(np.abs(poles[i]) < 1+1e-10)[0]
            poles[i] = poles[i][idxs]
            residues[i] = residues[i][idxs]

        ns = self._ns
        self.r_poles_ = poles[0]
        self.c_poles_ = poles[1]
        self.r_amps_ = residues[0]/(1-(poles[0][:, np.newaxis])**ns)
        self.c_amps_ = residues[1]/(1-(poles[1][:, np.newaxis])**ns)
        if self.damping > 0.:
            self.r_poles_ *= np.pow(self._rational.damping, -1/(ns-1))
            self.c_poles_ *= np.pow(self._rational.damping, -1/(ns-1))

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
        q:float=1/3,
        min_scale:int=-10
    ):
        self.model = model
        self.max_order = max_order
        self.radius = radius
        self.q = q
        self.min_scale = min_scale

    def _add_poles(
        self, clusters, new_set, order, ns):
        radius = self.radius
        radius = dist(0., 0.6, ns) if radius is None else radius
        q = self.q

        mean_clusters = [np.mean(list(c.values())) for c in clusters]
        mean_clusters = np.array(mean_clusters)
        std_clusters = [
            _std_new(list(c.values()), ns) if len(c) > 1 else radius
            for c in clusters]
        std_clusters = np.array(std_clusters)
        # Compute distance matrix.
        Pa, Pb = np.meshgrid(mean_clusters, new_set, indexing='ij')
        D = dist(Pa, Pb, ns)
        del Pa, Pb

        pop_idxs = []
        while np.min(D) < np.inf:
            # Get index of minimum distance without flattening.
            idx = np.unravel_index(np.argmin(D), D.shape)
            if D[idx] < np.min([radius, np.power(std_clusters[idx[0]], q)]):
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
        self, poles, amps, level, ns
    ):
        n_orders = len(amps)

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
        clusters = [c for c in clusters if len(c) > n_orders//2]

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
        while (scale > min_scale) and (len(poles) > n_orders//2):
            clusters_, poles, amps = self._level_clustering(
                poles, amps, level, ns)
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

        # Get intersection of orders.
        if len(clusters) == 0:
            return clusters, None

        orders_clusters = [list(c.keys()) for c in clusters]
        common_orders = orders_clusters[0]
        for oc in orders_clusters[1:]:
            common_orders = np.intersect1d(
                common_orders, oc, assume_unique=True)
        if len(common_orders) > 0:
            min_order = np.min(common_orders)
        else:
            min_order = None

        return clusters, min_order

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
            self.model.set_params(order=order, copy_y=True)
            if order == min_order:
                self.model.fit(y, parity, seed_freqs=seed_freqs, reuse=True)
            else:
                self.model.fit(y, parity, reuse=True)
            real_order = len(self.model.r_poles_) + 2*len(self.model.c_poles_)
            self.model.pole_pruning(tol=0.)

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

        clusters, min_order = self._clustering(
            poles_set, amps_set, ns=ns)
        self.min_order_ = min_order
        self.clusters_ = clusters
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

        stats = []
        for c, amps_c in zip(self.clusters_, self.clusters_amps_):
            p_mean = np.mean(list(c.values()))
            p_std = _std_new(list(c.values()), self._ns)
            a_mean = np.mean(list(amps_c.values()))
            a_std = np.std(list(amps_c.values()))
            stats.append({
                'p. mean': p_mean,
                'p. std': p_std,
                'a. mean': a_mean,
                'a. std': a_std,
            })
        return stats

    def plot(self, scale, ax=None):
        if not hasattr(self, 'amps_set_'):
            raise ValueError('You must run find_clusters() first.')

        # Compute maximum amplitude across all orders.
        grand_max = np.max([e[0] for e in self.amps_set_.values()])
        level = grand_max / (2**scale)

        poles_scale = {}
        for order, set_ in self.amps_set_.items():
            idxs_scale = np.nonzero((set_ > level/8) & (set_ <= level))[0]
            poles_scale[order] = self.poles_set_[order][idxs_scale]

        if ax is None:
            _, ax = plt.subplots(nrows=1)
        else:
            ax = ax

        ax.set_title(f'Level [{level/8:.2f}, {level:.2f}]')
        for i, poles_ in poles_scale.items():
            ax.plot(poles_.real, poles_.imag, 'o', label=f'order {i}')

        # Plot a unit circle.
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='k')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-.05, 1.1])

        # Draw the diameter at y=0 (only across the unit circle)
        ax.plot([-1.0, 1.0], [0.0, 0.0], color='k', lw=0.5)

        ax.set_aspect('equal')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        return ax
