#!/usr/bin/env python

import logging
import bisect

import numpy as np
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

def rational_function_sym(poles, residues, z):
    if len(poles[1]) > 0:
        r = (1 / (z[:, np.newaxis] - poles[1][np.newaxis, :])) @ residues[1]
        r += (1 / (z[:, np.newaxis] - np.conj(poles[1])[np.newaxis, :])) @ np.conj(residues[1])
    if len(poles[0]) > 0:
        r += (1 / (z[:, np.newaxis] - poles[0][np.newaxis, :])) @ residues[0]
    return r


def _get_max_order(max_order, shape):
    N, m = shape
    max_val = m*N//(m+1)
    if max_order is None:
        max_order_ = max_val
    else:
        if max_order > max_val:
            msg = f'The number of poles must be less than half of the number of samples. \
            max_order readjusted to {max_val}'
            logger.warning(msg, stacklevel=2)
        max_order_ = min(max_order, max_val)

    return max_order_

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

class RatAppSym:

    def __init__(self, x, rank_tol=1e-2, mode='(n,n)'):
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        x[1:] = 0.5 * (x[1:] + np.conj(x[-1:0:-1]))
        self.x = x
        self.rank_tol = rank_tol
        self.mode = mode
        N, m = x.shape
        if m > 1:
            y = np.concatenate(
                (np.real(x[:N//2+1]), np.imag(x)[1:(N+1)//2]), axis=0)
            # Check for rank deficiency.
            s = scipy.linalg.svdvals(y, overwrite_a=True)
            rank = np.nonzero(s/s[0] < rank_tol)[0]
            if len(rank) > 0:
                logger.info(f'Rank deficient data.')
                rank = rank[0]
            else:
                rank = m
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
        m = N // 2 + 1
        if freqs is None:
            # Choose rank + 1 peaks as initial frequencies.
            abs_v = np.linalg.norm(self.x, axis=1)
            abs_v = np.concatenate((abs_v, abs_v))
            peaks, h = scipy.signal.find_peaks(
                abs_v,
                height=np.max(abs_v)/5,
                distance=np.max((N/(8*(self._rank+1)), 2))
            )
            idxs = np.argsort(h['peak_heights'])[::-1]
            peaks = peaks[idxs]%N
            _, idxs = np.unique(peaks, return_index=True)
            peaks = peaks[np.sort(idxs)]
            freqs = peaks[peaks < m]
        else:
            freqs = np.asarray(freqs)

        if freqs.ndim != 1:
            raise ValueError('freqs must be a 1D array.')
        if not (np.all(freqs >= 0) and np.all(freqs < m)):
            raise ValueError('freqs must be in the range [0, N//2+1).')
        if len(freqs) != len(np.unique(freqs)):
            raise ValueError('freqs must be unique.')

        freqs = list(freqs)
        freqs.reverse()
        freqs_ = []
        c = 0
        while (c < self._rank+1) and (len(freqs) > 0):
            p = freqs.pop()
            freqs_.append(p)
            if (p == 0) or (N%2 == 0 and p == N//2):
                c += 1
            else:
                c += 2
        # Generate additional random indices if c < rank + 1.
        if c < self._rank + 1:
            logger.warning(
                'Not enough freqs found. Adding random indices.', stacklevel=2)
            diff = np.setdiff1d(np.arange(m), freqs_, assume_unique=True)
            rng = np.random.default_rng()
            rng.shuffle(diff)
            diff = list(diff)
            while c < self._rank+1:
                p = diff.pop()
                freqs_.append(p)
                if (p == 0) or (N%2 == 0 and p == N//2):
                    c += 1
                else:
                    c += 2

        self._freqs = np.array(freqs_)

    def _initialize_sets(self, seed_freqs):
        N = self.x.shape[0]
        m = N // 2 + 1
        gG = {'index': list(range(m)), 'data': list(self.x[:m])}
        gS = {}

        gS['index'] = [gG['index'][f] for f in seed_freqs]
        gS['data'] = [gG['data'][f] for f in seed_freqs]
        for f in seed_freqs:
            idx = bisect.bisect_left(gG['index'], f)
            gG['index'].pop(idx)
            gG['data'].pop(idx)
        return gS, gG

    def _fit(self, gS, gG):
        N = self.x.shape[0]
        rank = self._rank
        n_freqs = len(gS['data'])
        n_comps = len(gS['data'][0])
        ωN = np.exp(-2j * np.pi / N)
        # Locate zero and N/2.
        endsS = []
        if gG['index'][0] != 0:
            endsS.append(gS['index'].index(0))
        if N%2 == 0 and (gG['index'][-1] != N//2):
            endsS.append(gS['index'].index(N//2))
        endsS = np.sort(np.array(endsS, dtype=np.int64))

        Cp = ωN**(-np.array(gS['index']))[:, np.newaxis] - ωN**(-np.array(gG['index']))[np.newaxis, :]
        Cm = ωN**(np.array(gS['index']))[:, np.newaxis] - ωN**(-np.array(gG['index']))[np.newaxis, :]
        Lp = np.array(gS['data'])[:, np.newaxis] - np.array(gG['data'])[np.newaxis, :]
        Lm = np.conj(np.array(gS['data']))[:, np.newaxis] - np.array(gG['data'])[np.newaxis, :]
        Lp = Lp / Cp[..., np.newaxis]
        Lm = Lm / Cm[..., np.newaxis]
        mr = 2*len(gS['index']) - len(endsS)
        # Adapt matrix L for real and imaginary parts.
        LR = np.zeros(
            (mr, 2*len(gG['data']), n_comps),
            dtype=np.float64)
        # Add frequencies 0 and N/2 if they are in gS.
        ends_ = 2*endsS - np.arange(len(endsS))
        LR[ends_, ::2] = np.real(Lp[endsS])
        LR[ends_, 1::2] = np.imag(Lp[endsS])
        # Add inner frequencies.
        inners = np.setdiff1d(np.arange(n_freqs), endsS, assume_unique=True)
        inners_ = np.setdiff1d(np.arange(mr), ends_, assume_unique=True)
        LR[inners_[::2], ::2] = np.real(Lp[inners]) + np.real(Lm[inners])
        LR[inners_[::2], 1::2] = np.imag(Lp[inners]) + np.imag(Lm[inners])
        LR[inners_[1::2], ::2] = -np.imag(Lp[inners]) + np.imag(Lm[inners])
        LR[inners_[1::2], 1::2] = np.real(Lp[inners]) - np.real(Lm[inners])
        # Take into account that 0 and N/2 appear only once.
        if gG['index'][0] == 0:
            LR[:, :2] = LR[:, :2] / np.sqrt(2)
        if N%2 == 0 and (gG['index'][-1] == N//2):
            LR[:, -2:] = LR[:, -2:] / np.sqrt(2)
        LR = LR.reshape(mr, -1).T
        del Lp, Lm

        if self.mode == '(n-1,n)':
            # Construct inclusion matrix into the space that satisfies (3.14) of [From ESPRIT to ESPIRA].
            gS_ = np.zeros((mr, n_comps), dtype=np.float64)
            gS_[ends_, :] = np.real(np.array(gS['data'])[endsS])
            gS_[inners_[::2], :] = 2*np.real(np.array(gS['data'])[inners])
            gS_[inners_[1::2], :] = -2*np.imag(np.array(gS['data'])[inners])
            In = scipy.linalg.qr(gS_, pivoting=True)[0]
            In = In[:, rank:]
            # Compute weights to find best rational approximation.
            LR = LR @ In

        eigval, w = scipy.linalg.svd(
            LR,
            overwrite_a=True,
            full_matrices=False)[1:]
        logger.debug(f'Lowest eigenvalue: {eigval[-1]}')
        if self.mode == '(n-1,n)':
            w = In @ w[-1]
        w_ = np.zeros((n_freqs,), dtype=np.complex128)
        w_[endsS] = w[ends_]
        w_[inners] = w[inners_[::2]] + 1j * w[inners_[1::2]]
        w = w_

        # Compute rational function at G frequencies.
        rp = w[:, np.newaxis] * np.array(gS['data'])
        p = (1/Cp).T @ rp + (1/Cm[inners]).T @ np.conj(rp)[inners]
        q = (1/Cp).T @ w + (1/Cm[inners]).T @ np.conj(w)[inners]

        return p/(q[:, np.newaxis]), w

    def _normal_form(self, barycentric):
        N = self.x.shape[0]
        gS, endsS, w = barycentric
        S = np.array(gS['index'])
        gS = np.array(gS['data'])
        inners = np.setdiff1d(np.arange(len(S)), endsS, assume_unique=True)
        w = np.concatenate((w, np.conj(w[inners])))
        M = 2*len(S) - len(endsS) - 1
        ωN = np.exp(-2j * np.pi / N)

        a = np.zeros((M+2, M+2), dtype=np.complex128)
        a[1:, 0] = 1
        a[0, 1:] = w
        a[1:, 1:] = np.diag(ωN**(np.concatenate((-S, S[inners]))))

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

        C = np.zeros((len(S), M+1), dtype=np.complex128)
        lr, lc = len(poles[0]), len(poles[1])
        if lr > 0:
            C[:, :lr] = 1/(ωN**(-S)[:, np.newaxis] - poles[0][np.newaxis, :])
        if lc > 0:
            C[:, lr:lr+lc] = 1/(ωN**(-S)[:, np.newaxis] - poles[1][np.newaxis, :])
            C[:, lr:lr+lc] += 1/(ωN**(-S)[:, np.newaxis] - np.conj(poles[1])[np.newaxis, :])
            C[:, lr+lc:-1] = 1j/(ωN**(-S)[:, np.newaxis] - poles[1][np.newaxis, :])
            C[:, lr+lc:-1] -= 1j/(ωN**(-S)[:, np.newaxis] - np.conj(poles[1])[np.newaxis, :])
        C[:, -1] = 1
        C = np.concatenate((np.real(C), np.imag(C[inners])), axis=0)
        residues = scipy.linalg.solve(
            C,
            np.concatenate((np.real(gS), np.imag(gS[inners])), axis=0)
        )

        if self.mode == '(n-1,n)':
            if np.linalg.norm(residues[-1]) > 1e-8:
                msg = 'The constant term of rational function is not close to zero.'
                logger.error(msg, stacklevel=2)
            poly = (0.,)
        elif self.mode == '(n,n)':
            poly = (residues[-1],)
        residues = residues[:-1]
        residues = [residues[:lr], residues[lr:lr+lc]+1j*residues[lr+lc:]]

        for i in range(2):
            idxs = np.argsort(np.linalg.norm(residues[i], axis=1))[::-1]
            residues[i] = residues[i][idxs]
            poles[i] = poles[i][idxs]

        return tuple(poles), tuple(residues), poly

    def count_freqs(self, idxs):
        N = self.x.shape[0]
        idxs = np.array(idxs)
        idxs = np.sort(idxs)
        n_freqs = 0
        if idxs[0] == 0:
            n_freqs += 1
        if N%2 == 0 and (idxs[-1] == N//2):
            n_freqs += 1
        n_freqs = 2*len(idxs) - n_freqs
        return n_freqs

    def fit(self, tol=1e-3, max_order=None):
        N, m = self.x.shape
        max_order_ = _get_max_order(max_order, (N, m))

        if not hasattr(self, '_freqs'):
            self.set_seed_freqs()
        if self._rank > max_order_:
            msg = f'The minimum order is {self._rank}, which is greater than max_order {max_order_}.'
            raise ValueError(msg)

        n_freqs = self.count_freqs(self._freqs)
        if n_freqs >= max_order_ + 1:
            count = max_order_//2
            n_freqs = self.count_freqs(self._freqs[:count])
            while n_freqs < max_order_ + 1:
                count += 1
                n_freqs = self.count_freqs(self._freqs[:count])
            gS, gG = self._initialize_sets(self._freqs[:count])
        else:
            gS, gG = self._initialize_sets(self._freqs)
        r, w = self._fit(gS, gG)
        n_freqs = self.count_freqs(gS['index'])

        logger.debug(f'==== step: 0 ====')
        status = 0
        step = 1
        while status == 0:
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            error = np.linalg.norm(gG['data'] - r)/np.sqrt(N)
            logger.debug(f'error: {error}')

            if error < tol:
                status = 1
            elif n_freqs >= max_order_ + 1:
                status = 2
            else:
                logger.debug(f'==== step: {step} ====')
                _update_sets(gS, gG, idx)
                r, w = self._fit(gS, gG)
                if (gS['index'][-1] == 0) or (N%2 == 0 and gS['index'][-1] == N//2):
                    n_freqs += 1
                else:
                    n_freqs += 2
            step += 1

        if len(gS['index']) > len(self._freqs):
            self._freqs = np.array(gS['index'])

        endsS = []
        if gG['index'][0] != 0:
            endsS.append(gS['index'].index(0))
        if N%2 == 0 and (gG['index'][-1] != N//2):
            endsS.append(gS['index'].index(N//2))
        endsS = np.sort(np.array(endsS, dtype=np.int64))
        barycentric = (gS, endsS, w)
        p, res, poly = self._normal_form(barycentric)

        return p, res, poly, error


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

def exp_sum_R(poles, amps, t, fs=1):
    r = 0
    if poles[0] is not None:
        r += (poles[0][np.newaxis, :]**(t[:, np.newaxis] * fs)) @ amps[0]
    if poles[1] is not None:
        r += 2 * np.real((poles[1][np.newaxis, :]**(t[:, np.newaxis] * fs)) @ amps[1])
    return r

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


class EspiraR:

    def __init__(self, x, rank_tol=1e-2, mode='(n,n)'):
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
        self._rational = RatAppSym(y, rank_tol=rank_tol, mode=mode)

    def fit(self, tol=1e-3, max_order=None):
        N = self._rational.x.shape[0]

        poles, residues, poly = self._rational.fit(
            tol=tol, max_order=max_order)[:3]
        amps = 2*[None]
        poles = list(poles)
        for i in [0, 1]:
            amps[i] = residues[i] / (1 - (poles[i][:, np.newaxis])**N)
            # Remove false damping.
            poles[i] *= np.power(2, 2 / N)

        amps, poles = list(amps), list(poles)
        for i in range(2):
            idxs = np.argsort(np.linalg.norm(amps[i], axis=1))[::-1]
            amps[i] = amps[i][idxs]
            poles[i] = poles[i][idxs]

        return tuple(amps), tuple(poles), poly

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
        self, model, ns:int,
    ):
        self.model = model
        self.ns = ns

    def _add_poles(
        self,clusters, new_set, order, ns, q, radius
    ):
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
        self, poles, amps, level, ns, q=0.5, radius=None
    ):
        radius = dist(0., 0.6, ns) if radius is None else radius
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
                clusters, poles_, order, ns, q, radius)
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

    def _clustering(
        self, poles, amps, ns,
        q=0.5, radius=None, min_scale=None
    ):
        radius = dist(0., 0.6, ns) if radius is None else radius
        poles = {k: v.copy() for k, v in poles.items()}
        amps = {k: v.copy() for k, v in amps.items()}
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
                poles, amps, level, ns, q=q, radius=radius)
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

    def find_clusters(
            self, max_order,
            q:float=0.5, radius:float=None, min_scale:float=None):
        amps_set, poles_set = {}, {}
        min_order = self.model._rational._rank
        real_order = -1
        for order in range(min_order, max_order+1):
            if order <= real_order:
                continue
            amps_fit, poles_fit = self.model.fit(max_order=order, tol=0.)
            real_order = len(poles_fit[0]) + 2*len(poles_fit[1])

            amps_ = np.concatenate((amps_fit[0], amps_fit[1]), dtype=complex)
            amps_ = np.linalg.norm(amps_, axis=1)
            poles_ = np.concatenate((poles_fit[0], poles_fit[1]), dtype=complex)
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
            poles_set, amps_set, ns=self.ns,
            q=q, radius=radius, min_scale=min_scale)
        self.min_order_ = min_order
        self.clusters_ = clusters
        return clusters

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
        ax.set_ylim([-1.1, 1.1])

        ax.set_aspect('equal')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        return ax

def pole_pruning(amps, poles, ns, stol=1e-5):
    amps, poles = list(amps), list(poles)
    
    # Remove unstable poles.
    for i in [0, 1]:
        idxs = np.nonzero(np.abs(poles[i]) <= 1)[0]
        poles[i] = poles[i][idxs]
        amps[i] = amps[i][idxs]
    
    # Remove spurious poles.
    avg_norm = 0
    if len(poles[0]) > 0:
        avg_norm += np.sum((amps[0]**2)*(_inner_prod(ns, poles[0])[:, np.newaxis]))
    if len(poles[1]) > 0:
        avg_norm += np.sum(
            amps[1]*np.conj(amps[1])*_inner_prod(ns, poles[1])[:, np.newaxis])
    avg_norm = np.sqrt(np.real(avg_norm))
    for i in [0, 1]:
        idxs = np.nonzero(np.linalg.norm(amps[i], axis=1) >= stol*avg_norm)[0]
        poles[i] = poles[i][idxs]
        amps[i] = amps[i][idxs]

    return tuple(amps), tuple(poles)
