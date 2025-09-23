#!/usr/bin/env python

import logging
from itertools import product
import bisect

import numpy as np
import scipy

logger = logging.getLogger(__name__)


def fit_amplitudes(data, poles):
    # Validate input data.
    data = np.array(data)
    if data.ndim != 3:
        msg = f'Expected a 3D array for data, got an array of dimension {data.ndim}.'
        raise ValueError(msg)

    poles = np.array(poles)
    if poles.ndim != 1:
        msg = f'Expected a 1D array for poles, got an array of dimension {poles.ndim}.'
        raise ValueError(msg)

    # Adjust data using lstsq.
    n_responses, n_exitations, N_acc = data.shape
    ωN = np.exp(-2j * np.pi / N_acc)
    z = ωN**(-np.arange(N_acc))
    data = np.fft.fft(data, norm='forward', axis=-1)
    data = np.reshape(data, (-1, N_acc)).T
    norms = np.linalg.norm(data, axis=0)
    residues, errors, _, _ = np.linalg.lstsq(
            1 / (z[:, np.newaxis] - poles[np.newaxis, :]),
            ωN**(np.arange(N_acc))[:, np.newaxis] * data,
            rcond=None
            )
    errors = np.sqrt(errors) / norms
    errors = np.reshape(errors.T, (n_responses, n_exitations))

    # Compute amplitudes.
    amplitudes = N_acc * residues / (1 - (poles[:, np.newaxis])**N_acc)

    # Go back to original shape.
    residues = np.reshape(residues.T, (n_responses, n_exitations, len(poles)))
    amplitudes = np.reshape(amplitudes.T, (n_responses, n_exitations, len(poles)))

    return residues, amplitudes, errors

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


def _get_max_order(max_order, N):
    if max_order is None:
        max_order_ = N // 2 - 1
    else:
        if max_order > N // 2 - 1:
            msg = f'The number of poles must be less than half of the number of samples. \
            max_order readjusted to {N // 2 - 1}'
            logger.warning(msg, stacklevel=2)
        max_order_ = min(max_order, N // 2 - 1)

    return max_order_

def _update_sets(gS, gG, idx):
    gS['index'].append(gG['index'][idx])
    gS['data'].append(gG['data'][idx])
    gG['index'].pop(idx)
    gG['data'].pop(idx)
    return np.array(gS['index'])

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

    def __init__(self):
        pass

    @staticmethod
    def _initialize_sets(x, rank):
        N = x.shape[0]
        gG = {'index': list(range(N)), 'data': list(x)}
        gS = {'index': [], 'data': []}

        # Choose rank + 1 peaks as initial frequencies.
        abs_v = np.linalg.norm(x, axis=1)
        abs_v = np.concatenate((abs_v, abs_v))
        peaks, h = scipy.signal.find_peaks(
            abs_v,
            height=np.max(abs_v)/5,
            distance=np.max((N//(4*(rank+1)), 2))
        )
        idxs = np.argsort(h['peak_heights'])
        peaks = peaks[idxs]
        peaks = np.unique(peaks%N)
        peaks_ = list(peaks)
        peaks = []
        c = 0
        while c < rank+1:
            try:
                p = peaks_.pop()
            except IndexError:
                break
            peaks.append(p)
            c += 1
        # Generate additional random indices if c < rank + 1.
        if c < rank + 1:
            logger.warning(
                'Not enough peaks found. Adding random indices.', stacklevel=2)
            diff = np.setdiff1d(np.arange(N), peaks, assume_unique=True)
            rng = np.random.default_rng()
            rng.shuffle(diff)
            diff = list(diff)
            while c < rank+1:
                p = diff.pop()
                peaks.append(p)
                c += 1

        gS['index'].extend(gG['index'][p] for p in peaks)
        gS['data'].extend(gG['data'][p] for p in peaks)
        for p in peaks:
            idx = bisect.bisect_left(gG['index'], p)
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
        r = np.array([w[i] * gS['data'][i] for i in range(n_freqs)])
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
    
    def fit(self, x, tol=1e-3, max_order=None, rank_tol=0.01):
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        N, m = x.shape
        if m > 1:
            # Check for rank deficiency.
            s = scipy.linalg.svdvals(x, overwrite_a=False)
            test = np.sqrt(np.cumsum(s**2) / np.sum(s**2))
            test = np.nonzero(test >= 1-rank_tol)[0]
            if len(test) > 1:
                logger.info(f'Rank deficient data.')
            rank = test[0]+1
        else:
            rank = 1

        max_order_ = _get_max_order(max_order, x.shape[0])

        gS, gG = self._initialize_sets(x, rank)
        self.freqs_ = np.array(gS['index'])
        r, w = self._fit(N, gS, gG, rank)

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
                self.freqs_ = _update_sets(gS, gG, idx)
                r, w = self._fit(N, gS, gG, rank)
            step += 1
        
        barycentric = (gS, w)
        p, res = self._normal_form(N, barycentric)

        return p, res, error

class RatAppSym:

    def __init__(self):
        pass

    @staticmethod
    def _initialize_sets(x, rank):
        N = x.shape[0]
        m = N // 2 + 1
        gG = {'index': list(range(m)), 'data': list(x[:m])}
        gS = {'index': [], 'data': []}

        # Choose rank + 1 peaks as initial frequencies.
        abs_v = np.linalg.norm(x, axis=1)
        abs_v = np.concatenate((abs_v, abs_v))
        peaks, h = scipy.signal.find_peaks(
            abs_v,
            height=np.max(abs_v)/5,
            distance=np.max((N//(8*(rank+1)), 2))
        )
        idxs = np.argsort(h['peak_heights'])
        peaks = peaks[idxs]
        peaks = np.unique(peaks%N)
        peaks = peaks[peaks < m]
        peaks_ = list(peaks)
        peaks = []
        c = 0
        while c < rank+1:
            try:
                p = peaks_.pop()
            except IndexError:
                break
            peaks.append(p)
            if (p == 0) or (N%2 == 0 and p == N//2):
                c += 1
            else:
                c += 2
        # Generate additional random indices if c < rank + 1.
        if c < rank + 1:
            logger.warning(
                'Not enough peaks found. Adding random indices.', stacklevel=2)
            diff = np.setdiff1d(np.arange(m), peaks, assume_unique=True)
            rng = np.random.default_rng()
            rng.shuffle(diff)
            diff = list(diff)
            while c < rank+1:
                p = diff.pop()
                peaks.append(p)
                if (p == 0) or (N%2 == 0 and p == N//2):
                    c += 1
                else:
                    c += 2

        gS['index'].extend(gG['index'][p] for p in peaks)
        gS['data'].extend(gG['data'][p] for p in peaks)
        for p in peaks:
            idx = bisect.bisect_left(gG['index'], p)
            gG['index'].pop(idx)
            gG['data'].pop(idx)
        return gS, gG

    @staticmethod
    def _fit(N, gS, gG, rank):
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

    @staticmethod
    def _normal_form(N, barycentric):
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
        poles = (poles_r, poles_c)
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

        if np.linalg.norm(residues[-1]) > 1e-8:
            msg = 'The constant term of rational function is not close to zero.'
            logger.error(msg, stacklevel=2)
        residues = residues[:-1]
        residues = [residues[:lr], residues[lr:lr+lc]+1j*residues[lr+lc:]]

        return poles, tuple(residues)
    
    def fit(self, x, tol=1e-3, max_order=None, rank_tol=0.01):
        x[1:] = 0.5 * (x[1:] + np.conj(x[-1:0:-1]))
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        N, m = x.shape
        if m > 1:
            y = np.concatenate(
                (np.real(x[:N//2+1]), np.imag(x)[1:(N+1)//2]), axis=0)
            # Check for rank deficiency.
            s = scipy.linalg.svdvals(y, overwrite_a=True)
            test = np.sqrt(np.cumsum(s**2) / np.sum(s**2))
            test = np.nonzero(test >= 1-rank_tol)[0]
            if len(test) > 1:
                logger.info(f'Rank deficient data.')
            rank = test[0]+1
            del y
        else:
            rank = 1

        max_order_ = _get_max_order(max_order, x.shape[0])

        gS, gG = self._initialize_sets(x, rank)
        n_freqs = 0
        if gG['index'][0] != 0:
            n_freqs += 1
        if N%2 == 0 and (gG['index'][-1] != N//2):
            n_freqs += 1
        n_freqs = 2*len(gS['index']) - n_freqs
        r, w = self._fit(N, gS, gG, rank)

        logger.debug(f'==== step: 0 ====')
        succeed = False
        step = 1
        while not succeed:
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            error = np.linalg.norm(gG['data'] - r, axis=1)[idx]
            logger.debug(f'error: {error}')

            if error < tol:
                succeed = True
            elif n_freqs >= max_order_ + 1:
                msg = 'Convergence failed after the maximum number of poles is reached.\n'
                msg += f'The error is {error}'
                logger.warning(msg, stacklevel=2)
                break
            else:
                logger.debug(f'==== step: {step} ====')
                _update_sets(gS, gG, idx)
                r, w = self._fit(N, gS, gG, rank)
                if (gS['index'][-1] == 0) or (N%2 == 0 and gS['index'][-1] == N//2):
                    n_freqs += 1
                else:
                    n_freqs += 2
            step += 1

        endsS = []
        if gG['index'][0] != 0:
            endsS.append(gS['index'].index(0))
        if N%2 == 0 and (gG['index'][-1] != N//2):
            endsS.append(gS['index'].index(N//2))
        endsS = np.sort(np.array(endsS, dtype=np.int64))
        barycentric = (gS, endsS, w)
        p, res = self._normal_form(N, barycentric)

        return p, res, error


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
    r = (poles[0][np.newaxis, :]**(t[:, np.newaxis] * fs)) @ amps[0]
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

    def __init__(self, fs:float=1):
        self.fs = fs
        self._rational = RatApp()

    def fit(self, x, max_order=None, RatApp_kwargs=None):
        x = np.copy(x)
        RatApp_kwargs = {} if RatApp_kwargs is None else RatApp_kwargs            
        RatApp_kwargs['max_order'] = max_order
        N = x.shape[0]

        # Add a small damping to stabilize the algorithm.
        false_damping = np.power(2, -2 / N)
        x *= (false_damping**np.arange(N))[:, np.newaxis]
        x = np.fft.fft(x, axis=0, norm='forward')
        ωN = np.exp(-2j * np.pi / N)
        x = (ωN**np.arange(N))[:, np.newaxis] * x
        poles, residues = self._rational.fit(
            x, **RatApp_kwargs)[:2]
        amps = N * residues / (1 - (poles[:, np.newaxis])**N)
        self.amps_ = amps

        # Remove false damping.
        poles *= np.power(2, 2 / N)
        self.poles_ = poles

        return amps, self.freqs_

    @property
    def freqs_(self):
        return np.log(self.poles_) * self.fs


class EspiraR:

    def __init__(self, fs:float=1):
        self.fs = fs
        self._rational = RatAppSym()

    def fit(self, x, max_order=None, RatApp_kwargs=None):
        x = np.copy(x)
        RatApp_kwargs = {} if RatApp_kwargs is None else RatApp_kwargs            
        RatApp_kwargs['max_order'] = max_order
        N = x.shape[0]

        # Add a small damping to stabilize the algorithm.
        false_damping = np.power(2, -2 / N)
        x *= (false_damping**np.arange(N))[:, np.newaxis]
        x = np.fft.fft(x, axis=0, norm='forward')
        ωN = np.exp(-2j * np.pi / N)
        x = (ωN**np.arange(N))[:, np.newaxis] * x
        poles, residues = self._rational.fit(
            x, **RatApp_kwargs)[:2]
        amps = 2*[None]
        poles = list(poles)
        for i in [0, 1]:
            amps[i] = N * residues[i] / (1 - (poles[i][:, np.newaxis])**N)
            # Remove false damping.
            poles[i] *= np.power(2, 2 / N)
        self.amps_ = tuple(amps)
        self.poles_ = tuple(poles)

        return self.amps_, self.freqs_

    @property
    def freqs_(self):
        return tuple([np.log(p) * self.fs for p in self.poles_])
