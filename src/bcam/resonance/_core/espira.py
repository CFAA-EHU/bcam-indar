#!/usr/bin/env python

import logging
from itertools import product
import bisect

import numpy as np
import scipy

logger = logging.getLogger(__name__)


def _validate_data(array):

        array = np.asarray(array)

        # If input is scalar, raise error.
        if array.ndim == 0:
            raise ValueError('Expected 1D or 2D array, got scalar instead.')
        elif array.ndim > 2:
            raise ValueError('Expected 1D or 2D array, got more than 2 dimensions.')
        elif array.ndim == 1:
            array = np.expand_dims(array, axis=1)
    
        return array


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


class RatAppSym:

    def __init__(
        self,
        tol=1e-3,
        max_order=None
    ):
        self.tol = tol
        self.max_order = max_order

        self._removed = False

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

    def fit(self, x, tol=0.01):
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
            test = np.nonzero(test >= 1-tol)[0]
            if len(test) > 1:
                logger.info(f'Rank deficient data.')
            rank = test[0]+1
            del y
        else:
            rank = 1

        max_order_ = _get_max_order(self.max_order, x.shape[0])

        gS, gG = self._initialize_sets(x, rank)
        n_poles = 0
        for p in gS['index']:
            if (p == 0) or (N%2 == 0 and p == N//2):
                n_poles += 1
            else:
                n_poles += 2
        self.freqs_ = np.array(gS['index'])
        r, w = self._fit(N, gS, gG, rank)

        logger.debug(f'==== step: 0 ====')
        succeed = False
        step = 1
        while not succeed:
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            error = np.linalg.norm(gG['data'] - r, axis=1)[idx]
            logger.debug(f'error: {error}')

            if error < self.tol:
                succeed = True
            elif n_poles >= max_order_ + 1:
                msg = 'Convergence failed after the maximum number of poles is reached.\n'
                msg += f'The error is {error}'
                logger.warning(msg, stacklevel=2)
                break
            else:
                logger.debug(f'==== step: {step} ====')
                self.freqs_ = _update_sets(gS, gG, idx)
                r, w = self._fit(N, gS, gG, rank)
                if (gS['index'][-1] == 0) or (N%2 == 0 and gS['index'][-1] == N//2):
                    n_poles += 1
                else:
                    n_poles += 2
            step += 1
        self.error_ = error

        # TODO: correct this representation.
        barycentric = (gS, w)
        p, res = _normal_form(N, barycentric)
        self.poles_ = p
        self.residues_ = res

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

    def remove_spurious(self, rtol=1e-6):
        if not self._removed:
            r = np.linalg.norm(self.residues_, ord=2, axis=1)**2
            r = r / np.sum(r)
            idxs = np.argsort(r)
            stop = np.nonzero(np.sqrt(np.cumsum(r[idxs])) < rtol)[0]
            if len(stop) == 0:
                return
            else:
                stop = stop[-1]
                idxs = idxs[:stop+1]
                self.poles_ = np.delete(self.poles_, idxs)
                self.residues_ = np.delete(self.residues_, idxs, axis=0)
            self._removed = True

    def eval(self, z, pole_idxs=None):
        M = len(self.poles_)
        pole_idxs = np.arange(M) if pole_idxs is None else pole_idxs
        return rational_function(
            self.poles_[pole_idxs],
            self.residues_[pole_idxs],
            z)

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
        diff = np.setdiff1d(np.arange(m), peaks, assume_unique=True)
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

def _update_sets(gS, gG, idx):
    gS['index'].append(gG['index'][idx])
    gS['data'].append(gG['data'][idx])
    gG['index'].pop(idx)
    gG['data'].pop(idx)
    return np.array(gS['index'])
    
def _normal_form(N, barycentric):
    gS, w = barycentric
    S = np.array(gS['index'])
    gS = np.array(gS['data'])
    lastS = S[-1]
    M = len(S) - 1
    idxs = np.argsort(S)
    S = S[idxs]
    w = w[idxs]
    gS = gS[idxs]
    ωN = np.exp(-2j * np.pi / N)

    a = np.zeros((M+2, M+2), dtype=np.complex128)
    a[1:, 0] = 1
    a[0, 1:] = w
    a[1:, 1:] = np.diag(ωN**(-S))

    b = np.eye(M+2, dtype=np.complex128)
    b[0, 0] = 0

    poles, _ = scipy.linalg.eig(a, b, overwrite_a=True, overwrite_b=True)
    poles = poles[2:]

    idx = np.searchsorted(S, lastS)
    S = np.delete(S, idx)
    gS = np.delete(gS, idx, axis=0)
    C = ωN**(-S)[:, np.newaxis] - poles[np.newaxis, :]
    C = 1 / C
    residues = scipy.linalg.solve(C, gS)

    # Remove instable poles.
    idxs = np.nonzero(np.abs(poles) <= 1)
    poles, residues = poles[idxs], residues[idxs]
 
    return poles, residues

class RationalApproximation:
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

    def __init__(self, tol=1e-3, mode='normal', max_order=None):
        self.tol = tol
        self.mode = mode
        self.max_order = max_order

        self._removed = False

    def fit(self, x, tol=0.01):
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        N, m = x.shape
        if m > 1:
            # Check for rank deficiency.
            s = scipy.linalg.svdvals(x, overwrite_a=False)
            test = np.sqrt(np.cumsum(s**2) / np.sum(s**2))
            test = np.nonzero(test >= 1-tol)[0]
            if len(test) > 1:
                logger.info(f'Rank deficient data.')
            rank = test[0]+1
        else:
            rank = 1

        max_order_ = _get_max_order(self.max_order, x.shape[0])

        gS, gG = _initialize_sets(x, rank)
        self.freqs_ = np.array(gS['index'])
        r, w = self._fit(N, gS, gG, rank)

        logger.debug(f'==== step: 0 ====')
        succeed = False
        step = 1
        while not succeed:
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            error = np.linalg.norm(gG['data'] - r, axis=1)[idx]
            logger.debug(f'error: {error}')

            if error < self.tol:
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
        self.error_ = error
        
        barycentric = (gS, w)
        p, res = _normal_form(N, barycentric)
        self.poles_ = p
        self.residues_ = res

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

    def remove_spurious(self, rtol=1e-6):
        if not self._removed:
            r = np.linalg.norm(self.residues_, ord=2, axis=1)**2
            r = r / np.sum(r)
            idxs = np.argsort(r)
            stop = np.nonzero(np.sqrt(np.cumsum(r[idxs])) < rtol)[0]
            if len(stop) == 0:
                return
            else:
                stop = stop[-1]
                idxs = idxs[:stop+1]
                self.poles_ = np.delete(self.poles_, idxs)
                self.residues_ = np.delete(self.residues_, idxs, axis=0)
            self._removed = True

    def eval(self, z, pole_idxs=None):
        M = len(self.poles_)
        pole_idxs = np.arange(M) if pole_idxs is None else pole_idxs
        return rational_function(
            self.poles_[pole_idxs],
            self.residues_[pole_idxs],
            z)


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

    def __init__(
            self,
            tol=1e-3,
            max_order=None,
            fs=1,
            mode='complex'
        ):
        mode_map = {
            'complex': 'normal',
            'real': 'symmetric'
        }
        if mode not in mode_map.keys():
            msg = f'Expected mode either "complex" or "real", got {mode} instead.'
            raise ValueError(msg)
        if fs <= 0:
            raise ValueError('Expected positive sampling frequency.')

        self.tol = max(tol, 0)
        self.max_order = max_order
        self.fs = fs
        self.mode = mode

        self.rational_ = RationalApproximation(
            tol=self.tol,
            mode=mode_map[mode],
            max_order=self.max_order
        )

    def fit(self, X):
        X = _validate_data(X)
        X = np.copy(X)
        N = X.shape[0]
        # Add a small damping to stabilize the algorithm.
        false_damping = np.power(2, -2 / N)
        X *= (false_damping**np.arange(N))[:, np.newaxis]
        X = np.fft.fft(X, axis=0, norm='forward')
        ωN = np.exp(-2j * np.pi / N)
        X = (ωN**np.arange(N))[:, np.newaxis] * X
        self.rational_.fit(X)
        # Remove false damping.
        self.rational_.poles_ *= np.power(2, 2 / N)
        self._N = N

        return self

    def remove_spurious(self, rtol=1e-6):
        self.rational_.remove_spurious(rtol=rtol)
    
    @property
    def poles_(self):
        return self.rational_.poles_

    @property
    def Z_(self):
        return np.log(self.rational_.poles_) * self.fs

    @property
    def amplitudes_(self):
        '''
        Shape (n_poles, n_signal_components).
        '''
        N = self._N
        poles = self.rational_.poles_
        residues = self.rational_.residues_
        return N * residues / (1 - (np.power(2, -2 / N) * poles[:, np.newaxis])**N)

    def eval(self, t, freq_idxs=None):
        M = len(self.rational_.poles_)
        freq_idxs = np.arange(M) if freq_idxs is None else freq_idxs
        return exp_sum(
            self.rational_.poles_[freq_idxs],
            self.amplitudes_[freq_idxs],
            t,
            fs=self.fs
        )
