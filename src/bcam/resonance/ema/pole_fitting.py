#!/usr/bin/env python

import logging
import bisect

import numpy as np
import scipy
import pandas as pd
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from .helpers import Poles, HCoeffs
from bcam.resonance import Rational, ExpSum

logger = logging.getLogger(__name__)


# ========================
# Rational Approximation
# ========================

def pairing(A, B, ns:int, delta:float=None):

    delta = 0.8 if delta is None else delta

    # Compute distance matrix.
    if len(A) == 0 or len(B) == 0:
        return {}
    Pa, Pb = np.meshgrid(A, B, indexing='ij')
    D = dist(Pa, Pb, ns)
    del Pa, Pb

    closests = {}
    while np.min(D) < np.inf:
        # Get index of minimum distance without flattening.
        idx = np.unravel_index(np.argmin(D), D.shape)
        if D[idx] < dist(0., delta, ns):
            p_ = B[idx[1]]
            closests[idx[0]] = p_

            D[idx[0], :] = np.inf
            D[:, idx[1]] = np.inf
        else:
            D[idx[0], :] = np.inf

    return closests

# Define helpers for rational fitting.

def _get_residues(
        y, parity, poles, d:bool=False, cond=None, lapack_driver=None):
    N, rank = y.shape
    ns = 2*(N-1) + parity

    u = np.exp(-2j*np.pi/ns)
    # Construct Cauchy matrix.
    Cr = 1/(u**(-np.arange(N)[:, np.newaxis]) - poles.real[np.newaxis, :])
    if d:
        Cr = np.concatenate((Cr, np.ones(shape=(N, 1))), axis=1)
    Ci1 = 1/(u**(-np.arange(N)[:, np.newaxis]) - poles.cx[np.newaxis, :])
    Ci2 = 1/(u**(-np.arange(N)[:, np.newaxis]) - np.conj(poles.cx)[np.newaxis, :])
    C = np.concatenate(
        (Cr, Ci1 + Ci2, 1j*(Ci1 - Ci2)),
        axis=1
    )

    # Auxiliary indices for separating real and imaginary parts.
    ends = [0]
    if parity == 0:
        ends.append(N-1)
    ends = np.array(ends, dtype=np.int64)
    last = N-1+parity

    # Separate real and imaginary parts.
    C = np.concatenate(
        (np.real(C[ends]), np.sqrt(2)*np.real(C[1:last]), np.sqrt(2)*np.imag(C[1:last])),
        axis=0, dtype=float)
    y = np.concatenate(
        (np.real(y[ends]), np.sqrt(2)*np.real(y[1:last]), np.sqrt(2)*np.imag(y[1:last])),
        axis=0, dtype=float)

    # Solve for r.
    r = scipy.linalg.lstsq(
        C, y, overwrite_a=True, overwrite_b=True,
        cond=cond, lapack_driver=lapack_driver)[0]
    lr, lc = len(poles.real), len(poles.cx)
    if d:
        d = r[lr]
        r = HCoeffs(real=r[:lr], cx=r[lr+1:lr+1+lc] + 1j*r[lr+1+lc:])
    else:
        d = np.zeros(rank)
        r = HCoeffs(real=r[:lr], cx=r[lr:lr+lc] + 1j*r[lr+lc:lr+2*lc])

    return r, d

def _pole_pruning(poles:Poles, r:HCoeffs, tol:float):
    for part in ['real', 'cx']:
        idxs = np.nonzero(
            np.linalg.norm(getattr(r, part), axis=1) > tol)[0]
        setattr(poles, part, getattr(poles, part)[idxs])

    return poles

def _get_max_order(ns, rank):
    # Counting complex paramaters in a real time series
    # and in a symmetric rational function, we have that:
    max_order = int(np.floor(rank*ns/(rank+1)))

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
            compute_r:bool=True,
            prune_tol:float=0.,
            d:bool=False,
            lapack_driver:str=None,
            cond:float=None
        ):
        self.order = order
        self.compute_r = compute_r
        self.prune_tol = prune_tol
        self.d = d
        self.lapack_driver = lapack_driver
        self.cond = cond

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
        ns = 2*(N-1) + parity
        ωN = np.exp(-2j * np.pi / ns)

        # Locate zero and N-1 in S_ and G_.
        eS, eG = [], []
        innS, innG = [0, len(S_)], [0, len(G_)]
        if S_[0] == 0:
            eS.append(0)
            innS[0] = 1
        else:
            eG.append(0)
            innG[0] = 1

        if parity == 1:
            pass
        elif S_[-1] == N-1:
            eS.append(n_S-1)
            innS[1] = n_S - 1
        else:
            eG.append(n_G-1)
            innG[1] = n_G - 1
        innS, innG = np.s_[innS[0]:innS[1]], np.s_[innG[0]:innG[1]]
        eS = np.array(eS, dtype=np.int64)
        eG = np.array(eG, dtype=np.int64)
        M = 2*len(S_) - len(eS)

        # Construct Loewner matrix L.
        Cr = 1/(ωN**(-S_[eS])[:, np.newaxis]-ωN**(-G_)[np.newaxis, :])
        Lr = (gS_[eS][:, np.newaxis] - gG_[np.newaxis, :]) * Cr[..., np.newaxis]
        C1 = 1/(ωN**(-S_[innS])[:, np.newaxis]-ωN**(-G_)[np.newaxis, :])
        L1 = (gS_[innS][:, np.newaxis] - gG_[np.newaxis, :]) * C1[..., np.newaxis]
        C2 = 1/(ωN**(S_[innS])[:, np.newaxis]-ωN**(-G_)[np.newaxis, :])
        L2 = (np.conj(gS_[innS])[:, np.newaxis] - gG_[np.newaxis, :]) * C2[..., np.newaxis]
        L = np.concatenate((Lr, L1+L2, 1j*(L1-L2)), axis=0)
        del Lr, L1, L2, G_, gG_

        # Separate real and imaginary parts.
        L = np.concatenate(
            (np.real(L[:, eG]), np.sqrt(2)*np.real(L[:, innG]), np.sqrt(2)*np.imag(L[:, innG])),
            axis=1, dtype=float
        )
        L = L.reshape(M, -1).T

        if (not self.d) and (M > rank):
            # Construct inclusion matrix into the space that satisfies (3.14) of [From ESPRIT to ESPIRA].
            tmpS = np.concatenate(
                (np.real(gS_[eS]), 2*np.real(gS_[innS]), -2*np.imag(gS_[innS])),
                axis=0, dtype=float
            )
            In = scipy.linalg.qr(tmpS, pivoting=True)[0]
            In = In[:, rank:]

            # Compute weights to find best rational approximation.
            L = L @ In

        # Compute weights to find best rational approximation.
        eigval, w = scipy.linalg.svd(
            L,
            overwrite_a=True,
            full_matrices=False)[1:]
        logger.debug(f'Lowest eigenvalue: {eigval[-1]}')

        if (self.d) or (M <= rank):
            w = w[-1]
        else:
            w = In @ w[-1]

        w_ = np.zeros_like(S_, dtype=complex)
        w_[innS] = w[len(eS):len(S_)] + 1j*w[len(S_):]
        w_[eS] = w[:len(eS)]
        w = w_

        # Compute rational function at G frequencies.
        rp = w[:, np.newaxis] * gS_
        p = Cr.T@rp[eS] + C1.T@rp[innS] + C2.T@np.conj(rp[innS])
        q = Cr.T@w[eS] + C1.T@w[innS] + C2.T@np.conj(w[innS])

        return p/(q[:, np.newaxis]), w

    def _get_poles(self, barycentric, N, parity):
        ns = 2*(N-1) + parity
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
        ωN = np.exp(-2j * np.pi / ns)

        w = np.concatenate((w, np.conj(w[e[0]:e[1]])))
        a = np.zeros((M+2, M+2), dtype=np.complex128)
        a[1:, 0] = 1
        a[0, 1:] = w
        a[1:, 1:] = np.diag(ωN**(np.concatenate((-S, S[e[0]:e[1]]))))

        b = np.eye(M+2, dtype=np.complex128)
        b[0, 0] = 0

        poles = scipy.linalg.eigvals(a, b, overwrite_a=True)
        poles = poles[2:]

        poles = Poles.from_raw(poles)

        dim = poles.count()
        # Check that the number of poles is correct.
        if dim != M:
            msg = f'Expected {M} poles, got {dim} instead.'
            raise ValueError(msg)

        return poles

    def _fit(self, y, parity, indices):
        N = y.shape[0]
        gS, gG = self._initialize_sets(y, indices)
        n_freqs = self.count_freqs(gS['index'], N, parity)

        step = 1
        while n_freqs-1 < self.order:
            logger.debug(f'==== step: {step} ====')
            r, w = self._get_weights(gS, gG, parity)
            idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
            fr = gG['index'][idx]
            _update_sets(gS, gG, idx)

            if (fr == 0) or (parity == 0 and fr == N-1):
                n_freqs += 1
            else:
                n_freqs += 2
            step += 1
        self.n_poles_ = n_freqs-1
        logger.debug(f'==== step: {step} ====')
        r, w = self._get_weights(gS, gG, parity)
        idx = np.argmax(np.linalg.norm(gG['data'] - r, axis=1))
        fr = gG['index'][idx]
        if (fr == 0) or (parity == 0 and fr == N-1):
            n_freqs += 1
        else:
            n_freqs += 2

        # Store indices for reuse.
        if len(gS['index'])+1 > len(self.indices_):
            self.indices_ = gS['index'].copy()
            self.indices_.append(fr)
            self.indices_ = np.array(self.indices_)

        # Find poles and residues.
        barycentric = (np.array(gS['index']), w)
        poles = self._get_poles(barycentric, N, parity)

        return poles

    def fit(self, y, parity:bool=None):
        y = np.asarray(y)

        # Determine parity if not given.
        if parity is None:
            eps = np.finfo(y.dtype).eps
            tiny = np.imag(y[-1, :])
            parity = np.max(np.abs(tiny)) > 100*eps
        parity = int(parity)
        N, rank = y.shape
        ns = 2*(N-1)+parity

        max_order = _get_max_order(ns, rank)
        if self.order is None:
            self.order = rank
        elif self.order > max_order:
            msg = f'The order exceeds the maximum recommended order {max_order}.'
            logger.warning(msg, stacklevel=2)

        # If there are already indices from a previous fit, we reuse them.
        if hasattr(self, 'indices_'):
            indices = self.indices_
            n_freqs = self.count_freqs(indices, N, parity)
            if n_freqs-1 > self.order:
                n_freqs = 0
                i = 0
                for idx in indices:
                    if (idx == 0) or (parity == 0 and idx == N-1):
                        n_freqs += 1
                    else:
                        n_freqs += 2
                    i += 1
                    if n_freqs-1 >= self.order:
                        break
                indices = indices[:i]
        else:
            indices = [np.argmax(np.linalg.norm(y, axis=1))]
            self.indices_ = indices

        if self.order > max_order:
            msg = f'The order exceeds the maximum allowed order {max_order}.'
            raise ValueError(msg)

        poles = self._fit(y, parity, indices)
        self.poles_ = poles

        if self.prune_tol > 0.:
            self.r_, self.d_ = _get_residues(
                y, parity, poles, self.d,
                cond=self.cond, lapack_driver=self.lapack_driver)
            self.poles_ = _pole_pruning(self.poles_, self.r_, self.prune_tol)

        if self.compute_r:
            self.r_, self.d_ = _get_residues(
                y, parity, self.poles_, self.d,
                cond=self.cond, lapack_driver=self.lapack_driver)
            self._predict = Rational(
                self.poles_.full(), self.r_.full().T, self.d_)
        else:
            self.r_, self.d_ = None, None
            self._predict = lambda x : np.nan

        return self

    def predict(self, X):
        if self.r_ is None:
            msg = 'Cannot compute rational function values without residues. \
                Set compute_r=True when initializing the estimator.'
            logger.warning(msg)
        
        return self._predict(X).T

class VF(BaseEstimator):

    def __init__(
        self,
        *,
        order:int=1,
        poles:np.typing.ArrayLike=None,
        niter:int=1,
        compute_r:bool=True,
        d:bool=False,
        prune_tol:float=0.,
        cond:float=None,
        lapack_driver:str=None
    ):
        self.order = order
        self.poles = poles
        self.niter = niter
        self.compute_r = compute_r
        self.d = d
        self.prune_tol = prune_tol
        self.cond = cond
        self.lapack_driver = lapack_driver

    def _get_weights(self, y, parity, poles):
        N, rank = y.shape
        ns = 2*(N-1)+parity

        u = np.exp(2j*np.pi/ns)
        # Compute Cauchy matrix.
        Cr = 1/(u**np.arange(N)[:, np.newaxis] - poles.real[np.newaxis, :])
        Cr = np.concatenate((Cr, np.ones(shape=(N, 1))), axis=1)
        C1 = 1/(u**np.arange(N)[:, np.newaxis] - poles.cx[np.newaxis, :])
        C2 = 1/(u**np.arange(N)[:, np.newaxis] - np.conj(poles.cx)[np.newaxis, :])

        # Add constraint that rational function is symmetric.
        C = np.concatenate(
            (Cr, C1 + C2, 1j*(C1 - C2)),
            dtype=complex, axis=1)

        # Auxiliary indices for separating real and imaginary parts.
        ends = [0]
        if parity == 0:
            ends.append(N-1)
        ends = np.array(ends, dtype=np.int64)
        last = N-1+parity

        # Constraint vector.
        const_r = np.sum(Cr[ends], axis=0) + 2*np.sum(Cr[1:last], axis=0)
        const_r = np.real(const_r)
        const_c = np.sum(C1[ends], axis=0)
        const_c_r = const_c + np.sum(C1[1:last] + C2[1:last], axis=0)
        const_c_r = 2*np.real(const_c_r)
        const_c_i = const_c + np.sum(C1[1:last] - C2[1:last], axis=0)
        const_c_i = -2*np.imag(const_c_i)
        const = np.concatenate((const_r, const_c_r, const_c_i))/ns
        In = scipy.linalg.qr(const[:, np.newaxis], pivoting=True)[0]
        In = In[:, 1:]
        del Cr, C1, C2, const

        # Construct right-most columns of system matrix. Shape = (ns, n_poles, rank).
        R = -y[:, np.newaxis, :] * (C@In)[..., np.newaxis]

        if not self.d:
            n_real = len(poles.real)
            idxs = list(range(n_real)) + list(range(n_real+1, C.shape[1]))
            C = C[:, np.array(idxs)]

        # Separate real and imaginary parts.
        C = np.concatenate(
            (np.real(C[ends]), np.sqrt(2)*np.real(C[1:last]), np.sqrt(2)*np.imag(C[1:last])),
            axis=0, dtype=float
        )
        R = np.concatenate(
            (np.real(R[ends]), np.sqrt(2)*np.real(R[1:last]), np.sqrt(2)*np.imag(R[1:last])),
            axis=0, dtype=float
        )
        R = R.reshape((ns, -1))

        # Data to approximate. Shape = (ns, rank).
        b = np.concatenate(
            (np.real(y[ends]), np.sqrt(2)*np.real(y[1:last]), np.sqrt(2)*np.imag(y[1:last])),
            axis=0, dtype=float
        )

        if (self.lapack_driver is None) or (self.lapack_driver in ['gelsd', 'gelss']):
            U, sigma = scipy.linalg.svd(
                C, full_matrices=False, overwrite_a=True)[:2]
        elif self.lapack_driver == 'gelsy':
            U, sigma = scipy.linalg.qr(
                C, pivoting=True, overwrite_a=True, mode='economic')[:2]
            sigma = np.diag(np.abs(sigma))
        else:
            msg = f'Unknown LAPACK driver. Expected one of "gelsd", "gelss", or "gelsy", got {self.lapack_driver} instead.'
            raise ValueError(msg)
        del C

        # Determine rank of R.
        if self.cond is not None:
            R_rank = np.sum(sigma > sigma[0]*self.cond)
        else:
            # Get machine's epsilon for the data type of R.
            eps = np.finfo(R.dtype).eps
            R_rank = np.sum(sigma > 100*eps*sigma[0])
        U = U[:, :R_rank]
        S = R - U@(U.T@R)
        b = b - U@(U.T@b)
        del R, U, sigma
        S = S.reshape((ns, self.order, rank))
        S = S.transpose(1, 2, 0).reshape((-1, ns*rank)).T

        # Solve reduced LS problem.
        b = b.T.flatten().T
        t = scipy.linalg.lstsq(
            S, b, cond=self.cond, lapack_driver=self.lapack_driver,
            overwrite_a=True, overwrite_b=True)[0]
        t = In @ t

        n_real = len(poles.real) + 1 # +1 for the constant term.
        n_complex = len(poles.cx)
        w_r = t[:n_real-1]
        d = t[n_real-1] + 1
        w_c = t[n_real:n_real+n_complex] + 1j*t[n_real+n_complex:]

        return w_r, w_c, d

    def _get_poles(self, w_r, w_c, d, poles):
        w = np.concatenate((w_r, w_c, np.conj(w_c)), dtype=complex)
        M = np.diag(poles.full())
        M += -np.repeat(w[:, np.newaxis], M.shape[0], axis=1)/d

        # Compute poles as eigenvalues of M.
        poles = scipy.linalg.eigvals(M, overwrite_a=True)
        poles = Poles.from_raw(poles)

        if 2*len(poles.cx) + len(poles.real) != self.order:
            msg = f'Problems during computation of poles. \
                Expected {self.order} poles, got {2*len(poles.cx) + len(poles.real)} instead.'
            raise ValueError(msg)

        return poles

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

        max_order = _get_max_order(ns, rank)
        if self.order > max_order:
            msg = f'The order exceeds the maximum recommended order {max_order}.'
            logger.warning(msg, stacklevel=2)

        # Set initial poles.
        n_poles = self.order
        self.n_poles_ = n_poles # For compatibility with AAA.
        if self.poles is None:
            poles = 0.9*np.exp(1j*np.pi*np.arange(1, n_poles//2+1)/(n_poles//2+1))
            poles = Poles(cx=poles)
            if self.order % 2 == 1:
                poles.real = [0.9]
        elif isinstance(self.poles, tuple):
            if len(self.poles) == 2:
                poles = Poles(real=self.poles[0], cx=self.poles[1])
            else:
                msg = f'Expected a tuple of length 2 for poles, got a tuple of length {len(self.poles)} instead.'
                raise ValueError(msg)
        else:
            poles = Poles.from_raw(self.poles)

        if 2*len(poles.cx) + len(poles.real) != self.order:
            msg = f'Expected {self.order} poles, got {2*len(poles.cx) + len(poles.real)} instead.'
            raise ValueError(msg)

        for _ in range(self.niter):
            weights = self._get_weights(y, parity, poles)
            poles = self._get_poles(*weights, poles)
        self.poles_ = poles

        if self.prune_tol > 0.:
            self.r_, self.d_ = _get_residues(
                y, parity, poles, self.d,
                cond=self.cond, lapack_driver=self.lapack_driver)
            self.poles_ = _pole_pruning(self.poles_, self.r_, self.prune_tol)

        if self.compute_r:
            self.r_, self.d_ = _get_residues(
                y, parity, self.poles_, self.d,
                cond=self.cond, lapack_driver=self.lapack_driver)
            self._predict = Rational(
                self.poles_.full(), self.r_.full().T, self.d_)
        else:
            self.r_, self.d_ = None, None
            self._predict = lambda x : np.nan

        return self

    def predict(self, X):
        if self.r_ is None:
            msg = 'Cannot compute rational function values without residues. \
                Set compute_r=True when initializing the estimator.'
            logger.warning(msg)

        return self._predict(X).T


# ===============================
# Exponential Sums Decomposition
# ===============================

class SuperResolution(BaseEstimator):

    def __init__(
            self,
            *,
            order:int=2,
            rational_fitter={'method': 'AAA'},
            damping:float=0.,
            fs:float=1.,
            prune_tol:float=0.,
            compute_amps:bool=True
        ):
        self.order = order
        self.rational_fitter = rational_fitter
        self.damping = damping
        self.fs = fs
        self.prune_tol = prune_tol
        self.compute_amps = compute_amps

    @property
    def n_poles_(self):
        return self._rational_fitter.n_poles_

    @property
    def exps_(self):
        return np.emath.log(self.poles_.full())*self.fs

    def _get_amps(self, y):
        ns = y.shape[0]

        # Construct Cauchy matrix.
        Vr = self.poles_.real[np.newaxis, :]**(np.arange(ns)[:, np.newaxis])
        Vi = self.poles_.cx[np.newaxis, :]**(np.arange(ns)[:, np.newaxis])
        V = np.concatenate(
            (Vr, 2*np.real(Vi), -2*np.imag(Vi)),
            axis=1, dtype=float)

        # Solve for amplitudes.
        a = scipy.linalg.lstsq(
            V, y, overwrite_a=True, overwrite_b=True)[0]
        lr, lc = len(self.poles_.real), len(self.poles_.cx)
        amps = HCoeffs(real=a[:lr], cx=a[lr:lr+lc] + 1j*a[lr+lc:lr+2*lc])

        return amps

    def fit(self, y):
        y = np.asarray(y)
        ns = y.shape[0]
        N = ns//2+1

        if not hasattr(self, '_rational_fitter'):
            method = self.rational_fitter.get('method')
            r_dict = {k: item for k, item in self.rational_fitter.items() if k != 'method'}
            if method == 'AAA':
                self._rational_fitter = AAA(**r_dict)
            elif method == 'VF':
                self._rational_fitter = VF(**r_dict)
            else:
                msg = f'Unknown rational fitting method {method}.'
                raise ValueError(msg)
        self._rational_fitter.set_params(
            order=self.order, compute_r=False,
            prune_tol=0., d=False)

        if self.damping > 0.:
            y *= np.pow(self.damping, np.arange(ns)/(ns-1))[:, np.newaxis]

        ωN = np.exp(-2j*np.pi/ns)
        self._rational_fitter.fit(
            np.fft.rfft(y, axis=0)*(ωN**np.arange(N))[:, np.newaxis],
            parity=ns%2)

        self.poles_ = self._rational_fitter.poles_
        if self.damping > 0.:
            self.poles_.real *= np.pow(self.damping, -1/(ns-1))
            self.poles_.cx *= np.pow(self.damping, -1/(ns-1))

         # Remove unstable poles.
        for part in ['real', 'cx']:
            p_ = getattr(self.poles_, part)
            idxs = np.nonzero(np.abs(p_) < 1+1e-10)[0]
            setattr(self.poles_, part, p_[idxs])

        if self.prune_tol > 0.:
            self.amps_= self._get_amps(y)
            self.poles_ = _pole_pruning(self.poles_, self.amps_, self.prune_tol)

        if self.compute_amps:
            self.amps_ = self._get_amps(y)
            self._exp_sum = ExpSum(
                np.emath.log(self.poles_.full()), self.amps_.full().T)
        else:
            self.amps_ = None
            self._exp_sum = lambda x: np.nan

        return self

    def predict(self, X):
        if self.amps_ is None:
            msg = 'Cannot compute exponential sum values without amplitudes. \
                Set compute_amps=True when initializing the estimator.'
            logger.warning(msg)

        return np.real(self._exp_sum(X*self.fs)).T

# Stabilization algorithm

def _geometric_sum(r:float, ns:int):
    if ns <= 5:
        return sum(r**k for k in range(ns))

    delta = 1e-2
    eta = 1e-5

    # For r away from 1.
    idxs = np.nonzero(np.abs(1-r) > delta)[0]
    r[idxs] = (1 - r[idxs]**ns) / (1 - r[idxs])

    idxs = np.setdiff1d(
        np.arange(len(r)), idxs, assume_unique=True)
    if len(idxs) == 0:
        return r

    # For r**ns large.
    eps = np.finfo(r.dtype).eps

    sub_idxs = np.nonzero(np.abs(r[idxs])**ns > eta)[0]
    l = np.log(r[idxs[sub_idxs]])
    l = np.where(l == 0, eps, l)
    r[idxs[sub_idxs]] = np.sqrt(r[idxs[sub_idxs]]**(ns-1))
    r[idxs[sub_idxs]] *= np.sinh(ns*l/2) / np.sinh(l/2)

    r_idxs = np.setdiff1d(
        np.arange(len(idxs)), sub_idxs, assume_unique=True)
    if len(r_idxs) == 0:
        return r

    # For r close to 1 and r**ns small.
    L = int(np.floor(np.sqrt(ns)))
    M = ns//L
    r_ = r[idxs[r_idxs]].copy()
    r[idxs[r_idxs]] = _geometric_sum(r_.copy()**L, M)
    r[idxs[r_idxs]] *= _geometric_sum(r_.copy(), L)
    r[idxs[r_idxs]] += (r_**(M*L))*_geometric_sum(r_, ns%L)

    return r

def _inner_prod(ns:int, x, y=None):
    if y is None:
        r = np.abs(x)**2
    else:
        r = x*np.conj(y)
    shape = r.shape
    r = r.flatten()

    # Evaluate geometric sum recursively.
    r = _geometric_sum(r, ns) / ns

    r = r.reshape(shape)
    return r

def dist(x, y, ns:int):
    floats = isinstance(x, float) and isinstance(y, float)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    abs_x = np.sqrt(np.real(_inner_prod(ns, x)))
    abs_y = np.sqrt(np.real(_inner_prod(ns, y)))
    t1 = (abs_x - abs_y)**2
    t2 = 2*((abs_x*abs_y) - np.real(_inner_prod(ns, x, y)))

    if floats:
        return np.sqrt(t1 + t2)[0]
    else:
        return np.sqrt(t1 + t2)

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
        radius = dist(0., 0.5, ns) if radius is None else radius

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

    def fit(self, y):
        ns, rank = y.shape
        self._ns = ns

        # Validate orders.
        max_order = self.max_order
        min_order = 2
        max_order_ = _get_max_order(ns, rank)
        if (max_order is None) or (max_order > max_order_):
            max_order = max_order_

        n_poles = -1
        amps_set, poles_set = {}, {}
        for order in range(min_order, max_order+1, 2):
            self.model.set_params(order=order)
            self.model.fit(y)
            n_poles = self.model.n_poles_

            amps_ = np.concatenate(
                (self.model.amps_.real, self.model.amps_.cx),
                dtype=complex,
                axis=0)
            amps_ = np.linalg.norm(amps_, axis=1)
            poles_ = np.concatenate(
                (self.model.poles_.real, self.model.poles_.cx), dtype=complex)
            idxs = np.argsort(amps_)[::-1]
            amps_ = amps_[idxs]
            poles_ = poles_[idxs]
            amps_set[n_poles] = amps_
            poles_set[n_poles] = poles_

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
