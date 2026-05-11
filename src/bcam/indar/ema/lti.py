'''
Functions for Linear Time Invariant (LTI) models.
'''

import logging

import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data

logger = logging.getLogger(__name__)


def _convT(x, h):
    r'''
    Trasnpose convolution.

    Perform operation :math:`\sum_{k\le i < N} x_{i-k}h_i`, for :math:`i=0, \ldots, N-1`.
    
    Parameters
    ----------
    x : 2d array-like of shape (n_repetitions, n_samples)
    h : 1d array-like of shape (K, n_samples)

    Returns
    -------
    v : 2d array-like of shape (K, n_samples)
    '''
    _, N = x.shape

    v = scipy.signal.fftconvolve(
            h[:, :, ::-1],
            x[np.newaxis, :, :],
            axes=-1)[..., :N]
    v = v[:, :, ::-1]
    v = np.sum(v, axis=1)

    return v

class _Dloss(scipy.sparse.linalg.LinearOperator):
    r'''
    Derivative of the loss function to find the kernel of best linear fit.
    
    Parameters
    ----------
    x : 1d or 2d array-like of shape (n_repetitions, n_samples)
    '''
    
    def __init__(self, x, beta=0., mode='g'):
        self.x = x
        self.beta = beta
        self.mode = mode
        
        n_rep, N = x.shape
        M = n_rep*N

        if beta > 0.:
            if mode == 'g':
                M += N-1
            elif mode == 'a':
                M += N-2
            else:
                raise ValueError('Mode must be `g` or `a`.')

        super().__init__(shape=(M, N), dtype=x.dtype)

    def _matmat(self, h):
        n_rep, N = self.x.shape
        h = h.T
        v = scipy.signal.fftconvolve(
            h[:, np.newaxis, :],
            self.x[np.newaxis, :, :],
            axes=-1)[..., :N]
        v = v.reshape((-1, n_rep*N))
        if self.beta > 0.:
            p = h[:, 1:] - h[:, :-1]
            if self.mode == 'a':
                p = p[:, 1:]
            v = np.concatenate((v, self.beta * p), axis=1)

        return v.T

    def _rmatmat(self, h):
        n_rep, N = self.x.shape
        h0 = h[:n_rep*N].T
        h0 = h0.reshape((-1, n_rep, N))

        v = _convT(self.x, h0)

        if self.beta > 0.:
            h1 = h[n_rep*N:].T
            p = np.zeros((h1.shape[0], N))
            if self.mode == 'a':
                h1 = np.pad(
                    h1, ((0, 0), (1, 0)),
                    mode='constant', constant_values=0)

            p[:, 1:N-1] = -(h1[:, 1:] - h1[:, :-1])
            p[:, 0] = -h1[:, 0]
            p[:, -1] = h1[:, -1]

        v = (v + self.beta*p) if self.beta > 0. else v

        return v.T

class LTIKernel(BaseEstimator, RegressorMixin):
    '''
    Fit Linear Time Invariant kernel.

    A Linear Time Invariant (LTI) model assumes that
    the response `y` to an input `x` is given by
    
    .. math::
        y_t = \sum_{s\le t} K_{t-s} x_s + \epsilon_t,

    where `K` is the kernel to be estimated, and :math:`\epsilon` is a noise term.

    Parameters
    ----------
    penalty : float, optional
        Penalty parameter for the :math:`H^1` norm. Default is 0 (no penalty).
    mode : {'g', 'a'}, optional
        The mode of the penalty. For the `general` case, all points are taken into accout.
        For the `acceleration` case, the first point is not penalized.
        Maximum number of iterations for the CG solver. Default is None (no limit).

    Attributes
    ----------
    kernel_ : ndarray
        The estimated kernel after fitting the model.
    '''

    def __init__(
        self,
        *,
        dt:float=1.,
        alpha:float=0.,
        beta:float=0.,
        mode:str='g',
        atol:float=1e-6,
        btol:float=1e-6,
        maxiter:int=None,
        conlim:float=1e8,
        show:bool=False
    ):
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.atol = atol
        self.btol = btol
        self.maxiter = maxiter
        self.conlim = conlim
        self.show = show
        self.dt = dt

    def fit(self, X, y):
        r'''
        Fit LTI model.

        Parameters
        ----------
        X : array-like of shape (n_repetitions, n_time_samples).
            Input data.

        y : array-like of shape (n_repetitions, n_time_samples).
            Response data with the same shape of `X`.

        Returns
        -------
        self : object
            Fitted estimator.
        '''
        # Validate inputs.
        X = np.asarray(X)
        y = np.asarray(y)

        y = validate_data(self, X="no_validation", y=y, multi_output=True)

        if X.shape != y.shape:
            raise ValueError(
                "X and y must have the same shape."
            )

        # Set up minimization problem.
        lhs = _Dloss(X, beta=self.beta, mode=self.mode)
        rhs = y.flatten()
        if self.beta > 0.:
            _, N = X.shape
            if self.mode == 'g':
                pad = N-1
            elif self.mode == 'a':
                pad = N-2
            rhs = np.concatenate((rhs, np.zeros(pad)), axis=0)

        r = scipy.sparse.linalg.lsqr(
            lhs, rhs,
            damp=self.alpha,
            atol=self.atol,
            btol=self.btol,
            iter_lim=self.maxiter,
            conlim=self.conlim,
            show=self.show,
        )
        self.kernel_ = r[0]/self.dt
        self.info_ = {
            'istop': r[1],
            'itn': r[2],
            'normr': r[3],
            'normar': r[4],
            'norma': r[5],
            'conda': r[6],
            'normx': r[7],
        }

        return self

    def predict(self, X):
        r'''
        Predict response using the fitted LTI model.
        '''
        # Check if fitted.
        if not hasattr(self, 'kernel_'):
            raise ValueError("The model is not fitted yet. Call 'fit' first.")

        # Validate inputs.
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        if X.shape[0] > self.kernel_.shape[0]:
            raise ValueError("Input X is longer than the fitted kernel.")

        return self.dt*scipy.signal.fftconvolve(
            X, self.kernel_[np.newaxis, :], mode='full', axes=1)[:, :X.shape[1]]

    def score(self, X, y):
        r'''
        Compute the prediction RMS error.

        Parameters
        ----------
        X : array-like of shape (n_repetitions, n_time_samples).
            Input data.

        y : array-like of shape (n_repetitions, n_time_samples).
            True response data.

        Returns
        -------
        score : float
            RMS error.
        '''
        # Validate inputs.
        X = np.asarray(X)
        y = np.asarray(y)

        y = validate_data(self, X="no_validation", y=y, multi_output=True)

        if X.shape != y.shape:
            raise ValueError(
                "y must have the same shape as X."
            )

        y_pred = self.predict(X)

        # Compute R^2 score.
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y, axis=0, keepdims=True))**2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1-ss_res/ss_tot

def H1(X, y):
    # Validate inputs.
    X = np.asarray(X)
    y = np.asarray(y)
    if (X.ndim == 0) or (X.ndim > 2):
        raise ValueError("Input X must be a 1D or 2D array.")
    if X.shape != y.shape:
        raise ValueError("Input X and y must have the same shape.")
    if X.ndim == 1:
        X = X.reshape(1, -1)
        y = y.reshape(1, -1)

    X_fr = np.fft.rfft(X, axis=1)
    y_fr = np.fft.rfft(y, axis=1)
    H1_num = sum(
        np.conj(X_fr[rep])*y_fr[rep]
        for rep in range(X.shape[0]))
    H1_den = sum(
        np.abs(X_fr[rep])**2
        for rep in range(X.shape[0]))

    return H1_num/H1_den
