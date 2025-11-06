'''
Functions for Linear Time Invariant (LTI) models.
'''

import logging

import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data

logger = logging.getLogger(__name__)


def _conv(x, y):
    r'''
    Perform operation :math:`\sum_{k\le i < N} x_{i-k}y_i`, for :math:`i=0, \ldots, N-1`.

    Parameters
    ----------
    x, y : ndarrays
        Input arrays of shape `(N, K)`.
        If K > 1, the operation is performed column-wise.
    
    Returns
    -------
    x*y : ndarray
        The result of the convolution, with shape `(N, K)`.
        Each column is the result of the convolution of the corresponding column of `x` with `y`.
    '''
    N = x.shape[0]
    v = scipy.signal.fftconvolve(x, y[::-1], axes=0)[:N]
    return v[::-1]

class _Dloss(scipy.sparse.linalg.LinearOperator):
    r'''
    Derivative of the loss function to find the kernel of best linear fit.
    
    Parameters
    ----------
    x : 1d or 2d array-like of shape (n_samples, n_repetitions) or (n_samples,)
    '''
    
    def __init__(self, x):
        N = x.shape[0]
        self._x = x
        super().__init__(shape=(N, N), dtype=x.dtype)

    def _matmat(self, r):
        N = self._x.shape[0]
        x = self._x.reshape(N, 1, -1)
        v = scipy.signal.fftconvolve(r[..., np.newaxis], x, axes=0)[:N]
        v = _conv(x, v)
        return np.sum(v, axis=-1)
    
    def _adjoint(self):
        return self
    
class _DPenalty(scipy.sparse.linalg.LinearOperator):

    def __init__(self, N, mode='g', dtype=np.float64):
        self._N = N
        self.mode = mode
        if mode == 'g':
            self._my_matmat = self._matmat_g
        elif mode == 'a':
            self._my_matmat = self._matmat_a
        else:
            raise ValueError('Mode must be `g` or `a`.')
        super().__init__(shape=(N, N), dtype=dtype)

    def _matmat(self, r):
        return self._my_matmat(r)

    @staticmethod
    def _laplace(r):
        r_ = np.zeros_like(r)
        # Derivative term.
        r_[1:-1] = 2*r[1:-1]-(r[2:]+r[:-2])
        r_[0] = -(r[1]-r[0])
        r_[-1] = r[-1] - r[-2]
        return r_

    def _matmat_g(self, r):
        # Derivative term.
        r_ = self._laplace(r)
        # L2 norm.
        r_ += r
        return r_

    def _matmat_a(self, r):
        r_ = np.zeros_like(r)
        # Difference and projection.
        r_[1:] = self._laplace(r[1:])
        # L2 norm.
        r_ += r
        return r_

    def _adjoint(self):
        return self

class LTIKernel(BaseEstimator, RegressorMixin):
    r'''
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
    rtol : float, optional
        Relative tolerance for the conjugate gradient (CG) solver. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for the CG solver. Default is 0.
    maxiter : int, optional
        Maximum number of iterations for the CG solver. Default is None (no limit).
    callback : callable, optional
        Callback function to be passed to the CG solver.
        Check `scipy.sparse.linalg.cg` for more details.

    Attributes
    ----------
    kernel_ : ndarray
        The estimated kernel after fitting the model.
    '''

    def __init__(
        self,
        *,
        penalty:float=0.,
        mode:str='g',
        rtol:float=1e-5,
        atol:float=0.,
        maxiter:int=None,
        callback=None
    ):
        self.penalty = penalty
        self.mode = mode
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.callback = callback

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
                "y must have the same shape as X."
            )

        X = X.T
        y = y.T

        # Set up minimization problem.
        rhs = _conv(X, y)
        rhs = np.sum(rhs, axis=1)

        lhs = _Dloss(X)
        if self.penalty > 0:
            N = X.shape[0]
            lhs += self.penalty * _DPenalty(
                N, mode=self.mode, dtype=X.dtype)

        x, info = scipy.sparse.linalg.cg(
            lhs, rhs,
            x0=None,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            callback=self.callback
        )
        if info != 0:
            logger.warning(f'Conjugate gradient did not converge, info={info}')
        self.kernel_ = x

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

        return scipy.signal.fftconvolve(
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
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

    H1_num = sum(
        np.fft.rfft(X[:, rep])*np.conj(np.fft.rfft(y[:, rep]))
        for rep in range(X.shape[1]))
    H1_den = sum(
        np.abs(np.fft.rfft(X[:, rep]))**2
        for rep in range(X.shape[1]))
    return H1_num/H1_den
