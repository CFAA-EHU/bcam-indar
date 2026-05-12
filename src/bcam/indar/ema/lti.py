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
    Fit a Linear Time-Invariant (LTI) kernel.

    In a discrete LTI model, given an input :math:`x`,
    the response :math:`y` at a given time :math:`k` is

    .. math::
        y_k = dt\,\sum_{i=0}^k  h_{k-i}\,x_i + \\varepsilon_k,

    where :math:`dt` is the time step, :math:`h` is the Impulse Response Function (IRF) to be estimated
    (also known as the kernel), and :math:`\\varepsilon` is white noise.
    This class only admits SISO data, that is,
    the input and response are scalar time series, and the kernel is a 1D array.

    This class uses the matrix-free algorithm `LSMR <https://doi.org/10.1137/10079687X>`_, as
    implemented in :func:`scipy.sparse.linalg.lsmr`, to estimate the IRF.

    Parameters
    ----------
    dt : float
        Sampling time step.

    alpha, beta : float
        If `alpha` is positive, the :math:`\ell^2` norm of the kernel is penalized.
        If `beta` is positive, the :math:`\ell^2` norm of the kernel's derivative is penalized.

    mode : {'g', 'a'}
        When the mode is *general* (`g`), the derivative is penalized for all times, while
        for *acceleration* (`a`) mode, time :math:`k = 0` is not penalized.

    atol, btol : float
        `atol` is the relative tolerance in the entries of the input :math:`x`, and
        `btol` is the relative tolerance in the entries of the response :math:`y`.

    maxiter, conlim, show : int, int, bool
        See :func:`scipy.sparse.linalg.lsmr` for details.

    Attributes
    ----------
    kernel_ : array of shape (n_samples,)
        The estimated IRF, or kernel.

    info_ : dict
        Information about the optimization process, containing the following keys:

        - `istop`: reason for stopping.

        - `itn`: number of iterations.

        - `normr`: the norm of the residual.

        - `normar`: the norm of the projected residual.

        - `norma`: the estimate of the Frobenius norm of the matrix.

        - `conda`: the estimate of the condition number of the matrix.

        - `normx`: the norm of the solution.

    Examples
    --------
    We create synthetic data from a known kernel and fit the LTI model to check if the kernel is correctly estimated.

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> import numpy as np
        >>> from scipy.signal import fftconvolve
        >>> from bcam.indar import ema
        >>> # Create random input data.
        >>> rng = np.random.default_rng(123)
        >>> n_reps = 5 # Number of repetitions (trials)
        >>> N = 100 # Number of time samples
        >>> X = rng.normal(0, 1, size=(n_reps, N))
        >>> # Create a kernel and generate response data.
        >>> true_kernel = np.exp(-0.05*np.arange(N)) * np.cos(2*np.pi*0.1*np.arange(N))
        >>> y = fftconvolve(
        ...     X, true_kernel[np.newaxis, :], mode='full', axes=1)[:, :N]
        >>> # Add noise to the response.
        >>> y += rng.normal(0, .1, y.shape)
        >>> # Fit kernel from input/output data.
        >>> model = ema.LTIKernel(beta=5.)
        >>> model.fit(X, y)

    We check the prediction performance of the fitted model.

    .. plot::
        :context:
        :format: doctest
        :include-source: True

        >>> import matplotlib.pyplot as plt
        >>> plt.plot(true_kernel, label='True kernel')
        >>> plt.plot(model.kernel_, label='Estimated kernel')
        >>> plt.legend()
        >>> plt.show()
    '''

    def __init__(
        self,
        *,
        alpha:float=0.,
        beta:float=0.,
        mode:str='g',
        atol:float=1e-6,
        btol:float=1e-6,
        maxiter:int=None,
        conlim:float=1e8,
        show:bool=False,
        dt:float=1.,
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
        '''
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
        
        if self.mode not in ['g', 'a']:
            raise ValueError(
                "Mode must be 'g' or 'a'."
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
        '''
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
        '''
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
