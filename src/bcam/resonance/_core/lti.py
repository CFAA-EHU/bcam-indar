'''
This is ...
'''

import logging

import numpy as np
import scipy

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
    x : ndarray
        Input data of shape `(N, K)`, where `N` is the length of the time series and `K` is the number of repetitions.
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

    def __init__(self, N, dtype=np.float64):
        self._N = N
        super().__init__(shape=(N, N), dtype=dtype)

    def _matmat(self, r):
        N = self._N
        r_ = r * np.arange(N)[:, np.newaxis] / N
        r_ = np.fft.rfft(r_, axis=0)
        r_ = r_ * np.arange(N//2+1)[:, np.newaxis] / N
        r_ = np.fft.irfft(r_, axis=0)
        return r_
    
    def _adjoint(self):
        return self

class LTI_Kernel:
    r'''
    Derivative of the loss function to find the kernel of best linear fit.
    
    Parameters
    ----------
    x : ndarrays
        Input data of shape `(N, K)`, where `N` is the length of the time series and `K` is the number of repetitions.
    y : ndarray
        Response data of shape `(N, K)`, with the same shape as `x`.
    '''

    def __init__(self, x, y):
        x = np.asarray(x).squeeze()
        y = np.asarray(y).squeeze()
        if (x.ndim == 0) or (x.ndim > 2):
            raise ValueError("Input x must be a 1D or 2D array.")
        if x.shape != y.shape:
            raise ValueError("Input x and y must have the same shape.")
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
        
        self._x = x
        self._y = y

    def fit(self, alpha=0, cg_kwargs=None):
        rhs = _conv(self._x, self._y)
        rhs = np.sum(rhs, axis=1)

        lhs = _Dloss(self._x)
        if alpha > 0:
            N = self._x.shape[0]
            lhs += alpha * _DPenalty(N, dtype=self._x.dtype)

        cg_kwargs = {} if cg_kwargs is None else cg_kwargs
        r, info = scipy.sparse.linalg.cg(lhs, rhs, **cg_kwargs)
        if info != 0:
            logger.warning(f'Conjugate gradient did not converge, info={info}')

        return r

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = 64
    # Create a random vector.
    rng = np.random.default_rng(1235464)
    n_rep = 3
    t = np.arange(N)
    L = 5
    x = scipy.signal.windows.hann(L)
    x = np.pad(x, (3, N - L - 3))
    x = np.tile(x[:, np.newaxis], (1, n_rep))
    freqs = np.array([-2.5, -3.1]) + 2*np.pi*1j * np.array([10, 58])
    weights = np.array([1, 2])
    r_true = np.sum(
        weights[:, np.newaxis] * np.exp(freqs[:, np.newaxis] * (t/N)[np.newaxis, :]),
        axis=0)
    r_true = np.real(r_true)

    y0 = []
    for i in range(x.shape[1]):
        y0.append(np.convolve(x[:, i], r_true, mode='full')[:N])
    y0 = np.array(y0).T
    y = y0 + rng.normal(0, 0.3, (N, n_rep))

    fitter = LTI_Kernel(x, y)
    alpha = 10
    r = fitter.fit(alpha=alpha)

    fig, axs = plt.subplots(nrows=2, sharex=True)

    axs[0].plot(y[:, 0], label='y')
    axs[0].plot(y0[:, 0], label='y0')
    axs[0].legend()

    axs[1].plot(r, label='Estimated r', color='r')
    axs[1].plot(r_true, label='True r', color='b')
    axs[1].legend()
    print('error: ', np.linalg.norm(r - r_true))

    plt.show()