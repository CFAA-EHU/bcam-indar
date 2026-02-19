import numpy as np


class ExpSum:

    def __init__(
        self,
        exps,
        amps
    ):
        self.exps = np.atleast_1d(exps)
        self.amps = np.atleast_2d(amps)

        if exps.ndim > 1:
            msg = 'Expected a 1D-array for exps.'
            raise ValueError(msg)

        if amps.shape[-1] != len(exps):
            msg = 'The last dimension of amps should match the length of exps.'
            raise ValueError(msg)

    def __call__(self, t):
        t = np.atleast_1d(t)
        if t.ndim > 1:
            msg = f'Expected a 1D array for t, got an array of dimension {t.ndim}.'
            raise ValueError(msg)

        r = np.einsum(
            '...j,tj->...t',
            self.amps,
            np.exp(self.exps[np.newaxis, :]*t[:, np.newaxis]),
            dtype=complex
        )

        return r


class Kernel:

    def __init__(
        self,
        roots,
        amps,
        response:str='a',
    ):
        self.roots = np.atleast_1d(roots)
        self.amps = np.atleast_2d(amps)
        self.response = response

        if roots.ndim > 1:
            msg = 'Expected a 1D-array for roots.'
            raise ValueError(msg)

        if amps.shape[-1] != len(roots):
            msg = 'The last dimension of amps should match the length of roots.'
            raise ValueError(msg)

        self._factor = np.array([1.])
        if response == 'd':
            pass
        elif response == 'v':
            self._factor *= self.roots
        elif response == 'a':
            self._factor *= self.roots**2
        else:
            msg = f'Unknown response type: {response}'
            raise ValueError(msg)

    def __call__(self, t):
        t = np.atleast_1d(t)
        if t.ndim > 1:
            msg = f'Expected a 1D array for times, got an array of dimension {t.ndim}.'
            raise ValueError(msg)

        K = np.einsum(
            '...j,tj->...t',
            self.amps,
            self._factor[np.newaxis, :]*np.exp(self.roots[np.newaxis, :]*t[:, np.newaxis]),
            dtype=complex
        )
        K = np.imag(K)

        return K