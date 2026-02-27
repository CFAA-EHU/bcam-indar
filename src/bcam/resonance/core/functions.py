import numpy as np


class Rational:

    def __init__(
        self,
        poles,
        a,
        d=None
    ):
        self.poles = np.atleast_1d(poles)
        self.a = np.atleast_2d(a)
        self.d = np.atleast_1d(d) if d is not None else None

        if poles.ndim > 1:
            msg = 'Expected a 1D-array for poles.'
            raise ValueError(msg)

        if a.shape[-1] != len(poles):
            msg = 'The last dimension of a should match the length of poles.'
            raise ValueError(msg)

        if (self.d is not None) and (self.d.shape != a.shape[:-1]):
            msg = 'The shape of d should match the shape of a without the last dimension.'
            raise ValueError(msg)

    def __call__(self, z):
        z = np.atleast_1d(z)
        if z.ndim > 1:
            msg = f'Expected a 1D array for z, got an array of dimension {z.ndim}.'
            raise ValueError(msg)

        r = np.einsum(
            '...j,kj->...k',
            self.a,
            1 / (z[:, np.newaxis] - self.poles[np.newaxis, :]),
            dtype=complex)

        if self.d is not None:
            return r + self.d[..., np.newaxis]
        else:
            return r


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
            dtype=complex)

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

        self._factor = np.ones_like(self.roots)
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
