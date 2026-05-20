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
    '''
    Kernel or Impulse Response Function.

    Parameters
    ----------
    nat_freqs : array-like
        Natural frequencies of the system.
    
    amps : array-like
        Amplitudes corresponding to each natural frequency. The last dimension should match the length of nat_freqs.

    response : {'d', 'v', 'a'}, optional
        Type of response to compute:
        - 'd': Displacement response (default)
        - 'v': Velocity response
        - 'a': Acceleration response
    '''

    def __init__(
        self,
        nat_freqs,
        amps,
        response:str='a',
    ):
        self.nat_freqs = np.atleast_1d(nat_freqs)
        self.amps = np.atleast_2d(amps)
        self.response = response

        if self.nat_freqs.ndim > 1:
            msg = 'Expected a 1D-array for nat_freqs.'
            raise ValueError(msg)

        if self.amps.shape[-1] != len(self.nat_freqs):
            msg = 'The last dimension of amps should match the length of nat_freqs.'
            raise ValueError(msg)
        
        if response not in ['d', 'v', 'a']:
            msg = f'Unknown response type: {response}'
            raise ValueError(msg)

        self._factor = np.ones_like(self.nat_freqs)
        if response == 'd':
            pass
        elif response == 'v':
            self._factor *= self.nat_freqs
        elif response == 'a':
            self._factor *= self.nat_freqs**2

    def __call__(self, t):
        t = np.atleast_1d(t)
        if t.ndim > 1:
            msg = f'Expected a 1D array for times, got an array of dimension {t.ndim}.'
            raise ValueError(msg)

        K = np.einsum(
            '...j,tj->...t',
            self.amps,
            self._factor[np.newaxis, :]*np.exp(self.nat_freqs[np.newaxis, :]*t[:, np.newaxis]),
            dtype=complex
        )
        K = np.imag(K)

        return K
