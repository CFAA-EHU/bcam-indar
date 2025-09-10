#!/usr/bin/env python3
'''
This is ...
'''

import logging

import numpy as np
import scipy

from . import derivatives


logger = logging.getLogger(__name__)

# =================================
# Helper Functions
# =================================

def _validate_dims(M, C, K, check_symmetry=True):
    M, C, K = np.asarray(M), np.asarray(C), np.asarray(K)
    
    if M.ndim != 2:
        msg = f'Expected a 2-darray, got {M.ndim}-darray.'
        raise ValueError(msg)
    
    if M.shape[0] != M.shape[1]:
        msg = 'M must be a square array.'
        raise ValueError(msg)

    if (M.shape != C.shape) or (M.shape != K.shape):
        msg = 'Incompatible shapes for M, C, and K.'
        raise ValueError(msg)

    if check_symmetry:
        if not np.allclose(M, M.T):
            msg = 'M must be symmetric.'
            raise ValueError(msg)

        if not np.allclose(K, K.T):
            msg = 'K must be symmetric.'
            raise ValueError(msg)
    
    return M, C, K


# =================================
# Amplitudes
# =================================

def metric_amps(freqs, fs, ns, a_type='normal'):
    dof = len(freqs)
    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = np.exp(r/(2*fs)) * _sum_exp_weighted(r, fs, ns)
        r *= (4*fs**2)*np.sinh(x[np.newaxis, :]/(2*fs)) * np.sinh(y[:, np.newaxis]/(2*fs))
        return r

    m1 = _mult(freqs, np.conj(freqs))
    m2 = _mult(freqs, freqs)

    if a_type == 'normal':
        L = 2*dof
        m = np.zeros((L, L))
        m[:dof, :dof] = np.real(m1 - m2)
        m[dof:, :dof] = np.imag(m1 + m2)
        m[:dof, dof:] = m[dof:, :dof].T
        m[dof:, dof:] = np.real(m1 + m2)
    elif a_type == 'mechanical':
        L = 2*dof-1
        m = np.zeros((L, L))
        m[:dof, :dof] = np.real(m1 - m2)
        m[dof:, :dof] = trig_fft(np.imag(m1 + m2).T)[..., 1:].T
        m[:dof, dof:] = m[dof:, :dof].T
        m[dof:, dof:] = trig_fft(trig_fft(np.real(m1 + m2))[..., 1:].T)[..., 1:]
    m *= 0.5
    return m

def trig_fft(x):
    '''
    Trigonometric expansion of a real signal.
    '''
    N = x.shape[-1]
    x_hat = np.fft.rfft(x, axis=-1, norm='ortho')
    c = 2*np.ones(N//2 + 1)
    c[0] = 1
    if N % 2 == 0:
        c[-1] = 1
    x_hat = [
        np.sqrt(c) * np.real(x_hat), # cosine part
        -np.sqrt(2) * np.imag(x_hat)[..., 1:(N-1)//2+1] # sine part
    ]
    # Concatenate both parts along the last axis.
    x_hat = np.concatenate(x_hat, axis=-1)
    return x_hat

def trig_ifft(x):
    '''
    Inverse trigonometric expansion of a real signal.
    '''
    N = x.shape[-1]
    c = 2*np.ones(N//2 + 1)
    c[0] = 1
    if N % 2 == 0:
        c[-1] = 1
    x_inv = x[..., :N//2+1].astype(np.complex128)/np.sqrt(c)
    x_inv[..., 1:(N-1)//2+1] = x_inv[..., 1:(N-1)//2+1] - 1j*x[..., N//2 + 1:]/np.sqrt(2)
    x_inv = np.fft.irfft(x_inv, N, axis=-1, norm='ortho')
    return x_inv

def _sinh_m(x):
    eps = np.finfo(x.dtype).eps
    y = np.where(x, x, eps)
    return -np.exp(-y/2) * y / (2*np.sinh(y/2))

def _sum_exp_weighted(a, fs: float, ns: int):
    # TODO: add the case a = 0.
    T = ns / fs
    r = 1 + (1 - np.exp(a*T))/ns
    r = r - _sinh_m(a/fs) * np.exp(a/fs) * (1 - np.exp(a*T)) / (a*T)
    r = r * _sinh_m(a/fs) / a
    return r

def reshape_injection(x, n_out:int, n_in:int):
    '''
    2darray (n_in*(n_in+1)//2 + (n_out-n_in)*n_in, dof) to 3darray (n_out, n_in, dof).

    The returned slice [:n_in, :n_in] is symmetric.
    '''
    assert n_out >= n_in, 'n_out must be greater than or equal to n_in.'
    assert x.ndim == 2, 'Expected a 2D-array.'
    assert x.shape[0] == n_in*(n_in+1)//2 + (n_out-n_in)*n_in, 'Unexpected array shape.'

    L = x.shape[-1]
    x_ = np.zeros((n_out, n_in, L), dtype=x.dtype)
    x_[np.arange(n_in), np.arange(n_in)] = x[:n_in]
    c = n_in
    for i in range(1, n_in):
        cn = c + n_in - i
        x_[np.arange(n_in-i), i+np.arange(n_in-i)] = x[c:cn] / np.sqrt(2)
        x_[i+np.arange(n_in-i), np.arange(n_in-i)] = x[c:cn] / np.sqrt(2)
        c = cn
    if n_out > n_in:
        x_[n_in:, :] = x[c: c + (n_out-n_in)*n_in].reshape(n_out - n_in, n_in, -1)
    return x_

def reshape_projection(x):
    '''
    3darray (n_out, n_in, dof) to 2darray (flatten*, dof).

    When the array is flattened, the symmetry of x[:n_in, :n_in] is taken into account.
    '''
    assert x.ndim == 3, 'Expected a 3D-array.'
    n_out, n_in, L = x.shape
    assert n_out >= n_in, 'n_out must be greater than or equal to n_in.'

    x_ = np.zeros((n_in*(n_in+1)//2 + (n_out-n_in)*n_in, L), dtype=x.dtype)
    x_[:n_in] = x[np.arange(n_in), np.arange(n_in)]
    c = n_in
    for i in range(1, n_in):
        cn = c + n_in - i
        x_[c:cn] = x[np.arange(n_in-i), i+np.arange(n_in-i)]
        x_[c:cn] += x[i+np.arange(n_in-i), np.arange(n_in-i)]
        x_[c:cn] /= np.sqrt(2)
        c = cn
    if n_out > n_in:
        x_[c: c + (n_out - n_in)*n_in] = x[n_in:, :].reshape((n_out - n_in)*n_in, -1)
    return x_


class Amplitudes():

    def __init__(
            self,
            freqs, fs: float, ns: int,
            n_out:int=None, n_in:int=None,
            a_type:str='normal'):
        self.freqs = np.atleast_1d(freqs)
        self.fs = fs
        self.ns = ns
        self.n_out = len(freqs) if n_out is None else n_out
        self.n_in = n_out if n_in is None else n_in
        if a_type in ['normal', 'mechanical']:
            self.a_type = a_type
        else:
            raise ValueError(f"Unknown amplitude type: {a_type}")

    @property
    def values_(self):
        if not hasattr(self, 'raw_coeff_'):
            msg = 'Call fit() before accessing values_.'
            raise ValueError(msg)
        dof = len(self.freqs)
        x = self.raw_coeff_
        if self.a_type == 'normal':
            return x[..., :dof] + 1j*x[..., dof:]
        elif self.a_type == 'mechanical':
            r = np.concatenate(
                [np.zeros((*x.shape[:2], 1)), x[..., dof:]], axis=-1)
            r = trig_ifft(r).astype(np.complex128)
            return x[..., :dof] + 1j*r

    def _matrix(self, penalty: float):
        dof = len(self.freqs)
        L = 2*dof if self.a_type == 'normal' else 2*dof-1
        m = metric_amps(self.freqs, self.fs, self.ns, self.a_type)
        m += penalty * np.eye(L)
        return m

    def _rhs(self, y):
        freqs = self.freqs
        fs, ns = self.fs, self.ns

        def prod(t):
            freqs_ = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
            T = ns / fs
            t = t.reshape(1, -1)
            r_ = np.exp(freqs[:, np.newaxis] * t) * (1 - t/T)
            r_ *= freqs_[:, np.newaxis]
            return r_

        r = np.einsum(
            'ijt,kt->ijk',
            y,
            prod(np.arange(ns)/fs))
        r = r / fs
        if self.a_type == 'normal':
            r = np.concatenate(
                [np.imag(r), np.real(r)],
                axis=-1)
        elif self.a_type == 'mechanical':
            r = np.concatenate(
                [np.imag(r), trig_fft(np.real(r))[..., 1:]],
                axis=-1)
        # Project to space of 'symmetric' matrices.
        r = reshape_projection(r)
        return r.T

    def fit(self, y, penalty:float=0.):
        r = scipy.linalg.solve(
            self._matrix(penalty),
            self._rhs(y),
            assume_a='pos').T

        r = reshape_injection(r, self.n_out, self.n_in)
        self.raw_coeff_ = r
        return self.values_

# =================================
# Modal Parameters
# =================================

def reshape_modes_input(x, dof:int, n_out:int):
    '''
    Transform 1darray into (x, z) for use in PartialModesMap.
    '''
    assert len(x) == 2*n_out*dof - n_out*(n_out+1)//2, 'Unexpected length for x.'

    X = x[:dof*n_out].reshape(n_out, dof)
    
    Zu = np.zeros((n_out, n_out))
    init = dof*n_out
    for i in range(n_out-1):
        end = init + n_out - i - 1
        Zu[i, i+1:] = x[init:end]
        Zu[i+1:, i] = -x[init:end]
        init = end
    Zc = x[dof*n_out + n_out*(n_out-1)//2:].reshape(n_out, dof - n_out)
    Z = np.concatenate((Zu, Zc), axis=1)

    return X, Z

def reshape_modes_output(X, Z):
    '''
    Transform (x, z) into 1darray compatible with scipy.optimize.

    If z[:n_out, :n_out] is not anti-symmetric, then the upper triangle is used
    to construct an anti-symmetric matrix, and the lower triangle is ignored.
    '''
    n_out, dof = X.shape
    
    x = np.zeros(
        (2*n_out*dof - n_out*(n_out+1)//2,), dtype=X.dtype)
    x[:dof*n_out] = X.flatten()

    Zu = Z[:n_out, :n_out]
    init = dof*n_out
    for i in range(n_out-1):
        end = init + n_out - i - 1
        x[init:end] = Zu[i, i+1:]
        init = end
    x[dof*n_out + n_out*(n_out-1)//2:] = Z[:n_out, n_out:].flatten()

    return x

def basis_iterator(n_out:int, dof:int):
    e = np.zeros(2*n_out*dof - n_out*(n_out+1)//2)
    e[0] = 1
    for _ in range(len(e)):
        yield reshape_modes_input(e, dof, n_out)
        e = np.roll(e, 1)

def mode_to_amps(modes, n_out, n_in):
    return modes[:n_out, np.newaxis] * modes[np.newaxis, :n_in]

def amps_to_modes(amps):
    n_out, n_in, dof = amps.shape
    psi = np.zeros((n_out, dof), dtype=amps.dtype)
    psi[:n_in] = np.sqrt(amps[np.arange(n_in), np.arange(n_in)])

    dom_idxs = np.argmax(np.abs(psi[:n_in]), axis=0)
    dom = psi[dom_idxs, np.arange(dof)]
    psi = amps[:, dom_idxs, np.arange(dof)]
    psi /= dom[np.newaxis, :]
    return psi


class PartialModesMap:
    r'''
    A parameterization of mode shapes.

    Returns mode shapes with real part X with shape (N, dof), where N is the number of observations.
    The mode shapes are :math:`\Psi = X(I + i Z)`.

    The matrix Z is written in the basis :math:`\langle q_0, \ldots, q_{N-1}, e_{i_1}, \ldots \rangle`,
    where :math:`q_i` is an orthogonal basis for the range of X, and
    :math:`e_{i_j}` are elements of the canonical basis from coords, that is, coords = [i_1, i_2, ...].
    The matrix Z[:N, :N] is antisymmetric (the function uses the upper triangular part).

    Attributes
    ----------
    freqs : 1D-array
        Modal frequencies.
    coords : 1D-array
        Subset of canonical basis to complement the kernel of X.
    '''

    rtol = 1e-5
    atol = 1e-8

    def __init__(self, freqs, coords):
        assert freqs.ndim == 1, 'Expected a 1D-array for frequencies.'
        self.freqs = freqs

        assert coords.ndim == 1, 'Expected a 1D-array for coords.'
        assert len(coords) < len(freqs), 'Invalid length for coords.'
        self.coords = coords.astype(int)

        self._point = None
        self._vector = None
        self._psi = None
        self._dq = None

    @property
    def point(self):
        return self._point
    
    @point.setter
    def point(self, value):
        x, z = value
        assert x.ndim == 2, 'Expected a 2D-array for x.'
        n_out, dof = x.shape
        assert dof == len(self.freqs), 'x.shape[1] should equal dof.'
        assert n_out == len(self.coords), 'x.shape[0] should equal the number of observations.'
        assert x.shape == z.shape, 'Incompatible shapes for x and z.'
        assert np.allclose(z[:n_out, :n_out], -z[:n_out, :n_out].T), 'The matrix z must be skew-symmetric in the observed block.'
        if (self._point is None) or \
            not (np.allclose(x, self._point[0], rtol=self.rtol, atol=self.atol) & \
             np.allclose(z, self._point[1], rtol=self.rtol, atol=self.atol)):
            self._expensive_fun(x, z)
            self._point = value
            self._dq = None

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, value):
        dx, dz = value
        assert dx.ndim == 2, 'Expected a 2D-array for dx.'
        n_out, dof = dx.shape
        assert dof == len(self.freqs), 'dx.shape[1] should equal dof.'
        assert n_out == len(self.coords), 'dx.shape[0] should equal the number of observations.'
        assert dx.shape == dz.shape, 'Incompatible shapes for dx and dz.'
        if (self._vector is None) or \
            (self._dq is None) or \
            not (np.allclose(dx, self._vector[0], rtol=self.rtol, atol=self.atol) & \
            np.allclose(dz, self._vector[1], rtol=self.rtol, atol=self.atol)):
            self._compute_jac_expensive(dx, dz)
            self._vector = value

    def _inv_qe(self, a):
        '''
        Compute a@[q e]^{-1}, where [q e] = [q_0 ... q_{n_out-1} e_{i_1} ...].
        '''
        dof = len(self.freqs)
        n_out = len(self.coords)
        coords_c = np.setdiff1d(
            np.arange(dof), self.coords, assume_unique=True)
        q = self._grass[0]

        a_ = np.zeros((dof, dof), dtype=a.dtype)
        a_[:, coords_c] = a[:, n_out:]
        try:
            a_[:, self.coords] = scipy.linalg.solve(
                q[self.coords].T,
                (a[:, :n_out] - a_[:, coords_c]@q[coords_c, :]).T,
                assume_a='upper triangular').T
        except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            a_ = np.nan
        return a_

    def _expensive_fun(self, x, z):
        q, s, s_inv = derivatives.grass(x.T, self.coords)
        self._grass = (q, s, s_inv)

        z_ = q@z
        z_ = self._inv_qe(z_)
        self._z_ = z_
        if isinstance(z_, float) and np.isnan(z_):
            logging.warning('q[coords] is singular.')
            self._psi = np.nan
            return

        # Check mass positivity.
        dof = len(self.freqs)
        D = np.real(self.freqs), np.imag(self.freqs)
        m = D[0][:, np.newaxis] * z_.T
        m += m.T
        m[np.arange(dof), np.arange(dof)] += D[1]
        m -= (z_ * D[1][np.newaxis, :]) @ z_.T
        try:
            self._chk = scipy.linalg.cholesky(
                m, lower=False, overwrite_a=True)
        except scipy.linalg.LinAlgError:
            self._chk = np.nan
            logging.warning('Mass matrix is not positive-definite.')
            self._psi = np.nan
            return

        self._psi = x + 1j * x @ z_

    def __call__(self, x, z):
        """
        Evaluate the mode shapes at the given points.

        Parameters
        ----------
        x : 2D-array (N, dof)
            Real part of mode shapes.
        z : 2D-array (N, dof)
            Multiplicative factor of complex part of mode shapes.
        """
        self.point = (x, z)
        return self._psi

    def _jac_z_(self, dz, dq):
        _, z = self.point
        n_out, dof = len(self.coords), len(self.freqs)
        q = self._grass[0]

        # d(q@z@[q e]^{-1})
        # b1 = dq@z@[q e]^{-1} + q@dz@[q e]^{-1}.
        b1 = self._inv_qe(dq@z + q@dz)
        # b2 = [q e]d([q e]^{-1}) = -[dq 0][q e]^{-1}
        b2 = self._inv_qe(-np.pad(dq, ((0, 0), (0, dof-n_out))))

        return b1 + self._z_@b2

    def _compute_jac_expensive(self, dx, dz):
        # TODO: if dx, dz is just rescaled, it is not necessary to recompute dq either.
        if (dx == 0).all():
            self._dq = np.zeros_like(self._grass[0])
            self._ds = np.zeros_like(self._grass[1])
        else:
            self._dq, self._ds = derivatives.jac_grass(dx.T, self.coords, self._grass)

        self._dz_ = self._jac_z_(dz, self._dq)

    def jac(self, x, z):
        self.point = (x, z)
        dof = len(self.freqs)
        def dpsi(dx, dz):
            self.vector = (dx, dz)

            a = 1j * self._z_
            a[range(dof), range(dof)] += 1
            a = dx@a
            return a + 1j * x@self._dz_
        return dpsi

    def _hessp_z_(self, pz, pdq, pd2q):
        n_out, dof = len(self.coords), len(self.freqs)
        _, z = self.point
        _, dz = self.vector
        q = self._grass[0]
        dq = self._dq

        t1 = self._inv_qe(pd2q@z + pdq@dz + dq@pz)

        # a = [q e]d([q e]^{-1}) = -[dq 0][q e]^{-1}
        a = self._inv_qe(-np.pad(dq, ((0, 0), (0, dof-n_out))))
        t2 = self._inv_qe(pdq@z + q@pz)@a

        # pa = [q e]d([q e]^{-1})(p) = -[dq(p) 0][q e]^{-1}
        pa = self._inv_qe(-np.pad(pdq, ((0, 0), (0, dof-n_out))))
        t3 = self._inv_qe(dq@z + q@dz)@pa

        # b1 = a@pa
        b1 = a@pa 
        # b2 = d([q e] d[q e]^{-1}(p)) = -d([pdq 0][q e]^{-1})
        # b2 = -[pd2q 0][q e]^{-1} - [pdq 0]d([q e]^{-1})
        b2 = -self._inv_qe(np.pad(pd2q, ((0, 0), (0, dof-n_out))))
        b2_ = -self._inv_qe(np.pad(pdq, ((0, 0), (0, dof-n_out))))@a
        b2 = b2 + b2_
        # b1 + b2 = [q e]d(d[q e]^{-1}(p))
        t4 = self._inv_qe(q@z)@(b1 + b2)

        return t1 + t2 + t3 + t4

    def hessp(self, x, z, px, pz):
        self.point = (x, z)
        self._compute_jac_expensive(px, pz)
        pdq, pds, pdz_ = self._dq, self._ds, self._dz_

        def d2psi_p(dx, dz):
            self.vector = (dx, dz)
            dz_ = self._dz_
            cross = px@dz_ + dx@pdz_

            pd2q, _ = derivatives.hessp_grass(
                pdq, pds, self._dq, self._ds, self.coords, self._grass)
            pd2z_ = self._hessp_z_(pz, pdq, pd2q)
            return 1j*(cross + x@pd2z_)
        return d2psi_p

    def constraints(self, x, z):
        self.point = (x, z)
        q = self._grass[0]
        c_res = -np.sum(np.log(np.diag(q[self.coords])))
        if isinstance(self._chk, float) and np.isnan(self._chk):
            c_pos = np.inf
        else:
            c_pos = -np.sum(np.log(np.diag(self._chk)))
        return np.array([c_res, c_pos])

    def _dmass(self, dz_):
        D = np.real(self.freqs), np.imag(self.freqs)

        dm = D[0][:, np.newaxis] * dz_.T
        dm += dm.T
        dm_ = (self._z_ * D[1][np.newaxis, :]) @ dz_.T
        dm_ += dm_.T
        return dm - dm_

    def jac_constraints(self, x, z):
        self.point = (x, z)
        q = self._grass[0]

        def d_constr(dx, dz):
            self.vector = (dx, dz)
            dq = self._dq
            jac = np.zeros(2)
            
            jac[0] = -np.sum(np.diag(dq[self.coords])/np.diag(q[self.coords]))

            if isinstance(self._chk, float):
                jac[1] = 0
            else:
                dchk = self._dmass(self._dz_)
                dchk = derivatives.jac_cho(self._chk, dchk)
                jac[1] = -np.sum(np.diag(dchk)/np.diag(self._chk))
            return jac

        return d_constr

    def _hessp_m(self, pz, pdz_, pdq, pd2q):
        d2z_ = self._hessp_z_(pz, pdq, pd2q)
        d2m = self._dmass(d2z_)
        d2m_ = (pdz_ * np.imag(self.freqs)[np.newaxis, :]) @ self._dz_.T
        d2m_ += d2m_.T
        return d2m - d2m_

    def hessp_constraints(self, x, z, px, pz):
        self.point = (x, z)
        self._compute_jac_expensive(px, pz)
        pdq, pds, pdz_ = self._dq, self._ds, self._dz_
        pdchk = self._dmass(pdz_)
        pdchk = derivatives.jac_cho(self._chk, pdchk)

        def local(a, da, pda, pd2a):
            idx = (np.arange(len(a)), np.arange(len(a)))
            r = pd2a[idx] - pda[idx]*da[idx]/a[idx]
            return r/a[idx]

        def hessp_fun(dx, dz):
            self.vector = (dx, dz)
            q = self._grass[0]
            pd2q, _ = derivatives.hessp_grass(
                pdq, pds, self._dq, self._ds, self.coords, self._grass)
            hessp = np.zeros(2)

            hessp[0] = -np.sum(local(
                q[self.coords], self._dq[self.coords], pdq[self.coords], pd2q[self.coords]))

            if isinstance(self._chk, float):
                hessp[1] = 0
            else:
                dchk = self._dmass(self._dz_)
                dchk = derivatives.jac_cho(self._chk, dchk)
                d2chk = self._hessp_m(pz, pdz_, pdq, pd2q)

                d2chk = derivatives.jac_cho(self._chk, d2chk - pdchk.T@dchk - dchk.T@pdchk)
                hessp[1] = -np.sum(local(self._chk, dchk, pdchk, d2chk))
            return hessp

        return hessp_fun


class ModesProp:

    def __init__(
        self,
        freqs,
        fs:int,
        ns:int,
        n_out:int=None,
        n_in:int=None
    ):
        self.freqs = freqs
        self.fs = fs
        self.ns = ns
        self.n_out = len(freqs) if n_out is None else n_out
        self.n_in = n_out if n_in is None else n_in

        self._get_metric()

    def _get_metric(self):
        self._metric = metric_amps(
            self.freqs, self.fs, self.ns, a_type='normal')

    def _fun(self, x, amps, dof:int):
        x = x.reshape(self.n_out, dof)
        amps_ = mode_to_amps(x, self.n_out, self.n_in)

        dif = amps_ - amps
        dif = np.concatenate(
            [np.real(dif), np.imag(dif)], axis=-1)
        trans = np.einsum('ijk,kl->ijl', dif, self._metric)

        # Compute f(x).
        f = np.einsum('ijk,ijk', trans, dif)
        return f
    
    def _jac(self, x, amps, dof:int):
        x = x.reshape(self.n_out, dof)
        amps_ = mode_to_amps(x, self.n_out, self.n_in)

        dif = amps_ - amps
        dif = np.concatenate(
            [np.real(dif), np.imag(dif)], axis=-1)
        trans = np.einsum('ijk,kl->ijl', dif, self._metric)

        # Compute df(x).
        trans = trans[..., :dof]
        t1 = np.array(
            [np.sum(trans[q, :self.n_in] * x[:self.n_in], axis=0)
             for q in range(self.n_out)])
        t2 = np.array(
            [np.sum(trans[self.n_in:self.n_out, q] * x[self.n_in:self.n_out], axis=0)
             for q in range(self.n_in)])
        df = 2*np.concatenate(
            [2*t1[:self.n_in] + t2, t1[self.n_in:]], axis=0)
        return df.flatten()

    def _hessp(self, x, p, amps, dof:int):
        x = x.reshape(self.n_out, dof)
        p = p.reshape(self.n_out, dof)

        amps_ = mode_to_amps(x, self.n_out, self.n_in)
        A = amps_ - amps
        A = np.concatenate(
            [np.real(A), np.imag(A)], axis=-1)

        B = p[:, np.newaxis]*x[np.newaxis, :self.n_in]
        B += x[:, np.newaxis]*p[np.newaxis, :self.n_in]

        Ap = np.einsum('ijk,kl->ijl', A, self._metric[:, :dof])
        t1 = np.array(
            [np.sum(Ap[q, :self.n_in] * p[:self.n_in], axis=0)
             for q in range(self.n_out)])
        t2 = np.array(
            [np.sum(Ap[self.n_in:self.n_out, q] * p[self.n_in:self.n_out], axis=0)
             for q in range(self.n_in)])
        Ap = 2*np.concatenate(
            [2*t1[:self.n_in] + t2, t1[self.n_in:]], axis=0)

        Bx = np.einsum('ijk,kl->ijl', B, self._metric[:dof, :dof])
        t1 = np.array(
            [np.sum(Bx[q, :self.n_in] * x[:self.n_in], axis=0)
             for q in range(self.n_out)])
        t2 = np.array(
            [np.sum(Bx[self.n_in:self.n_out, q] * x[self.n_in:self.n_out], axis=0)
             for q in range(self.n_in)])
        Bx = 2*np.concatenate(
            [2*t1[:self.n_in] + t2, t1[self.n_in:]], axis=0)

        return (Ap + Bx).flatten()

    def fit(self, amps):
        dof = len(self.freqs)
        if dof != amps.shape[2]:
            msg = f'The number of frequencies (dof) must match the last dimension of the amplitudes.'
            raise ValueError(msg)

        x0 = np.real(amps_to_modes(amps))
        x0 = x0.flatten()
        bounds = scipy.optimize.Bounds(
            lb=-1*np.ones(self.n_out * dof),
            ub=1*np.ones(self.n_out * dof))
        res = scipy.optimize.dual_annealing(
            self._fun,
            x0=x0,
            bounds=bounds,
            args=(amps, dof),
            minimizer_kwargs={
                'method': 'Newton-CG',
                'jac': self._jac,
                'hessp': self._hessp
            })
        return res

class Modes:

    def __init__(
        self,
        freqs,
        amps,
        coords,
        fs:int,
        ns:int,
    ):
        self.freqs = freqs
        assert freqs.ndim == 1, 'Expected 1D array for frequencies.'
        
        self.amps = amps
        assert amps.ndim == 3, 'Expected 3D array for amplitudes.'
        assert amps.shape[2] == len(freqs), 'Incompatible shapes for frequencies and amplitudes.'
        
        self.fs = fs
        self.ns = ns
        self._modes_map = PartialModesMap(freqs, coords)

        self._get_metric()

    def _get_metric(self):
        self._metric = metric_amps(
            self.freqs, self.fs, self.ns, a_type='normal')

    def _fun(self, x):
        n_out, n_in, dof = self.amps.shape
        x_, z_ = reshape_modes_input(x, dof, n_out)

        modes = self._modes_map(x_, z_)
        amps = mode_to_amps(modes, n_out, n_in)
        diff = amps - self.amps
        diff = np.concatenate(
                [np.real(diff), np.imag(diff)], axis=-1)
        dist = np.einsum('ijk,kl,ijl->', diff, self._metric, diff)
        return dist

    def _jac_amps(self, p, dp):
        n_out, n_in, _ = self.amps.shape
        r = p[:n_out, np.newaxis] * dp[np.newaxis, :n_in]
        r += dp[:n_out, np.newaxis] * p[np.newaxis, :n_in]
        r = np.concatenate(
            [np.real(r), np.imag(r)], axis=-1)
        return r

    def _jac(self, x):
        n_out, n_in, dof = self.amps.shape
        x_, z_ = reshape_modes_input(x, dof, n_out)

        modes = self._modes_map(x_, z_)
        d_modes = self._modes_map.jac(x_, z_)

        amps = mode_to_amps(modes, n_out, n_in)
        diff = np.concatenate(
            [np.real(amps - self.amps), np.imag(amps - self.amps)], axis=-1)
        del amps
        diff = 2*np.einsum('ijk,kl->ijl', diff, self._metric)

        jac = [np.einsum('ijl,ijl', diff, self._jac_amps(modes, d_modes(dx, dz)))
               for dx, dz in basis_iterator(n_out, dof)]
        return np.array(jac)

    def _hessp_amp(self, modes, pd_modes, d_modes, d2_modes):
        n_in = self.amps.shape[1]

        d2_amps = d2_modes[:, np.newaxis]*modes[np.newaxis, :n_in]
        d2_amps += pd_modes[:, np.newaxis]*d_modes[np.newaxis, :n_in]
        d2_amps += d_modes[:, np.newaxis]*pd_modes[np.newaxis, :n_in]
        d2_amps += modes[:, np.newaxis]*d2_modes[np.newaxis, :n_in]

        d2_amps = np.concatenate(
            [np.real(d2_amps), np.imag(d2_amps)], axis=-1)
        return d2_amps

    def _hessp(self, x, p):
        n_out, n_in, dof = self.amps.shape
        x_, z_ = reshape_modes_input(x, dof, n_out)
        px, pz = reshape_modes_input(p, dof, n_out)

        modes = self._modes_map(x_, z_)
        amps = mode_to_amps(modes, n_out, n_in)
        diff = np.concatenate(
            [np.real(amps - self.amps), np.imag(amps - self.amps)], axis=-1)

        jac_modes = self._modes_map.jac(x_, z_)
        hessp_modes = self._modes_map.hessp(x_, z_, px, pz)
        pd_amps = self._jac_amps(modes, jac_modes(px, pz))

        def hess_f(dx, dz):
            d2_amps = self._hessp_amp(
                modes, jac_modes(px, pz), jac_modes(dx, dz), hessp_modes(dx, dz))
            d_amps = self._jac_amps(modes, jac_modes(dx, dz))

            d2_dist = 2*np.einsum(
                'ijk,kl,ijl->', d2_amps, self._metric, diff)
            d2_dist += 2*np.einsum(
                'ijk,kl,ijl->', pd_amps, self._metric, d_amps)
            return d2_dist

        hessp = [hess_f(dx, dz) for dx, dz in basis_iterator(n_out, dof)]
        return np.array(hessp)


def modal_to_system(mode_shapes, Z):
    '''Recover system matrices from mode shapes and complex frequencies.

    Parameters
    ----------
    mode_shapes : 2D-array
        Square matrix with mode shapes.
    Z : 1D-array
        Complex frequencies.

    Returns
    -------
    M, C, K : 2D-arrays
    '''
    ms = np.asarray(mode_shapes, dtype=np.complex128)
    Z = np.asarray(Z, dtype=np.complex128)
    if ms.shape[0] != ms.shape[1]:
        msg = 'Expected a square matrix for mode shapes.'
        raise ValueError(msg)
    if ms.shape[0] != Z.shape[0]:
        msg = 'Incompatible shapes for mode shapes and complex frequencies.'
        raise ValueError(msg)

    M = ((ms * (Z / Z.imag)[np.newaxis, :]) @ ms.T).imag
    M = np.linalg.inv(M)

    K = ((ms * (-1 / (Z * Z.imag))[np.newaxis, :]) @ ms.T).imag
    K = np.linalg.inv(K)

    C = ((ms * (Z**2 / Z.imag)[np.newaxis, :]) @ ms.T).imag
    C = -M @ C @ M

    return M, C, K

def system_to_modal(M, C, K):
    '''Recover mode shapes and complex frequencies from system matrices.
    
    Parameters
    ----------
    M, C, K : 2D-arrays
    
    Returns
    -------
    mode_shapes : 2D-array
    Z : 1D-array
    '''
    # Check parameters.
    N = M.shape[0]
    if N != M.shape[1]:
        msg = 'M must be a square matrix.'
        raise ValueError(msg)
    if (M.shape != C.shape) or (M.shape != K.shape):
        msg = 'Incompatible shapes for M, C, and K.'
        raise ValueError(msg)

    # Construct matrix for QEP.
    a = np.zeros((2 * N, 2 * N))
    b = np.zeros((2 * N, 2 * N))
    a[:N, N:] = K
    a[N:, :N] = -K
    a[N:, N:] = -C
    b[:N, :N] = K
    b[N:, N:] = M
    eigvals, eigvecs = scipy.linalg.eig(
        a, b,
        overwrite_a=True, overwrite_b=True)
    Z = eigvals[::2]
    mode_shapes = eigvecs[:N, ::2]
    
    # Order by size of complex frequency.
    idxs = np.argsort(np.abs(Z))
    mode_shapes = mode_shapes[:, idxs]
    Z = Z[idxs]

    # Mass normalization.
    # TODO: optimize.
    d =  np.diag(Z) @ mode_shapes.T @ M @ mode_shapes * np.diag(Z)
    d = d - mode_shapes.T @ K @ mode_shapes
    mu = 2j * Z * np.imag(Z) / np.diag(d)
    mu = np.sqrt(mu)
    mode_shapes = mode_shapes @ np.diag(mu)
    
    return mode_shapes, Z



# =================================
# Systems
# =================================

def randomSystem(
        dofs,
        mass_range=(1, 2),
        damping_range=(0.125, 0.25),
        freqs_range=(4, 8),
        damping_type='prop',
        seed=None
    ):
    rng = np.random.default_rng(seed)
    m = rng.uniform(*mass_range, dofs)
    zeta = rng.uniform(*damping_range, dofs)
    freqs = rng.uniform(*freqs_range, dofs)
    
    # Generate rotation matrices.
    if dofs > 1:
        U = scipy.stats.ortho_group(dim=dofs, seed=seed)
        Um = U.rvs()
        Uk = U.rvs()
        if damping_type == 'prop':
            Uc = Uk
        elif damping_type == 'nop':
            Uc = U.rvs()
        else:
            msg = 'Invalid damping type. Choose between "prop" and "nop".'
            raise ValueError(msg)
    else:
        Um = np.array([[1]])
        Uk, Uc = Um, Um
    
    modes_ = Um @ np.diag(1 / np.sqrt(m))
    invModes_ = np.diag(np.sqrt(m)) @ Um.T
    M = Um @ np.diag(m) @ Um.T
    C = invModes_.T @ Uc @ np.diag(2 * zeta * freqs) @ Uc.T @ invModes_
    K = invModes_.T @ Uk @ np.diag(freqs**2) @ Uk.T @ invModes_
    normal_modes = modes_ @ Uk

    if damping_type == 'prop':
        mode_shapes = normal_modes
        Z = freqs * (-zeta + 1j * np.sqrt(1 - zeta**2))
    else:
        A = normal_modes.T @ C @ normal_modes
        B = np.diag(freqs**2)
        mode_shapes, Z = system_to_modal(np.eye(dofs), A, B)
        mode_shapes = normal_modes @ mode_shapes

    # Sort by size of complex frequency.
    idxs = np.argsort(np.abs(Z))
    mode_shapes = mode_shapes[:, idxs]
    Z = Z[idxs]

    mechanical = {
        'mass': M,
        'damping': C,
        'stiffness': K
    }
    modal = {
        'frequencies': Z,
        'mode_shapes': mode_shapes
    }
    return mechanical, modal


class Spring:

    def __init__(self, M, C, K, force=None):
        '''Simulate a linear vibrating system.

        Set up an object representing a linear system
        to be used in JiTCODE.
        '''
        self.M, self.C, self.K = _validate_dims(M, C, K)
        self.force = np.zeros(self.M.shape[0]) if force is None else np.array(force, ndmin=1)
        if len(self.force) != self.M.shape[0]:
            msg = f'Expected a force with {self.M.shape[0]}-DoF,\
                got {len(self.force)} instead.'
            raise ValueError(msg)

        Minv = np.linalg.inv(self.M)
        self._A = Minv @ C
        self._B = Minv @ K
        self._force = Minv @ self.force

    def __iter__(self):
        dofs = self.M.shape[0]
        q = [y(i) for i in range(dofs)]
        p = [y(i + dofs) for i in range(dofs)]
        for i in range(dofs):
            yield p[i]
        for i in range(dofs):
            tmp = -sum(self._B[i, j] * q[j] for j in range(dofs))
            tmp += -sum(self._A[i, j] * p[j] for j in range(dofs))
            tmp += self._force[i]
            yield tmp

    def __len__(self):
        return 2 * self.M.shape[0]
