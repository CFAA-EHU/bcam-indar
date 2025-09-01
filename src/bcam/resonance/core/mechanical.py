'''
This is ...
'''

import logging

import numpy as np
import scipy

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

# Jacobian of the QR decomposition
# ----------------------------
def _jac_qr(x, dx, qr):
    # dx = dq@r + q@dr.
    q, r = qr
    # x.T@dx + x@dx.T = r.T@dr + dr.T@r. To prove this,
    # notice that q.T@q = I, so dq.T@q + q.T@dq = 0.
    a = x.T@dx
    a += a.T
    dr = np.zeros(r.shape, dtype=r.dtype)
    u = np.zeros(a.shape[1], dtype=x.dtype)

    # u_{ij} = \sum_{i<k} (r_{ki}dr_{kj} + r_{kj}dr_{ki}), for j >=i.
    # Then, a_{ij} = u_{ij} + r_{ii}dr_{ij} + r_{ij}dr_{ii}.
    for i in range(a.shape[0]-1):
        dr[i, i] = (a[i, i] - u[i])/(2*r[i, i])
        dr[i, i+1:] = (a[i, i+1:] - u[i+1:] - r[i, i+1:]*dr[i, i])/r[i, i]
        u[i+1:] = [
            np.sum(r[:i, i+1]*dr[:i, j] + dr[:i, i+1]*r[:i, j], axis=0)
            for j in range(i+1, a.shape[1])]
    dr[-1, -1] = (a[-1, -1] - u[-1])/(2*r[-1, -1])

    dq = scipy.linalg.solve(
        r.T, dx.T - dr.T@q.T, assume_a='lower triangular',
        overwrite_b=True).T
    return dq, dr

class jac_qr:
    '''
    Computes the derivative of the qr decomposition.

    Attributes
    ----------
    x : np.ndarray
        The input array.
    qr : tuple, optional
        The QR decomposition of the input array as (Q, R).
    '''

    def __init__(self, x, qr=None):
        self.x = np.atleast_2d(x)
        if self.x.ndim != 2:
            raise ValueError('Expected a 2D-array.')
        if self.x.shape[0] < self.x.shape[1]:
            raise ValueError('Expected a 2D-array with shape (N, M) and N >= M.')
        if qr is None:
            q, r = scipy.linalg.qr(
                x, overwrite_a=False, mode='economic', pivoting=False)
            idx = np.argwhere(np.diag(r) < 0)
            q[:, idx] *= -1
            r[idx, :] *= -1
            self.qr = (q, r)
        else:
            self.qr = qr

    def __call__(self, dx):
        r'''
        If :math:`f:x \mapsto (q, r)`, then this method returns
        :math:`df(x)(dx) = (dq, dr)`
        '''
        dx = np.atleast_2d(dx)
        if dx.shape != self.x.shape:
            raise ValueError('Expected a 2D-array with the same shape as x.')
        dq, dr = _jac_qr(self.x, dx, self.qr)
        return dq, dr

# =================================
# Amplitudes
# =================================

def _metric_amps(freqs, fs, ns, a_type='normal'):
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
        m[dof:, :dof] = _trig_fft(np.imag(m1 + m2).T)[..., 1:].T
        m[:dof, dof:] = m[dof:, :dof].T
        m[dof:, dof:] = _trig_fft(_trig_fft(np.real(m1 + m2))[..., 1:].T)[..., 1:]
    m *= 0.5
    return m

def _trig_fft(x):
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

def _trig_ifft(x):
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

def _reshape_injection(x, n_out:int, n_in:int):
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

def _reshape_projection(x):
    n_out, n_in, L = x.shape
    x_ = np.zeros_like(x, dtype=x.dtype)
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
            r = _trig_ifft(r).astype(np.complex128)
            return x[..., :dof] + 1j*r

    def _matrix(self, penalty: float):
        dof = len(self.freqs)
        L = 2*dof if self.a_type == 'normal' else 2*dof-1
        m = _metric_amps(self.freqs, self.fs, self.ns, self.a_type)
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
                [np.imag(r), _trig_fft(np.real(r))[..., 1:]],
                axis=-1)
        # Project to space of 'symmetric' matrices.
        r = _reshape_projection(r)
        return r.T

    def fit(self, y, penalty:float=0.):
        r = scipy.linalg.solve(
            self._matrix(penalty),
            self._rhs(y),
            assume_a='pos').T

        r = _reshape_injection(r, self.n_out, self.n_in)
        self.raw_coeff_ = r
        return self.values_

# =================================
# Modal Parameters
# =================================

def _reshape_mode_shapes_input(x, dof:int, n_out:int):
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

def _reshape_mode_shapes_output(X, Z):
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

def _mode_to_amps(modes, n_out, n_in):
    return modes[:n_out, np.newaxis] * modes[np.newaxis, :n_in]

def _amps_to_modes(amps):
    n_out, n_in, dof = amps.shape
    psi = np.zeros((n_out, dof), dtype=amps.dtype)
    psi[:n_in] = np.sqrt(amps[np.arange(n_in), np.arange(n_in)])

    dom_idxs = np.argmax(np.abs(psi[:n_in]), axis=0)
    dom = psi[dom_idxs, np.arange(dof)]
    psi = amps[:, dom_idxs, np.arange(dof)]
    psi /= dom[np.newaxis, :]
    return psi

def _inv_qe(q, a, lu, coords_div):
    r'''
    Computes :math:`\hat{a} = a\,[q e_\sigma]^{-1}`.
    '''
    dof, n_out = q.shape
    coords, coords_c = coords_div
    if coords.size == 0:
        a_ = a @ q.T
    else:
        a_ = np.zeros((dof, dof), dtype=a.dtype)
        a_[:, coords] = a[:, n_out:]
        a_[:, coords_c] = scipy.linalg.lu_solve(
            lu, (a[:, :n_out] - a_[:, coords]@q[coords, :]).T).T
    return a_

class _PartialModesMap:

    def __init__(self, freqs, coords):
        self.freqs = freqs
        self.point = (None, None)

        if coords is None:
            self._inv_qe = self._inv_qe_full
            self._z_new = self._z_new_full
            self._jac_z_ = self._jac_z_full
        else:
            coords_c = np.setdiff1d(
                np.arange(len(freqs)), coords, assume_unique=True)
            self.coords = (coords, coords_c)
            self._inv_qe = self._inv_qe_def
            self._z_new = self._z_new_def
            self._jac_z_ = self._jac_z_def

        self._psi = None

    def _inv_qe_full(self, q, a):
        return a @ q.T
    
    def _inv_qe_def(self, q, a):
        dof, n_out = q.shape
        coords, coords_c = self.coords
        a_ = np.zeros((dof, dof), dtype=a.dtype)
        a_[:, coords] = a[:, n_out:]
        a_[:, coords_c] = scipy.linalg.lu_solve(
            self._lu,
            (a[:, :n_out] - a_[:, coords]@q[coords, :]).T).T
        return a_

    def _z_new_full(self, z):
        return self._inv_qe(self._q, self._q@z)

    def _z_new_def(self, z):
        lu = scipy.linalg.lu_factor(self._q[self.coords[1]].T)
        self._lu = lu
        if np.min(np.abs(np.diag(lu[0]))) < 1e-12:
            logging.warning('Ill-defined coordinate patch.')
            return np.nan
        return self._inv_qe(self._q, self._q@z)

    def __call__(self, x, z):
        n_out, dof = x.shape
        q, r = scipy.linalg.qr(
            x.T, overwrite_a=False, mode='economic', pivoting=False)
        # Detect negative elements in the diagonal of R.
        idx = np.argwhere(np.diag(r) < 0)
        r[idx, :] *= -1
        q[:, idx] *= -1
        self._q, self._r = q, r
        c_X = np.abs(np.prod(r[np.arange(n_out), np.arange(n_out)]))
        if c_X < 1e-12:
            logging.warning('Real part does not have full rank.')

        z_ = self._z_new(z)
        self._z_ = z_
        if np.isnan(self._z_.any()):
            self._psi = np.nan
            return

        d = self.freqs.real, self.freqs.imag
        H = d[0][:, np.newaxis] * z_.T
        H += H.T
        H[np.arange(dof), np.arange(dof)] += d[1]
        H -= (z_ * d[1][np.newaxis, :]) @ z_.T
        try:
            self._chk = scipy.linalg.cholesky(
                H, lower=False, overwrite_a=True)
        except scipy.linalg.LinAlgError:
            self._chk = np.nan
            logging.warning('Mass matrix is not positive-definite.')
            self._psi = np.nan
            return

        self._psi = x + 1j * x @ z_
        return self._psi

    def _jac_z_full(self, z, dz):
        q = self._q
        dq, _ = self._dqr
        return dq@z@q.T + q@dz@q.T + q@z@dq.T

    def _jac_z_def(self, z, dz):
        n_out, dof = z.shape
        q = self._q
        dq, _ = self._dqr

        # d(q@z_@[q e]^{-1})
        # b1 = dq@z_@[q e]^{-1} + q@dz_@[q e]^{-1}.
        b1 = _inv_qe(q, dq@z + q@dz)
        # b2 = [q e]d([q e]^{-1}) = -[dq 0][q e]^{-1}
        b2 = _inv_qe(q, -np.pad(dq, (0, dof-n_out)))

        return b1 + self._z_@b2

    def jac(self, x, z, dx, dz):
        dq, dr = _jac_qr(x, dx, (self._q, self._r))
        self._dqr = (dq, dr)
        z_ = self._z_
        _, dof = x.shape

        a = 1j * z_
        a[range(dof), range(dof)] += 1
        a = dx@a

        dz_ = self._jac_z_(z, dz)

        return a + 1j * x@dz_

def partial_modes_map(
    x, z, freqs, coords=None):
    r'''
    A parameterization of mode shapes.

    Returns mode shapes with real part X with shape (N, dof), where N is the number of observations.
    The mode shapes are :math:`\Psi = X(I + i Z)`.

    The argument Z in the function is written in the basis :math:`\langle q_0, \ldots, q_{N-1}, e_{i_1}, \ldots \rangle`,
    where :math:`q_i` is an orthogonal basis of the range of X, and
    :math:`e_{i_j}` are elements of the canonical basis from coords, that is, coords = [i_1, i_2, ...].
    The matrix Z[:N, :N] is antisymmetric (the function uses the upper triangular part).

    Parameters
    ----------
    X : 2D-array (N, dof)
        Real part of mode shapes.
    Z : 2D-array (N, dof)
        Multiplicative factor of complex part of mode shapes.
    freqs : 1D-array
        Modal frequencies.
    coords : 1D-array, optional
        Subset of canonical basis to complement the kernel of X.
        By default, it takes [N, N+1, ..., dof-1].
    Returns
    -------
    psi : 2D-array
        Array where each column is a mode shape.
    '''

    x = np.asarray(x)
    z = np.asarray(z)
    if x.shape[0] > x.shape[1]:
        msg = 'Expected a 2D-array with more DoF (columns) than observations (rows).'
        raise ValueError(msg)
    if x.shape != z.shape:
        msg = 'Incompatible shapes for X and Z.'
        raise ValueError(msg)
    z = np.triu(z, k=1)
    z[:x.shape[0], :x.shape[0]] = z[:x.shape[0], :x.shape[0]] - z[:x.shape[0], :x.shape[0]].T

    freqs = np.asarray(freqs).squeeze()
    if freqs.ndim > 1:
        msg = 'Expected a 1D-array for frequencies.'
        raise ValueError(msg)
    freqs = np.atleast_1d(freqs)

    if coords is not None:
        coords = np.atleast_1d(coords, dtype=int)
        if coords.ndim > 1:
            msg = 'Expected a 1D-array for coords.'
            raise ValueError(msg)
        elif len(coords) != x.shape[1] - x.shape[0]:
            msg = 'Incompatible shapes for coords and mode shapes.'
            raise ValueError(msg)
        coords = np.sort(np.unique(coords))
    elif x.shape[0] < x.shape[1]:
        coords = np.arange(x.shape[0], x.shape[1])

    modes_map = _PartialModesMap(freqs, coords)
    psi = modes_map(x, z)
    if np.isnan(psi.any()):
        return psi
    else:
        psi *= np.sqrt(np.imag(freqs))[np.newaxis, :]
        return psi


def _jac_amps(p, dp, n_out, n_in):
    r = p[:n_out, np.newaxis] * dp[np.newaxis, :n_in]
    r += dp[:n_out, np.newaxis] * p[np.newaxis, :n_in]
    return r

def _jac_dist(p, dp, metric):
    p = np.concatenate(
        [np.real(p), np.imag(p)], axis=-1)
    dp = np.concatenate(
        [np.real(dp), np.imag(dp)], axis=-1)
    return 2*np.einsum('ijk,kl,ijl->', p, metric, dp)

def _fun(x, z, freqs, coords, n_out, n_in, metric, amps0):
    modes_map = _PartialModesMap(freqs, coords)
    modes = modes_map(x, z)
    amps = _mode_to_amps(modes, n_out, n_in)
    diff = amps - amps0
    diff = np.concatenate(
            [np.real(diff), np.imag(diff)], axis=-1)
    dist = np.einsum('ijk,kl,ijl->', diff, metric, diff)
    return dist

def _jac_fun(x, z, dx, dz, freqs, coords, n_out, n_in, metric, amps0):
    modes_map = _PartialModesMap(freqs, coords)
    modes = modes_map(x, z)
    amps = _mode_to_amps(modes, n_out, n_in)

    def composition(dx_, dz_):
        d_modes = modes.jac(x, z, dx_, dz_)
        d_amps = _jac_amps(modes, d_modes, n_out, n_in)
        return _jac_dist(amps - amps0, d_amps, metric)

    return composition(dx, dz)

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
        self._metric = _metric_amps(
            self.freqs, self.fs, self.ns, a_type='normal')

    def _fun(self, x, amps, dof:int):
        x = x.reshape(self.n_out, dof)
        amps_ = _mode_to_amps(x, self.n_out, self.n_in)

        dif = amps_ - amps
        dif = np.concatenate(
            [np.real(dif), np.imag(dif)], axis=-1)
        trans = np.einsum('ijk,kl->ijl', dif, self._metric)

        # Compute f(x).
        f = np.einsum('ijk,ijk', trans, dif)
        return f
    
    def _jac(self, x, amps, dof:int):
        x = x.reshape(self.n_out, dof)
        amps_ = _mode_to_amps(x, self.n_out, self.n_in)

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

        amps_ = _mode_to_amps(x, self.n_out, self.n_in)
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

        x0 = np.real(_amps_to_modes(amps))
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
        self._metric = _metric_amps(
            self.freqs, self.fs, self.ns, a_type='normal')

    def _fun(self, x, amps, dof:int):
        x = x.reshape(self.n_out, dof)
        amps_ = _mode_to_amps(x, self.n_out, self.n_in)

        dif = amps_ - amps
        dif = np.concatenate(
            [np.real(dif), np.imag(dif)], axis=-1)
        trans = np.einsum('ijk,kl->ijl', dif, self._metric)

        # Compute f(x).
        f = np.einsum('ijk,ijk', trans, dif)
        return f
    

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dof, n_out, n_in = 4, 3, 2
    rng = np.random.default_rng()
    freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
    x0 = 0.1*rng.normal(size=(n_out, dof))
    z0 = 1e-3*rng.normal(size=(n_out, dof))
    amps0 = _mode_to_amps(x0, n_out, n_in)
    coords = np.arange(n_out, dof)

    ns, fs = 210, 100
    metric = _metric_amps(freqs, fs, ns)

    xi = 0.1*rng.normal(size=(n_out, dof))
    zi = 1e-3*rng.normal(size=(n_out, dof))
    dx = 0.1*rng.normal(size=(n_out, dof))
    dz = 1e-3*rng.normal(size=(n_out, dof))
    ll = 1e-2 * np.arange(-40, 41)
    f_line = [
        _fun(xi+l*dx, zi+l*dz, freqs, coords, n_out, n_in, metric, amps0)
        for l in ll]
    f_line = np.array(f_line)

    fig, ax = plt.subplots(ncols=1, sharex=True, figsize=(5, 5))
    fig.suptitle('Test derivatives of objective for real mode shapes fitting')

    ax.set_title('1st order')
    ax.plot(ll, f_line)
    # ax.axhline(0, color='k', linestyle='--', linewidth=1)

    # ax[1].set_title('2nd order')
    # ax[1].plot(L, f_line - (fx + (dfx@v)*L + 0.5*(ddfxp@v)*(L**2)))
    # # ax[1].axhline(0, color='k', linestyle='--', linewidth=1)
    plt.show()



    # dof, n_out, n_in = 4, 3, 2
    # rng = np.random.default_rng()
    # freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
    # X = 0.1*rng.normal(size=(n_out, dof))
    # amps0 = _mode_to_amps(X, n_out, n_in)
    # amps = amps0 + 1e-4*(rng.normal(size=amps0.shape) + 1j*rng.normal(size=amps0.shape))
    # ns, fs = 210, 100

    # modes = _ModesProp(
    #     freqs, fs, ns, n_out=n_out, n_in=n_in)
    # res = modes.fit(amps)
    # print('Original amps:\n', amps0)
    # print('Noisy amps:\n', amps)
    # print('===================')
    # print('Original:\n', X)
    # print('Fitted:\n', res.x.reshape(n_out, dof))

    # print(res.success, res.message)


    # # ================================
    # # Test reshapes amplitudes
    # # ================================
    # seed = 1234345
    # rng = np.random.default_rng(seed)
    # dof, n_out, n_in = 4, 3, 2

    # x = rng.normal(
    #     size=(n_in*(n_in+1)//2 + (n_out-n_in)*n_in, 2*dof - 1))
    # ix = _reshape_injection(x, n_out=n_out, n_in=n_in)
    # pix = _reshape_projection(ix)

    # print('- Test reshapes: ', np.allclose(x, pix))


    # # ================================
    # # Test reshapes mode shapes
    # # ================================
    # dof, n_out, n_in = 4, 3, 2
    # coords = np.arange(n_out, dof)
    # rng = np.random.default_rng(1268)
    # freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)

    # # Reference mode shape.
    # X0 = 0.1*rng.normal(size=(n_out, dof))
    # Z0 = 0.01*rng.normal(size=(n_out, dof))
    # Z0 = np.triu(Z0, k=1)
    # Z0[:n_out, :n_out] = Z0[:n_out, :n_out] - Z0[:n_out, :n_out].T
    # psi0 = _partial_modes_map(X0, Z0, freqs, coords=coords)[0]
    # amps0 = _mode_to_amps(psi0, n_out, n_in)
    # x = _reshape_mode_shapes_output(X0, Z0)
    # X0_, Z0_ = _reshape_mode_shapes_input(x, dof, n_out)

    # print('- Test reshapes: ', np.allclose(X0_, X0), np.allclose(Z0_, Z0))


    # # ================================
    # # Test loss function and df
    # # for real mode shapes
    # # ================================
    # dof, n_out, n_in = 4, 3, 2
    # rng = np.random.default_rng()
    # freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
    # X = 0.1*rng.normal(size=(n_out, dof))
    # amps = _mode_to_amps(X, n_out, n_in)
    # ns, fs = 210, 100

    # modes = _ModesProp(
    #     freqs, fs, ns, n_out=n_out, n_in=n_in)
    # x = rng.normal(scale=.1, size=n_out*dof)
    # v = rng.normal(scale=.1, size=n_out*dof)
    # fx, dfx = modes._fun(x, amps, dof)
    # ddfxp = modes._hessp(x, v, amps, dof)

    # L = 1e-3 * np.arange(-40, 41)
    # f_line = [modes._fun(x + l*v, amps, dof)[0] for l in L]
    # f_line = np.array(f_line)

    # fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(10, 5))
    # fig.suptitle('Test derivatives of objective for real mode shapes fitting')

    # ax[0].set_title('1st order')
    # ax[0].plot(L, f_line - (fx + (dfx@v)*L))
    # ax[0].axhline(0, color='k', linestyle='--', linewidth=1)

    # ax[1].set_title('2nd order')
    # ax[1].plot(L, f_line - (fx + (dfx@v)*L + 0.5*(ddfxp@v)*(L**2)))
    # ax[1].axhline(0, color='k', linestyle='--', linewidth=1)
    # plt.show()
