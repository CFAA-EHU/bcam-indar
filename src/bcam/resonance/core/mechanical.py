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

# =================================
# Amplitudes
# =================================

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

def _local_matrix(
        freqs, fs: float, ns: int, penalty: float = 0.0):
    N = len(freqs)
    r = np.zeros((2*N, 2*N))

    def _mult(x, y):
        r = x[np.newaxis, :] + y[:, np.newaxis]
        r = np.exp(r/(2*fs)) * _sum_exp_weighted(r, fs, ns)
        r *= (4*fs**2)*np.sinh(x[np.newaxis, :]/(2*fs)) * np.sinh(y[:, np.newaxis]/(2*fs))
        return r

    r1 = _mult(freqs, np.conj(freqs))
    r2 = _mult(freqs, freqs)

    r[:N, :N] = np.real(r1 - r2)
    r[N:, :N] = np.imag(r1 + r2)
    r[:N, N:] = r[N:, :N].T
    r[N:, N:] = np.real(r1 + r2)
    r *= 0.5
    r += penalty * np.eye(2*N)

    return r

def _reshape_injection(
        x, dof: int, n_out:int, n_in:int):
    L = x.shape[-1]
    x_ = np.zeros((n_out, n_in, 2*dof - 1, L), dtype=x.dtype)
    x = x.reshape(
        n_in*(n_in+1)//2 + (n_out-n_in)*n_in, 2*dof - 1, L)
    x_[(np.arange(n_in), np.arange(n_in))] = x[:n_in]
    c = n_in
    for i in range(1, n_in):
        cn = c + n_in - i
        x_[(np.arange(n_in-i), i+np.arange(n_in-i))] = x[c:cn]/np.sqrt(2)
        x_[(i+np.arange(n_in-i), np.arange(n_in-i))] = x[c:cn]/np.sqrt(2)
        c = cn
    x_[n_in:, :] = x[c: c + (n_out-n_in)*n_in].reshape(n_out - n_in, n_in, 2*dof - 1, L)/np.sqrt(2)
    return x_

def _reshape_projection(
        x, dof: int, n_out:int, n_in:int):
    L = x.shape[-1]
    x_ = np.zeros(
        (n_in*(n_in+1)//2 + (n_out-n_in)*n_in, 2*dof - 1, L),
        dtype=x.dtype)
    x_[:n_in] = x[(np.arange(n_in), np.arange(n_in))]
    c = n_in
    for i in range(1, n_in):
        cn = c + n_in - i
        x_[c:cn] = (x[(np.arange(n_in-i), i+np.arange(n_in-i))] + x[(i+np.arange(n_in-i), np.arange(n_in-i))]) / np.sqrt(2)
        c = cn
    x_[c: c + (n_out - n_in)*n_in] = x[n_in:, :].reshape((n_out - n_in)*n_in, 2*dof - 1, L) * np.sqrt(2)
    return x_.reshape(-1, L)


class _PreCoeff_to_Kernel(scipy.sparse.linalg.LinearOperator):
    
    def __init__(
            self,
            freqs, fs: float, ns: int,
            n_out:int, n_in:int,
            penalty:float=0.0):
        self.freqs = freqs
        self.fs = fs
        self.ns = ns
        self.n_out = n_out
        self.n_in = n_in
        self.penalty = penalty

        dof = len(self.freqs)
        M = (n_in*(n_in+1)//2 + (n_out-n_in)*n_in) * (2*dof - 1)
        super().__init__(
            dtype=np.float64,
            shape=(M, M))

        # Change basis to mean im = 0.
        lm = _local_matrix(freqs, fs, ns, penalty)
        plm = np.zeros((2*dof-1, 2*dof-1), dtype=lm.dtype)
        plm[dof:, :dof] = _trig_fft(lm[dof:, :dof].T)[..., 1:].T
        plm[:dof, dof:] = plm[dof:, :dof].T
        plm[dof:, dof:] = _trig_fft(_trig_fft(lm[dof:, dof:])[..., 1:].T)[..., 1:]
        self._local_matrix = lm

    def _matmat(self, x):
        dof = len(self.freqs)
        r = _reshape_injection(x, dof, self.n_out, self.n_in)
        M_local = self._local_matrix
        r = np.einsum('lk,ijkm->ijlm', M_local, r)
        return _reshape_projection(r, dof, self.n_out, self.n_in)

    def _adjoint(self):
        return self


class _PreCoeff():

    def __init__(
            self,
            freqs, fs: float, ns: int,
            n_out:int=None, n_in:int=None):
        self.freqs = freqs
        self.fs = fs
        self.ns = ns
        self.n_out = len(freqs) if n_out is None else n_out
        self.n_in = n_out if n_in is None else n_in

    @property
    def amplitudes_(self):
        dof = len(self.freqs)
        x = self.raw_coeff_
        r = np.concatenate(
            [np.zeros((*x.shape[:2], 1)), x[..., dof:]], axis=-1)
        r = _trig_ifft(r).astype(np.complex128)
        r = x[..., :dof] + 1j*r
        return r

    def _rhs(self, y):
        freqs = self.freqs
        fs, ns = self.fs, self.ns

        def prod(t):
            T = ns / fs
            t = t.reshape(1, -1)
            freqs_ = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
            r_ = np.exp(freqs[:, np.newaxis] * t) * (1 - t/T)
            r_ *= freqs_[:, np.newaxis]
            return r_

        r = np.einsum(
            'ijt,kt->ijk',
            y,
            prod(np.arange(ns)/fs))
        r = r / fs
        r = np.concatenate(
            [np.imag(r), _trig_fft(np.real(r))[..., 1:]],
            axis=-1
        )

        return r

    def fit(
            self, y,
            penalty:float =0.0, cg_kwargs=None):
        dof = len(self.freqs)
        cg_kwargs = {} if cg_kwargs is None else cg_kwargs
        lhs = _PreCoeff_to_Kernel(
            self.freqs, self.fs, self.ns,
            self.n_out, self.n_in,
            penalty=penalty)
        rhs = self._rhs(y)
        rhs = _reshape_projection(
            rhs[..., np.newaxis], dof, self.n_out, self.n_in)
        r, info = scipy.sparse.linalg.cg(lhs, rhs, **cg_kwargs)
        if info != 0:
            logger.warning(f'Conjugate gradient did not converge, info={info}')

        r = _reshape_injection(
            r[..., np.newaxis], dof, self.n_out, self.n_in)
        self.raw_coeff_ = r[..., 0]
        return self.amplitudes_

# =================================
# Modal Parameters
# =================================

def _partial_mode_shapes_map(x, freqs, dof, n_outputs):
    No = n_outputs
    a = x[:dof*No].reshape(No, dof)
    q, r, _ = scipy.linalg.qr(
        a.T, overwrite_a=False, mode='full', pivoting=False)
    r = np.min(r[(np.arange(No), np.arange(No))])
    if r < 1e-8:
        logging.warning('Real part does not have full rank.')
        return np.nan

    lu = x[dof*No: dof*No + No*(No-1)//2]
    l_ = np.zeros((No, No))
    init = 0
    for i in range(No-1):
        end = init + No - i - 1
        l_[i, i+1:] = lu[init: end]
        l_[i+1:, i] = -lu[init: end]
        init = end
    lu = l_
    lc = x[dof*No + No*(No-1)//2:].reshape(No, dof - No)
    l = np.concatenate((lu, lc), axis=1)

    d = freqs.real, freqs.imag
    tmp = q @ l.T
    H = (d[0][:, np.newaxis] * q[:, :No]).T @ tmp
    H += H.T
    H += (d[1][:, np.newaxis] * q[:, :No]).T @ q[:, :No]
    H += -tmp.T @ (d[1][:, np.newaxis] * tmp)

    try:
        scipy.linalg.cholesky(H, lower=False, overwrite_a=True)
    except scipy.linalg.LinAlgError:
        logging.warning('Mass matrix is not positive-definite.')
        return np.nan

    l = q[:, :No] @ l @ q.T

    psi = a @ (np.eye(dof) + 1j * l)
    psi = psi * np.sqrt(np.imag(freqs))[np.newaxis, :]

    return psi

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
        damping_type='proportional',
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
        if damping_type == 'proportional':
            Uc = Uk
        elif damping_type == 'non-proportional':
            Uc = U.rvs()
        else:
            msg = 'Invalid damping type. Choose between "proportional" and "non-proportional".'
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

    if damping_type == 'proportional':
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

    seed = 1234345
    rng = np.random.default_rng(seed)
    dof, n_out, n_in = 4, 3, 2
    L = 2
    
    x = rng.normal(
        size=((n_in*(n_in+1)//2 + (n_out-n_in)*n_in)*(2*dof - 1), L))
    ix = _reshape_injection(x, dof, n_out=n_out, n_in=n_in)
    pix = _reshape_projection(ix, dof, n_out=n_out, n_in=n_in)

    print('test reshapes: ', np.allclose(x, pix))
