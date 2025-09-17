#!/usr/bin/env python

from itertools import product
import warnings

import numpy as np
import scipy
from jitcode import jitcode, y, t
import scipy.linalg
import scipy.optimize
import symengine as se


def _validate_dims(M, C, K, check_symmetry=True):
    M, C, K = np.asarray(M), np.asarray(C), np.asarray(K)
    
    if M.ndim != 2:
        msg = f'Expected a 2-darray, got {M.ndim}-darray.'
        raise ValueError(msg)
    
    if M.shape[0] != M.shape[1]:
        msg = 'M must be a square array.'
        raise ValueError(msg)
    
    if M.shape != C.shape or M.shape != K.shape:
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

def _compatibility_linear(x, No, N):
    if No > N:
        msg = f'Expected No <= N, got No = {No}, N = {N}.'
        raise ValueError(msg)
    
    r = No * (No + 1) // 2
    d = No * N
    if len(x) != 2 * d - r:
        msg = f'Expected {2 * d - r} elements, got {len(x)}.'
        raise ValueError(msg)
    
    beta = x[:d].reshape((No, N))
    M = np.zeros((r, d))
    k = 0
    for i, j in product(range(No), repeat=2):
        if i > j:
            continue
        M[k, i * N: (i + 1) * N] = beta[j]
        M[k, j * N: (j + 1) * N] = beta[i]
        k += 1
    Q = scipy.linalg.qr(M.T, overwrite_a=True, pivoting=True)[0]
    alpha = Q[:, r:] @ x[d:]
    alpha = alpha.reshape((No, N))

    return alpha, beta

def mode_shapes_generator(X, L, D=None, check=True):
    '''
    Generates mode shapes with real part X.
    The algorithm creates a pair of matrices (X, Y) such that
    X @ D @ Y.T + Y @ D @ X.T = 0, where
    D is a positive diagonal matrix; by default D = I.

    Parameters
    ----------
    X : ndarray
        Real part of mode shapes.
    L : ndarray
        Parameterization of the complex part of the mode shapes.
        Same shape as X, but it only uses the upper triangular part,
        and ignores the remaining entries.
    D : ndarray, optional
        Diagonal factors.
    seed : int, optional
        Seed for random number generator.

    Returns
    -------
    X + 1j * Y : ndarray
        Mode shapes.
    '''
    X = np.asarray(X)
    N = X.shape[0]
    if check:
        if np.ndim(D) > 1:
            msg = f'Invalid shape for D. Expected (N,), got {D.shape}.'
            raise ValueError(msg)
        if X.shape[0] > X.shape[1]:
            msg = f'Expected a (N, M) matrix X with N <= M, got N > M.'
            raise ValueError(msg)
    D = np.ones(N) if D is None else D
    
    # Redifine X, Y so that now X @ Y.T + Y @ X.T = 0.
    X = X * np.sqrt(D)[np.newaxis, :]
    Y = np.zeros_like(X)

    for k in range(N):
        y_m = np.linalg.lstsq(
            X[:k+1],
            -Y[:k+1] @ X[k],
            rcond=None
            )
        y_m = y_m[0]
        _, _, V = np.linalg.svd(X[:k+1])
        V = V[k+1:].T
        Y[k] = y_m + V @ L[k, k+1:]
    mode_shapes = X + 1j * Y

    # Check linear independence.
    if check:
        rank = np.linalg.matrix_rank(mode_shapes)
        if rank < N:
            msg = f'Mode shapes are not linearly independent.'
            warnings.warn(msg, stacklevel=2)

    # Remove the D factor.
    mode_shapes = mode_shapes * np.sqrt(1 / D)[np.newaxis, :]
    return mode_shapes

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
    N = M.shape[0]
    if N != M.shape[1]:
        msg = 'M must be a square matrix.'
        raise ValueError(msg)
    if M.shape != C.shape or M.shape != K.shape:
        msg = 'Incompatible shapes for M, C, and K.'
        raise ValueError(msg)

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


def tensor_mode_shapes(amplitudes, poles, excitations, fs, Ni):
    '''
    Parameters
    ----------
    amplitude : 3D-array
        Shape (n_responses, n_exitations, n_poles).
    poles : 1D-array
        The poles equal exp(Z / fs).
    excitations : 2D-array
        Shape (n_times, n_excitations).
    fs : 2-tuple
        Sampling frequencies for excitation and response, respectively.
    Ni : int
        Sample where the response starts, in response's sample rate.

    Returns
    -------
    tensor : 3D-array
        Shape (n_responses, n_excitations, n_poles).
    '''
    # Select poles with positive imaginary part.
    idx = np.nonzero(poles.imag > -1e-6)[0]
    poles = poles[idx]
    amplitudes = amplitudes[..., idx]
    
    N = excitations.shape[0]
    # tmp = exp(-Zt) with shape (n_times, n_poles)
    tmp = poles[np.newaxis, :] ** (-(fs[1]/fs[0]) * np.arange(N)[:, np.newaxis])
    tmp = np.expand_dims(tmp, axis=1) # (n_times, 1, n_poles)
    # Laplace of f (Lf) has shape (n_excitations, n_poles)
    Lf = scipy.integrate.simpson(
            excitations[..., np.newaxis] * tmp, # (n_times, n_excitations, n_poles)
            dx = 1/(fs[0]),
            axis=0)
    
    Z = np.log(poles) * fs[1]
    Z_ = Z[np.newaxis, ...] # (1, n_poles)
    # factor has shape (n_excitations, n_poles).
    factor = 2j * Z_.imag * (poles[np.newaxis, :]**(-Ni)) / ((Z_**2) * Lf)
    # tensor has shape (n_responses, n_excitations, n_poles)
    tensor = amplitudes * factor[np.newaxis, :]
    return tensor


# === Systems ===

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

    return (M, C, K), (Z, mode_shapes)


class BeamSystem:

    def __init__(self, density, flexibility, h=1, seed=None):
        self._density = np.asarray(density)
        self._flexibility = np.asarray(flexibility)
        self._h = h
        self._seed = seed

    def generate(self, damping=None):
        '''
        Parameters
        ----------
        damping : 2-tuple, optional
            Range for random damping values.
        '''
        density = self._density
        flexibility = self._flexibility
        h = self._h
        n_nodes = len(density)

        M = np.zeros((n_nodes-1, n_nodes-1))
        K = np.zeros((n_nodes-1, n_nodes-1))

        ref_M = np.zeros((3, 3, 2))
        ref_K = np.zeros((3, 3, 2))
        for i, j, k in product(range(-1, 2), range(-1, 2), range(2)):
            base_l = scipy.interpolate.BSpline.basis_element(np.arange(-1, 3) + i)
            base_r = scipy.interpolate.BSpline.basis_element(np.arange(-1, 3) + j)
            coeff = scipy.interpolate.BSpline.basis_element(np.arange(-1, 2) + k)
            # Polynomial of degree 5.
            f = lambda x: coeff(x) * base_l(x) * base_r(x)
            # Polynomial of degree 1.
            g = lambda x: coeff(x) * base_l.derivative(2)(x) * base_r.derivative(2)(x)
            # Quadrature exact for polynomials of degree 2n - 1.
            I, _ = scipy.integrate.fixed_quad(f, 0, 1, n=3)
            ref_M[i+1, j+1, k] = I
            I, _ = scipy.integrate.fixed_quad(g, 0, 1, n=1)
            ref_K[i+1, j+1, k] = I

        # Compute cells contributions to M and K.
        def _fill(r, l, k):
            I = h * ref_M[r+1, l+1, k]
            M[cell-1 + l, cell-1 + r] += density[cell + k] * I
            I = h * ref_K[r+1, l+1, k]
            K[cell-1 + l, cell-1 + r] += flexibility[cell + k] * I

        cell = 0
        r, l = 1, 1
        for k in range(2):
            _fill(r, l, k)

        cell = 1
        for r, l, k in product(range(2), repeat=3):
            _fill(r, l, k)

        for cell in range(2, n_nodes - 1):
            for r, l, k in product(range(-1, 2), range(-1, 2), range(2)):
                _fill(r, l, k)

        # Generate damping matrix.
        if damping is None:
            C = np.zeros((n_nodes - 1, n_nodes - 1))
        else:
            rng = np.random.default_rng(self._seed)
            C = rng.uniform(*damping, n_nodes - 1)
            diag_l = 0.5 * rng.uniform(*damping, n_nodes - 2)
            diag_ll = 0.25 * rng.uniform(*damping, n_nodes - 3)
            C = np.diag(C)
            for i in range(n_nodes - 2):
                C[i+1, i] = diag_l[i]
            for i in range(n_nodes - 3):
                C[i+2, i] = diag_ll[i]
            C = h * (C + C.T) / 2
        
        return M, C, K
    
    @staticmethod
    def basis2bending(c):
        n_nodes = len(c) + 1
        c = np.concatenate((2 * [0], c, 2 * [0]))
        nodes = list(range(n_nodes + 2))
        nodes = 2 * [nodes[0] - 1] + nodes + 2 * [nodes[-1] + 1]
        bending = scipy.interpolate.BSpline(nodes, c, 2)
        
        return bending


# === Simulations ===

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


class Hammer:

    def __init__(self, M, C, K, sigma=0.1, dt=None):
        self.M, self.C, self.K = _validate_dims(M, C, K)
        self.sigma = sigma
        self.dt = self.sigma/4 if dt is None else dt
        
        self._force = se.exp(-0.5 * ((t - 5 * sigma) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

    def simulate(
            self,
            exitation_points,
            T=10,
            solver_name='dopri5'
        ):
        '''
        Parameters
        ----------
        exitation_points : 1D or 2D-array
            If a 1D-array, it must have the same length as the number of DoFs.
            If a 2D-array, it must have shape (DoFs, N), where N is the number of exitation points.
        '''
        dofs = self.M.shape[0]

        # Validate the exitation_points.
        exitation_points = np.array(exitation_points)
        if exitation_points.ndim == 1:
            exitation_points = exitation_points[:, np.newaxis]
        elif exitation_points.ndim > 2:
            msg = f'Invalid number of dimensions for exitation_point. Expected 1 or 2, got {exitation_points.ndim}.'
            raise ValueError(msg)
        
        if exitation_points.shape[0] != dofs:
            msg = f'Invalid shape for exitation_point. Expected ({dofs}, ...), got ({exitation_points.shape[0]}, ...).'
            raise ValueError(msg)
        exitation_points = exitation_points / np.linalg.norm(exitation_points, axis=0, keepdims=True)
        n_excitations = exitation_points.shape[1]
        
        dt = self.dt
        n_samples = round(T / dt)
        data = np.zeros((dofs, n_excitations, n_samples))
        for n, point in enumerate(exitation_points.T):
            system = Spring(
                self.M, self.C, self.K, point * self._force)

            initial_state = np.zeros(2*dofs)
            solver = jitcode(system)
            solver.set_integrator(solver_name)
            solver.set_initial_value(initial_state, 0.0)
            solution = []
            for time in dt * np.arange(0, n_samples+1):
                solution.append(solver.integrate(time))
            solution = np.array(solution)

            acc = solution[:, dofs:]
            acc = (acc[2:] - acc[:-2]) / (2 * dt)
            acc = np.append(acc, np.zeros((1, dofs)), axis=0)
            acc = np.roll(acc, 1, axis=0)
            data[:, n, :] = acc.T

        impact = se.lambdify(t, [self._force])
        impact = impact(dt * np.arange(0, n_samples))

        return data, impact


class SymmetricModel:

    def __init__(self, tensor, Z, excitation_idxs):
        '''
        Parameters
        ----------
        tensor : 3D-array
            Shape (n_res, n_ext, n_dof).
        '''
        # Tensor is reshaped to (n_dof, n_res, n_ext)
        # to access faster to the last two axes.
        self.tensor = np.transpose(tensor, axes=(2, 0, 1))
        self.Z = Z
        self.excitation_idxs = np.asarray(excitation_idxs)

        if self.tensor.shape[0] != Z.shape[0]:
            msg = f'Invalid tensor shape. Expected {(self.tensor.shape[0], ...)}, got {Z.shape[0]}.'
            raise ValueError(msg)
        
        if self.tensor.shape[2] != self.excitation_idxs.shape[0]:
            msg = f'Invalid tensor shape. Expected {(..., self.excitation_idxs.shape[0])}, got {self.tensor.shape[2]}.'
            raise ValueError(msg)


    def fastFit(self):
        n_dof, n_res, n_ext = self.tensor.shape
        guess = np.zeros((n_dof, n_res), dtype=np.complex128)
        _guess = np.zeros((n_dof, n_ext), dtype=np.complex128)
        d_idxs = self.excitation_idxs
        
        for dof in range(n_dof):
            tmp = self.tensor[dof][(d_idxs, np.arange(n_ext))]
            guess[dof, d_idxs] = np.sqrt(tmp)

        Dom_idxs = np.argmax(np.abs(guess[:, d_idxs]), axis=1)
        Dom = guess[(np.arange(n_dof), d_idxs[Dom_idxs])]
        weak = np.setdiff1d(
            np.arange(n_res), d_idxs, assume_unique=True)
        for dof in range(n_dof):
            guess[dof, weak] = self.tensor[dof, weak, Dom_idxs[dof]] / Dom[dof]
            _guess[dof] = self.tensor[dof, d_idxs, Dom_idxs[dof]] / Dom[dof]
            _guess[dof] = np.sign(_guess[dof].real)
        guess[:, d_idxs] *= _guess
        self._guess_ = guess

        return self
    
    @property
    def guess_(self):
        return self._guess_.T
    
    def _basic_setup(self, x):
        n_dof, n_res, n_ext = self.tensor.shape
        x = x[::2] + 1j * x[1::2]
        return n_dof, n_res, n_ext, x.reshape((n_dof, n_res))
    
    def _loss(self, x):
        _, n_res, n_ext, x = self._basic_setup(x)
        tensor_ = x[:, :n_res, np.newaxis] * x[:, np.newaxis, :n_ext]
        return np.sum(
            np.linalg.norm(tensor_ - self.tensor, ord='fro', axis=(1, 2))**2)
    
    def _jac_loss(self, x):
        n_dof, n_res, n_ext, x = self._basic_setup(x)
        tensor_ = x[:, :n_res, np.newaxis] * x[:, np.newaxis, :n_ext]
        A = tensor_ - self.tensor
        S1 = A @ np.conj(x[:, :n_ext, np.newaxis])
        S2 = np.pad(
            np.swapaxes(A, 1, 2) @ np.conj(x[:, :n_res, np.newaxis]),
            ((0, 0), (0, n_res - n_ext), (0, 0)))
        # Sum and remove last axis.
        partial = np.squeeze(S1 + S2, axis=-1)
        r = np.zeros((n_dof, 2 * n_res))
        r[:, ::2], r[:, 1::2] = 2 * partial.real, 2 * partial.imag
        
        return r.flatten()


class SymmetricModel2:

    def __init__(self, tensor, Z):
        self._tensor = np.asarray(tensor).transpose((2, 0, 1))
        self.Z = Z

        n_dof, n_res, n_ext = self._tensor.shape
        mask = np.zeros((n_res, n_ext), dtype=bool)
        for i, j in product(range(n_ext), repeat=2):
            mask[i, j] = (j > i)
        self.mask = mask
        mask_extended = np.expand_dims(mask, axis=2)
        mask_extended = np.repeat(mask_extended, n_dof * n_res, axis=2)
        self.mask_extended = mask_extended

    @property
    def tensor(self):
        self._tensor.transpose((1, 2, 0))

    def _fitAbs(self):
        n_dof, n_res, n_ext = self._tensor.shape
        normr = np.zeros(n_dof)
        abs_mode_shapes = np.zeros((n_res, n_dof))
        # TODO: Use CSR sparse matrix.
        A = np.zeros((n_res * n_ext, n_res))
        for j in range(n_res):
            A[j * n_ext: (j+1) * n_ext, j] += 1
        for n in range(n_ext):
            A[n::n_ext, n] += 1

        for dof in range(n_dof):
            X = np.abs(self._tensor[dof])
            absModes, _, _, normr_ = scipy.sparse.linalg.lsmr(
                A * X.flatten()[:, np.newaxis],
                (X * np.log(X)).flatten())[:4]
            abs_mode_shapes[:, dof] = np.exp(absModes)
            normr[dof] = normr_

        self.abs_mode_shapes_ = abs_mode_shapes
        self.absNormr_ = np.array(normr)
        absTensor = abs_mode_shapes[:n_res, np.newaxis, :] * abs_mode_shapes[np.newaxis, :n_ext, :]
        absTensor /= self.Z[np.newaxis, np.newaxis, :].imag
        absTensor /= np.linalg.norm(absTensor, ord=2, axis=-1, keepdims=True)
        self._absTensor = absTensor # (n_res, n_ext, n_dof)

        return self
    
    def _loss(self, x):
        n_dof, n_res, n_ext = self._tensor.shape
        x = x.reshape((n_dof, n_res))
        tmp = x[:, :n_res, np.newaxis] + x[:, np.newaxis, :n_ext]
        tmp = tmp - np.angle(self._tensor)
        tmp = (np.exp(1j * tmp) - 1) * np.conj(np.exp(1j * tmp) - 1)
        tmp *= self._tensor * np.conj(self._tensor)
        return np.sum(tmp).real

    def _jac_loss(self, x):
        n_dof, n_res, n_ext = self._tensor.shape
        x = x.reshape((n_dof, n_res))
        tmp = x[:, :n_res, np.newaxis] + x[:, np.newaxis, :n_ext]
        tmp = tmp - np.angle(self._tensor)
        tmp = 2 * np.sin(tmp)
        tmp = self._tensor * np.conj(self._tensor) * tmp
        S1 = np.sum(tmp, axis=2)
        S2 = np.pad(
            np.sum(tmp, axis=1),
            ((0, 0), (0, n_res - n_ext)))
        return (S1 + S2).flatten().real
    
    def _compatibility(self, x):
        n_dof, n_res, n_ext = self._tensor.shape
        x = x.reshape((n_dof, n_res))
        x = x[:, :n_res, np.newaxis] + x[:, np.newaxis, :n_ext]
        S = self._absTensor * np.sin(x.transpose((1, 2, 0)))
        S = np.sum(S, axis=-1)
        # S has shape (n_res, n_ext) before compressing.
        S = np.ma.masked_array(S, mask=self.mask)
        return S.compressed()
    
    def _jac_compatibility(self, x):
        n_dof, n_res, n_ext = self._tensor.shape
        x = x.reshape((n_dof, n_res))
        x = x[:, :n_res, np.newaxis] + x[:, np.newaxis, :n_ext]
        # S has shape (n_res, n_ext, n_dof).
        S = self._absTensor * np.cos(x.transpose((1, 2, 0)))

        # TODO: Use CSR sparse matrix. Is it worth it? How sparse is it?
        jac = np.zeros((n_res, n_ext, n_dof, n_res))
        for k in range(n_ext):
            jac[k, :, :, k] += S[k, :, :]
            jac[:, k, :, k] += S[:, k, :]
        for k in range(n_ext, n_res):
            jac[k, :, :, k] += S[k, :, :]
        jac = jac.reshape((n_res, n_ext, n_dof * n_res))
        jac = np.ma.masked_array(jac, mask=self.mask_extended)
        jac_ = np.zeros(
            ((n_ext * (n_ext + 1)) // 2 + (n_res - n_ext) * n_ext, n_dof * n_res))
        for k in range(n_dof * n_res):
            jac_[:, k] = jac[..., k].compressed()
        return jac_
    
    # TODO: Add Hessians.

    def _fitAngle(self, x0=None):
        n_dof, n_res, _ = self._tensor.shape

        constraint = scipy.optimize.NonlinearConstraint(
            fun=self._compatibility,
            lb=0, ub=0,
            jac=self._jac_compatibility,
            # hess='3-point',
            )
        
        x0 = np.zeros(n_dof * n_res) if x0 is None else np.asarray(x0)
        r = scipy.optimize.minimize(
            self._loss,
            x0=x0,
            method='SLSQP',
            jac=self._jac_loss,
            # hess='3-point',
            constraints=constraint,
            options={'disp': False},
            )

        print('Success : ', r.success)
        phase = r.x.reshape((n_dof, n_res)).T
        phase = np.exp(1j * phase)
        self.mode_shapes_ = self.abs_mode_shapes_ * phase

        return self

def fastGuess(tensor):
    tensor = np.transpose(tensor, axes=(2, 0, 1))
    n_dof, n_res, n_ext = tensor.shape
    guess = np.zeros((n_dof, n_res))
    _guess = np.zeros((n_dof, n_ext))

    for dof in range(n_dof):
        tmp = tensor[dof][(np.arange(n_ext), np.arange(n_ext))]
        guess[dof, :n_ext] = np.sqrt(tmp)

    Dom_idxs = np.argmax(np.abs(guess[:, :n_ext]), axis=1)
    Dom = guess[(np.arange(n_dof), Dom_idxs)]
    for dof in range(n_dof):
        guess[dof, n_ext:] = tensor[dof, n_ext:, Dom_idxs[dof]] / Dom[dof]
        _guess[dof] = tensor[dof, :n_ext, Dom_idxs[dof]] / Dom[dof]
        _guess[dof] = np.sign(_guess[dof].real)
    guess[:, :n_ext] *= _guess

    return guess.T

class SymmetricModel3:
    
    def __init__(self, tensor, Z):
        self.tensor = np.asarray(tensor)
        self.Z = Z

    def fit(self):
        pass
