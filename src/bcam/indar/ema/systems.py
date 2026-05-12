from itertools import product

import numpy as np
import scipy
import jitcode


# === Systems ===

class Beam:

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
        y = jitcode.y
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
