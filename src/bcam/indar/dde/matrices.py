'''
Finite dimensional approximations of Differential Delay Equations (DDEs)
'''

# ==== Imports ====
from itertools import product

import numpy as np
import scipy
import sympy

# ==== Functions ====
def inv_cayley(x, a=1, b=1):
    return (b*x - a) / (x + 1)

def cofactor_matrix(M):
    M = np.asarray(M)
    size = M.shape[0]
    C = np.zeros_like(M)
    # remove row i and column j from M.
    for i, j in product(range(size), repeat=2):
        C[i, j] = (-1)**(i+j) * np.linalg.det(M[np.arange(size)!=i, :][:, np.arange(size)!=j])
    return C

def jacobi(M, dM):
    return np.trace(cofactor_matrix(M).T @ dM)

def test_eigenvalue(B0, B1, eigval, tau=1):
    r'''Test the quality of the approximation of the eigenvalue.

    For the delay equation :math:`y'(t) = B_0 y(t) + B_1 y(t - \tau)`,
    test how well the eigenvalue `eigval` solves the characteristic equation  :math:`\Delta(\lambda) := \det(\lambda I - B_0 - e^{-\tau\lambda} B_1) = 0`.
    
    Parameters
    ----------
    B0 : array_like
        The matrix B0.
    B1 : array_like
        The matrix B1.
    eigval : float
        The eigenvalue to test.
    tau : float, optional
        The time delay. The default is 1.

    Returns
    -------
    float
        :math:`\frac{|\Delta(\lambda)|}{|\Delta'(\lambda)|}`.
    '''
    if np.isnan(eigval):
        return np.nan
    B0 = np.array(B0)
    B1 = np.array(B1)
    B0, B1 = _validate_coeffs(B0, B1)
    N = B0.shape[0]
    Id = np.identity(N)
    L = eigval * Id - B0 - np.exp(-tau * eigval) * B1
    dL = Id + tau * np.exp(-tau * eigval) * B1
    d_dis = jacobi(L, dL)
    return np.abs(np.linalg.det(L) / d_dis)

def _validate_coeffs(B0, B1):
    if np.iscomplexobj([B0, B1]):
        dtype = complex
    elif np.isrealobj([B0, B1]):
        dtype = float
    else:
        msg = 'B0 and B1 must be either real or complex.'
        raise ValueError(msg)
    if B0.shape != B1.shape:
        raise ValueError('B0 and B1 must have the same shape.')
    if B0.ndim > 2:
        raise ValueError('B0 and B1 must be at most 2D.')
    if B0.size == 1:
        d = 2 - B0.ndim
        B0 = np.expand_dims(B0, axis=list(range(d)))
        B1 = np.expand_dims(B1, axis=list(range(d)))
    if B0.shape[0] != B0.shape[1]:
                raise ValueError('B0 and B1 must be square matrices.')
    return np.array(B0, dtype=dtype), np.array(B1, dtype=dtype)

# ==== Semi-discretization matrices ====
class Semi_Discretization:

    def __init__(self, degree):
        if degree not in [0, 1]:
            raise ValueError("Degree must be 0 or 1.")
        self.degree = degree
        if degree == 0:
            self._matrix = self._semi_matrix_0
        elif degree == 1:
            self._matrix = self._semi_matrix_1
    
    def matrix(self, B0, B1, M:int, tau:float=1, check=True):
        '''
        Build the semi-discretization matrix with B-splines of order 1.
        
        Parameters
        ----------
        B_0 : array_like
            The matrix B0.
        B_1 : array_like
            The matrix B1.
        M : int
            The number of time steps.
        tau : float, optional
            The time delay. The default is 1.
    
        Returns
        -------
        csr_matrix
            The semi-discretization matrix.
        '''
        B0 = np.array(B0)
        B1 = np.array(B1)
        self._dtype = complex
        if check:
            B0, B1 = _validate_coeffs(B0, B1)
            self._dtype = B0.dtype

        return self._matrix(B0, B1, M, tau)

    def _semi_matrix_0(self, B0, B1, M:int, tau):
        # correct effect of time delay.
        B0, B1 = tau * B0, tau * B1
        # basic definitions.
        h = 1 / M
        N = np.shape(B0)[0]
        Id = scipy.sparse.eye_array(N, format='csr')
        A = scipy.linalg.inv(B0)
        B = scipy.linalg.expm(h * B0)
        # build the matrix.
        # subdiagonal elements.
        S_t = scipy.sparse.diags_array(
            np.ones(M), offsets=0, shape=(M, M), format='csr')
        S_t = scipy.sparse.kron(S_t, Id, format='csr')
        # (M - 1, 0) block
        S_bl = ((B - np.eye(N)) @ A @ B1)
        # (M - 1, M - 1) block
        S_br = B

        bottom = scipy.sparse.block_array(
            [[scipy.sparse.csr_array((N, (M-1)*N)),
             S_br]], format='csr')
        S = scipy.sparse.block_array(
            [[None, S_t],
             [S_bl, bottom]],
             format='csr', dtype=self._dtype)

        return S

    def _semi_matrix_1(self, B0, B1, M:int, tau=1):
        # correct effect of time delay.
        B0, B1 = tau * B0, tau * B1
        # basic definitions.
        h = 1 / M
        N = np.shape(B0)[0]
        Id = np.identity(N)
        A = scipy.linalg.inv(B0)
        B = scipy.linalg.expm(h * B0)
        # build the matrix.
        # subdiagonal elements.
        S_t = scipy.sparse.diags_array(
            np.ones(M), offsets=0, shape=(M, M), format='csr')
        S_t = scipy.sparse.kron(S_t, Id, format='csr')
        # (M - 1, 0) block
        S_M0 = A @ (B + (1/h)*A - (1/h)*A @ B) @ B1
        # (M - 1, 1) block
        S_M1 = -A @ (Id + (1/h)*A - (1/h)*A @ B) @ B1
        # (M - 1, M - 1) block
        S_br = B

        bottom = scipy.sparse.block_array(
            [[S_M1,
             scipy.sparse.csr_array((N, (M-2)*N)),
             S_br]], format='csr')
        S = scipy.sparse.block_array(
            [[None, S_t],
             [S_M0, bottom]],
            format='csr', dtype=self._dtype)

        return S
    
    def eigvals(self, B0, B1, M, tau=1):
        r'''Approximated characteristic roots.
        
        Eigenvalues are approximated modulo :math:`2\pi / (h\tau)`.
        '''
        S = self.matrix(B0, B1, M, tau)
        eigs = scipy.linalg.eigvals(S.todense(), overwrite_a=True)
        eigs = np.log(eigs) * M / tau

        return eigs

# ==== Tustin matrices ====
class Tustin:
    '''
    Construction of Tustin matrices for delay differential equations (DDE).

    Parameters
    ----------
    degree : int
        The degree of the B-spline basis functions.
        If the method is 'trigonometric', this parameter is ignored.

    method : str, optional
        The method to use for the construction of the matrices.
    '''

    def __init__(self, degree=None, method='bspline'):
        self.degree = degree
        if method in ['bspline', 'trigonometric']:
            self.method = method
        else:
            raise ValueError("Method must be either 'bspline' or 'trigonometric'.")

        if method == 'bspline':
            if degree is None:
                raise ValueError("Degree must be specified for the B-spline method.")
            self._base = scipy.interpolate.BSpline.basis_element
            self._setup_bspline()
            self._matrix = self._bspline_matrix
        elif method == 'trigonometric':
            self._matrix = self._trig_matrix
    
    def _psi_b(self, x, k:int, nu:int=0):
        '''
        Define boundary base functions.
        '''
        # 0 <= k < degree-1.
        degree = self.degree
        x = np.asarray(x)
        yn, y0, yp = x[x < 0], x[(x >= 0) & (x <= k+3)], x[x > k+3]
        # This is the integral of the base.
        nodes = np.concatenate((np.zeros(degree-k), np.arange(1, k+4)))
        I_psi = self._base(nodes).derivative(nu=nu+1)
        r = np.concatenate(
            (np.zeros_like(yn), I_psi(y0), np.zeros_like(yp)))
        return r / 2

    def _psi(self, x, k=0, nu:int=0):
        '''
        Define inner base functions.
        '''
        degree = self.degree
        x = np.asarray(x)
        x = x - k
        yn, y0, yp = x[x < 0], x[(x >= 0) & (x <= degree+2)], x[x > degree+2]
        # This is the integral of the base.
        I_psi = self._base(np.arange(0, degree+3))
        I_psi = I_psi.derivative(nu=nu+1)
        r = np.concatenate(
            (np.zeros_like(yn), I_psi(y0), np.zeros_like(yp)))
        return r / 2

    def _setup_bspline(self):
        '''
        Perform as many computations as possible for the bspline method, given only the degree.
        '''
        degree = self.degree
        fixed_quad = scipy.integrate.fixed_quad

        n_2 = np.sqrt(np.sinh(1) - 4 * np.sinh(0.5)**2)
        s = sympy.Symbol('s', real=True)
        phi = [
            1 + 0*s,
            sympy.sinh(s + 1/2) / np.sqrt(np.sinh(1)),
            (-2 * sympy.sinh(1/2) + sympy.cosh(s + 1/2)) / n_2
        ]
        I_phi = [sympy.integrate(p, s) for p in phi]
        I_phi = [p - p.subs(s, -1) for p in I_phi]

        # Fixed block (3 x 3).
        # -------------------
        self._K_ff = [np.zeros((3, 3)) for _ in range(3)]
        self._K_ff[0][0, :] = [p.subs(s, 0) for p in phi]
        self._K_ff[1][0, 0] = 1
        self._K_ff[2][0, 0] = 0.5
        for i, j in [(1, 0), (0, 1), (2, 1)]:
            H1 = I_phi[j] * phi[i] + phi[j] * sympy.diff(phi[i], s)
            H1 = sympy.integrate(H1, (s, -1, 0))
            self._K_ff[2][i, j] = H1

        # Boundary X boundary block (2(degree - 1) x 2(degree - 1)).
        # ---------------------------------------------------------
        # TODO: accelerate computations using Vermeulen.
        # [0, degree-1) even, [degree-1, 2*degree-1) odd.
        self._S_bb = [np.zeros((degree-1, degree-1)) for _ in range(2)]
        self._K_bb = [np.zeros((degree-1, degree-1)) for _ in range(2)]
        for i, j in product(range(degree-1), repeat=2):
            if j > i:
                continue
            # S entries.
            IS0, IS1 = 0, 0
            for l in range(j+3):
                IS0 += fixed_quad(
                    lambda x: self._psi_b(x, j, nu=0) * self._psi_b(x, i, nu=0),
                    l, l+1, n=degree+1)[0]
                IS1 += fixed_quad(
                        lambda x: self._psi_b(x, j, nu=1) * self._psi_b(x, i, nu=1),
                        l, l+1, n=degree)[0]
            IS0, IS1 = IS0, IS1
            self._S_bb[0][i, j] = IS0
            self._S_bb[1][i, j] = IS1
            if j == i:
                continue
            self._S_bb[0][j, i] = IS0
            self._S_bb[1][j, i] = IS1

            # K entries.
            IK0, IK1 = 0, 0
            for l in range(j+3):
                IK0 += fixed_quad(
                    lambda x: self._psi_b(x, j, nu=-1) * self._psi_b(x, i, nu=0),
                    l, l+1, n=degree+1)[0]
                IK1 += fixed_quad(
                    lambda x: self._psi_b(x, j, nu=0) * self._psi_b(x, i, nu=1),
                    l, l+1, n=degree)[0]
            IK0, IK1 = IK0, IK1
            self._K_bb[0][i, j] = IK0
            self._K_bb[0][j, i] = -IK0
            self._K_bb[1][i, j] = IK1
            self._K_bb[1][j, i] = -IK1

        # Inner X inner block ((M-degree-1) x (M-degree-1)).
        # -------------------------------------------------
        self._S_ii = [np.zeros(degree+2) for _ in range(2)]
        self._K_ii = [np.zeros(degree+2) for _ in range(2)]
        inner = self._base(np.arange(0, 2*degree+5))
        ii = np.arange(0, degree+2)
        # S entries.
        self._S_ii[0] = -inner.derivative(nu=2)(degree+2+ii) / 4
        self._S_ii[1] = inner.derivative(nu=4)(degree+2+ii) / 4

        # K entries.
        self._K_ii[0][1:] = -inner.derivative(nu=1)(degree+2+ii[1:]) / 4
        self._K_ii[1][1:] = inner.derivative(nu=3)(degree+2+ii[1:]) / 4

        # Fixed X boundary block.
        # -----------------------
        self._K_fb = np.zeros(degree-1)
        self._K_fb = np.arange(3, degree+2) / (2*(degree+2))

        # Fixed X inner block.
        # --------------------
        self._K_fi = 0.5

        # Boundary X inner block.
        # -----------------------
        # S entries.
        self._S_bi = [np.zeros((degree-1, degree+1)) for _ in range(2)]
        for i, j in product(range(degree-1), range(degree+1)):
            IS0, IS1 = 0, 0
            for l in range(j, i+3):
                IS0 += fixed_quad(
                    lambda x: self._psi(x, k=j, nu=0) * self._psi_b(x, i, nu=0),
                    l, l+1, n=degree+1)[0]
                IS1 += fixed_quad(
                    lambda x: self._psi(x, k=j, nu=1) * self._psi_b(x, i, nu=1),
                    l, l+1, n=degree)[0]
            self._S_bi[0][i, j] = IS0
            self._S_bi[1][i, j] = IS1

        # K entries.
        self._K_bi = [np.zeros((degree-1, degree+1)) for _ in range(2)]
        for i, j in product(range(degree-1), range(degree+1)):
            IK0, IK1 = 0, 0
            for l in range(j, i+3):
                IK0 += fixed_quad(
                    lambda x: self._psi(x, k=j, nu=-1) * self._psi_b(x, k=i, nu=0),
                    l, l+1, n=degree+1)[0]
                IK1 += fixed_quad(
                    lambda x: self._psi(x, k=j, nu=0) * self._psi_b(x, k=i, nu=1),
                    l, l+1, n=degree)[0]
            self._K_bi[0][i, j] = IK0
            self._K_bi[1][i, j] = IK1

    def matrix(self, B0, B1, M:int, tau:float=1, cayley=(1, 1), check=True):
        '''
        Construct matrix.

        Parameters
        ----------
        B0, B1 : array_like
            Matrices defining the delay differential equation.

        M : int
            Dimension of the space of scalar functions.
        
        tau : float, optional
            Time delay. The default is 1.

        cayley : tuple, optional
            Cayley transform to transform the system.
            If cayley = (a, b), the transform is given by :math:`\frac{a+z}{b-z}`.
            The default is (1, 1).

        check : bool, optional
            Check the input matrices. The default is True.

        Returns
        -------
        S : csr_matrix
            Representation of the identity in the B-spline basis.
        K : csr_matrix
            Representation of K in the B-spline basis.
        '''
        if self.method == 'bspline' and (M < 2 * (self.degree + 1)):
            msg = 'M for size less than 2 * (degree + 1) has not been implemented yet.'
            raise ValueError(msg)
        B0 = np.array(B0)
        B1 = np.array(B1)
        if check:
            B0, B1 = _validate_coeffs(B0, B1)
        self._dtype = B0.dtype

        return self._matrix(B0, B1, M, tau, cayley)

    def _bspline_matrix(self, B0, B1, M, tau, cayley):
        # correct effect of time delay.
        B0, B1 = tau * B0, tau * B1
        # basic definitions.
        a, b = cayley
        sc = a + b
        deg = self.degree
        N = np.shape(B0)[0]
        Id = scipy.sparse.eye_array(N, format='csr')
        A = sc * scipy.linalg.inv(b * np.eye(N) - B0 - np.exp(-b) * B1)
        B = A @ (b * np.eye(N) - B0)

        # build the matrix.
        # Fixed block (3 x 3).
        # -------------------
        S_ff = scipy.sparse.eye_array(3*N, format='csr')
        K_ff = scipy.sparse.csr_array((3*N, 3*N))
        for m, sm in zip([A, B, -sc * np.eye(N)], self._K_ff):
            K_ff += np.kron(m, sm)

        # Boundary X boundary block (2(degree - 1) x 2(degree - 1)).
        # ---------------------------------------------------------
        S_bb = self._S_bb[0] / M**2 + self._S_bb[1]
        K_bb = -sc * (self._K_bb[0] / M**2 + self._K_bb[1]) / M
        
        # Inner X inner block ((M-degree-1) x (M-degree-1)).
        # -------------------------------------------------
        S_ii_ = self._S_ii[0] / M**2 + self._S_ii[1]
        S_ii = [np.repeat(val, M-deg-1-j) for j, val in enumerate(S_ii_[1:])]
        S_ii = S_ii[::-1] + [np.repeat(S_ii_[0], M-deg-1)] + S_ii
        S_ii = scipy.sparse.diags_array(
            S_ii, offsets=np.arange(-deg-1, deg+2),
            shape=(M-deg-1, M-deg-1), format='csr')

        K_ii_ = -sc * (self._K_ii[0] / M**2 + self._K_ii[1]) / M
        K_ii = [np.repeat(val, M-deg-2-j) for j, val in enumerate(K_ii_[1:])]
        K_ii = K_ii[::-1] + [-e for e in K_ii]
        offsets = list(range(-deg-1, 0)) + list(range(1, deg+2))
        K_ii = scipy.sparse.diags_array(
            K_ii, offsets=offsets,
            shape=(M-deg-1, M-deg-1), format='csr')

        # Fixed X boundary block.
        # -----------------------
        c2 = -2 * np.sinh(1/2) / np.sqrt(np.sinh(1) - 4 * np.sinh(0.5)**2)
        K_fb = np.zeros((3, deg-1))
        K_fb[0] = -sc * self._K_fb / M**(5/2)
        K_fb[2] = -sc * c2 * self._K_fb / M**(5/2)

        # Fixed X inner block.
        # --------------------
        K_fi = np.zeros((3, M-deg-1))
        K_fi[0, :] = -sc * self._K_fi / M**(5/2)
        K_fi[2, :] = -sc * c2 * self._K_fi / M**(5/2)
        upper_K = np.concatenate(
            (K_fb, K_fi, K_fb[:, ::-1]), axis=-1)
        upper_K = scipy.sparse.kron(Id, upper_K, format='csr')

        # Boundary X inner block.
        # -----------------------
        S_bi = self._S_bi[0] / M**2 + self._S_bi[1]
        K_bi = -sc * (self._K_bi[0] / M**2 + self._K_bi[1]) / M

        if deg > 1:
            empty = scipy.sparse.csr_array((deg-1, (M-2*(deg+1))))
            sub_upper = scipy.sparse.block_array(
                [[S_bi, empty]], format='csr')
            sub_bottom = scipy.sparse.block_array(
                [[empty, S_bi[::-1, ::-1]]], format='csr')
            S_ii = scipy.sparse.block_array(
                [[S_bb, sub_upper, None],
                 [sub_upper.T, S_ii, sub_bottom.T],
                 [None, sub_bottom, S_bb[::-1, ::-1]]], format='csr')
            
            sub_upper = scipy.sparse.block_array(
                [[K_bi, empty]], format='csr')
            sub_bottom = scipy.sparse.block_array(
                [[empty, -K_bi[::-1, ::-1]]], format='csr')
            K_ii = scipy.sparse.block_array(
                [[K_bb, sub_upper, None],
                 [-sub_upper.T, K_ii, -sub_bottom.T],
                 [None, sub_bottom, -K_bb[::-1, ::-1]]], format='csr')
        S_ii = scipy.sparse.kron(Id, S_ii, format='csr')
        K_ii = scipy.sparse.kron(Id, K_ii, format='csr')

        S = scipy.sparse.block_array(
            [[S_ff, None],
             [None, S_ii]],
             format='csr', dtype=self._dtype)
        K = scipy.sparse.block_array(
            [[K_ff, upper_K],
             [-upper_K.T, K_ii]],
             format='csr', dtype=self._dtype)

        return S, K

    def _trig_matrix(self, B0, B1, M, tau, cayley):
        # correct effect of time delay.
        B0, B1 = tau * B0, tau * B1
        # basic definitions.
        a, b = cayley
        sc = a + b
        N = np.shape(B0)[0]
        Id = scipy.sparse.eye_array(N, format='csr')
        A = sc * scipy.linalg.inv(b * np.eye(N) - B0 - np.exp(-b) * B1)
        B = A @ (b * np.eye(N) - B0)

        # build the matrix.
        # (0, 0) block
        K_00 = -(sc/2) * np.eye(N) + A + B
        # (M, 0) block
        K_M0 = -sc * Id * np.cosh(1/2) / np.sqrt(np.sinh(1))
        # (0, M) block
        tmp = sc * (np.cosh(1/2) - 2 * np.sinh(1/2)) * np.eye(N) + (A * np.sinh(1/2))
        K_0M = tmp / np.sqrt(np.sinh(1))
        # Diagonal block.
        diag = np.zeros(2*(M-1), dtype=complex)
        diag[::2] = sc / (np.pi * 2j * np.arange(1, M)) # negative values of n.
        diag[1::2] = -diag[::2] # positive values of n.
        K_diag = scipy.sparse.diags_array(
            diag, offsets=0, shape=(2*(M-1), 2*(M-1)), format='csr')
        K_diag = scipy.sparse.kron(K_diag, Id, format='csr')
        # 0th column
        diag = np.zeros((2*(M-1), 1), dtype=complex)
        diag[::2, 0] = -sc / (np.pi * 2j * np.arange(1, M) * np.sqrt(1 + (2 * np.pi * np.arange(1, M))**2))
        diag[1::2, 0] = -diag[::2, 0]
        K_0c = scipy.sparse.kron(diag, Id, format='csr')
        # 0th row
        K_0r = np.kron(diag.T, np.eye(N))
        diag = np.zeros((1, 2*(M-1)))
        diag[0, ::2] = 1 / np.sqrt(1 + (2 * np.pi * np.arange(1, M))**2)
        diag[0, 1::2] = diag[0, ::2]
        K_0r = K_0r + np.kron(diag, A)
        # Mth column
        diag = np.zeros((2*(M-1), 1), dtype=complex)
        diag[::2, 0] = -(2*sc) * np.sinh(1/2) / (np.sqrt(1 + (2 * np.pi * np.arange(1, M))**2) * np.sqrt(np.sinh(1)))
        diag[1::2, 0] = diag[::2, 0]
        K_Mc = scipy.sparse.kron(diag, Id, format='csr')

        K = scipy.sparse.block_array(
            [[K_00, K_0r, K_0M],
             [K_0c, K_diag, K_Mc],
             [K_M0, None, None]], format='csr')

        return scipy.sparse.eye_array(2*M*N, format='csr'), K
    
    def eigvals(self, B0, B1, M, tau=1, cayley=(1, 1)):
        '''Approximated characteristic roots.'''
        S, K = self.matrix(B0, B1, M, tau, cayley)
        eigs = scipy.linalg.eigvals(K.todense(), b=S.todense(), overwrite_a=True)
        eigs = inv_cayley(-1+eigs, *cayley) / tau

        return eigs
    
    def eigs(self, B0, B1, M, eigs_kwargs=None, tau=1, cayley=(1, 1)):
        eigs_kwargs = {} if eigs_kwargs is None else eigs_kwargs
        N = B0.shape[0]
        degree = self.degree

        S, K = self.matrix(B0, B1, M, tau, cayley)
        if self.method == 'bspline':
            asOp = scipy.sparse.linalg.aslinearoperator
            Sinv = _Sinv(S, M, N, degree)
            Id = _opId((M+degree)*N, dtype=self._dtype)
            mu = scipy.sparse.linalg.eigs(
                -Id + Sinv @ asOp(K),
                which='LM',
                return_eigenvectors=False,
                **eigs_kwargs
            )
        elif self.method == 'trigonometric':
            mu = scipy.sparse.linalg.eigs(
                -S+K,
                which='LM',
                **eigs_kwargs
            )
        mu = inv_cayley(mu, *cayley) / tau

        return mu

class _Sinv(scipy.sparse.linalg.LinearOperator):

    def __init__(self, S, M, N, degree):
        super().__init__(dtype=S.dtype, shape=S.shape)
        self.M = M
        self.N = N
        self.degree = degree

        D = M + degree - 3
        S_banded = np.zeros((degree+2, D))
        for j in range(degree+2):
            S_banded[j, :D-j] = S[3*N:3*N+D, 3*N:3*N+D].diagonal(j)
        self._cho = scipy.linalg.cholesky_banded(
            S_banded, overwrite_ab=True, lower=True, check_finite=False
        )

    def _matmat(self, V):
        N = self.N
        D = self.M + self.degree - 3
        # X = np.concatenate(
        #     list(V[j*D+3*N:(j+1)*D+3*N] for j in range(N)), axis=-1)
        # X = scipy.linalg.cho_solve_banded(
        #     (self._cho, True), X)
        # div = X.shape[1] // N
        # for j in range(N):
        #     V[j*D+3*N:(j+1)*D+3*N] = X[:, j*div:(j+1)*div]
        for j in range(N):
            V[3*N+j*D:3*N+(j+1)*D] = scipy.linalg.cho_solve_banded(
                (self._cho, True), V[3*N+j*D:3*N+(j+1)*D],
                overwrite_b=True
            )

        return V

    def _adjoint(self):
        return self

class _opId(scipy.sparse.linalg.LinearOperator):

    def __init__(self, N, dtype):
        super().__init__(dtype=dtype, shape=(N, N))
    
    def _matvec(self, x):
        return x
    
    def _adjoint(self):
        return self

