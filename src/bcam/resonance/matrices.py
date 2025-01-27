#!/usr/bin/env python

# ==== Imports ====
from itertools import product

import numpy as np
from scipy.linalg import inv, expm
from scipy.sparse import csr_matrix

# ==== Functions ====
inv_cayley = lambda x : (x - 1) / (x + 1)

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

def test_eigenvalue(B0, B1, eigval):
    r"""Test the quality of the approximation of the eigenvalue.

    For the delay equation :math:`y'(t) = B_0 y(t) + B_1 y(t - 1)`,
    test how well the eigenvalue `eigval` solves the characteristic equation  :math:`\Delta(\lambda) := \det(\lambda I - B_0 - e^{-\lambda} B_1) = 0`.
    
    Parameters
    ----------
    B0 : array_like
        The matrix B0.
    B1 : array_like
        The matrix B1.
    eigval : float
        The eigenvalue to test.

    Returns
    -------
    float
        :math:`\frac{|\Delta(\lambda)|}{|\Delta'(\lambda)|}`.
    """
    if np.isnan(eigval):
        return np.nan
    B0 = np.asarray(B0)
    B1 = np.asarray(B1)
    N = B0.shape[0]
    Id = np.identity(N)
    L = eigval * Id - B0 - np.exp(-eigval) * B1
    dL = Id + np.exp(-eigval) * B1
    d_dis = jacobi(L, dL)
    return np.abs(np.linalg.det(L) / d_dis)

def _transfer_matrix(M, data, row_ind, col_ind, m1:int, m2:int):
    N1, N2 = M.shape
    M = [(M[i, j], m1 + i, m2 + j)
         for i, j in product(range(N1), range(N2))]
    local_data, local_row, local_col = list(zip(*M))
    data += local_data
    row_ind += local_row
    col_ind += local_col

# ==== Semi-discretization matrices ====
def initial_semi_discretization_matrix(B_0, B_1, M:int, tau=1):
    """Build the semi-discretization matrix with B-splines of order 1.
    
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
    """
    # correct effect of time delay.
    B_0 = np.copy(tau * B_0)
    B_1 = np.copy(tau * B_1)
    h = 1 / (M-1)
    # basic definitions.
    N = np.shape(B_0)[0]
    Id = np.identity(N)
    A = inv(B_0)
    B = expm(h * B_0)
    # build the matrix.
    data = []
    row_ind, col_ind = [], []
    # (M - 1, 0) block
    S = ((B - Id) @ A @ B_1)
    _transfer_matrix(
        S, data, row_ind, col_ind, (M - 1) * N, 0)
    # (M - 1, M - 1) block
    S = B
    _transfer_matrix(
        S, data, row_ind, col_ind, (M - 1) * N, (M - 1) * N)
    # subdiagonal elements.
    for m in range(1, M):
        S = Id
        _transfer_matrix(
            S, data, row_ind, col_ind, (m-1) * N, m * N)

    return csr_matrix((data, (row_ind, col_ind)), shape=(M*N, M*N))

def semi_discretization_matrix(B_0, B_1, M:int, tau=1):
    """Build the semi-discretization matrix with B-splines of order 2.
    
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
    """
    # correct effect of time delay.
    B_0 = np.copy(tau * B_0)
    B_1 = np.copy(tau * B_1)
    h = 1 / (M-1)
    # basic definitions.
    N = np.shape(B_0)[0]
    Id = np.identity(N)
    A = inv(B_0)
    B = expm(h * B_0)
    # build the matrix.
    data = []
    row_ind, col_ind = [], []
    # (M - 1, 0) block
    S = A @ (B + (1/h)*A - (1/h)*A @ B) @ B_1
    _transfer_matrix(
        S, data, row_ind, col_ind, (M - 1) * N, 0)
    # (M - 1, 1) block
    S = -A @ (Id + (1/h)*A - (1/h)*A @ B) @ B_1
    _transfer_matrix(
        S, data, row_ind, col_ind, (M - 1) * N, N)
    # (M - 1, M - 1) block
    S = B
    _transfer_matrix(
        S, data, row_ind, col_ind, (M - 1) * N, (M - 1) * N)

    # subdiagonal elements.
    for m in range(1, M):
        S = Id
        _transfer_matrix(
            S, data, row_ind, col_ind, (m-1) * N, m * N)

    return csr_matrix((data, (row_ind, col_ind)), shape=(M*N, M*N))

# ==== Tustin matrices ====
def tustin_matrix_unstable(B_0, B_1, M:int, tau=1):
    # correct effect of time delay.
    B_0 = np.copy(tau * B_0)
    B_1 = np.copy(tau * B_1)
    # basic definitions.
    N = np.shape(B_0)[0]
    Id = np.identity(N, dtype=complex)
    A = 2 * inv(Id - B_0 - np.exp(-1) * B_1)
    B = A @ (Id - B_0)

    # build the matrix.
    data = []
    row_ind, col_ind = [], []
    # (0, 0) block
    T = -2 * Id + A + B
    _transfer_matrix(
        T, data, row_ind, col_ind, 0, 0)
    # (M, 0) block
    T = -2 * Id
    _transfer_matrix(
        T, data, row_ind, col_ind, (2*M-1)*N, 0)
    # (0, M) block
    T = (1/6) * Id + 0.5 * A
    _transfer_matrix(
        T, data, row_ind, col_ind, 0, (2*M-1)*N)
    # (M, M) block
    T = -Id
    _transfer_matrix(
        T, data, row_ind, col_ind, (2*M-1)*N, (2*M-1)*N)

    for n in range(1, M):
        # even indices are for positive values of n.
        # diagonal elements.
        T = -(1 + (1 / (np.pi * 1j * n))) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n)*N, (2*n)*N)
        # 0th row
        T = A + (1 / (np.pi * 1j * n)) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, 0, (2*n)*N)
        # 0th column
        T = (1 / (np.pi * 1j * n)) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n)*N, 0)
        # Mth column
        T = (1 / (2 * np.pi * 1j * n)) * (1 + 1 / (np.pi * 1j * n)) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n)*N, (2*M-1)*N)

        # odd indices are for negative values of n.
        # diagonal elements.
        T = -(1 - (1 / (np.pi * 1j * n))) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n-1)*N, (2*n-1)*N)
        # 0th row
        T = A - (1 / (np.pi * 1j * n)) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, 0, (2*n-1)*N)
        # 0th column
        T = -(1 / (np.pi * 1j * n)) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n-1)*N, 0)
        # Mth column
        T = (1 / (2 * np.pi * 1j * n)) * (-1 + 1 / (np.pi * 1j * n)) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n-1)*N, (2*M-1)*N)

    return csr_matrix((data, (row_ind, col_ind)), shape=(2*M*N, 2*M*N))

def tustin_matrix_stable(B_0, B_1, M:int, tau=1):
    # correct effect of time delay.
    B_0 = np.copy(tau * B_0)
    B_1 = np.copy(tau * B_1)
    # basic definitions.
    N = np.shape(B_0)[0]
    Id = np.identity(N, dtype=complex)
    A = 2 * inv(Id - B_0 - np.exp(-1) * B_1)
    B = A @ (Id - B_0)

    # build the matrix.
    data = []
    row_ind, col_ind = [], []
    # (0, 0) block
    T = -2 * Id + A + B
    _transfer_matrix(
        T, data, row_ind, col_ind, 0, 0)
    # (M, 0) block
    T = -2 * Id * np.cosh(1/2) / np.sqrt(np.sinh(1))
    _transfer_matrix(
        T, data, row_ind, col_ind, (2*M-1)*N, 0)
    # (0, M) block
    tmp = 2 * (np.cosh(1/2) - 2 * np.sinh(1/2)) * Id + (A * np.sinh(1/2))
    T = tmp / np.sqrt(np.sinh(1))
    _transfer_matrix(
        T, data, row_ind, col_ind, 0, (2*M-1)*N)
    # (M, M) block
    T = -Id
    _transfer_matrix(
        T, data, row_ind, col_ind, (2*M-1)*N, (2*M-1)*N)

    for n in range(1, M):
        den = np.sqrt(1 + (2 * np.pi * n)**2)
        # even indices are for positive values of n.
        # diagonal elements.
        T = -(1 + (1 / (np.pi * 1j * n))) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n)*N, (2*n)*N)
        # 0th row
        T = (A + Id / (np.pi * 1j * n)) / den
        _transfer_matrix(
            T, data, row_ind, col_ind, 0, (2*n)*N)
        # 0th column
        T = Id / (np.pi *1j * n * den)
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n)*N, 0)
        # Mth column
        T = -4 * np.sinh(1/2) * Id / (den * np.sqrt(np.sinh(1)))
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n)*N, (2*M-1)*N)

        # odd indices are for negative values of n.
        # diagonal elements.
        T = -(1 - (1 / (np.pi * 1j * n))) * Id
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n-1)*N, (2*n-1)*N)
        # 0th row
        T = (A - Id / (np.pi * 1j * n)) / den
        _transfer_matrix(
            T, data, row_ind, col_ind, 0, (2*n-1)*N)
        # 0th column
        T = -Id / (np.pi *1j * n * den)
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n-1)*N, 0)
        # Mth column
        T = -4 * np.sinh(1/2) * Id / (den * np.sqrt(np.sinh(1)))
        _transfer_matrix(
            T, data, row_ind, col_ind, (2*n-1)*N, (2*M-1)*N)

    return csr_matrix((data, (row_ind, col_ind)), shape=(2*M*N, 2*M*N))


if __name__ == '__main__':
    rng = np.random.default_rng(seed=12342)
    B0 = rng.uniform(-5, 5, size=(3, 3))
    B1 = rng.uniform(-5, 5, size=(3, 3))
    M = 10
    C = tustin_matrix_stable(B0, B1, M)
    print(C)
