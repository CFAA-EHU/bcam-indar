import logging

import numpy as np
import scipy

logger = logging.getLogger(__name__)


# Grassmannian decomposition
# --------------------------
# TODO: the complexity of grass should be in between LU and QR.
# Improve the algorithm.
def grass(x, coords):
    '''
    Computes Grassmannian decomposition of x.

    The matrix x is decomposed as x = q s, where the columns of q are orthogonal,
    and q[coords] is lower triangular.

    Parameters
    ----------
    x : np.ndarray, (N, M)
        2D-array with N >= M.
    coords : 1D-array
        Indices of rows of x to be used.

    Returns
    -------
    q : np.ndarray, (N, M)
    s : np.ndarray, (M, M)
    inv_s : np.ndarray, (M, M)
        Inverse of s.
    '''
    assert x.ndim == 2, 'x should be a 2D-array.'
    n, m = x.shape
    assert n >= m, 'x should be a 2D-array with shape (N, M) and N >= M.'
    if n == m:
        return np.eye(n), x, None

    assert len(coords) == m, 'Incompatible length for coords.'
    coords_c = np.setdiff1d(
        np.arange(n), coords, assume_unique=True)
    x_ = x.copy()
    p, l, u = scipy.linalg.lu(
        x_[coords].T,
        overwrite_a=True, permute_l=False, p_indices=True)
    x_[coords] = u.T
    p_inv = np.argsort(p)
    x_[coords_c] = scipy.linalg.solve(
        l, x[coords_c][:, p_inv].T, assume_a='lower triangular').T
    x_ = x_[:, ::-1]

    q, r = scipy.linalg.qr(
        x_, mode='economic', overwrite_a=True)
    q = q[:, ::-1]
    s = r[::-1][:, ::-1]@(l[p].T)
    inv_s = np.eye(m)[p_inv]
    inv_s = scipy.linalg.solve(
        l, inv_s, assume_a='lower triangular',
        overwrite_b=True, overwrite_a=True)
    inv_s = scipy.linalg.solve(
        (r[::-1][:, ::-1]).T, inv_s, assume_a='upper triangular',
        overwrite_b=True, overwrite_a=True).T
    idx = np.argwhere(np.diag(q) < 0)
    q[:, idx] = -q[:, idx]
    s[idx, :] = -s[idx, :]
    inv_s[:, idx] = -inv_s[:, idx]

    return q, s, inv_s

def jac_grass(dx, coords, grass):
    # From dx = q ds + dq s we get q.T dx = ds + q.T dq s and
    # (pi q)^{-1}(pi dx) = ds + (pi q)^{-1}(pi dq) s,
    # where pi is the projection to the rows in coords.
    q, s, inv_s = grass
    if inv_s is None:
        return np.zeros_like(q), q.T@dx

    a = scipy.linalg.solve(
        q[coords], dx[coords], assume_a='lower triangular')
    a = (q.T@dx - a)@inv_s
    a = np.triu(a, k=1)
    a = a - a.T
    ds = q.T@dx - a@s
    dq = (dx - q@ds)@inv_s

    return dq, ds

def jac_grass_minimal(dx, coords, grass):
    q, _, inv_s = grass
    v = scipy.linalg.solve(
        q[coords], dx[coords], assume_a='lower triangular')
    v = (q.T@dx - v)@inv_s
    a = np.triu(v, k=1)
    a = a - a.T
    dq_r = q[coords]@(a - v)

    return dq_r

def hessp_grass(pdq, pds, dq, ds, coords, grass):
    q, s, inv_s = grass
    if inv_s is None:
        return np.zeros_like(pdq), np.zeros_like(pds)

    a = -pdq@ds - dq@pds
    b = -dq.T@pdq - pdq.T@dq
    qTpd2q = scipy.linalg.solve(
        q[coords], a[coords], assume_a='lower triangular')
    qTpd2q = (q.T@a - qTpd2q)@inv_s
    qTpd2q = np.triu(qTpd2q, k=1)
    qTpd2q = qTpd2q + (np.tril(b, k=-1) - qTpd2q.T)
    qTpd2q[range(q.shape[1]), range(q.shape[1])] = np.diag(b)/2

    pd2s = q.T@a - qTpd2q@s
    pd2q = (a - q@pd2s)@inv_s

    return pd2q, pd2s


# Jacobian of Cholesky decomposition
# ----------------------------------
def jac_cho(u, dx):
    assert dx.ndim == 2, 'dx should be a 2D-array.'
    assert dx.shape[0] == dx.shape[1], 'dx should be a square array.'
    assert u.shape == dx.shape, 'Incompatible shapes for u and dx.'
    assert np.allclose(dx, dx.T), 'dx should be symmetric.'
    du = np.zeros_like(u)
    y = np.zeros(u.shape[1], dtype=u.dtype)

    # dx_{ij} = \sum_{i<k} (u_{ki}du_{kj} + u_{kj}du_{ki}), for j >=i.
    # Then, dx_{ij} = u_{ij} + u_{ii}du_{ij} + u_{ij}du_{ii}.
    for i in range(dx.shape[0]-1):
        du[i, i] = (dx[i, i] - y[i])/(2*u[i, i])
        du[i, i+1:] = (dx[i, i+1:] - y[i+1:] - u[i, i+1:]*du[i, i])/u[i, i]
        y[i+1:] = [
            np.sum(u[:i+1, i+1]*du[:i+1, j] + du[:i+1, i+1]*u[:i+1, j], axis=0)
            for j in range(i+1, dx.shape[1])]
    du[-1, -1] = (dx[-1, -1] - y[-1])/(2*u[-1, -1])

    return du

def hess_cho(u, dux, duy):
    a = -(dux.T@duy + duy.T@dux)
    d2uxy = jac_cho(u, a)
    return d2uxy

# Jacobian of the QR decomposition
# --------------------------------
def jac_qr(x, dx, qr):
    '''
    Computes the derivative of the qr decomposition at x.

    Parameters
    ----------
    x : np.ndarray
    dx : np.ndarray
    qr : tuple, optional
        The QR decomposition of x as (q, r), where q.shape == x.shape.

    Returns
    -------
    dq : np.ndarray
        The derivative of the rotation q.
    dr : np.ndarray
        The derivative of the upper triangular matrix r.
    '''
    assert x.ndim == 2, 'x should be a 2D-array.'
    assert x.shape[0] >= x.shape[1], 'x should be a 2D-array with shape (N, M) and N >= M.'
    assert dx.shape == x.shape, 'x and dx should have the same shape.'
    # dx = dq@r + q@dr.
    q, r = qr
    assert q.shape == x.shape, 'Incompatible shapes for q and x.'
    # x.T@dx + x@dx.T = r.T@dr + dr.T@r. To prove this,
    # notice that q.T@q = I, so dq.T@q + q.T@dq = 0.
    a = x.T@dx
    a += a.T
    dr = np.zeros_like(r)
    u = np.zeros(a.shape[1], dtype=x.dtype)

    # u_{ij} = \sum_{i<k} (r_{ki}dr_{kj} + r_{kj}dr_{ki}), for j >=i.
    # Then, a_{ij} = u_{ij} + r_{ii}dr_{ij} + r_{ij}dr_{ii}.
    for i in range(a.shape[0]-1):
        dr[i, i] = (a[i, i] - u[i])/(2*r[i, i])
        dr[i, i+1:] = (a[i, i+1:] - u[i+1:] - r[i, i+1:]*dr[i, i])/r[i, i]
        u[i+1:] = [
            np.sum(r[:i+1, i+1]*dr[:i+1, j] + dr[:i+1, i+1]*r[:i+1, j], axis=0)
            for j in range(i+1, a.shape[1])]
    dr[-1, -1] = (a[-1, -1] - u[-1])/(2*r[-1, -1])

    dq = scipy.linalg.solve(
        r.T, dx.T - dr.T@q.T, assume_a='lower triangular',
        overwrite_b=True).T
    return dq, dr

def pivot_to_permutation(piv):
    perm = np.arange(len(piv))
    for i in range(len(piv)):
        perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
    return perm

def jac_lu(dx, lu_piv):
    '''
    Computes the derivative of the lu decomposition at x.

    If x = plu, where p is a permutation, then
    dx = p(dl@u + l@du).

    Parameters
    ----------
    dx : np.ndarray
    (lu, piv) : tuple
        Factorization of the coefficient matrix a, as given by lu_factor.

    Returns
    -------
    dlu : np.ndarray
        The derivative of the LU decomposition.
        The zero diagonal terms of l are not stored.
    '''
    lu, piv = lu_piv
    assert dx.shape == lu.shape, 'Incompatible shapes for dx and lu.'
    dx = dx[pivot_to_permutation(piv)].copy()
    dlu = np.zeros_like(lu)

    dlu[0] = dx[0]
    for i in range(1, lu.shape[0]):
        # We start with j > i.
        dlu[i, :i] = scipy.linalg.solve(
            np.triu(lu[:i, :i]).T,
            dx[i, :i] - lu[i, :i]@np.triu(dlu[:i, :i]),
            assume_a='lower triangular')
        # w = \sum_{1\le k<i} dl_{ik}u_{kj} + l_{ik}du_{kj}, for i <= j.
        w = dlu[i, :i]@lu[:i, i:] + lu[i, :i]@dlu[:i, i:]
        # This gives us du_{ij}, for j \ge i.
        dlu[i, i:] = dx[i, i:] - w

    return dlu
