import pytest

import numpy as np
import scipy

from bcam.resonance._core import mechanical, derivatives

# Derivatives tests
# -----------------

class TestGrass:

    @pytest.fixture(scope='class')
    def initial(self):
        rng = np.random.default_rng()
        n, m = 5, 3
        x = rng.normal(size=(n, m))
        # Generate m integers in [0, n-1] without repetition.
        coords = rng.choice(n, size=m, replace=False)
        coords = np.sort(coords)

        return [x, coords]

    def test_grass(self, initial):
        x, coords = initial
        q, s, inv_s = derivatives.grass(x, coords)
        assert np.allclose(x, q@s)
        assert np.allclose(np.eye(len(coords)), s@inv_s)

    def test_jac(self, initial):
        x, coords = initial
        rng = np.random.default_rng()
        dx = rng.normal(size=x.shape)

        q, s, s_inv = derivatives.grass(x, coords)
        dq, ds = derivatives.jac_grass(dx, coords, (q, s, s_inv))
        assert np.allclose(dx, dq@s + q@ds)
        assert np.allclose(q.T@dq, -dq.T@q)
        assert np.allclose(np.triu(dq[coords], k=1), 0)

    def test_jac_minimal(self, initial):
        x, coords = initial
        rng = np.random.default_rng()
        dx = rng.normal(size=x.shape)

        q, s, s_inv = derivatives.grass(x, coords)
        dq = derivatives.jac_grass(dx, coords, (q, s, s_inv))[0]
        dq_r = derivatives.jac_grass_minimal(dx, coords, (q, s, s_inv))
        assert np.allclose(dq_r, dq[coords])

    def test_hessp(self, initial):
        x, coords = initial
        rng = np.random.default_rng()
        dx = rng.normal(size=x.shape)
        p = rng.normal(size=x.shape)
    
        q, s, s_inv = derivatives.grass(x, coords)
        dq, ds = derivatives.jac_grass(dx, coords, (q, s, s_inv))
        pdq, pds = derivatives.jac_grass(p, coords, (q, s, s_inv))
        pd2q, pd2s = derivatives.hessp_grass(pdq, pds, dq, ds, coords, (q, s, s_inv))

        assert np.allclose(pdq@ds + dq@pds + pd2q@s + q@pd2s, 0)
        assert np.allclose(dq.T@pdq + pdq.T@dq + q.T@pd2q + pd2q.T@q, 0)
        assert np.allclose(np.triu(pd2q[coords], k=1), 0)

def test_jac_cho():
    rng = np.random.default_rng()
    n = 4
    u = rng.normal(size=(n, n))
    dx = rng.normal(size=(n, n))
    dx = (dx + dx.T)/2
    u = np.triu(u)
    du = derivatives.jac_cho(u, dx)
    assert np.allclose(dx, du.T@u+ u.T@du)

def test_jac_qr():
    rng = np.random.default_rng()
    x = rng.normal(size=(4, 3))
    q, r = scipy.linalg.qr(
        x, overwrite_a=False, mode='economic', pivoting=False)
    idx = np.argwhere(np.diag(r) < 0)
    q[:, idx] *= -1
    r[idx, :] *= -1
    dx = rng.normal(size=(4, 3))
    dq, dr = derivatives.jac_qr(x, dx, (q, r))
    assert np.allclose(dx, dq@r + q@dr)

def test_jac_lu():
    rng = np.random.default_rng()
    n = 4
    x = rng.normal(size=(n, n))
    dx = rng.normal(size=(n, n))
    lu_piv = scipy.linalg.lu_factor(x)
    dlu = derivatives.jac_lu(dx, lu_piv)

    lu, piv = lu_piv
    l, u = np.tril(lu, k=-1)+np.eye(n), np.triu(lu)
    dl, du = np.tril(dlu, k=-1), np.triu(dlu)
    dx = dx[derivatives.pivot_to_permutation(piv)]
    assert np.allclose(dx, dl@u + l@du)

# Mechanical tests
# ----------------

def test_reshape_modes():
    dof, n_out = 4, 3
    rng = np.random.default_rng(1268)

    # Reference mode shape.
    x0 = 0.1*rng.normal(size=(n_out, dof))
    z0 = 0.01*rng.normal(size=(n_out, dof))
    z0 = np.triu(z0, k=1)
    z0[:n_out, :n_out] = z0[:n_out, :n_out] - z0[:n_out, :n_out].T
    x = mechanical.reshape_modes_output(x0, z0)
    x0_, z0_ = mechanical.reshape_modes_input(x, dof, n_out)

    assert np.allclose(x0_, x0), np.allclose(z0_, z0)

class TestModeMap:

    @pytest.fixture(scope='class')
    def mode_shapes(self):
        dof, n_out = 6, 4
        rng = np.random.default_rng(123455)

        x = rng.normal(size=(n_out, dof))
        z = rng.normal(scale=1e-3, size=(n_out, dof))
        z[:n_out, :n_out] = (z[:n_out, :n_out] - z[:n_out, :n_out].T)/2
        freqs = -rng.uniform(0.1, 10, size=dof) + 1j * rng.uniform(size=dof)
        coords = np.arange(n_out)
        modes = mechanical.PartialModesMap(freqs, coords)
        psi = modes(x, z)
        psi *= np.sqrt(np.imag(freqs))[np.newaxis, :]

        return psi, freqs

    def test_null(self, mode_shapes):
        psi, freqs = mode_shapes
        test_obj = np.imag(psi / np.imag(freqs)[np.newaxis, :] @ psi.T)
        assert np.allclose(test_obj, 0)

    def test_mass_inv(self, mode_shapes):
        psi, freqs = mode_shapes
        test_obj = np.imag(psi * (freqs / np.imag(freqs))[np.newaxis, :] @ psi.T)
        scipy.linalg.cholesky(test_obj, lower=False, overwrite_a=True)

    def test_stiffness_inv(self, mode_shapes):
        psi, freqs = mode_shapes
        test_obj = -np.imag(psi / (freqs * np.imag(freqs))[np.newaxis, :] @ psi.T)
        scipy.linalg.cholesky(test_obj, lower=False, overwrite_a=True)

class TestAmplitudes:

    @staticmethod
    def kernel(ns, fs, a, freqs):
        dof = len(freqs)
        t = np.arange(ns) / fs
        K = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
        K = a * K.reshape(1, 1, dof)
        K = np.expand_dims(K, axis=2)
        K = K * np.exp(freqs[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, dof)
        K = np.imag(np.sum(K, axis=-1))
        return K
    
    def test_reshape(self):
        rng = np.random.default_rng(1234345)
        dof, n_out, n_in = 4, 3, 2

        x = rng.normal(
            size=(n_in*(n_in+1)//2 + (n_out-n_in)*n_in, 2*dof - 1))
        ix = mechanical.reshape_injection(x, n_out=n_out, n_in=n_in)
        pix = mechanical.reshape_projection(ix)

        assert np.allclose(x, pix)

    def test_amps(self):
        rng = np.random.default_rng(12345)
        dof = 4
        freqs = -rng.uniform(1, 3, dof) + 2j*np.pi*rng.uniform(1, 50, dof)
        ns, fs = 200, 100
        n_out, n_in = 3, 2

        amps = rng.normal(scale=1, size=(n_out, n_in, dof)).astype(np.complex128)
        amps += 1j * rng.normal(scale=.1, size=(n_out, n_in, dof))
        amps[:n_in] = 0.5 * (amps[:n_in] + amps[:n_in].transpose((1, 0, 2)))
        data = self.kernel(ns, fs, amps, freqs)

        model = mechanical.Amplitudes(
            freqs=freqs,
            fs=fs,
            ns=ns,
            n_out=n_out,
            n_in=n_in
        )
        amps_fit = model.fit(data, penalty=0.)
        error = np.linalg.norm(amps_fit - amps, axis=-1) / np.linalg.norm(amps, axis=-1)
        assert np.max(error) < 1e-2

    def test_mech_amps(self):
        dof = 4
        modal = mechanical.randomSystem(
            dof,
            mass_range=(1, 2),
            damping_range=(0.02, 0.05),
            freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
            damping_type='nop',
            seed=123456)[1]
        psi = modal['mode_shapes']
        freqs = modal['frequencies']
        ns, fs = 210, 100
        n_out, n_in = 3, 2

        amps = psi.reshape(dof, 1, dof) * psi.reshape(1, dof, dof)
        amps = amps * (1/np.imag(freqs)).reshape(1, 1, dof)
        amps = amps[:n_out, :n_in]
        data = self.kernel(ns, fs, amps, freqs)

        model = mechanical.Amplitudes(
            freqs=freqs,
            fs=fs,
            ns=ns,
            n_out=n_out,
            n_in=n_in,
            a_type='mechanical')
        amps_fit = model.fit(data)
        error = np.linalg.norm(amps_fit - amps, axis=-1) / np.linalg.norm(amps, axis=-1)
        assert np.max(error) < 1e-2
