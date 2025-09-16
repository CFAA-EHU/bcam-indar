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


# Tests for modes
# ---------------

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

class TestModes:
    
    @pytest.fixture(scope='class')
    def mode_shapes(self):
        dof, n_out = 6, 4
        rng = np.random.default_rng(123455)

        freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
        coords = np.arange(n_out)
        modes = mechanical.PartialModesMap(freqs, coords)
        psi = np.nan
        while isinstance(psi, float):
            x = rng.normal(size=(n_out, dof))
            z = 5e-2*rng.normal(size=(n_out, dof))
            z[:n_out, :n_out] = z[:n_out, :n_out] - z[:n_out, :n_out].T
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

    @pytest.fixture(scope='class')
    def initial(self):
        dof, n_out = 6, 4
        rng = np.random.default_rng()
        freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
        coords = np.arange(n_out)
        modes = mechanical.PartialModesMap(freqs, coords)
        modes.atol = 1e-20
        modes.rtol = 1e-15

        return modes, n_out, dof
    
    def test_jac(self, initial):
        modes, n_out, dof = initial
        rng = np.random.default_rng()

        modes_i = np.nan
        while isinstance(modes_i, float):
            xi = rng.normal(size=(n_out, dof))
            zi = 5e-2*rng.normal(size=(n_out, dof))
            zi[:n_out, :n_out] = zi[:n_out, :n_out] - zi[:n_out, :n_out].T
            modes_i = modes(xi, zi)

        dx = rng.normal(size=(n_out, dof))
        dz = rng.normal(size=(n_out, dof))
        dz[:n_out, :n_out] = dz[:n_out, :n_out] - dz[:n_out, :n_out].T

        eval = rng.normal(size=(n_out, dof))

        jac_ana = np.sum(eval * modes.jac(xi, zi)(dx, dz))
        test = False
        for exp in range(3, 10):
            delta = 10**-exp
            modes_f = modes(xi + delta*dx, zi + delta*dz)
            jac_num = np.sum(eval * (modes_f - modes_i)/delta)
            test = np.allclose(jac_num, jac_ana, rtol=1e-5, atol=0.)
            if test:
                break
        assert test
    
    def test_hessp(self, initial):
        modes, n_out, dof = initial
        rng = np.random.default_rng()

        modes_i = np.nan
        while isinstance(modes_i, float):
            xi = rng.normal(size=(n_out, dof))
            zi = 5e-2*rng.normal(size=(n_out, dof))
            zi[:n_out, :n_out] = zi[:n_out, :n_out] - zi[:n_out, :n_out].T
            modes_i = modes(xi, zi)

        dx = rng.normal(size=(n_out, dof))
        dz = rng.normal(size=(n_out, dof))
        dz[:n_out, :n_out] = dz[:n_out, :n_out] - dz[:n_out, :n_out].T

        vx = rng.normal(size=(n_out, dof))
        vz = rng.normal(size=(n_out, dof))
        vz[:n_out, :n_out] = vz[:n_out, :n_out] - vz[:n_out, :n_out].T

        eval = rng.normal(size=(n_out, dof))

        hessp_ana = np.sum(eval * modes.hessp(xi, zi, vx, vz)(dx, dz))
        jac_modes_i = modes.jac(xi, zi)(vx, vz)
        test = False
        for exp in range(3, 10):
            delta = 10**-exp
            jac_modes_f = modes.jac(xi + delta*dx, zi + delta*dz)(vx, vz)
            hessp_num = np.sum(eval * (jac_modes_f - jac_modes_i)/delta)
            test = np.allclose(hessp_num, hessp_ana, rtol=1e-5, atol=0.)
            if test:
                break
        assert test

class TestModesFitting:

    def test_prop(self):
        dof, n_out, n_in = 4, 3, 2
        rng = np.random.default_rng()
        freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
        modes_m = rng.normal(size=(n_out, dof))
        amps_m = mechanical.mode_to_amps(modes_m, n_out, n_in)

        ns, fs = 210, 100
        modes = mechanical.RealModes(freqs, amps_m, fs, ns)
        modes_fit = modes.fit()
        # The result is unique up to a sign flip in each mode.
        modes_fit *= np.sign(modes_m[0, :]/modes_fit[0, :])[np.newaxis, :]

        assert modes.success_
        assert np.allclose(modes_fit, modes_m, rtol=1e-4, atol=0.)

    def test_complex(self):
        dof, n_out, n_in = 4, 3, 2
        rng = np.random.default_rng()
        freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
        coords = np.arange(n_out)
        modes_m = np.nan
        while isinstance(modes_m, float):
            xm = rng.normal(size=(n_out, dof))
            zm = 5e-2*rng.normal(size=(n_out, dof))
            zm[:n_out, :n_out] = zm[:n_out, :n_out] - zm[:n_out, :n_out].T
            modes_m = mechanical.PartialModesMap(freqs, coords)(xm, zm)
        amps_m = mechanical.mode_to_amps(modes_m, n_out, n_in)

        ns, fs = 210, 100
        # Fit as proportional as initial guess.
        modes = mechanical.RealModes(
                freqs, amps_m, fs, ns)
        modes_real = modes.fit()

        # Check that real modes are not good enough.
        # The result is unique up to a sign flip in each mode.
        modes_real *= np.sign(np.real(modes_m[0, :]/modes_real[0, :]))[np.newaxis, :]
        assert not np.allclose(modes_real, modes_m, rtol=1e-4, atol=0.)

        modes = mechanical.ComplexModes(freqs, coords, amps_m, fs, ns)
        x0 = (modes_real, np.zeros_like(modes_real))
        modes_fit = modes.fit(x0, options={'verbose': 2, 'gtol': 1e-1})
        # The result is unique up to a sign flip in each mode.
        modes_fit *= np.sign(np.real(modes_m[0, :]/modes_fit[0, :]))[np.newaxis, :]

        assert modes.success_
        assert np.allclose(modes_fit, modes_m, rtol=1e-4, atol=0.)

# Fitting tests
# -------------

def test_trig_fft():
    rng = np.random.default_rng()
    n = 10
    x = rng.normal(size=n)
    x = x - np.mean(x)
    x_fft = mechanical.trig_fft(x)
    x_ifft = mechanical.trig_ifft(x_fft)
    assert np.allclose(x, x_ifft)

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
        data_pred = self.kernel(ns, fs, amps_fit, freqs)
        assert np.allclose(data, data_pred, atol=0.)

    def test_mech_amps(self):
        dof = 4
        modal = mechanical.randomSystem(
            dof,
            mass_range=(1, 2),
            damping_range=(0.02, 0.05),
            freqs_range=(2 * np.pi * 1, 2 * np.pi * 20),
            damping_type='nop',
            seed=123456)[1]
        freqs = modal['frequencies']
        modes = modal['mode_shapes']
        modes = modes * (1/np.sqrt(np.imag(freqs)))[np.newaxis, :]
        ns, fs = 210, 100
        n_out, n_in = 3, 2

        amps = mechanical.mode_to_amps(modes, n_out, n_in)
        data = self.kernel(ns, fs, amps, freqs)

        model = mechanical.Amplitudes(
            freqs=freqs,
            fs=fs,
            ns=ns,
            n_out=n_out,
            n_in=n_in,
            a_type='mechanical')
        amps_fit = model.fit(data, penalty=0.)
        data_pred = self.kernel(ns, fs, amps_fit, freqs)
        assert np.allclose(data, data_pred, atol=0.)
