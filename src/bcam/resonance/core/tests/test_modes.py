from xml.parsers.expat import model
import pytest

import numpy as np
import scipy

import bcam.resonance.core.mechanical as mechanical


class TestModeMap:

    @pytest.fixture(scope='class')
    def mode_shapes(self):
        dof, n_out = 6, 4
        rng = np.random.default_rng(123455)

        X = rng.normal(size=(n_out, dof))
        Z = rng.normal(scale=1e-3, size=(n_out, dof))
        Z[:n_out, :n_out] = (Z[:n_out, :n_out] - Z[:n_out, :n_out].T)/2
        freqs = -rng.uniform(0.1, 10, size=dof) + 1j * rng.uniform(size=dof)
        psi = mechanical.partial_modes_map(X, Z, freqs)

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

def test_jac_qr():
    rng = np.random.default_rng()
    X = rng.normal(size=(4, 3))
    q, r = scipy.linalg.qr(
        X, overwrite_a=False, mode='economic', pivoting=False)
    idx = np.argwhere(np.diag(r) < 0)
    q[:, idx] *= -1
    r[idx, :] *= -1
    dX = rng.normal(size=(4, 3))
    dq, dr = mechanical.jac_qr(X)(dX)
    assert np.allclose(dX, dq@r + q@dr)
