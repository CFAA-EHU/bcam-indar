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
        psi = mechanical.partial_mode_shapes_map(X, Z, freqs)

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

def test_amplitudes():
    rng = np.random.default_rng(12345)
    dof = 4
    freqs = -rng.uniform(1, 3, dof) + 2j*np.pi*rng.uniform(1, 50, dof)
    ns, fs = 200, 100
    n_out, n_in = 3, 2

    def kernel(t, a, freqs):
        K = 2*fs*np.exp(freqs/(2*fs)) * np.sinh(freqs/(2*fs))
        K = a * K.reshape(1, 1, dof)
        K = np.expand_dims(K, axis=2)
        K = K * np.exp(freqs[np.newaxis, :] * t[:, np.newaxis]).reshape(1, 1, ns, dof)
        K = np.imag(np.sum(K, axis=-1))
        return K

    t = np.arange(ns) / fs

    amps = rng.normal(scale=1, size=(n_out, n_in, dof)).astype(np.complex128)
    amps += 1j * rng.normal(scale=.1, size=(n_out, n_in, dof))
    amps[:n_in] = 0.5 * (amps[:n_in] + amps[:n_in].transpose((1, 0, 2)))
    data = kernel(t, amps, freqs)

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
