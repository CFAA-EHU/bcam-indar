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
