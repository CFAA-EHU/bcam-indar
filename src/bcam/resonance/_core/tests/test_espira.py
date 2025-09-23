import pytest

import numpy as np
import scipy

from bcam.resonance._core import espira


# ============================
# Test Rational Approximation
# ============================

class Test_RatApp:

    def test_rational_fit(self):
        M = 10
        N = 2 * (M + 1) + 10 # N >= 2 * (M + 1)
        rng = np.random.default_rng()

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0, 1, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, 1)) + 1j * rng.normal(0, 2, (M, 1))

        # Construct signal.
        ωN = np.exp(-2j * np.pi / N)
        x = espira.rational_function(poles, residues, ωN**(-np.arange(N)))

        # Fit the signal.
        res = espira.RatApp()
        poles_fit, residues_fit = res.fit(x, tol=1e-6)[:2]
        x_fit = espira.rational_function(poles_fit, residues_fit, ωN**(-np.arange(N)))

        assert len(poles_fit) == M
        assert np.allclose(np.sort_complex(poles_fit), np.sort_complex(poles))
        assert np.allclose(x_fit, x)

    def test_vector_rational_fit(self):
        M = 10
        L = 4
        N = 2 * (M + 1) + 40 # N >= 2 * (M + 1)
        rng = np.random.default_rng()

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0, 1, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, L)) + 1j * rng.normal(0, 2, (M, L))

        # Construct signal.
        ωN = np.exp(-2j * np.pi / N)
        x = espira.rational_function(poles, residues, ωN**(-np.arange(N)))

        # Fit the signal.
        res = espira.RatApp()
        poles_fit, residues_fit = res.fit(x, tol=1e-6)[:2]
        x_fit = espira.rational_function(poles_fit, residues_fit, ωN**(-np.arange(N)))

        assert len(poles_fit) == M
        assert np.allclose(np.sort_complex(poles_fit), np.sort_complex(poles))
        assert np.allclose(x_fit, x)

    def test_deficient_rational_fit(self):
        M = 10
        L = 3
        N = 2 * (M + 1) + 40 # N >= 2 * (M + 1)
        rng = np.random.default_rng()

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0, 1, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, L-1)) + 1j * rng.normal(0, 2, (M, L-1))
        add = rng.normal(0, 1, (L-1, 1))
        residues = np.concatenate((residues, residues@add), axis=1)

        # Construct signal.
        ωN = np.exp(-2j * np.pi / N)
        x = espira.rational_function(poles, residues, ωN**(-np.arange(N)))

        # Fit the signal.
        res = espira.RatApp()
        poles_fit, residues_fit = res.fit(x, tol=1e-6)[:2]
        x_fit = espira.rational_function(poles_fit, residues_fit, ωN**(-np.arange(N)))

        assert len(poles_fit) == M
        assert np.allclose(np.sort_complex(poles_fit), np.sort_complex(poles))
        assert np.allclose(x_fit, x)


class Test_RatAppSym:

    @staticmethod
    def _fit_symmetric(poles, residues, N):
        # Construct signal.
        ωN = np.exp(-2j * np.pi / N)
        x = espira.rational_function_sym(poles, residues, ωN**(-np.arange(N)))

        # Fit the signal.
        res = espira.RatAppSym()
        poles_fit, residues_fit = res.fit(x, tol=1e-6)[:2]
        rr, rc = residues_fit
        # Remove spurious poles with very small residues.
        idx_r = np.nonzero(np.linalg.norm(rr, axis=1) > 1e-5)[0]
        idx_c = np.nonzero(np.linalg.norm(rc, axis=1) > 1e-5)[0]
        poles_fit = (poles_fit[0][idx_r], poles_fit[1][idx_c])
        residues_fit = (rr[idx_r], rc[idx_c])

        return x, poles_fit, residues_fit

    def test_symmetric_fit(self):
        # Test only complex poles
        # ------------------------
        rng = np.random.default_rng()
        M = 6
        N = 2 * (2*M + 1) + 10
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0.01, 0.49, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, 1)) + 1j * rng.normal(0, 2, (M, 1))
        poles = (np.array([]), poles)
        residues = (np.array([]), residues)

        x, poles_fit, residues_fit = self._fit_symmetric(poles, residues, N)
        x_fit = espira.rational_function_sym(poles_fit, residues_fit, ωN**(-np.arange(N)))

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M)
        assert np.allclose(np.sort_complex(poles_fit[1]), np.sort_complex(poles[1]))
        assert np.allclose(x_fit, x)
        
        # Test with real and complex poles.
        # ------------------------
        MR, MC = 2, 4
        N = 2 * (2*MC + MR + 1) + 10
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, MC)
        phase = rng.uniform(0.01, 0.49, MC)
        poles_C = r * np.exp(2j * np.pi * phase)
        poles_R = np.zeros(MR, dtype=np.float64)
        poles_R[0] = rng.uniform(0.7, 0.9)
        poles_R[1] = -rng.uniform(0.7, 0.9)
        poles = (poles_R, poles_C)
        residues_R = rng.normal(0, 2, (MR, 1))
        residues_C = rng.normal(0, 2, (MC, 1)) + 1j * rng.normal(0, 2, (MC, 1))
        residues = (residues_R, residues_C)

        x, poles_fit, residues_fit = self._fit_symmetric(poles, residues, N)
        x_fit = espira.rational_function_sym(poles_fit, residues_fit, ωN**(-np.arange(N)))

        assert (len(poles_fit[0]) == MR) and (len(poles_fit[1]) == MC)
        for i in [0, 1]:
            assert np.allclose(
                np.sort_complex(poles_fit[i]), np.sort_complex(poles[i]))
        assert np.allclose(x_fit, x)
        
    def test_vector_fit(self):
        # Test only complex poles
        # ------------------------
        rng = np.random.default_rng()
        M = 6
        L = 3
        N = 2 * (2*M + 1) + 10
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0.01, 0.49, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, L)) + 1j * rng.normal(0, 2, (M, L))
        poles = (np.array([]), poles)
        residues = (np.array([]), residues)

        x, poles_fit, residues_fit = self._fit_symmetric(poles, residues, N)

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M)
        assert np.allclose(np.sort_complex(poles_fit[1]), np.sort_complex(poles[1]))
        assert np.allclose(
            espira.rational_function_sym(poles_fit, residues_fit, ωN**(-np.arange(N))), x)

# ============
# Test ESPIRA
# ============

class Test_ESPIRA:

    def test_espira(self):
        M, L = 5, 3
        N = 2 * (M + 1) + 10 # N >= 2 * (M + 1)
        rng = np.random.default_rng()

        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(-0.5, 0.5, M)
        poles = r * np.exp(2j * np.pi * phase)
        amps = rng.normal(0, 2, (M, L)) + 1j * rng.normal(0, 2, (M, L))

        x = espira.exp_sum(poles, amps, np.arange(N), fs=1)
        self.x = np.array(x)

        # ==== Fit exponential sum ====
        res = espira.Espira()
        amps_fit = res.fit(x, RatApp_kwargs={'tol': 1e-6})[0]
        poles_fit = res.poles_
        x_fit = espira.exp_sum(poles_fit, amps_fit, np.arange(N), fs=1)

        assert len(poles_fit) == M
        assert np.allclose(x_fit, x)

    def test_espira_real(self):
        # Create a random generator.
        M, L = 4, 2
        N = 2 * (M + 1) + 10 # N >= 2 * (M + 1)
        rng = np.random.default_rng()

        r = rng.uniform(0.7, 0.9, M//2)
        phase = rng.uniform(0.1, 0.4, M//2)
        poles = r * np.exp(2j * np.pi * phase)
        poles = np.concatenate((poles, np.conj(poles)))
        amps = rng.normal(0, 2, (M//2, L)) + 1j * rng.normal(0, 2, (M//2, L))
        amps = np.concatenate((amps, np.conj(amps)), axis=0)

        x = espira.exp_sum(poles, amps, np.arange(N), fs=1)
        self.x = np.array(x)

        # ==== Fit exponential sum ====
        res = espira.EspiraR()
        amps_fit = res.fit(x, RatApp_kwargs={'tol': 1e-6})[0]
        poles_fit = res.poles_
        x_fit = espira.exp_sum_R(poles_fit, amps_fit, np.arange(N), fs=1)

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M//2)
        assert np.allclose(x_fit, x)
