import pytest

import numpy as np

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
        res = espira.RatApp(x, rank_tol=1e-6)
        poles_fit, residues_fit = res.fit(tol=1e-6)[:2]
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
        res = espira.RatApp(x, rank_tol=1e-6)
        poles_fit, residues_fit = res.fit(tol=1e-6)[:2]
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
        res = espira.RatApp(x, rank_tol=1e-6)
        poles_fit, residues_fit = res.fit(tol=1e-6)[:2]
        x_fit = espira.rational_function(poles_fit, residues_fit, ωN**(-np.arange(N)))

        assert len(poles_fit) == M
        assert np.allclose(np.sort_complex(poles_fit), np.sort_complex(poles))
        assert np.allclose(x_fit, x)


class Test_RatAppSym:

    def test_symmetric_fit_complex(self):
        # Test only complex poles
        # ------------------------
        rng = np.random.default_rng()
        M = 6
        N = 2 * (2*M + 1) + 10 + rng.integers(0, 2)
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0.01, 0.49, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, 1)) + 1j * rng.normal(0, 2, (M, 1))
        idxs = np.argsort(poles)
        poles = poles[idxs]
        residues = residues[idxs]
        poles_ = np.concatenate((poles, np.conj(poles)))
        residues_ = np.concatenate((residues, np.conj(residues)), axis=0)

        # Construct signal.
        y = espira.rational_function(poles_, residues_, ωN**(-np.arange(N//2+1)))

        # Fit the signal.
        model = espira.RatAppSym(order=2*M)
        model.fit(y, N%2)
        model.pole_pruning(tol=1e-5)

        poles_fit = [model.r_poles_, model.c_poles_]
        residues_fit = [model.r_residues_, model.c_residues_]

        idxs = np.argsort(poles_fit[1])
        poles_fit[1] = poles_fit[1][idxs]
        residues_fit[1] = residues_fit[1][idxs]

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M)
        assert np.allclose(poles_fit[1], poles)
        assert np.allclose(residues_fit[1], residues)

    def test_symmetric_fit_real_complex(self):
        # Test with real and complex poles.
        # ------------------------
        rng = np.random.default_rng()
        MR, MC = 2, 4
        N = 2 * (2*MC + MR + 1) + 10 + rng.integers(0, 2)
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, MC)
        phase = rng.uniform(0.01, 0.49, MC)
        poles_C = r * np.exp(2j * np.pi * phase)
        idxs = np.argsort(poles_C)
        poles_C = poles_C[idxs]
        poles_R = np.zeros(MR, dtype=np.float64)
        poles_R[0] = -rng.uniform(0.7, 0.9)
        poles_R[1] = rng.uniform(0.7, 0.9)
        poles = [poles_R, poles_C]
        residues_R = rng.normal(0, 2, (MR, 1))
        residues_C = rng.normal(0, 2, (MC, 1)) + 1j * rng.normal(0, 2, (MC, 1))
        residues_C = residues_C[idxs]
        residues = [residues_R, residues_C]
        poles_ = np.concatenate((poles_R, poles_C, np.conj(poles_C)))
        residues_ = np.concatenate((residues_R, residues_C, np.conj(residues_C)), axis=0)

        # Construct signal.
        y = espira.rational_function(poles_, residues_, ωN**(-np.arange(N//2+1)))

        # Fit the signal.
        model = espira.RatAppSym(order=2*MC + MR)
        model.fit(y, N%2)
        model.pole_pruning(tol=1e-5)

        poles_fit = [model.r_poles_, model.c_poles_]
        residues_fit = [model.r_residues_, model.c_residues_]

        for i in range(2):
            idxs = np.argsort(poles_fit[i])
            poles_fit[i] = poles_fit[i][idxs]
            residues_fit[i] = residues_fit[i][idxs]

        assert (len(poles_fit[0]) == MR) and (len(poles_fit[1]) == MC)
        for i in [0, 1]:
            assert np.allclose(poles_fit[i], poles[i])
            assert np.allclose(residues_fit[i], residues[i])

    def test_vector_fit(self):
        # Test only complex poles
        # ------------------------
        rng = np.random.default_rng()
        M = 6
        L = 3
        N = 2 * (2*M + 1) + 10 + rng.integers(0, 2)
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0.01, 0.49, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, L)) + 1j * rng.normal(0, 2, (M, L))
        idxs = np.argsort(poles)
        poles = poles[idxs]
        residues = residues[idxs]
        poles_ = np.concatenate((poles, np.conj(poles)))
        residues_ = np.concatenate((residues, np.conj(residues)), axis=0)

        # Construct signal.
        y = espira.rational_function(poles_, residues_, ωN**(-np.arange(N//2+1)))

        # Fit the signal.
        model = espira.RatAppSym(order=2*M)
        model.fit(y, N%2)
        model.pole_pruning(tol=1e-5)

        poles_fit = [model.r_poles_, model.c_poles_]
        residues_fit = [model.r_residues_, model.c_residues_]

        idxs = np.argsort(poles_fit[1])
        poles_fit[1] = poles_fit[1][idxs]
        residues_fit[1] = residues_fit[1][idxs]

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M)
        assert np.allclose(poles_fit[1], poles)
        assert np.allclose(residues_fit[1], residues)

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

        x = espira.exp_sum(poles, amps, np.arange(N))
        self.x = np.array(x)

        # ==== Fit exponential sum ====
        res = espira.Espira(x, rank_tol=1e-6)
        amps_fit, poles_fit = res.fit(tol=1e-6)
        x_fit = espira.exp_sum(poles_fit, amps_fit, np.arange(N), fs=1)

        assert len(poles_fit) == M
        assert np.allclose(x_fit, x)

    def test_espira_real(self):
        # Create a random generator.
        M, L = 4, 2
        N = 2*(M+1) + 10 # N >= 2 * (M + 1)
        rng = np.random.default_rng()

        r = rng.uniform(0.7, 0.9, M//2)
        phase = rng.uniform(0.1, 0.4, M//2)
        poles = r * np.exp(2j * np.pi * phase)
        poles_ = np.concatenate((poles, np.conj(poles)))
        amps = rng.normal(0, 2, (M//2, L)) + 1j * rng.normal(0, 2, (M//2, L))
        amps_ = np.concatenate((amps, np.conj(amps)), axis=0)
        idxs = np.argsort(poles)
        poles = poles[idxs]
        amps = amps[idxs]

        x = espira.exp_sum(poles_, amps_, np.arange(N), fs=1)
        x = np.real(x)
        y = np.fft.rfft(x, axis=0)

        # ==== Fit exponential sum ====
        model = espira.EspiraR(order=2*M)
        model.fit(y, N%2)
        # Remove spurious poles with very small amplitude.
        model.pole_pruning(tol=1e-5)
        
        poles_fit = [model.r_poles_, model.c_poles_]
        amps_fit = [model.r_amps_, model.c_amps_]
        idxs = np.argsort(poles_fit[1])
        poles_fit[1] = poles_fit[1][idxs]
        amps_fit[1] = amps_fit[1][idxs]

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M//2)
        assert np.allclose(poles_fit[1], poles)
        assert np.allclose(amps_fit[1], amps)


class Test_Rational:

    def test_fit_complex(self):
        # Test only complex poles
        # ------------------------
        rng = np.random.default_rng()
        M = 1
        N = 2 * (2*M + 1) + 10 + 0
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.8, 0.9, M)
        phase = rng.uniform(0.01, 0.49, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, 1)) + 1j * rng.normal(0, 2, (M, 1))
        idxs = np.argsort(poles)
        poles = poles[idxs]
        residues = residues[idxs]
        poles_ = np.concatenate((poles, np.conj(poles)))
        residues_ = np.concatenate((residues, np.conj(residues)), axis=0)

        # Construct signal.
        y = espira.rational_function(poles_, residues_, ωN**(-np.arange(N//2+1)))

        # Fit the signal.
        model = espira.Rational(order=2*M)
        model.fit(y, N%2)
        model.pole_pruning(tol=1e-5)

        poles_fit = [model.r_poles_, model.c_poles_]
        residues_fit = [model.r_residues_, model.c_residues_]

        idxs = np.argsort(poles_fit[1])
        poles_fit[1] = poles_fit[1][idxs]
        residues_fit[1] = residues_fit[1][idxs]

        assert (len(poles_fit[0]) == 0) and (len(poles_fit[1]) == M)
        assert np.allclose(poles_fit[1], poles)
        assert np.allclose(residues_fit[1], residues)

    def test_fit_real_complex(self):
        # Test with real and complex poles.
        # ------------------------
        rng = np.random.default_rng()
        MR, MC = 3, 5
        N = 2 * (2*MC + MR + 1) + 10 + rng.integers(0, 2)
        ωN = np.exp(-2j * np.pi / N)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, MC)
        phase = rng.uniform(0.01, 0.49, MC)
        poles_C = r * np.exp(2j * np.pi * phase)
        poles_R = np.zeros(MR, dtype=np.float64)
        poles_R = rng.choice([-1, 1], size=MR)*rng.uniform(0.7, 0.9, MR)
        poles = [poles_R, poles_C]

        residues_R = rng.normal(0, 2, (MR, 1))
        residues_C = rng.normal(0, 2, (MC, 1)) + 1j * rng.normal(0, 2, (MC, 1))
        residues = [residues_R, residues_C]

        # Sort poles and residues.
        for i in range(2):
            idxs = np.argsort(poles[i])
            poles[i] = poles[i][idxs]
            residues[i] = residues[i][idxs]
        poles_ = np.concatenate((poles_R, poles_C, np.conj(poles_C)))
        residues_ = np.concatenate((residues_R, residues_C, np.conj(residues_C)), axis=0)

        # Construct signal.
        y = espira.rational_function(poles_, residues_, ωN**(-np.arange(N//2+1)))

        # Fit the signal.
        model = espira.Rational(order=2*MC + MR)
        model.fit(y, N%2)
        model.pole_pruning()

        poles_fit = [model.r_poles_, model.c_poles_]
        residues_fit = [model.r_residues_, model.c_residues_]

        for i in range(2):
            idxs = np.argsort(poles_fit[i])
            poles_fit[i] = poles_fit[i][idxs]
            residues_fit[i] = residues_fit[i][idxs]

        assert (len(poles_fit[0]) == MR) and (len(poles_fit[1]) == MC)
        for i in [0, 1]:
            assert np.allclose(poles_fit[i], poles[i])
            assert np.allclose(residues_fit[i], residues[i])
