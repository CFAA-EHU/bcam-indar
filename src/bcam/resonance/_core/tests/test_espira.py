import pytest

import numpy as np
import scipy

from bcam.resonance._core import espira


class Test_RationalApproximation:

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
        res = espira.RationalApproximation(tol=1e-6)
        res.fit(x)

        assert len(res.poles_) == M
        assert np.allclose(np.sort_complex(res.poles_), np.sort_complex(poles))
        assert np.allclose(res.eval(ωN**(-np.arange(N))), x)

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
        res = espira.RationalApproximation(tol=1e-6)
        res.fit(x, tol=1e-6)

        assert len(res.poles_) == M
        assert np.allclose(np.sort_complex(res.poles_), np.sort_complex(poles))
        assert np.allclose(res.eval(ωN**(-np.arange(N))), x)

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
        res = espira.RationalApproximation(tol=1e-6)
        res.fit(x, tol=1e-6)

        assert len(res.poles_) == M
        assert np.allclose(np.sort_complex(res.poles_), np.sort_complex(poles))
        assert np.allclose(res.eval(ωN**(-np.arange(N))), x)

    def test_symmetric_fit(self):
        pass
