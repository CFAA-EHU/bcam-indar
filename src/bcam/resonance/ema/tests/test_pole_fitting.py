import pytest

import numpy as np

from bcam.resonance.ema import pole_fitting


# ============================
# Test Rational Approximation
# ============================

def random_rational(
        n_real_poles, n_complex_pairs, rank=1, ns=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_poles = n_real_poles + 2*n_complex_pairs
    if ns is None:
        ns = (rank+1)*2*n_poles//rank + 10 + rng.integers(0, 2)
    u = np.exp(2j * np.pi / ns)

    # Create complex frequencies and residues.
    poles_r = rng.uniform(0.8, 0.95, n_real_poles)
    poles_r = rng.choice([-1, 1], size=n_real_poles) * poles_r
    rad = rng.uniform(0.8, 0.95, n_complex_pairs)
    phase = rng.uniform(0.01, 0.49, n_complex_pairs)
    poles_cx = rad * np.exp(2j * np.pi * phase)

    r_r = rng.normal(size=(n_real_poles, rank))
    r_cx = rng.normal(size=(n_complex_pairs, rank)) + 1j * rng.normal(size=(n_complex_pairs, rank))
    
    idxs = np.argsort(poles_r)
    poles_r = poles_r[idxs]
    r_r = r_r[idxs]
    idxs = np.argsort(poles_cx)
    poles_cx = poles_cx[idxs]
    r_cx = r_cx[idxs]
    
    poles = np.concatenate((poles_r, poles_cx, np.conj(poles_cx)))
    r = np.concatenate((r_r, r_cx, np.conj(r_cx)), axis=0)

    # Construct signal.
    y = pole_fitting.rational(poles, r, 0., u**np.arange(ns//2+1))

    return (poles_r, poles_cx), (r_r, r_cx), (y, ns%2)


class Test_AAA:

    def test_scalar_complex(self):
        # Test only complex poles
        # ------------------------
        parent_rng = np.random.default_rng()
        n_trials = 5
        children = parent_rng.spawn(n_trials)
        for rng in children:
            M = rng.integers(1, 10)
            poles, r, y = random_rational(
                n_real_poles=0, n_complex_pairs=M, rank=1, rng=rng)

            # Fit the signal.
            model = pole_fitting.AAA(
                order=2*M, prune_tol=1e-5)
            model.fit(*y)

            for part in ['real', 'cx']:
                p_ = getattr(model.poles_, part)
                idxs = np.argsort(p_)
                setattr(model.poles_, part, p_[idxs])
                setattr(model.r_, part, getattr(model.r_, part)[idxs])

            assert (len(model.poles_.real) == 0) and (len(model.poles_.cx) == M)
            assert np.allclose(model.poles_.cx, poles[1])
            assert np.allclose(model.r_.cx, r[1])

    def test_scalar_mixed(self):
        # Test only complex poles
        # ------------------------
        parent_rng = np.random.default_rng()
        n_trials = 5
        children = parent_rng.spawn(n_trials)
        for rng in children:
            M = rng.integers(1, 5)
            L = rng.integers(1, 5)
            poles, r, y = random_rational(
                n_real_poles=L, n_complex_pairs=M, rank=1, rng=rng)

            # Fit the signal.
            model = pole_fitting.AAA(
                order=2*M+L, prune_tol=1e-5)
            model.fit(*y)

            for part in ['real', 'cx']:
                p_ = getattr(model.poles_, part)
                idxs = np.argsort(p_)
                setattr(model.poles_, part, p_[idxs])
                setattr(model.r_, part, getattr(model.r_, part)[idxs])

            assert (len(model.poles_.real) == L) and (len(model.poles_.cx) == M)
            assert np.allclose(model.poles_.real, poles[0])
            assert np.allclose(model.poles_.cx, poles[1])
            assert np.allclose(model.r_.real, r[0])
            assert np.allclose(model.r_.cx, r[1])

    def test_vector_mixed(self):
        # Test only complex poles
        # ------------------------
        parent_rng = np.random.default_rng()
        n_trials = 5
        children = parent_rng.spawn(n_trials)
        for rng in children:
            M = rng.integers(2, 5)
            L = rng.integers(1, 5)
            poles, r, y = random_rational(
                n_real_poles=L, n_complex_pairs=M, rank=3, rng=rng)

            # Fit the signal.
            model = pole_fitting.AAA(
                order=2*M+L, prune_tol=1e-5)
            model.fit(*y)

            for part in ['real', 'cx']:
                p_ = getattr(model.poles_, part)
                idxs = np.argsort(p_)
                setattr(model.poles_, part, p_[idxs])
                setattr(model.r_, part, getattr(model.r_, part)[idxs])

            assert (len(model.poles_.real) == L) and (len(model.poles_.cx) == M)
            assert np.allclose(model.poles_.real, poles[0])
            assert np.allclose(model.poles_.cx, poles[1])
            assert np.allclose(model.r_.real, r[0])
            assert np.allclose(model.r_.cx, r[1])


class Test_VF:

    def test_scalar_complex(self):
        # Test only complex poles
        # ------------------------
        parent_rng = np.random.default_rng()
        n_trials = 5
        children = parent_rng.spawn(n_trials)
        for rng in children:
            M = rng.integers(1, 10)
            poles, r, y = random_rational(
                n_real_poles=0, n_complex_pairs=M, rank=1, rng=rng)

            # Fit the signal.
            model = pole_fitting.VF(order=2*M, niter=2)
            model.fit(*y)

            for part in ['real', 'cx']:
                p_ = getattr(model.poles_, part)
                idxs = np.argsort(p_)
                setattr(model.poles_, part, p_[idxs])
                setattr(model.r_, part, getattr(model.r_, part)[idxs])

            assert (len(model.poles_.real) == 0) and (len(model.poles_.cx) == M)
            assert np.allclose(model.poles_.cx, poles[1])
            assert np.allclose(model.r_.cx, r[1])

    def test_scalar_mixed(self):
        # Test only complex poles
        # ------------------------
        parent_rng = np.random.default_rng()
        n_trials = 5
        children = parent_rng.spawn(n_trials)
        for rng in children:
            M = rng.integers(1, 5)
            L = rng.integers(1, 5)
            poles, r, y = random_rational(
                n_real_poles=L, n_complex_pairs=M, rank=1, rng=rng)

            # Fit the signal.
            model = pole_fitting.VF(order=2*M+L, niter=2)
            model.fit(*y)

            for part in ['real', 'cx']:
                p_ = getattr(model.poles_, part)
                idxs = np.argsort(p_)
                setattr(model.poles_, part, p_[idxs])
                setattr(model.r_, part, getattr(model.r_, part)[idxs])

            assert (len(model.poles_.real) == L) and (len(model.poles_.cx) == M)
            assert np.allclose(model.poles_.real, poles[0])
            assert np.allclose(model.poles_.cx, poles[1])
            assert np.allclose(model.r_.real, r[0])
            assert np.allclose(model.r_.cx, r[1])

    def test_vector_mixed(self):
        # Test only complex poles
        # ------------------------
        parent_rng = np.random.default_rng()
        n_trials = 5
        children = parent_rng.spawn(n_trials)
        for rng in children:
            M = rng.integers(2, 5)
            L = rng.integers(1, 5)
            poles, r, y = random_rational(
                n_real_poles=L, n_complex_pairs=M, rank=3, rng=rng)

            # Fit the signal.
            model = pole_fitting.VF(order=2*M+L, niter=2)
            model.fit(*y)

            for part in ['real', 'cx']:
                p_ = getattr(model.poles_, part)
                idxs = np.argsort(p_)
                setattr(model.poles_, part, p_[idxs])
                setattr(model.r_, part, getattr(model.r_, part)[idxs])

            assert (len(model.poles_.real) == L) and (len(model.poles_.cx) == M)
            assert np.allclose(model.poles_.real, poles[0])
            assert np.allclose(model.poles_.cx, poles[1])
            assert np.allclose(model.r_.real, r[0])
            assert np.allclose(model.r_.cx, r[1])

# ============
# Test SuperResolution
# ============

class Test_SuperResolution:

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

        x = pole_fitting.exp_sum(poles_, amps_, np.arange(N), fs=1)
        x = np.real(x)
        y = np.fft.rfft(x, axis=0)

        # ==== Fit exponential sum ====
        model = pole_fitting.Espira(
            order=2*M, copy_y=False)
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
