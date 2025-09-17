#!/usr/bin/env python

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import bcam.vibrations.espira as espira


class Test_ESPIRA:

    def __init__(self, N, M, n_comps=1, dt=1, seed=None):
        self.N = N
        self.M = M
        self.n_comps = n_comps
        self.dt = dt
        self.seed = seed

        self.x = None
        self.original_freqs = None
        self.original_decays = None
        self.original_amplitudes = None

    def exp_sum_fit(self, espira_kwargs=None):
        # ==== Create exponential sum ====
        # Create a random generator.
        rng = np.random.default_rng(self.seed)
        M = self.M
        L = self.n_comps
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(-0.5, 0.5, M)
        z = r * np.exp(2j * np.pi * phase)
        amp = rng.normal(0, 2, (M, L)) + 1j * rng.normal(0, 2, (M, L))
        self.original_amplitudes = amp
        self.original_decays = np.log(np.abs(z)) / self.dt
        self.original_freqs = phase / self.dt

        x = espira.exp_sum(z, amp, np.arange(self.N), fs=1)
        self.x = np.array(x)

        # ==== Fit exponential sum ====
        espira_kwargs = espira_kwargs if espira_kwargs is not None else {}
        r = espira.Espira(fs=1/self.dt, **espira_kwargs)
        r.fit(x)
        del x
        r.remove_spurious()
        print(f'Error: {r.rational_.error_}')

        _, axs = plt.subplots()
        y = r.eval(self.dt * np.arange(self.N))
        axs.plot(
            np.arange(self.N),
            2 + np.log10(np.linalg.norm(self.x - y, axis=1)) - np.log10(np.linalg.norm(self.x, axis=1)),
            '-o'
        )
        axs.set_xlabel('Time')
        axs.set_ylabel(r'$\log_{10}$ of relative error (%)')

        return axs

    def exp_sum_fit_real(self, espira_kwargs=None):
        # ==== Create exponential sum ====
        # Create a random generator.
        rng = np.random.default_rng(self.seed)
        M = self.M
        L = self.n_comps
        r = rng.uniform(0.2, 0.9, M//2)
        phase = rng.uniform(0.1, 0.4, M//2)
        z = r * np.exp(2j * np.pi * phase)
        z = np.concatenate((z, np.conj(z)))
        amp = rng.normal(0, 2, (M//2, L)) + 1j * rng.normal(0, 2, (M//2, L))
        amp = np.concatenate((amp, np.conj(amp)), axis=0)
        self.original_amplitudes = amp
        self.original_decays = np.log(np.abs(z)) / self.dt
        self.original_freqs = np.angle(z) / self.dt

        x = espira.exp_sum(z, amp, np.arange(self.N), fs=1)
        self.x = np.array(x)

        # ==== Fit exponential sum ====
        espira_kwargs = espira_kwargs if espira_kwargs is not None else {}
        r = espira.Espira(mode='real', fs=1/self.dt, **espira_kwargs)
        r.fit(x)
        del x
        r.remove_spurious()
        print(f'Error: {r.rational_.error_}')

        _, axs = plt.subplots()
        y = r.eval(self.dt * np.arange(self.N))
        axs.plot(
            np.arange(self.N),
            2 + np.log10(np.linalg.norm(self.x - y, axis=1)) - np.log10(np.linalg.norm(self.x, axis=1)),
            '-o'
        )
        axs.set_xlabel('Time')
        axs.set_ylabel(r'$\log_{10}$ of relative error (%)')

        return axs


class Test_RationalApproximation:

    def __init__(self, N, M, n_comps=1, seed=None):
        self.N = N
        self.M = M
        self.n_comps = n_comps
        self.seed = seed

        self.x = None
        self.original_poles = None
        self.original_residues = None
        self.rational = None

    def rational_fit(self, rational_kwargs=None):
        N, M = self.N, self.M
        L = self.n_comps
        rng = np.random.default_rng(self.seed)

        # Create complex frequencies and residues.
        r = rng.uniform(0.7, 0.9, M)
        phase = rng.uniform(0, 1, M)
        poles = r * np.exp(2j * np.pi * phase)
        residues = rng.normal(0, 2, (M, L)) + 1j * rng.normal(0, 2, (M, L))

        # Construct signal.
        ωN = np.exp(-2j * np.pi / N)
        x = espira.rational_function(poles, residues, ωN**(-np.arange(N)))

        # Fit the signal.
        rational_kwargs = rational_kwargs if rational_kwargs is not None else {}
        r = espira.RationalApproximation(**rational_kwargs)
        r.fit(x)
        r.remove_spurious()
        print(f'Number of poles: {len(r.poles_)}')

        self.original_poles = poles
        self.original_residues = residues
        self.x = x
        self.rational = r

    def symmetric_rational_fit(self, rational_kwargs=None):
        N, M = self.N, self.M
        L = self.n_comps
        rng = np.random.default_rng(self.seed)

        # Create complex frequencies and coefficients.
        r = rng.uniform(0.4, 0.9, M//2)
        phase = rng.uniform(0.1, 0.4, M//2)
        poles = r * np.exp(2j * np.pi * phase)
        poles = np.concatenate((poles, np.conj(poles)))
        residues = rng.normal(0, 2, (M//2, L)) + 1j * rng.normal(0, 2, (M//2, L))
        residues = np.concatenate((residues, np.conj(residues)), axis=0)

        # Construct signal.
        ωN = np.exp(-2j * np.pi / N)
        x = espira.rational_function(poles, residues, ωN**(-np.arange(N)))

        # Fit the signal.
        rational_kwargs = rational_kwargs if rational_kwargs is not None else {}
        r = espira.RationalApproximation(mode='symmetric', **rational_kwargs)
        r.fit(x)
        r.remove_spurious()
        print(f'Number of poles: {len(r.poles_)}')

        self.original_poles = poles
        self.original_residues = residues
        self.x = x
        self.rational = r

    def plot_poles(self):
        z = self.original_poles
        z_r = self.rational.poles_
        
        fig, axs = plt.subplots()
        fig.suptitle('Recovered complex frequencies')
        axs.scatter(z.real, z.imag, color='b', label='Original')
        axs.scatter(z_r.real, z_r.imag, color='r', alpha=0.4, label='Retrieved')
        # Draw a unit circle.
        axs.add_patch(patches.Circle((0, 0), 1, fill=False, edgecolor='black'))
        axs.set_aspect('equal')
        axs.set_xlabel('Real')
        axs.set_ylabel('Imaginary')
        axs.legend(loc='upper right')

        return axs

    def plot_approximation(self, comp=0):
        x = self.x
        N = len(x)
        ωN = np.exp(-2j * np.pi / N)
        x_r = self.rational.eval(ωN**(-np.arange(N)))

        fig, axs = plt.subplots()
        fig.suptitle('Polynomial approximation')
        axs.plot(np.abs(x[:, comp]), '-o', color='b', label='Original')
        axs.plot(np.abs(x_r[:, comp]), '-o', color='r', alpha=0.4, label='Retrieved')
        axs.legend()
        axs.set_ylabel(f'|x{comp}|')

        return axs


# %%
if __name__ == '__main__':
    import logging

    logger = logging.getLogger(espira.__name__)
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

# # %%
# # ==== Test normal AAA (1DoF) ====
#     M = 10
#     N = 2 * (M + 1) + 10 # N >= 2 * (M + 1)
#     kwargs = {'max_order': None, 'tol': 1e-6}
#     test = Test_RationalApproximation(N, M, seed=None)
#     test.rational_fit(rational_kwargs=kwargs)
#     test.plot_poles()
#     test.plot_approximation()
#     plt.show()

# # %%
# # ==== Test symmetric AAA (1DoF) ====
#     M = 10 # Even number
#     N = 2 * (M + 1) + 1 # N >= 2 * (M + 1)
#     kwargs = {'max_order': None, 'tol': 1e-6}
#     test = Test_RationalApproximation(N, M, seed=None)
#     test.symmetric_rational_fit(rational_kwargs=kwargs)
#     test.plot_poles()
#     test.plot_approximation()
#     plt.show()

# # %%
# # ==== Test normal AAA (nDoF) ====
#     M = 10
#     L = 4
#     N = 2 * (M + 1) + 40 # N >= 2 * (M + 1)
#     kwargs = {'max_order': None, 'tol': 1e-6}
#     test = Test_RationalApproximation(N, M, n_comps=L, seed=None)
#     test.rational_fit(rational_kwargs=kwargs)
#     test.plot_poles()
#     test.plot_approximation(comp=3)
#     plt.show()

# # %%
# # ==== Test symmetric AAA (nDoF) ====
#     M = 10 # Even number
#     L = 4
#     N = 2 * (M + 1) + 10 # N >= 2 * (M + 1)
#     kwargs = {'max_order': None, 'tol': 1e-10}
#     test = Test_RationalApproximation(N, M, n_comps=L, seed=None)
#     test.symmetric_rational_fit(rational_kwargs=kwargs)
#     test.plot_poles()
#     test.plot_approximation(comp=1)
#     plt.show()

# # %%
# # ==== Test ESPIRA complex ====
#     M = 5
#     L = 2
#     N = 2 * (M + 1) + 10 # N >= 2 * (M + 1)
#     kwargs = {'max_order': None, 'tol': 1e-6}
#     test = Test_ESPIRA(N, M, n_comps=L, seed=None)
#     test.exp_sum_fit(espira_kwargs=kwargs)
#     plt.show()

# # %%
# # ==== Test ESPIRA real ====
#     M = 10
#     L = 2
#     N = 2 * (M + 1) + 5 # N >= 2 * (M + 1)
#     test = Test_ESPIRA(N, M, n_comps=L, seed=None)
#     kwargs = {'max_order': None, 'tol': 1e-6}
#     test.exp_sum_fit_real(espira_kwargs=kwargs)
#     plt.show()
