#!/usr/bin/env python

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from bcam.resonance._core import espira


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

# %%
if __name__ == '__main__':
    import logging

    logger = logging.getLogger(espira.__name__)
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

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
