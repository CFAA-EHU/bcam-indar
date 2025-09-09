'''
This is ...
'''

import numpy as np
import matplotlib.pyplot as plt

from bcam.resonance._core import mechanical


# # ==========================
# # Test loss function and df
# # for complex mode shapes
# # ==========================
# dof, n_out, n_in = 4, 3, 2
# rng = np.random.default_rng()
# freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
# x0 = rng.normal(size=(n_out, dof))
# amps0 = mechanical.mode_to_amps(x0, n_out, n_in)
# coords = np.arange(n_out)

# ns, fs = 210, 100
# mechanical.PartialModesMap.atol = 1e-20
# mechanical.PartialModesMap.rtol = 1e-15
# modes = mechanical.Modes(freqs, amps0, coords, fs, ns)

# xi = rng.normal(size=(n_out, dof))
# zi = 5e-2*rng.normal(size=(n_out, dof))
# zi[:n_out, :n_out] = zi[:n_out, :n_out] - zi[:n_out, :n_out].T
# pi = mechanical.reshape_modes_output(xi, zi)

# dx = rng.normal(size=(n_out, dof))
# dz = rng.normal(size=(n_out, dof))
# dz[:n_out, :n_out] = dz[:n_out, :n_out] - dz[:n_out, :n_out].T
# dp = mechanical.reshape_modes_output(dx, dz)

# ll = np.linspace(-1e-2, 1e-2, 500)
# f_line = [modes._fun(pi+l*dp) for l in ll]
# f_line = np.array(f_line)
# fpi = modes._fun(pi)
# df = modes._jac(pi) @ dp
# d2f = modes._hessp(pi, dp) @ dp

# fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(10, 5))
# fig.suptitle('Test derivatives of objective for real mode shapes fitting')

# ax[0].set_title('1st order')
# ax[0].plot(ll, f_line - (fpi + df*ll))
# ax[0].axhline(0, color='k', linestyle='--', linewidth=1)

# ax[1].set_title('2nd order')
# ax[1].plot(ll, f_line - (fpi + df*ll + 0.5*d2f*(ll**2)))
# ax[1].axhline(0, color='k', linestyle='--', linewidth=1)
# plt.show()

# # ================================
# # Test constrains and its jacobian
# # ================================
# dof, n_out = 4, 3
# rng = np.random.default_rng()
# freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
# coords = np.arange(n_out)

# modes = mechanical.PartialModesMap(freqs, coords)

# xi = 0.1*rng.normal(size=(n_out, dof))
# zi = 1e-3*rng.normal(size=(n_out, dof))
# zi[:n_out, :n_out] = zi[:n_out, :n_out] - zi[:n_out, :n_out].T
# dx = 0.1*rng.normal(size=(n_out, dof))
# dz = 1e-3*rng.normal(size=(n_out, dof))
# dz[:n_out, :n_out] = dz[:n_out, :n_out] - dz[:n_out, :n_out].T

# constr = modes.constraints(xi, zi)
# vec = rng.normal(size=2)
# ll = 1e-3 * np.arange(-40, 41)
# constr_line = [
#     modes.constraints(xi + l*dx, zi + l*dz) @ vec for l in ll]
# constr_line = np.array(constr_line)
# const_i = vec @ modes.constraints(xi, zi)
# jac_const = modes.jac_constraints(xi, zi)
# jac_const = vec @ jac_const(dx, dz)

# fig, ax = plt.subplots(ncols=1, sharex=True, figsize=(5, 5))
# fig.suptitle('Test derivatives of constraints for complex mode shapes fitting')

# ax.set_title('1st order')
# ax.plot(ll, constr_line - (const_i + jac_const*ll))
# ax.axhline(0, color='k', linestyle='--', linewidth=1)

# plt.show()


# # ================================
# # Test loss function and df
# # for real mode shapes
# # ================================
# dof, n_out, n_in = 4, 3, 2
# rng = np.random.default_rng()
# freqs = -rng.uniform(-2, -1, dof) + 1j*rng.uniform(2*np.pi, 2*np.pi*20, dof)
# X = 0.1*rng.normal(size=(n_out, dof))
# amps = _mode_to_amps(X, n_out, n_in)
# ns, fs = 210, 100

# modes = _ModesProp(
#     freqs, fs, ns, n_out=n_out, n_in=n_in)
# x = rng.normal(scale=.1, size=n_out*dof)
# v = rng.normal(scale=.1, size=n_out*dof)
# fx, dfx = modes._fun(x, amps, dof)
# ddfxp = modes._hessp(x, v, amps, dof)

# L = 1e-3 * np.arange(-40, 41)
# f_line = [modes._fun(x + l*v, amps, dof)[0] for l in L]
# f_line = np.array(f_line)

# fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(10, 5))
# fig.suptitle('Test derivatives of objective for real mode shapes fitting')

# ax[0].set_title('1st order')
# ax[0].plot(L, f_line - (fx + (dfx@v)*L))
# ax[0].axhline(0, color='k', linestyle='--', linewidth=1)

# ax[1].set_title('2nd order')
# ax[1].plot(L, f_line - (fx + (dfx@v)*L + 0.5*(ddfxp@v)*(L**2)))
# ax[1].axhline(0, color='k', linestyle='--', linewidth=1)
# plt.show()
