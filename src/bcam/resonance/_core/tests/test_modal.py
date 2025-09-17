#!/usr/bin/env python

# %%
import numpy as np
import scipy
import scipy.optimize

import bcam.vibrations.linearModel as linearModel

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

# %%
# # ==== Test consistency of mode shapes ====
#     # This test should ideally return zero.
#     N = 5
#     seed = 1234
#     rng = np.random.default_rng(seed)
#     X = rng.normal(size=(N, N))
#     L = rng.normal(size=(N, N))
#     L = np.triu(L, k=1)
#     Z = rng.uniform(-1, -0.5, size=N) + 1j * rng.uniform(5, 10, size=N)
#     mode_shapes = linearModel.mode_shapes_generator(X, L, D=1/Z.imag)
#     M, C, K = linearModel.modal_to_system(mode_shapes, Z)

#     idxs = np.argsort(np.abs(Z))
#     Z = Z[idxs]
#     mode_shapes = mode_shapes[:, idxs]
#     M, C, K = linearModel.modal_to_system(mode_shapes, Z)
#     mode_shapes_r, Z_r = linearModel.system_to_modal(M, C, K)

#     print('Mode shapes: ', np.allclose(np.abs(mode_shapes), np.abs(mode_shapes_r)))
#     print('Frequencies: ', np.allclose(Z, Z_r))

#     L = np.zeros((2 * N, 2 * N), dtype=np.complex128)
#     L[:N, :N] = Φ
#     L[:N, N:] = np.conj(L[:N, :N])
#     L[N:, :N] = Φ @ np.diag(Z)
#     L[N:, N:] = np.conj(L[N:, :N])

#     tmp = 0.5j * np.concatenate((1/Z.imag, 1/Z.imag))
#     invL = np.zeros_like(L)
#     invL[:N, :N] = -np.diag(1/Z) @ Φ.T
#     invL[:N, N:] = -Φ.T
#     invL[N:, :N] = -np.conj(invL[:N, :N])
#     invL[N:, N:] = -np.conj(invL[:N, N:])
#     invL = np.diag(tmp) @ invL
#     tmp = np.zeros((2 * N, 2 * N), dtype=np.complex128)
#     tmp[:N, :N] = -K
#     tmp[N:, N:] = M
#     invL = invL @ tmp

#     print('$L L^{-1} = I$ : ', np.allclose(L @ invL, np.eye(2 * N)))
#     print('$L^{-1} L = I$ : ', np.allclose(invL @ L, np.eye(2 * N)))

# # %%
# # ==== Test class SymmetricModel : loss (no constraints) ====
#     rng = np.random.default_rng(123)
#     n_ext, n_res, n_dof = sorted(rng.integers(1, 6, 3))
#     Z = rng.uniform(-1, -0.5, size=n_dof) + 1j * rng.uniform(5, 10, size=n_dof)

#     # Generate tensor product without constraints.
#     x = np.random.normal(0, 1, 2 * n_dof * n_res)
#     x0 = x + np.random.normal(0, 1e-1, 2 * n_dof * n_res)
#     y = x[::2] + 1j * x[1::2]
#     y = y.reshape((n_dof, n_res))
#     tensor = y[:, :n_res, np.newaxis] * y[:, np.newaxis, :n_ext]
#     tensor = tensor.transpose((1, 2, 0))
#     test = linearModel.SymmetricModel(tensor, Z, np.arange(n_ext))

#     r = scipy.optimize.minimize(
#         test._loss,
#         x0 = x0,
#         method='Newton-CG',
#         jac=test._jac_loss)
#     x_r = r.x
#     print('Find vectors: ', np.allclose(x, x_r))

# %%
# ==== Test class SymmetricModel : loss with constraints ====
    # TODO
    # tensor = fake_modes[:n_res] @ fake_modes[:n_ext].T
    # x = fake_modes[:n_res].T.flatten()
    # x_ = np.zeros(2 * dofs * n_res)
    # x_[::2] = x.real
    # x_[1::2] = x.imag
    # x = x_
    # compatibility_jac(x).shape

    # comp = scipy.optimize.NonlinearConstraint(
    #     fun=compatibility,
    #     lb=0, ub=0,
    #     jac=compatibility_jac)

    # r = scipy.optimize.minimize(
    #     min_tensor,
    #     x0 = x0,
    #     method='trust-constr',
    #     jac=min_tensor_jac,
    #     constraints=comp)
    # x_r = r.x
    # print(f'Original: {x}')
    # print(f'Recovered: {x_r}')

# # %%
# # ==== Test class SymmetricModel2 ====
#     # Generate tensor product without constraints.
#     n_res, n_ext = 10, 10
#     x = np.random.normal(0, 1, n_res) + 1j * np.random.normal(0, 1, n_res)
#     tensor = x[:n_res, np.newaxis] * x[np.newaxis, :n_ext]
#     tensor = tensor[..., np.newaxis]

#     # Test absolute value.
#     test = linearModel.SymmetricModel2(tensor, np.array([1]))
#     test._fitAbs()
#     print(f'Gets x : {np.allclose(np.abs(x), test.abs_mode_shapes_.T)}')

#     # # Add noise.
#     # tensor_n = tensor \
#     #     + np.random.normal(0, 0.1, (n_res, n_ext, 1)) + 1j * np.random.normal(0, 0.1, (n_res, n_ext, 1))
#     # test = linearModel.SymmetricModel2(tensor_n)
#     # test._fitAbs()
#     # print(f'Original: {np.abs(x)}')
#     # print(f'Recovered: {test.abs_mode_shapes_.T}')

#     # Test angles.
#     test = linearModel.SymmetricModel2(tensor, np.array([1]))
#     print('Loss : ', test._loss(np.angle(x)))

#     r = scipy.optimize.minimize(
#         test._loss,
#         x0 = np.zeros(n_res),
#         method='Newton-CG',
#         jac=test._jac_loss)
#     x_r = r.x
#     print(r.success)
#     print(x / np.abs(x))
#     print(np.exp(1j * x_r))

# # %%
#     seed = None
#     n_res, n_ext, n_dof = 3, 3, 3
#     rng = np.random.default_rng(seed)
#     Z = rng.uniform(-1, -0.5, size=n_dof) + 1j * rng.uniform(1, 10, size=n_dof)
#     mode_shapes = linearModel.mode_shapes_generator(
#         rng.normal(size=(n_res, n_dof)),
#         D=1/Z.imag,
#         scale=1,
#         seed=seed
#     )
#     tensor = mode_shapes[:n_res, np.newaxis, :] * mode_shapes[np.newaxis, :n_ext, :]

#     test = linearModel.SymmetricModel2(tensor, Z)
#     test._fitAbs()
#     # print('Abs Test: ', np.allclose(np.abs(mode_shapes), test.abs_mode_shapes_))

#     # print('_compatibility test : ', 
#     #       np.allclose(test._compatibility(np.angle(mode_shapes).T), 0))
    
#     guess = np.zeros(n_res * n_dof)
#     for _ in range(1):
        
#         test._fitAngle(x0=guess.flatten())
#         print('Loss : ', test._loss(np.angle(test.mode_shapes_).T))
#         print('Compatibility test : ',
#               np.max(np.abs(test._compatibility(guess))))

#         guess = linearModel.mode_shapes_generator(
#             X=rng.normal(size=(n_res, n_dof)),
#             D=1/Z.imag,
#             scale=1,
#             seed=None
#         )
#         guess = np.angle(guess).T

        # print('mode shape : \n', mode_shapes)
        # print('retrieved mode shape : \n', test.mode_shapes_)
        # print('Loss : ', test._loss(np.angle(test.mode_shapes_).T))

        # print('Compatibility test : ',
        #       np.max(np.abs(test._compatibility(np.angle(test.mode_shapes_).T))))

# %%
# ===== Test SymmetricModel3 =====
    # Generate tensor product without constraints.
    n_res, n_ext, n_dof = 2, 2, 2
    seed = 123
    mask = np.ones((n_res, n_ext), dtype=bool)
    mask = np.triu(mask, k=1)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_res, n_dof))
    L = rng.normal(scale=0.1, size=(n_res, n_dof))
    L = np.triu(L, k=1)
    Z = rng.uniform(-1, -0.5, size=n_dof) + 1j * rng.uniform(5, 10, size=n_dof)
    mode_shapes = linearModel.mode_shapes_generator(X, L, D=1/Z.imag)

    # Target tensor.
    tensor = mode_shapes[:n_res, np.newaxis, :] * mode_shapes[np.newaxis, :n_ext, :]
    guess = linearModel.fastGuess(tensor.real)

    # x = rng.normal(size=(2*n_dof - 1) * n_res - n_res * (n_res - 1) // 2)
    def fun(x):
        # Length of x is (2*n_dof - 1) * n_res - n_res * (n_res - 1) // 2.
        X = x[:n_res * n_dof].reshape((n_res, n_dof))
        L = np.zeros_like(X)
        start = n_res * n_dof
        for k in range(n_res):
            L[k, k+1:] = x[start:start + n_dof - k - 1]
            start += n_dof - k - 1

        mode_shapes = linearModel.mode_shapes_generator(X, L, D=1/Z.imag)
        r = tensor - mode_shapes[:n_res, np.newaxis, :] * mode_shapes[np.newaxis, :n_ext, :]
        return np.linalg.norm(r.flatten())

    x1 = (X.flatten(), np.ma.masked_array(L, mask=~mask).compressed())
    x1 = np.concatenate(x1)
    # # print(fun(x1))
    guess = (guess.flatten(), np.zeros((n_dof - 1) * n_res - n_res * (n_res - 1) // 2))
    guess = np.concatenate(guess)

    # res = scipy.optimize.basinhopping(
    #     fun,
    #     x0=guess,
    # )
    res = scipy.optimize.minimize(
        fun,
        x0=guess,
        # method='BFGS'
    )
    print('Guess : ', guess)
    print('Expected : ', x1)
    print(res)

# %%