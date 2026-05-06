import pytest

import numpy as np
import scipy

from bcam.indar.ema import lti

def test_lti_kernel_fit():
    rng = np.random.default_rng()
    N = 100
    n_rep = 8

    # Create random input data.
    X = rng.normal(0, 1, (n_rep, N))

    # Create a random kernel.
    true_kernel = rng.normal(0, 1, N)

    # Generate response data by convolving input with the kernel.
    y = scipy.signal.fftconvolve(
        X, true_kernel[np.newaxis, :], mode='full', axes=1)[:, :N]

    # Fit LTI model to recover the kernel.
    model = lti.LTIKernel(show=False, atol=1e-10, btol=1e-10)
    model.fit(X, y)

    # Check that the estimated kernel is close to the true kernel (up to a sign).
    assert np.allclose(model.kernel_, true_kernel, rtol=1e-5, atol=0.)

