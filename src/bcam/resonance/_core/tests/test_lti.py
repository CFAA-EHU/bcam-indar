import pytest

import numpy as np
import scipy

from bcam.resonance._core import lti

def test_lti_kernel_fit():
    rng = np.random.default_rng()
    N = 200
    K = 5

    # Create random input data.
    X = rng.normal(0, 1, (N, K))

    # Create a random kernel.
    true_kernel = rng.normal(0, 1, N)

    # Generate response data by convolving input with the kernel.
    y = scipy.signal.fftconvolve(X, true_kernel[:, np.newaxis], mode='full', axes=0)[:N]

    # Fit LTI model to recover the kernel.
    model = lti.LTIKernel(rtol=1e-10)
    model.fit(X, y)
    estimated_kernel = model.kernel_

    # Check that the estimated kernel is close to the true kernel (up to a sign).
    assert np.allclose(estimated_kernel, true_kernel, rtol=1e-5, atol=0.)
    assert np.allclose(model.score(X, y), 1.0, rtol=1e-5, atol=0.)
