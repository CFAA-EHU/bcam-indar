'''
A sub-package for Experimental Modal Analysis (EMA).

We assume that the system is modeled by the equation

.. math::
    M \\ddot{x} + C \\dot{x} + K x = f

where :math:`x` is the displacement in general coordinates, 
:math:`f` is an external force, and
:math:`M`, :math:`C` and :math:`K` are the mass, damping and stiffness matrices, respectively.
We also assume that these matrices are symmetric and positive definite.

Essential steps for processing EMA data are:

1. Estimate the Response Function.
2. Find the order and the poles of the system.
3. Estimate the mode shapes, either real or complex.

Step 1 can be applied to any input-output model that is
Linear Time-Invariant (LTI) and causal.
The Impulse Response Function (IRF) is a matrix-valued time series, but
for mechanical systems it acquires a special structure, in particular,
it is an exponential sum.

In step 2 the Impulse Response Function (IRF) is approximated by an exponential sum, so
the package offers methods to estimate the exponents and the number of terms in the sum.
However, not every exponential sum is a valid IRF of a mechanical system, and
the amplitudes must have a special form.

The amplitudes are approximated by the mode shapes in step 3, and
the package allows the user to choose between real and complex mode shapes.
With this information, it is possible to estimate the mass, damping and stiffness matrices of the system.
'''

from .lti import *
from .pole_fitting import *
from .mechanical import *
from .systems import *
