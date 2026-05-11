'''
A sub-package for Experimental Modal Analysis (EMA).

Essential steps in the analysis of EMA data are:

1. Estimate of the Impulse Response Function (IRF).
2. Find the order and the poles of the system.
3. Estimate the mode shapes, either real or complex.
'''

from .lti import *
from .pole_fitting import *
from .mechanical import *
from .systems import *
