'''
Indar
=====

This package provides resources for the analysis of vibrations.

Available subpackages
---------------------
ema
    Tools for Experimental Modal Analysis (EMA).
dde
    Tools for Delay Differential Equations (DDEs).
'''

from . import core
from .core import *

__version__ = '1.0.2'

submodules = {
    'dde',
    'ema',
}

__all__ = list(
    submodules |
    set(core.__all__) |
    {'__version__'}
)

def __dir__():
    return __all__
