'''
Indar
=====
'''

from . import core
from .core import *

__version__ = '1.0.1'

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
