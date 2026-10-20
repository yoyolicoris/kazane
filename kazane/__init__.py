from importlib.metadata import version

from .decimate import Decimate
from .sinc import sinc_kernel
from .upsample import Upsample

__version__ = version('kazane')

__all__ = ['Decimate', 'Upsample', 'sinc_kernel']
