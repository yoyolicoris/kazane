from importlib.metadata import PackageNotFoundError, version

from .decimate import Decimate
from .sinc import sinc_kernel
from .upsample import Upsample

try:
    __version__ = version('kazane')
except PackageNotFoundError:  # pragma: no cover - source tree without installing
    __version__ = '0+unknown'

__all__ = ['Decimate', 'Upsample', 'sinc_kernel']
