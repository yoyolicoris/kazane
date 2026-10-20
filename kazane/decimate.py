from collections.abc import Callable

import torch
from torch import nn

from .fftconv import (
    _custom_fft_conv1d,
    _fft_conv1d_with_spectrum,
    _kernel_spectrum,
    _next_pow2,
)
from .sinc import sinc_kernel
from .upsample import _manual_reflect_pad, _pad_to_block_2


class Decimate(nn.Module):
    r"""Downsamples a signal by an integer factor using a windowed sinc filter.

    The signal is reflect-padded, low-pass filtered with a windowed sinc
    kernel (see :func:`~kazane.sinc_kernel`) and keeps every :attr:`q`-th
    sample. The filtering is computed with FFT convolution on overlapping
    blocks.

    Args:
        q (int, optional): downsample factor, must be greater than 1.
            Default: ``2``
        num_zeros (int, optional): number of zero crossings of the sinc filter
            kept on each side of its centre. Default: ``16``
        window_func (Callable, optional): function that takes a window length
            and returns a window tensor, applied to the sinc kernel.
            Default: :func:`torch.hann_window`
        **kwargs: arguments passed through to :func:`~kazane.sinc_kernel`,
            e.g. ``roll_off``.

    Shape:
        - Input: :math:`(*, T)` where :math:`*` means any number of
          dimensions including none.
        - Output: :math:`(*, T / q)`, for :math:`T` divisible by :math:`q`.

    Examples::

        >>> import torch
        >>> import kazane
        >>> decimater = kazane.Decimate(4)
        >>> x = torch.randn(1000)
        >>> decimater(x).shape
        torch.Size([250])
        >>> decimater = kazane.Decimate(3, num_zeros=24, roll_off=0.9)
        >>> decimater(torch.randn(8, 2, 44100)).shape
        torch.Size([8, 2, 14700])

    """
    BLOCK_RATIO: int = 16

    def __init__(self,
                 q: int = 2,
                 num_zeros: int = 16,
                 window_func: Callable[[int],
                                       torch.Tensor] = torch.hann_window,
                 **kwargs):
        assert q > 1, "q should be greater than 1"
        super().__init__()
        kernel = sinc_kernel(num_zeros, q, **kwargs)
        N = kernel.numel()
        kernel *= window_func(N, dtype=torch.float64)
        kernel /= q

        self.stride = q
        self.padding = N // 2
        self.register_buffer('kernel', kernel.view(1, 1, -1).float())

        # FFT size is chosen first (q times a power of 2, fastest on both CPU
        # and CUDA), block length follows
        per_phase = (N + q - 1) // q
        self.fft_size = q * _next_pow2(per_phase * self.BLOCK_RATIO)
        self.block_length = self.fft_size - 2 * self.padding
        # stored as real [P, F, 2] so that .to(dtype) keeps the imaginary part
        self.register_buffer('kernel_spectrum',
                             torch.view_as_real(
                                 _kernel_spectrum(self.kernel, self.fft_size)),
                             persistent=False)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(-1, 1, shape[-1])

        block_length = self.block_length
        out_size = shape[-1] // self.stride
        if shape[-1] < block_length:
            x = _manual_reflect_pad(x, (self.padding, self.padding))
            y = _custom_fft_conv1d(x, self.kernel, stride=self.stride)
        else:
            x = _pad_to_block_2(x, block_length, self.padding)
            num_blocks = x.shape[-2]
            x = x.reshape(-1, 1, x.shape[-1])
            W = torch.view_as_complex(self.kernel_spectrum)
            y = _fft_conv1d_with_spectrum(x, W,
                                          self.fft_size,
                                          self.kernel.shape[-1],
                                          self.stride)
            y = y.view(-1, num_blocks * y.shape[-1])[..., :out_size]

        return y.view(shape[:-1] + (-1,))

