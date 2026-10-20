from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .fftconv import (
    _custom_fft_conv1d,
    _fft_conv1d_with_spectrum,
    _kernel_spectrum,
    _manual_reflect_pad,
    _next_pow2,
    _pad_to_block,
)
from .sinc import sinc_kernel


class Upsample(nn.Module):
    r"""Upsamples a signal by an integer factor using a windowed sinc filter.

    The signal is reflect-padded, zero-stuffed by :attr:`q` and low-pass
    filtered with a windowed sinc kernel (see :func:`~kazane.sinc_kernel`).
    The filtering is computed with FFT convolution on overlapping blocks.

    Args:
        q (int, optional): upsample factor, must be greater than 1.
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
        - Output: :math:`(*, T \times q)`.

    Examples::

        >>> import torch
        >>> import kazane
        >>> upsampler = kazane.Upsample(3)
        >>> x = torch.randn(8, 2, 44100)
        >>> upsampler(x).shape
        torch.Size([8, 2, 132300])
        >>> upsampler = kazane.Upsample(2, window_func=torch.blackman_window)
        >>> upsampler(x).shape
        torch.Size([8, 2, 88200])

    """

    BLOCK_RATIO: int = 16

    def __init__(
        self,
        q: int = 2,
        num_zeros: int = 16,
        window_func: Callable[[int], torch.Tensor] = torch.hann_window,
        **kwargs,
    ):
        assert q > 1, 'q should be greater than 1'
        super().__init__()
        kernel = sinc_kernel(num_zeros, q, **kwargs)
        N = kernel.numel()
        kernel *= window_func(N, dtype=torch.float64)
        kernel = F.pad(kernel.float(), [q - 1, 0]).view(-1, q).t().reshape(q, 1, -1).flip(0)
        self.stride = q
        self.padding = N // 2 // q
        self.register_buffer('kernel', kernel)

        # FFT size is chosen first (power of 2, fastest on both CPU and CUDA),
        # block length follows
        self.fft_size = _next_pow2(kernel.shape[-1] * self.BLOCK_RATIO + 2 * self.padding)
        self.block_length = self.fft_size - 2 * self.padding
        # stored as real [P, F, 2] so that .to(dtype) keeps the imaginary part
        self.register_buffer(
            'kernel_spectrum',
            torch.view_as_real(_kernel_spectrum(kernel, self.fft_size)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(-1, 1, shape[-1])

        if shape[-1] < self.block_length:
            x = _manual_reflect_pad(x, (self.padding, self.padding))
            y = _custom_fft_conv1d(x, self.kernel)
        else:
            x = _pad_to_block(x, self.block_length, self.padding)
            num_blocks = x.shape[-2]
            x = x.reshape(-1, 1, x.shape[-1])
            W = torch.view_as_complex(self.kernel_spectrum)
            y = _fft_conv1d_with_spectrum(x, W, self.fft_size, self.kernel.shape[-1])
            y = (
                y.view(-1, num_blocks, self.stride, self.block_length)
                .transpose(1, 2)
                .reshape(-1, self.stride, num_blocks * self.block_length)[..., : shape[-1]]
            )

        y = y.transpose(1, 2).contiguous()
        return y.view(shape[:-1] + (-1,))
