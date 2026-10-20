from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .fftconv import (
    _custom_fft_conv1d,
    _fft_conv1d_with_spectrum,
    _kernel_spectrum,
    _next_pow2,
)
from .sinc import sinc_kernel


def _manual_reflect_pad(x: torch.Tensor, padding: tuple[int, int]):
    flipped_x = x.flip(-1)
    tensors = [x]
    left_padding, right_padding = padding
    step = False
    while left_padding > 0:
        if step:
            tensors.insert(0, x[..., -left_padding-1:-1])
        else:
            tensors.insert(0, flipped_x[..., -left_padding-1:-1])
        left_padding -= x.shape[-1] - 1
        step = not step

    step = False
    while right_padding > 0:
        if step:
            tensors.append(x[..., 1:right_padding + 1])
        else:
            tensors.append(flipped_x[..., 1:right_padding + 1])
        right_padding -= x.shape[-1] - 1
        step = not step
    x = torch.cat(tensors, -1)
    return x


def _pad_to_block_2(x: torch.Tensor, block_size: int, padding: int):
    offset = x.shape[-1] % block_size
    if offset:
        offset = block_size - offset
    paddings = (padding, offset + padding)
    if max(paddings) >= x.shape[-1]:
        x = _manual_reflect_pad(x, paddings)
    else:
        x = F.pad(x, paddings, mode='reflect')
    return x.unfold(-1, block_size + 2 * padding, block_size)


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
        kernel = F.pad(kernel.float(), [q - 1, 0]).view(-1,
                                                        q).t().reshape(q, 1, -1).flip(0)
        self.stride = q
        self.padding = N // 2 // q
        self.register_buffer('kernel', kernel)

        # FFT size is chosen first (power of 2, fastest on both CPU and CUDA),
        # block length follows
        kernel_len = kernel.shape[-1]
        self.fft_size = _next_pow2(
            kernel_len * self.BLOCK_RATIO + 2 * self.padding)
        self.block_length = self.fft_size - 2 * self.padding
        # stored as real [P, F, 2] so that .to(dtype) keeps the imaginary part
        self.register_buffer('kernel_spectrum',
                             torch.view_as_real(
                                 _kernel_spectrum(kernel, self.fft_size)),
                             persistent=False)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(-1, 1, shape[-1])

        block_length = self.block_length
        if shape[-1] < block_length:
            x = _manual_reflect_pad(x, (self.padding, self.padding))
            y = _custom_fft_conv1d(x, self.kernel)
        else:
            q = self.kernel.shape[0]
            x = _pad_to_block_2(x, block_length, self.padding)
            num_blocks = x.shape[-2]
            x = x.reshape(-1, 1, x.shape[-1])
            W = torch.view_as_complex(self.kernel_spectrum)
            y = _fft_conv1d_with_spectrum(x, W,
                                          self.fft_size,
                                          self.kernel.shape[-1])
            y = y.view(-1, num_blocks, q, block_length).transpose(1,
                                                                  2).reshape(-1, q, num_blocks * block_length)[..., :shape[-1]]

        y = y.transpose(1, 2).contiguous()
        return y.view(shape[:-1] + (-1,))
