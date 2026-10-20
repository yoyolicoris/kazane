import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fft import irfft, rfft


def _smooth_even(n: int) -> int:
    # Smallest even 2/3/5-smooth integer >= n
    m = max(n, 2)
    m += m % 2
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 2


def _next_pow2(n: int) -> int:
    return 1 << (max(n, 2) - 1).bit_length()


def _kernel_spectrum(weight: Tensor, s: int) -> Tensor:
    # Conjugated spectrum of the kernel(s), shape [-1, s // 2 + 1]. It also carries
    # the 1/s normalisation, so the inverse transform below runs unnormalised.
    weight = weight.reshape(-1, weight.shape[-1])
    return rfft(weight, n=s).conj() / s


def _fft_conv1d_with_spectrum(
    input: Tensor, W: Tensor, s: int, kernel_len: int, stride: int = 1
) -> Tensor:
    # s % (2 * stride) == 0 and input.shape[-1] <= s are required.
    # The result is a (possibly non-contiguous) slice of the inverse transform.
    output_size = (input.shape[-1] - (kernel_len - 1) - 1) // stride + 1

    Y = rfft(input, n=s) * W
    if stride > 1:
        # Decimating by `stride` in time sums `stride` shifted copies of the spectrum,
        # every other copy mirrored. The unfolded windows overlap by one bin, which is
        # exactly the bin shared by two neighbouring copies.
        step = s // stride // 2
        copies = Y.unfold(-1, step + 1, step)
        Y = copies[..., ::2, :].sum(-2) + copies[..., 1::2, :].sum(-2).flip(-1).conj()
    return irfft(Y, n=s // stride, norm='forward')[..., :output_size]


def _custom_fft_conv1d(input: Tensor, weight: Tensor, stride: int = 1) -> Tensor:
    kernel_len = weight.shape[-1]
    # FFT length with s % (2 * stride) == 0 and s // stride smooth
    s = stride * _smooth_even((max(input.shape[-1], kernel_len) + stride - 1) // stride)
    W = _kernel_spectrum(weight, s)
    return _fft_conv1d_with_spectrum(input, W, s, kernel_len, stride)


def _manual_reflect_pad(x: torch.Tensor, padding: tuple[int, int]):
    flipped_x = x.flip(-1)
    tensors = [x]
    left_padding, right_padding = padding
    step = False
    while left_padding > 0:
        if step:
            tensors.insert(0, x[..., -left_padding - 1 : -1])
        else:
            tensors.insert(0, flipped_x[..., -left_padding - 1 : -1])
        left_padding -= x.shape[-1] - 1
        step = not step

    step = False
    while right_padding > 0:
        if step:
            tensors.append(x[..., 1 : right_padding + 1])
        else:
            tensors.append(flipped_x[..., 1 : right_padding + 1])
        right_padding -= x.shape[-1] - 1
        step = not step
    return torch.cat(tensors, -1)


def _pad_to_block(x: torch.Tensor, block_size: int, padding: int):
    offset = x.shape[-1] % block_size
    if offset:
        offset = block_size - offset
    paddings = (padding, offset + padding)
    if max(paddings) >= x.shape[-1]:
        x = _manual_reflect_pad(x, paddings)
    else:
        x = F.pad(x, paddings, mode='reflect')
    return x.unfold(-1, block_size + 2 * padding, block_size)
