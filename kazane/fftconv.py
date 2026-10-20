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
    m = 2
    while m < n:
        m = m * 2
    return m


def _kernel_spectrum(weight: Tensor, s: int) -> Tensor:
    # Conjugated spectrum of the kernel(s), shape [-1, s // 2 + 1]
    weight = weight.reshape(-1, weight.shape[-1])
    W = rfft(weight, n=s)
    torch.view_as_real(W)[..., 1].mul_(-1)
    return W


def _fft_conv1d_with_spectrum(
    input: Tensor, W: Tensor, s: int, kernel_len: int, stride: int = 1
) -> Tensor:
    # s % (2 * stride) == 0 and input.shape[-1] <= s are required
    output_size = (input.shape[-1] - (kernel_len - 1) - 1) // stride + 1

    X = rfft(input, n=s)
    Y = X * W

    # handle stride
    if stride > 1:
        new_n_fft = s // stride
        step_size = new_n_fft // 2
        strided_Y_size = step_size + 1

        tmpY = torch.view_as_real(Y)
        Y_real, Y_imag = tmpY[..., 0], tmpY[..., 1]

        unfolded_Y_real = Y_real.unfold(-1, strided_Y_size, step_size)
        unfolded_Y_imag = Y_imag[..., 1:].unfold(-1, strided_Y_size - 2, step_size)
        Y_pos_real, Y_pos_imag = (
            unfolded_Y_real[..., ::2, :].sum(-2),
            unfolded_Y_imag[..., ::2, :].sum(-2),
        )
        Y_neg_real, Y_neg_imag = (
            unfolded_Y_real[..., 1::2, :].sum(-2).flip(-1),
            unfolded_Y_imag[..., 1::2, :].sum(-2).flip(-1),
        )

        Y_real = Y_pos_real.add_(Y_neg_real)
        Y_imag = Y_pos_imag.add_(Y_neg_imag, alpha=-1)
        Y_imag = F.pad(Y_imag, [1, 1])

        Y = torch.view_as_complex(torch.stack((Y_real, Y_imag), -1)).div_(stride)

    output = irfft(Y, n=s // stride)

    # Remove extra padded values
    output = output[..., :output_size].contiguous()
    return output


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
