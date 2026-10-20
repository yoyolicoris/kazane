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


def _fft_size(length: int, stride: int) -> int:
    # FFT length s >= length with s % (2 * stride) == 0 and s // stride smooth
    return stride * _smooth_even((length + stride - 1) // stride)


def _kernel_spectrum(weight: Tensor, s: int) -> Tensor:
    # Conjugated spectrum of the kernel(s), shape [-1, s // 2 + 1]
    weight = weight.reshape(-1, weight.shape[-1])
    W = rfft(weight, n=s)
    # W.imag.mul_(-1)
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
        n_fft = s
        new_n_fft = n_fft // stride
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
    s = _fft_size(max(input.shape[-1], kernel_len), stride)
    W = _kernel_spectrum(weight, s)
    return _fft_conv1d_with_spectrum(input, W, s, kernel_len, stride)
