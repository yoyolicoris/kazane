"""Regression sweeps against direct (non-FFT) reference implementations."""

import pytest
import torch
import torch.nn.functional as F

from kazane import Decimate, Upsample
from kazane.fftconv import _manual_reflect_pad

TOL = 1e-4
QS = [2, 3, 4, 5, 6, 8]


def lengths(block_length):
    """Lengths on both sides of the short/long threshold."""
    ls = {42, 46, 50, 300}
    for base in (block_length, 2 * block_length):
        ls.update(range(base - 1, base + 2))
    return sorted(ls)


def ref_decimate(m, x):
    x = _manual_reflect_pad(x.reshape(-1, 1, x.shape[-1]), (m.padding, m.padding))
    return F.conv1d(x, m.kernel, stride=m.stride)[:, 0]


def ref_upsample(m, x):
    # zero-stuffing by q followed by a true convolution with the (time
    # reversed) windowed sinc kernel; y[i * q + c] = full[N - 1 + i * q + c]
    q, pad = m.stride, m.padding
    # polyphase kernel [q, 1, Lp] (flipped over phases) back to a full kernel
    k = m.kernel.flip(0)[:, 0].t().reshape(-1)[q - 1 :]
    x = _manual_reflect_pad(x.reshape(-1, 1, x.shape[-1]), (pad, pad))
    full = F.conv_transpose1d(x, k.flip(-1).view(1, 1, -1), stride=q)[:, 0]
    start = k.numel() - 1
    return full[..., start : start + (x.shape[-1] - 2 * pad) * q]


def make_inputs(T):
    g = torch.Generator().manual_seed(T)
    yield torch.randn(3, T, generator=g)
    # non-contiguous
    yield torch.randn(T, 2, 3, generator=g).permute(1, 2, 0)


@pytest.mark.parametrize('q', QS)
def test_decimate_matches_reference(q):
    m = Decimate(q)
    for T in lengths(m.block_length):
        for x in make_inputs(T):
            y = m(x)
            ref = ref_decimate(m, x)
            assert y.shape[:-1] == x.shape[:-1]
            assert y.shape[-1] >= T // q
            y = y.reshape(-1, y.shape[-1])
            n = y.shape[-1]
            err = (y - ref[..., :n]).abs().max().item()
            assert err < TOL, f'q={q} T={T} shape={tuple(x.shape)}: {err:g}'


@pytest.mark.parametrize('q', QS)
def test_upsample_matches_reference(q):
    m = Upsample(q)
    for T in lengths(m.block_length):
        for x in make_inputs(T):
            y = m(x)
            ref = ref_upsample(m, x)
            assert y.shape == x.shape[:-1] + (T * q,)
            err = (y.reshape(-1, T * q) - ref).abs().max().item()
            assert err < TOL, f'q={q} T={T} shape={tuple(x.shape)}: {err:g}'


@pytest.mark.parametrize('cls,ref', [(Decimate, ref_decimate), (Upsample, ref_upsample)])
@pytest.mark.parametrize('q', [2, 3, 4])
def test_double(cls, ref, q):
    m = cls(q).double()
    assert m.kernel_spectrum.dtype == torch.float64
    for T in (50, m.block_length + 7, 3 * m.block_length + 1):
        x = torch.randn(2, T, dtype=torch.float64)
        y = m(x)
        assert y.dtype == torch.float64
        r = ref(m, x)
        y = y.reshape(-1, y.shape[-1])
        err = (y - r[..., : y.shape[-1]]).abs().max().item()
        assert err < 1e-5, f'q={q} T={T}: {err:g}'


@pytest.mark.parametrize('cls', [Decimate, Upsample])
def test_load_state_dict_refreshes_spectrum(cls):
    src, dst = cls(2, roll_off=0.8), cls(2)
    assert list(src.state_dict().keys()) == ['kernel']
    dst.load_state_dict(src.state_dict())
    # one input on each side of the short/long threshold
    for T in (50, 2 * src.block_length):
        x = torch.randn(2, T)
        torch.testing.assert_close(dst(x), src(x))
