import pytest
import torch

from kazane import Upsample

from .test_decimate import make_sweep, make_tone

device = 'cpu'
dtype = torch.float64


@pytest.mark.parametrize('q', [2, 3, 5, 7])
@pytest.mark.parametrize('zeros,rms', [(64, 1e-4), (32, 1e-4), (16, 4e-4)])
def test_quality_sine(q, zeros, rms):
    FREQ = 512.0
    DURATION = 2
    sr = 22050
    sr_new = sr * q
    up = Upsample(q, zeros, roll_off=0.945).to(dtype)

    x = make_tone(FREQ, sr, DURATION)
    y = make_tone(FREQ, sr_new, DURATION)
    y_pred = up(x)

    idx = slice(sr_new // 2, -sr_new // 2)

    err = torch.mean(torch.abs(y[idx] - y_pred[idx])).item()
    assert err <= rms, f'{err:g} > {rms:g}'


@pytest.mark.parametrize('q', [2, 3])
@pytest.mark.parametrize('zeros,rms', [(64, 5e-2), (32, 5e-2), (16, 5e-2)])
def test_quality_sweep(q, zeros, rms):
    FREQ = 4096
    DURATION = 5
    sr = 22050
    sr_new = sr * q
    up = Upsample(q, zeros, roll_off=0.945).to(dtype)

    x = make_sweep(FREQ, sr, DURATION)
    y = make_sweep(FREQ, sr_new, DURATION)
    y_pred = up(x)

    idx = slice(sr_new // 2, -sr_new // 2)

    err = torch.mean(torch.abs(y[idx] - y_pred[idx])).item()
    assert err <= rms, f'{err:g} > {rms:g}'


def test_short_input():
    up = Upsample(2, 32)
    x = torch.randn(8)
    up(x)
