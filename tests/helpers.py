from math import log2, pi

import torch

dtype = torch.float64


def make_tone(freq, sr, duration):
    return torch.sin(2 * pi * freq / sr * torch.arange(int(sr * duration), dtype=dtype))


def make_sweep(freq, sr, duration):
    return torch.sin(
        torch.cumsum(
            2
            * pi
            * torch.logspace(
                log2(2.0 / sr),
                log2(float(freq) / sr),
                steps=int(duration * sr),
                base=2.0,
                dtype=dtype,
            ),
            dim=-1,
        )
    )
