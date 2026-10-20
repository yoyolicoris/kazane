# Kazane: simple sinc interpolation for 1D signal in PyTorch


[![build](https://github.com/yoyolicoris/kazane/actions/workflows/ci.yml/badge.svg)](https://github.com/yoyolicoris/kazane/actions/workflows/ci.yml)
[![Release](https://github.com/yoyolicoris/kazane/actions/workflows/release.yml/badge.svg)](https://github.com/yoyolicoris/kazane/actions/workflows/release.yml)
[![PyPI version](https://badge.fury.io/py/kazane.svg)](https://badge.fury.io/py/kazane)

Kazane utilize FFT based convolution to provide fast sinc interpolation for 1D signal when your sample rate only needs to change by an integer amounts; If you need to change by a fraction amounts, checkout [julius](https://github.com/adefossez/julius).

## Installation

```commandline
pip install kazane
```
or 
``` commandline
pip install git+https://github.com/yoyolicoris/kazane
```
for latest version.

## Usage

```python
import kazane
import torch

signal = torch.randn(8, 2, 44100)

# downsample by an amount of 3
decimater = kazane.Decimate(3)
resampled_signal = decimater(signal)

# upsample by an amount of 2
upsampler = kazane.Upsample(2)
resampled_signal = upsampler(signal)

# you can also control number of zeros, roll-off frequency of the sinc interpolation kernel
decimater = kazane.Decimate(3, num_zeros=24, roll_off=0.9)

# use other types of window function for the sinc kernel
upsampler = kazane.Upsample(2, window_func=torch.blackman_window)
```

## Benchmarks on CUDA
Using the benchmark scripts at [bench](./bench), you can see that FFT can gives some speed improvements when the sample rate changes with some common integer numbers.
```
[---------- Down sample ----------]
               |  julius  |  kazane
2 threads: ------------------------
      rate: 2  |   52.2   |   52.4 
      rate: 3  |   66.5   |   36.1 
      rate: 5  |   94.8   |   30.0 
      rate: 7  |  121.7   |   42.3 

Times are in milliseconds (ms).

[----------- Up sample -----------]
               |  julius  |  kazane
2 threads: ------------------------
      rate: 2  |   48.8   |   39.0 
      rate: 3  |   68.1   |   51.6 
      rate: 5  |  112.5   |   78.9 
      rate: 7  |  159.4   |  108.0 

Times are in milliseconds (ms).
```
