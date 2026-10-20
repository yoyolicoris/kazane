# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dev env is managed by pixi ([pyproject.toml](pyproject.toml), `[tool.pixi.*]`); the `kazane` package is installed editable. Environments: `default` (py313, CPU torch, test + dev tools), `py311`..`py314`, `bench` (needs julius), `cuda` (CUDA torch; use for GPU benchmarks).

```bash
pixi run test                                   # pytest with branch coverage; fails below 100%
pixi run pytest tests/test_upsample.py -k name  # single file / test
pixi run lint && pixi run format-check          # ruff (config in pyproject.toml)
pixi run build && pixi run smoke                # sdist+wheel, then install wheel and check version
pixi run -e bench bench-upsample                # also bench-decimate (designed for CUDA: use -e cuda)
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint/format-check, the test matrix over py311..py314, and build+smoke, all via pixi with `locked: true` (commit `pixi.lock` with `[tool.pixi]` changes). The version comes from git tags via hatch-vcs: release by pushing a `v*` tag; [release.yml](.github/workflows/release.yml) publishes to PyPI with trusted publishing. Checkouts need full history (`fetch-depth: 0`).

## Architecture

Small PyTorch library (`kazane/`) for sinc resampling of 1D signals by **integer** factors using FFT convolution; fractional rates are delegated to `julius`. Public API (`kazane/__init__.py`): `Decimate`, `Upsample`, `sinc_kernel`. All inputs are `[*, T]`; leading dims are flattened to a batch and restored.

- [sinc.py](kazane/sinc.py): windowless sinc kernel with `num_zeros` zero crossings and `q` taps per crossing. `Upsample`/`Decimate` multiply it by `window_func` (default Hann) in float64, then cast to float32 buffer.
- [fftconv.py](kazane/fftconv.py): `_fft_conv1d_with_spectrum` (core; `_custom_fft_conv1d` is a wrapper that computes the spectrum and a smooth FFT size on the fly, used by the short paths). It does rfft → multiply by conjugated kernel spectrum (cross-correlation) → irfft, and implements **stride in the frequency domain** by folding/aliasing spectrum segments (real/imag handled separately, then divided by stride) instead of subsampling afterward. Also holds the shared padding helpers `_manual_reflect_pad` and `_pad_to_block`. Edit with care: `Decimate` depends on this.
- [upsample.py](kazane/upsample.py): kernel is reshaped into `q` polyphase filters `[q, 1, L]` and flipped; output channels are interleaved via `transpose(1, 2)`.
- [decimate.py](kazane/decimate.py): single kernel, `stride=q`, kernel pre-divided by `q`.

### Block processing / padding (shared pattern in Upsample, Decimate)

`forward` has two paths, chosen by `shape[-1] < self.block_length`. `__init__` picks `fft_size` first (power of 2; `q *` power of 2 for Decimate) from the class constant `BLOCK_RATIO`, then `block_length = fft_size - 2*padding`. Power-of-2 sizes were fastest on CPU and CUDA; prime factors in the size are very slow. The long path uses the kernel spectrum cached in the non-persistent `kernel_spectrum` buffer (stored as real `[P, F, 2]` because `.to(dtype)` drops the imag part of complex buffers):
- **Short input**: pad once and run one FFT conv.
- **Long input**: pad, `unfold` into overlapping blocks (block + 2·padding, hop = block), run the FFT conv over all blocks as extra batch, then fold the results back and trim to the exact output length.

Reflect padding (`F.pad(mode='reflect')`) requires pad < input length, so `_manual_reflect_pad` handles pad ≥ length by repeatedly mirroring; `tests/test_padding.py` covers this.
