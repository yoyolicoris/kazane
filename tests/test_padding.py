import pytest
import torch

from kazane.fftconv import _pad_to_block


@pytest.mark.parametrize('x_shape', [(1, 1, 16), (8, 4, 32)])
@pytest.mark.parametrize('block_size', [8, 16, 32, 64])
@pytest.mark.parametrize('padding', [4, 16, 64])
def test_pad_to_block(x_shape, block_size, padding):
    x = torch.randn(*x_shape)
    _pad_to_block(x, block_size, padding)
