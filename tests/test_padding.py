import pytest
import torch

from kazane.upsample import _pad_to_block_2


@pytest.mark.parametrize('x_shape', [(1, 1, 16), (8, 4, 32)])
@pytest.mark.parametrize('block_size', [8, 16, 32, 64])
@pytest.mark.parametrize('padding', [4, 16, 64])
def test_pad_to_block(x_shape, block_size, padding):
    x = torch.randn(*x_shape)
    _pad_to_block_2(x, block_size, padding)
