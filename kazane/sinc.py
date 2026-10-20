import torch


def sinc_kernel(num_zeros, precision, roll_off=0.945, dtype=torch.float64):
    r"""Computes a windowless sinc interpolation kernel.

    The kernel has :math:`2 \times \text{num\_zeros} \times \text{precision} + 1`
    taps and is symmetric around its centre:

    .. math::
        \text{out}_t = r \, \text{sinc}\left(\frac{r t}{\text{precision}}\right),
        \quad t = -\text{num\_zeros} \times \text{precision}, \dots,
        \text{num\_zeros} \times \text{precision}

    where :math:`r` is :attr:`roll_off` and :math:`\text{sinc}` is the
    normalized sinc function (see :func:`torch.sinc`).

    Args:
        num_zeros (int): number of zero crossings of the sinc filter kept on
            each side of its centre.
        precision (int): number of filter coefficients per zero crossing.
        roll_off (float, optional): cut-off frequency as a fraction of the
            Nyquist frequency. Default: ``0.945``
        dtype (:class:`torch.dtype`, optional): data type of the kernel.
            Default: ``torch.float64``

    Returns:
        Tensor: the kernel, of shape :math:`(2 \times \text{num\_zeros} \times
        \text{precision} + 1)`.

    Examples::

        >>> import torch
        >>> import kazane
        >>> kazane.sinc_kernel(2, 2).shape
        torch.Size([9])
        >>> kazane.sinc_kernel(2, 2)
        tensor([-0.0539, -0.2051,  0.0547,  0.6342,  0.9450,  0.6342,  0.0547, -0.2051,
                -0.0539], dtype=torch.float64)

    """
    t = torch.arange(-num_zeros * precision, num_zeros *
                     precision + 1, dtype=dtype) / precision * roll_off
    return torch.sinc(t) * roll_off

