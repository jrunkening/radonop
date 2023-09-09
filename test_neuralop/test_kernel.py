import torch
from neuralop.kernel import FourierKernel


def test_fourier_kernel():
    batch_size = 10
    in_channels  = 3
    out_channels = 5
    signals_shape = (100, 101, 102)
    modes_shape = (50, 51, 51) # mode size should <= scil(signal_size / 2)

    signals = torch.rand(batch_size, in_channels, *signals_shape)

    kernel = FourierKernel(in_channels, out_channels, modes_shape, torch.nn.ReLU())

    assert kernel(signals).shape == (batch_size, out_channels, *signals_shape)


def test_fourier_kernel_gpu():
    batch_size = 10
    in_channels  = 3
    out_channels = 5
    signals_shape = (100, 101, 102)
    modes_shape = (50, 51, 51) # mode size should <= scil(signal_size / 2)

    signals = torch.rand(batch_size, in_channels, *signals_shape).to("cuda")

    kernel = FourierKernel(in_channels, out_channels, modes_shape, torch.nn.ReLU()).to("cuda")

    assert kernel(signals).shape == (batch_size, out_channels, *signals_shape)
