import torch
from neuralop.operator import FourierNeuralOperator


def test_fourier_kernel():
    batch_size = 10
    in_channels  = 3
    out_channels = 1
    signals_shape = (20, 21, 22)
    modes_shape = (10, 11, 11) # mode size should <= scil(signal_size / 2)

    signals = torch.rand(batch_size, in_channels, *signals_shape)

    op = FourierNeuralOperator(
        channels=(in_channels, 64, 64, 64, 64, 64, 128, out_channels),
        modes=modes_shape,
        activate=torch.nn.ReLU(),
    )

    assert op(signals).shape == (batch_size, out_channels, *signals_shape)


def test_fourier_kernel_gpu():
    batch_size = 10
    in_channels  = 3
    out_channels = 5
    signals_shape = (20, 21, 22)
    modes_shape = (10, 11, 11) # mode size should <= scil(signal_size / 2)

    signals = torch.rand(batch_size, in_channels, *signals_shape).to("cuda")

    op = FourierNeuralOperator(
        channels=(in_channels, 64, 64, 64, 64, 64, 128, out_channels),
        modes=modes_shape,
        activate=torch.nn.ReLU(),
    ).to("cuda")

    assert op(signals).shape == (batch_size, out_channels, *signals_shape)
