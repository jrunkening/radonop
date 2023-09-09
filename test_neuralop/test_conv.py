import torch
from neuralop.conv import SpectralConv


def test_spectral_conv():
    batch_size = 10
    in_channels  = 3
    out_channels = 5
    signals_shape = (10, 11, 12, 13, 14, 15, 16)
    modes_shape = (5, 6, 6, 7, 7, 8, 8) # mode size should <= scil(signal_size / 2)

    signals = torch.rand(batch_size, in_channels, *signals_shape)

    conv = SpectralConv(in_channels, out_channels, modes_shape)

    assert conv(signals).shape == (batch_size, out_channels, *signals_shape)


def test_spectral_conv_gpu():
    batch_size = 10
    in_channels  = 3
    out_channels = 5
    signals_shape = (100, 101, 102)
    modes_shape = (50, 51, 51) # mode size should <= scil(signal_size / 2)

    signals = torch.rand(batch_size, in_channels, *signals_shape).to("cuda")

    conv = SpectralConv(in_channels, out_channels, modes_shape).to("cuda")

    assert conv(signals).shape == (batch_size, out_channels, *signals_shape)
