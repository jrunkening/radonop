import torch
from torch import nn
from neuralop.conv import SpectralConv


class FourierKernel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: tuple, activate, dtype=torch.float32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activate = activate
        self.dtype = dtype

        self.linear = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.conv = SpectralConv(in_channels, out_channels, modes, dtype=dtype)

    def forward(self, xs: torch.tensor):
        xs_affined = self.linear(xs.permute(0, *range(2, xs.dim()), 1)).permute(0, xs.dim()-1, *range(1, xs.dim()-1))
        xs_conved = self.conv(xs)

        return self.activate(xs_affined + xs_conved)
