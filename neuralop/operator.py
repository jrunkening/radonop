import torch
from torch import nn
from neuralop.kernel import FourierKernel


class FourierNeuralOperator(nn.Module):
    def __init__(self, channels: tuple, modes: tuple, activate, dtype=torch.float32) -> None:
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.activate = activate

        self.lift = nn.Linear(channels[0], channels[1], dtype=dtype)
        self.integral_kernel = nn.Sequential(
            FourierKernel(channels[1], channels[2], modes, activate, dtype=dtype),
            FourierKernel(channels[2], channels[3], modes, activate, dtype=dtype),
            FourierKernel(channels[3], channels[4], modes, activate, dtype=dtype),
            FourierKernel(channels[4], channels[5], modes, nn.Identity(), dtype=dtype),
        )
        self.project = nn.Sequential(
            nn.Linear(channels[5], channels[6], dtype=dtype),
            activate,
            nn.Linear(channels[6], channels[7], dtype=dtype),
        )

    def forward(self, xs: torch.tensor):
        xs_lifted = self.lift(xs.permute(0, *range(2, xs.dim()), 1)).permute(0, xs.dim()-1, *range(1, xs.dim()-1))
        xs_mapped = self.integral_kernel(xs_lifted)
        xs_projected = self.project(xs_mapped.permute(0, *range(2, xs_mapped.dim()), 1)).permute(0, xs_mapped.dim()-1, *range(1, xs_mapped.dim()-1))

        return xs_projected
