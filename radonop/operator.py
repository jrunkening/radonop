from pathlib import Path
import torch
import torch.nn as nn
from neuralop.operator import FourierNeuralOperator


PARAMS_PATH = Path(__file__).parent.joinpath("parameters")


class InverseRadonOperator(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activate, dtype=torch.float32) -> None:
        super().__init__()

        self.op = FourierNeuralOperator(
            channels=(in_channels, 64, 64, 64, 64, 64, 128, out_channels),
            modes=modes,
            activate=activate,
            dtype=dtype
        )

    def forward(self, xs):
        return self.op(xs)
