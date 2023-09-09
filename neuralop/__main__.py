import torch
from torch import nn

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes: tuple, dtype=torch.float32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = torch.tensor(modes)
        self.dtype = dtype

        self.weights, self.weights_conj = self.build_weights()

    def glorot_uniform_complex(self, shape, scale):
        return torch.complex(
            nn.init.xavier_uniform_(torch.empty(shape, dtype=self.dtype)),
            nn.init.xavier_uniform_(torch.empty(shape, dtype=self.dtype))
        ) / scale

    def build_weights(self):
        weights_shape = (self.out_channels, self.in_channels, torch.prod(self.modes))
        scale = torch.complex(torch.tensor(self.in_channels * self.out_channels, dtype=self.dtype), torch.tensor(0, dtype=self.dtype))

        weights = nn.Parameter(self.glorot_uniform_complex(weights_shape, scale))
        weights_conj = nn.Parameter(self.glorot_uniform_complex(weights_shape, scale))

        return weights, weights_conj


    def forward(self, xs: torch.tensor):
        xs_spectrum = torch.fft.rfftn(xs, dim=[*range(2, xs.dim())])
        xs_spectrum_applied = self.apply_spectral_pattern(xs_spectrum)
        xs_applied = torch.fft.irfftn(xs_spectrum_applied, s=xs.shape[2:], dim=[*range(2, xs_spectrum_applied.dim())])

        return xs_applied

    def apply_spectral_pattern(self, xs_spectrum):
        xs_spectrum_applied = torch.zeros(
            xs_spectrum.size(0), self.out_channels, *xs_spectrum.shape[2:],
            dtype=xs_spectrum.dtype, device=xs_spectrum.device
        )

        slices, slices_conj = [slice(None, None), slice(None, None), slice(None, self.modes[-1])], [slice(None, None), slice(None, None), slice(None, self.modes[-1])]
        for mode in self.modes[:-1]:
            slices.insert(2, slice(None, mode))
            slices_conj.insert(2, slice(-mode, None))
        xs_spectrum_sliced, xs_spectrum_sliced_conj = xs_spectrum[*slices], xs_spectrum[*slices_conj]

        shape = xs_spectrum_sliced.shape

        xs_spectrum_sliced_reshaped = xs_spectrum_sliced.reshape(*shape[:2], -1)
        xs_spectrum_sliced_conj_reshaped = xs_spectrum_sliced_conj.reshape(*shape[:2], -1)
        xs_spectrum_sliced_reshaped_applied = torch.einsum("bim,oim->bom", xs_spectrum_sliced_reshaped, self.weights)
        xs_spectrum_sliced_conj_reshaped_applied = torch.einsum("bim,oim->bom", xs_spectrum_sliced_conj_reshaped, self.weights_conj)
        xs_spectrum_sliced_applied = xs_spectrum_sliced_reshaped_applied.reshape(shape[0], self.out_channels, *shape[2:])
        xs_spectrum_sliced_conj_applied = xs_spectrum_sliced_conj_reshaped_applied.reshape(shape[0], self.out_channels, *shape[2:])

        xs_spectrum_applied[slices_conj] = xs_spectrum_sliced_conj_applied
        xs_spectrum_applied[slices] = xs_spectrum_sliced_applied

        return xs_spectrum_applied


if __name__ == "__main__":
    print(SpectralConv(2, 5, (2,2,2,3,3))(torch.ones(2,2,3,4,5,6,7, dtype=torch.float32)).shape)
