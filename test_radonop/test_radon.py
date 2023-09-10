import numpy as np
import qutip as q
import torch
from radonop.radon import radon, iradon


DIM = 185


class CatState:
    def __init__(self, dim=DIM, alpha_lim=2+2j) -> None:
        super().__init__()
        self.dim = dim
        self.alpha_lim = alpha_lim

    def gen(self, alpha):
        return (q.displace(self.dim, alpha) + q.displace(self.dim, -alpha)) * q.fock(self.dim, 0)

    def rand(self):
        alpha = self.alpha_lim.real * np.random.rand() + self.alpha_lim.imag * np.random.rand() * 1j

        return self.gen(alpha)


def gen_data(state_generator, xs):
    w = torch.tensor(q.wigner(state_generator.rand(), xs, xs))[None, None, :, :]
    pdf = radon(w)
    pdf /= pdf.sum(dim=-2)

    return pdf.type(torch.FloatTensor), w.type(torch.FloatTensor)


def test_iradon():
    pdf, w = gen_data(CatState(alpha_lim=2+2j), xs=np.linspace(-6, 6, 128))

    iradon(pdf, device="cuda")
