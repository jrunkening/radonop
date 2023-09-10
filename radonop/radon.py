import torch
from torchvision.transforms.functional import rotate
from neuralop.loss import H1Loss
from radonop.operator import InverseRadonOperator, PARAMS_PATH


def radon(image: torch.tensor):
    assert image.size(2) == image.size(3)

    image_rotated = torch.zeros_like(image)

    n_theta = image_rotated.size(3)
    for step in range(n_theta):
        rotation = rotate(image, step*360/n_theta)
        image_rotated[:, :, :, step] = torch.sum(rotation, dim=2)

    return image_rotated


def iradon(image_transformed: torch.tensor, tolerance=1e-1, learning_rate=1e-4, weight_decay=1e-4, device="cpu"):
    image_channels = image_transformed.size(1)
    modes = torch.ceil(torch.tensor(image_transformed.shape[2:]) / 2).type(torch.IntTensor).tolist()
    params_file = PARAMS_PATH.joinpath(f"iradon_{modes}.pth")

    image_transformed = torch.cat((image_transformed, gen_grid(image_transformed)), dim=1)

    operator = InverseRadonOperator(
        in_channels=image_transformed.size(1),
        out_channels=image_channels,
        modes=modes,
        activate=torch.nn.GELU()
    )
    if params_file.exists():
        operator.load_state_dict(torch.load(params_file))
    operator = operator.to(device)
    operator.train()
    optimizer = torch.optim.Adagrad(operator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    while (loss := train(image_transformed, image_channels, operator, H1Loss(d=2), optimizer, device)) > tolerance:
        print(f"\r\033[0K loss: {loss:.5f} > tolerance: {tolerance}", end="", flush=True)
    print() # offset log

    operator = operator.to("cpu")
    torch.save(operator.state_dict(), params_file)

    return operator(image_transformed).detach()


def gen_grid(image_transformed: torch.tensor):
    shape = image_transformed.shape[2:]

    grid_intensity, grid_theta = torch.meshgrid(
        torch.linspace(0, 2*torch.pi, shape[0], dtype=image_transformed.dtype),
        torch.linspace(0, 1, shape[1], dtype=image_transformed.dtype),
        indexing='xy'
    )

    return torch.cat((grid_intensity[None, None, :, :], grid_theta[None, None, :, :]), dim=1)


def train(image_transformed: torch.tensor, image_channels: int, operator: InverseRadonOperator, loss_fn, optimizer, device):
    # get data
    xs = image_transformed.to(device)
    ys = xs[:, :image_channels, ...].view(xs.shape[0], image_channels, *xs.shape[2:])

    # calculate loss
    pred = radon(operator(xs))
    loss = loss_fn(pred, ys)

    # back propagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
