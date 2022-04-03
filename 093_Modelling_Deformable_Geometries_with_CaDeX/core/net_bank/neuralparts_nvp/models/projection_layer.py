import torch
from torch import nn

from .siren import Siren


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dims):
        super().__init__()
        self._input_dims = input_dims

    @property
    def proj_dims(self):
        return self._input_dims

    def forward(self, x):
        return x


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2*proj_dims),
            nn.ReLU(),
            nn.Linear(2*proj_dims, proj_dims)
        )

    @property
    def proj_dims(self):
        return self._proj_dims

    def forward(self, x):
        return self.proj(x)


class SirenProjectionLayer(ProjectionLayer):
    def __init__(self, input_dims, proj_dims, hidden_dims, hidden_layers):
        super().__init__(input_dims, proj_dims)
        self.proj = Siren(input_dims, hidden_dims, hidden_layers, proj_dims)


class FixedPositionalEncoding(ProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__(input_dims, proj_dims)
        pi = 3.141592653589793
        ll = proj_dims//2
        self.sigma = pi * torch.pow(2, torch.linspace(0, ll-1, ll)).view(1, -1)

    @property
    def proj_dims(self):
        return self._proj_dims*3

    def forward(self, x):
        device = x.device
        return torch.cat([
            torch.sin(
                x[:, :, :, :, None] * self.sigma[None, None, None].to(device)
            ),
            torch.cos(
                x[:, :, :, :, None] * self.sigma[None, None, None].to(device)
            )
        ], dim=-1).view(x.shape[0], x.shape[1], x.shape[2], -1)


class GaussianRandomFourierFeatures(ProjectionLayer):
    def __init__(self, input_dims, proj_dims, gamma=1.0):
        super().__init__(input_dims, proj_dims)
        self._two_pi = 6.283185307179586
        self._gamma = gamma
        ll = proj_dims//2
        self.register_buffer("B", torch.randn(3, ll))

    def forward(self, x):
        xB = x.matmul(self.B * self._two_pi * self._gamma)
        return torch.cat([torch.cos(xB), torch.sin(xB)], dim=-1)


def get_projection_layer(**kwargs):
    type = kwargs["type"]

    if type == "identity":
        return IdentityProjection(3)
    elif type == "simple":
        return ProjectionLayer(3, kwargs.get("proj_dims", 128))
    elif type == "siren":
        return SirenProjectionLayer(
            3,
            proj_dims=kwargs.get("proj_dims", 128),
            hidden_dims=kwargs.get("hidden_dims", 256),
            hidden_layers=kwargs.get("hidden_layers", 3)
        )
    elif type == "fixed_positional_encoding":
        return FixedPositionalEncoding(3, kwargs.get("proj_dims", 10))
    elif type == "gaussianrff":
        return GaussianRandomFourierFeatures(
            3, kwargs.get("proj_dims", 10), kwargs.get("gamma", 1.0)
        )
