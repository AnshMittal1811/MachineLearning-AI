
"""Representation network."""

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Pyramid(nn.Module):
    """Pyramid.

    Args:
        n_channel (int, optional): Number of channel of images.
        n_target (int, optional): Dimension of viewpoints.
    """

    def __init__(self, n_channel: int = 3, n_target: int = 7) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channel + n_target, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """Represents r given images `x` and viewpoints `v`.

        Args:
            x (torch.Tensor): Images tensor, size `(batch, c, 64, 64)`.
            v (torch.Tensor): Viewpoints tensor, size `(batch, 7)`.

        Returns:
            r (torch.Tensor): Representation tensor, size `(batch, 256, 1, 1)`.
        """

        v = v.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        r = self.conv(torch.cat([x, v], dim=1))

        return r


class Tower(nn.Module):
    """Tower or Pool.

    Args:
        n_channel (int, optional): Number of channel of images.
        n_target (int, optional): Dimension of viewpoints.
        do_pool (bool, optional): If `True`, average pooling layer is added.
    """

    def __init__(self, n_channel: int = 3, n_target: int = 7,
                 do_pool: bool = False) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(n_channel, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256 + n_target, 256, kernel_size=3, stride=1,
                               padding=1)
        self.conv6 = nn.Conv2d(256 + n_target, 128, kernel_size=3, stride=1,
                               padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        self.do_pool = do_pool
        self.pool = nn.AvgPool2d(16)

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """Represents r given images `x` and viewpoints `v`.

        Args:
            x (torch.Tensor): Images tensor, size `(batch, c, 64, 64)`.
            v (torch.Tensor): Viewpoints tensor, size `(batch, 7)`.

        Returns:
            r (torch.Tensor): Representation tensor, size `(batch, 256, 1, 1)`
                if `do_pool` is `True`, `(batch, 256, 16, 16)` otherwise.
        """

        # First skip-connected conv block
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        h = F.relu(self.conv3(skip_in))
        h = F.relu(self.conv4(h)) + skip_out

        # Second skip-connected conv block
        v = v.view(-1, 7, 1, 1).repeat(1, 1, 16, 16)
        skip_in = torch.cat([h, v], dim=1)
        skip_out = F.relu(self.conv5(skip_in))

        h = F.relu(self.conv6(skip_in))
        h = F.relu(self.conv7(h)) + skip_out

        r = F.relu(self.conv8(h))

        if self.do_pool:
            r = self.pool(r)

        return r


class Simple(nn.Module):
    """Simple.

    Args:
        n_channel (int, optional): Number of channel of images.
        n_target (int, optional): Dimension of viewpoints.
    """

    def __init__(self, n_channel: int = 3, n_target: int = 7) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channel + n_target, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """Represents r given images `x` and viewpoints `v`.

        Args:
            x (torch.Tensor): Images tensor, size `(batch, c, 64, 64)`.
            v (torch.Tensor): Viewpoints tensor, size `(batch, 7)`.

        Returns:
            r (torch.Tensor): Representation tensor, size
                `(batch, 32, 16, 16)`.
        """

        v = v.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        r = self.conv(torch.cat([x, v], dim=1))

        return r
