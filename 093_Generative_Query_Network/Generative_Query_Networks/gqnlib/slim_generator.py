
"""Generator for SLIM (Spatial Language Integrating Model)."""

from typing import Tuple

import torch
from torch import nn, Tensor

from .utils import kl_divergence_normal
from .generation import Conv2dLSTMCell


class LatentEncoder(nn.Module):
    """Latent encoder p(z|x).

    Args:
        x_channel (int): Number of channels in input observation.
        e_channel (int): Number of channels in the conv. layer mapping input
            images to LSTM input.
        h_channel (int): Number of channels in LSTM layer.
        z_channel (int): Number of channels in the stochastic latent in each
            DRAW step.
    """

    def __init__(self, x_channel: int, e_channel: int, h_channel: int,
                 z_channel: int, stride: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(x_channel, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, e_channel, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.lstm_cell = Conv2dLSTMCell(e_channel + z_channel, h_channel,
                                        kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(h_channel, z_channel * 2, kernel_size=5,
                               stride=1, padding=2)

    def forward(self, x: Tensor, z: Tensor, h: Tensor, c: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward p(z|x, z_prev, h_prev, c_prev).

        Args:
            x (torch.Tensor): Representations, size `(b, r, 64, 64)`.
            z (torch.Tensor): Previous latents, size `(b, z, 8, 8)`.
            h (torch.Tensor): Previous hidden states, size `(b, h, 8, 8)`.
            c (torch.Tensor): Previous cell states, size `(b, h, 8, 8)`.

        Returns:
            mu (torch.Tensor): Mean of `z` distribution, size `(b, z, 8, 8)`.
            logvar (torch.Tensor): Log variance of `z` distribution, size
                `(b, z, 8, 8)`.
            h (torch.Tensor): Current hidden states, size `(b, h, 8, 8)`.
            c (torch.Tensor): Current cell states, size `(b, h, 8, 8)`.
        """

        lstm_input = self.conv(x)
        h, c = self.lstm_cell(torch.cat([lstm_input, z], dim=1), (h, c))
        mu, logvar = torch.chunk(self.conv2(h), 2, dim=1)

        return mu, logvar, h, c


class VisualDecoder(nn.Module):
    """Visual decoder u = f(z, v, r).

    Args:
        h_channel (int): Number of channels in LSTM layer.
        d_channel (int): Number of channels in the conv. layer mapping the
            canvas state to the LSTM input.
        z_channel (int): Number of channels in the latents.
        r_dim (int): Dimension size of representations.
        u_channel (int): Number of channels in the hidden layer between
            LSTM states and the canvas.
        v_dim (int): Dimension size of viewpoints.
        stride (int): Kernel size of transposed conv. layer.
    """

    def __init__(self, h_channel: int, d_channel: int, z_channel: int,
                 r_dim: int, u_channel: int, v_dim: int, stride: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            u_channel, d_channel, kernel_size=stride, stride=stride)
        self.lstm_cell = Conv2dLSTMCell(
            z_channel + v_dim + r_dim + d_channel, h_channel,
            kernel_size=5, stride=1, padding=2)
        self.deconv = nn.ConvTranspose2d(
            h_channel, u_channel, kernel_size=stride, stride=stride)

    def forward(self, z: Tensor, v: Tensor, r: Tensor, u: Tensor, h: Tensor,
                c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Render u = M(z, v)

        Args:
            z (torch.Tensor): Latent states, size `(b, z, 8, 8)`.
            v (torch.Tensor): Query viewpoints, size `(b, v)`.
            r (torch.Tensor): Context representations, size `(b, r, 1, 1)`.
            u (torch.Tensor): Canvas for images, size `(b, u, h*st, w*st)`.
            h (torch.Tensor): Previous hidden states, size `(b, h, 8, 8)`.
            c (torch.Tensor): Previous cell states, size `(b, h, 8, 8)`.

        Returns:
            u (torch.Tensor): Aggregated canvas, size `(b, u, h*st, w*st)`.
            h (torch.Tensor): Current hidden states, size `(b, h, 8, 8)`.
            c (torch.Tensor): Current cell states, size `(b, h, 8, 8)`.
        """

        # Resize viewpoints and representations
        batch, _, height, width = z.size()
        v = v.contiguous().view(batch, -1, 1, 1).repeat(1, 1, height, width)
        r = r.contiguous().view(batch, -1, 1, 1).repeat(1, 1, height, width)

        lstm_input = self.conv(u)
        h, c = self.lstm_cell(torch.cat([z, v, r, lstm_input], dim=1), (h, c))
        u = u + self.deconv(h)

        return u, h, c


class SlimGenerator(nn.Module):
    """Generator class for SLIM.

    Args:
        x_channel (int, optional): Number of channels in input observation.
        u_channel (int, optional): Number of channels in the hidden layer
            between LSTM states and the canvas.
        r_dim (int, optional): Dimension size of representations.
        e_channel (int, optional): Number of channels in the conv. layer
            mapping input images to LSTM input.
        d_channel (int, optional): Number of channels in the conv. layer
            mapping the canvas state to the LSTM input.
        h_channel (int, optional): Number of channels in LSTM layer.
        z_channel (int, optional): Number of channels in the stochastic latent
            in each DRAW step.
        stride (int, optional): Kernel size of transposed conv. layer.
        v_dim (int, optional): Dimension size of viewpoints.
        n_layer (int, optional): Number of recurrent layers in DRAW.
        scale (int, optional): Scale of image in generation process.
    """

    def __init__(self, x_channel: int = 3, u_channel: int = 128,
                 r_dim: int = 256, e_channel: int = 128, d_channel: int = 128,
                 h_channel: int = 128, z_channel: int = 3, stride: int = 2,
                 v_dim: int = 4, n_layer: int = 8, scale: int = 4) -> None:
        super().__init__()

        self.u_channel = u_channel
        self.h_channel = h_channel
        self.z_channel = z_channel
        self.stride = stride
        self.n_layer = n_layer
        self.scale = scale

        self.encoder = LatentEncoder(
            x_channel, e_channel, h_channel, z_channel, stride)
        self.decoder = VisualDecoder(
            h_channel, d_channel, z_channel, r_dim, u_channel, v_dim, stride)

        self.translation = nn.ConvTranspose2d(u_channel, x_channel,
                                              kernel_size=4, stride=4)

    def forward(self, x_q: Tensor, v_q: Tensor, r_c: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Inferences with query pair `(x_q, v_q)` and context representation
        `r_c`.

        Args:
            x_q (torch.Tensor): True queried iamges, size `(b, c, h, w)`.
            v_q (torch.Tensor): Query of viewpoints, size `(b, v)`.
            r_c (torch.Tensor): Representation of context, size `(b, r, x, y)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            kl_loss (torch.Tensor): Calculated KL loss, size `(b,)`.
        """

        # Data size
        batch_size, _, h, w = x_q.size()
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Encoder initial state
        h_enc = x_q.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_enc = x_q.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Decoder initial state
        h_dec = x_q.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = x_q.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = x_q.new_zeros((batch_size, self.u_channel, h_scale * self.stride,
                           w_scale * self.stride))

        # Latent state
        z = x_q.new_zeros((batch_size, self.z_channel, h_scale, w_scale))

        # KL loss
        kl_loss = x_q.new_zeros((batch_size,))

        for _ in range(self.n_layer):
            # Posterior sample
            q_mu, q_logvar, h_enc, c_enc = self.encoder(x_q, z, h_enc, c_enc)
            z = q_mu + (0.5 * q_logvar).exp() * torch.randn_like(q_logvar)

            # Generator state update
            u, h_dec, c_dec = self.decoder(z, v_q, r_c, u, h_dec, c_dec)

            # Calculate KL loss (prior: N(0, I))
            p_mu = torch.zeros_like(q_mu)
            p_logvar = torch.zeros_like(q_logvar)
            _kl_tmp = kl_divergence_normal(
                q_mu, q_logvar.exp(), p_mu, p_logvar.exp(), reduce=False)
            kl_loss += _kl_tmp.sum([1, 2, 3])

        canvas = self.translation(u)

        return canvas, kl_loss

    def sample(self, v_q: Tensor, r_c: Tensor,
               x_shape: Tuple[int, int] = (64, 64)) -> Tensor:
        """Samples images from the prior given viewpoint and representations.

        Args:
            v_q (torch.Tensor): Query of viewpoints, size `(b, v)`.
            r_c (torch.Tensor): Representation of context, size `(b, r, x, y)`.
            x_shape (tuple of int, optional): Sampled x shape.

        Returns:
            canvas (torch.Tensor): Sampled data, size `(b, c, h, w)`.
        """

        # Data size
        batch_size = v_q.size(0)
        h, w = x_shape
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Decoder initial state
        h_dec = v_q.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = v_q.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = v_q.new_zeros((batch_size, self.u_channel, h_scale * self.stride,
                           w_scale * self.stride))

        for _ in range(self.n_layer):
            # Sample prior z ~ N(0, I)
            z = torch.randn(batch_size, self.z_channel, h_scale, w_scale)

            # Generator state update
            u, h_dec, c_dec = self.decoder(z, v_q, r_c, u, h_dec, c_dec)

        canvas = self.translation(u)

        return canvas
