
"""Prior, Posterior, and Renderer for Consistent GQN."""

from typing import Tuple

import torch
from torch import nn, Tensor

from .utils import kl_divergence_normal
from .generation import Conv2dLSTMCell


class LatentDistribution(nn.Module):
    """Latent Distribution p(z|r).

    * prior: p(z|r_c)
    * posterior: p(z|r_c, r_q)

    Args:
        r_channel (int): Number of channels in representation.
        e_channel (int): Number of channels in the conv. layer mapping input
            images to LSTM input (nf_enc).
        h_channel (int): Number of channels in LSTM layer (nf_to_hidden).
        z_channel (int): Number of channels in the stochastic latent in each
            DRAW step (nf_z).
    """

    def __init__(self, r_channel: int, e_channel: int, h_channel: int,
                 z_channel: int, stride: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(r_channel, e_channel, kernel_size=stride,
                               stride=stride)
        self.lstm_cell = Conv2dLSTMCell(e_channel + z_channel, h_channel,
                                        kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(h_channel, z_channel * 2, kernel_size=5,
                               stride=1, padding=2)

    def forward(self, r: Tensor, z: Tensor, h: Tensor, c: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Converts r -> z.

        Args:
            r (torch.Tensor): Representations, size `(b, r, 16, 16)`.
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

        lstm_input = self.conv1(r)
        h, c = self.lstm_cell(torch.cat([lstm_input, z], dim=1), (h, c))
        mu, logvar = torch.chunk(self.conv2(h), 2, dim=1)

        return mu, logvar, h, c


class Renderer(nn.Module):
    """Renderer M_gamma(z, v_q)

    Args:
        h_channel (int): Number of channels in LSTM layer (nf_to_hidden).
        d_channel (int): Number of channels in the conv. layer mapping the
            canvas state to the LSTM input (nf_dec).
        z_channel (int): Number of channels in the stochastic latent in each
            DRAW step (nf_z).
        u_channel (int): Number of channels in the hidden layer between
            LSTM states and the canvas (nf_to_obs).
        v_dim (int): Dimension size of viewpoints.
        stride (int): Kernel size of transposed conv. layer (stride_to_obs).
    """

    def __init__(self, h_channel: int, d_channel: int, z_channel: int,
                 u_channel: int, v_dim: int, stride: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(u_channel, d_channel, kernel_size=stride,
                              stride=stride)
        self.lstm_cell = Conv2dLSTMCell(z_channel + v_dim + d_channel,
                                        h_channel, kernel_size=5, stride=1,
                                        padding=2)
        self.deconv = nn.ConvTranspose2d(h_channel, u_channel,
                                         kernel_size=stride, stride=stride)

    def forward(self, z: Tensor, v: Tensor, u: Tensor, h: Tensor, c: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Render u = M(z, v)

        Args:
            z (torch.Tensor): Latent states, size `(b, z, 8, 8)`.
            v (torch.Tensor): Query viewpoints, size `(b, n, v)`.
            u (torch.Tensor): Canvas for images, size `(b*n, u, h*st, w*st)`.
            h (torch.Tensor): Previous hidden states, size `(b*n, h, 8, 8)`.
            c (torch.Tensor): Previous cell states, size `(b*n, h, 8, 8)`.

        Returns:
            u (torch.Tensor): Aggregated canvas, size `(b*n, u, h*st, w*st)`.
            h (torch.Tensor): Current hidden states, size `(b*n, h, 8, 8)`.
            c (torch.Tensor): Current cell states, size `(b*n, h, 8, 8)`.
        """

        # Resize latents and viewpoints
        _, z_dim, h_dim, w_dim = z.size()
        b, n, v_dim = v.size()

        # z: (b, z, h, w) -> (b * n, z, h, w)
        z = z.repeat_interleave(n, dim=0)

        # v: (b, n, v) -> (b * n, v, h, w)
        v = v.contiguous().view(b * n, v_dim, 1, 1).repeat(1, 1, h_dim, w_dim)

        lstm_input = self.conv(u)
        h, c = self.lstm_cell(torch.cat([z, v, lstm_input], dim=1), (h, c))
        u = u + self.deconv(h)

        return u, h, c


class DRAWRenderer(nn.Module):
    """DRAW Renderer class for generation and inference.

    Args:
        x_channel (int, optional): Number of channels in the observations.
        u_channel (int, optional): Number of channels in the hidden layer
            between LSTM states and the canvas (nf_to_obs).
        r_channel (int, optional): Number of channels in representation.
        e_channel (int, optional): Number of channels in the conv. layer
            mapping input images to LSTM input (nf_enc).
        d_channel (int, optional): Number of channels in the conv. layer
            mapping the canvas state to the LSTM input (nf_dec).
        h_channel (int, optional): Number of channels in LSTM layer
            (nf_to_hidden).
        z_channel (int, optional): Number of channels in the stochastic latent
            in each DRAW step (nf_z).
        stride (int, optional): Kernel size of transposed conv. layer
            (stride_to_obs).
        v_dim (int, optional): Dimension size of viewpoints.
        n_layer (int, optional): Number of recurrent layers.
        scale (int, optional): Scale of image generation process.
    """

    def __init__(self, x_channel: int = 3, u_channel: int = 128,
                 r_channel: int = 32, e_channel: int = 128,
                 d_channel: int = 128, h_channel: int = 64, z_channel: int = 3,
                 stride: int = 2, v_dim: int = 7, n_layer: int = 8,
                 scale: int = 4) -> None:
        super().__init__()

        self.u_channel = u_channel
        self.h_channel = h_channel
        self.z_channel = z_channel
        self.stride = stride
        self.n_layer = n_layer
        self.scale = scale

        self.prior = LatentDistribution(r_channel, e_channel, h_channel,
                                        z_channel, stride)
        self.posterior = LatentDistribution(r_channel * 2, e_channel,
                                            h_channel, z_channel, stride)
        self.renderer = Renderer(h_channel, d_channel, z_channel, u_channel,
                                 v_dim, stride)

        self.translation = nn.ConvTranspose2d(u_channel, x_channel,
                                              kernel_size=4, stride=4)

    def forward(self, x: Tensor, v: Tensor, r_c: Tensor, r_q: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Inferences given query pair (x, v) and representation r.

        Args:
            x (torch.Tensor): Query iamges `x_q`, size `(b, n, c, h, w)`.
            v (torch.Tensor): Query viewpoints `v_q`, size `(b, n, v)`.
            r_c (torch.Tensor): Representation of context, size `(b, c, h, w)`.
            r_q (torch.Tensor): Representation of query, size `(b, c, h, w)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size
                `(b, n, c, h, w)`.
            kl_loss (torch.Tensor): Calculated KL loss, size `(b,)`.
        """

        # Data size
        batch_size, n, _, h, w = x.size()
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Prior initial state
        h_phi = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_phi = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Posterior initial state
        h_psi = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_psi = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Renderer initial state
        h_rnd = x.new_zeros((batch_size * n, self.h_channel, h_scale, w_scale))
        c_rnd = x.new_zeros((batch_size * n, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = x.new_zeros((batch_size * n, self.u_channel, h_scale * self.stride,
                         w_scale * self.stride))

        # Latent state
        z = x.new_zeros((batch_size, self.z_channel, h_scale, w_scale))

        # KL loss value
        kl_loss = x.new_zeros((batch_size,))

        for _ in range(self.n_layer):
            # Prior
            p_mu, p_logvar, h_phi, c_phi = self.prior(r_c, z, h_phi, c_phi)

            # Posterior
            q_mu, q_logvar, h_psi, c_psi = self.posterior(
                torch.cat([r_c, r_q], dim=1), z, h_psi, c_psi)

            # Posterior sample
            z = q_mu + (0.5 * q_logvar).exp() * torch.randn_like(q_logvar)

            # Generator state update
            u, h_rnd, c_rnd = self.renderer(z, v, u, h_rnd, c_rnd)

            # Calculate loss
            _kl_tmp = kl_divergence_normal(q_mu, q_logvar.exp(), p_mu,
                                           p_logvar.exp(), reduce=False)
            kl_loss += _kl_tmp.sum([1, 2, 3])

        canvas = self.translation(u)

        # Reshape
        _, *canvas_dim = canvas.size()
        canvas = canvas.view(batch_size, n, *canvas_dim)

        return canvas, kl_loss

    def sample(self, v: Tensor, r: Tensor, x_shape: Tuple[int, int] = (64, 64)
               ) -> Tensor:
        """Samples images from the prior given viewpoint and representation.

        Args:
            v (torch.Tensor): Query viewpoints `v_q`, size `(b, n, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.
            x_shape (tuple of int, optional): Sampled x shape.

        Returns:
            canvas (torch.Tensor): Sampled data, size `(b, n, c, h, w)`.
        """

        batch_size, n, _ = v.size()
        h, w = x_shape
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Prior initial state
        h_phi = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_phi = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Renderer initial state
        h_rnd = v.new_zeros((batch_size * n, self.h_channel, h_scale, w_scale))
        c_rnd = v.new_zeros((batch_size * n, self.h_channel, h_scale, w_scale))

        # Latent state
        z = v.new_zeros((batch_size, self.z_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = v.new_zeros((batch_size * n, self.u_channel, h_scale * self.stride,
                         w_scale * self.stride))

        for _ in range(self.n_layer):
            # Sample prior
            p_mu, p_logvar, h_phi, c_phi = self.prior(r, z, h_phi, c_phi)
            z = p_mu + (0.5 * p_logvar).exp() * torch.randn_like(p_logvar)

            # Generator state update
            u, h_rnd, c_rnd = self.renderer(z, v, u, h_rnd, c_rnd)

        canvas = self.translation(u)

        # Reshape
        _, *canvas_dim = canvas.size()
        canvas = canvas.view(batch_size, n, *canvas_dim)

        return canvas
