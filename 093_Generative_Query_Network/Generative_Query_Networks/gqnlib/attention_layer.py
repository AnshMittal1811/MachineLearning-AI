
"""Attention Layer for Attetion GQN.

Ref)
D. Rosenbaum et al., "Learning models for visual 3D localization with implicit
mapping", http://arxiv.org/abs/1807.03149

S. Reed et al., "Few-shot Autoregressive Density Estimation: Towards Learning
to Learn Distributions", https://arxiv.org/abs/1710.10304
"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .generation import Conv2dLSTMCell
from .utils import kl_divergence_normal


class DictionaryEncoder(nn.Module):
    """Dictionary encoder for representation (look-up table of attention).

    Args:
        x_channel (int, optional): Number of channels for images.
    """

    def __init__(self, x_channel: int = 3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(x_channel, 32, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1),
        )

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward to calculate (key-value) pairs.

        1. Extract all image patches
        2. For each patch, calculate (key, value) pair.

        * key: encoded images by conv. net.
        * value: concatenation of encoded images, viewpoints, position in
            image, and key.

        Args:
            x (torch.Tensor): Images, size `(b, c, h, w)`.
            v (torch.Tensor): Viewpoints, size `(b, v)`.

        Returns:
            key (torch.Tensor): Dict key, size `(b*l, 64, 8, 8)`.
            value (torch.Tensor): Dict value, size `(b*l, c+v+2+64, 8, 8)`.
        """

        # Resize images: (64, 64) -> (32, 32)
        x = F.interpolate(x, (32, 32))
        _, c, *_ = x.size()

        # Extract 3x8x8 patches with an overlap of 4 pixels
        x = F.unfold(x, kernel_size=8, stride=4)
        x = x.permute(0, 2, 1)

        # Reshape: (b*l, c, 8, 8)
        b, l, _ = x.size()
        x = x.contiguous().view(b * l, c, 8, 8)

        # key: (b*l, 64, 8, 8)
        key = self.conv(x)

        # Positions of each patch
        pos_x = torch.arange(7, device=x.device).repeat(7).float()
        pos_y = torch.arange(7, device=x.device).repeat_interleave(7).float()

        pos_x = pos_x.view(-1, 1, 1, 1).repeat(b, 1, 8, 8)
        pos_y = pos_y.view(-1, 1, 1, 1).repeat(b, 1, 8, 8)

        # Expand given viewpoints
        v = v.view(b, -1, 1, 1).repeat(l, 1, 8, 8)

        # value: (b*l, x_channel + v_dim + 2 + key_channel, 8, 8)
        value = torch.cat([x, v, pos_x, pos_y, key], dim=1)

        return key, value


class ScaledDotProduct2DAttention(nn.Module):
    """Scaled Dot-Product Attention for 2-D images."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward method to return queried values.

        Args:
            q (torch.Tensor): Query of size `(b, c, h, w)`.
            k (torch.Tensor): Key of size `(d, c, h, w)`.
            v (torch.Tensor): Value of size `(d, v, h, w)`.

        Returns:
            y (torch.Tensor): Queried value of size `(b, v, h, w)`.
        """

        # Dimension of key
        d_k = k.size(1)

        # Transpose for calulation
        q = q.permute(2, 3, 0, 1)
        k = k.permute(2, 3, 1, 0)
        v = v.permute(2, 3, 0, 1)

        # Attetntion
        attn = q.matmul(k) / (d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        y = attn.matmul(v)

        # Revert shape
        y = y.permute(2, 3, 0, 1)

        return y


class AttentionGenerator(nn.Module):
    """Generator with attentive representation.

    Args:
        x_channel (int, optional): Number of channel in input images.
        v_dim (int, optional): Dimensions of viewpoints.
        z_channel (int, optional): Number of channel in latent variable.
        h_channel (int, optional): Number of channel in hidden states.
        u_channel (int, optional): Number of channel in hidden layer for
            canvas.
        n_layer (int, optional): Number of recurrent layers.
        scale (int, optional): Scale of image generation process.
        stride (int, optional): Kernel size of transposed conv. layer.
    """

    def __init__(self, x_channel: int = 3, v_dim: int = 7, z_channel: int = 64,
                 h_channel: int = 128, u_channel: int = 128, n_layer: int = 8,
                 scale: int = 4, stride: int = 2) -> None:
        super().__init__()

        self.x_channel = x_channel
        self.v_dim = v_dim
        self.z_channel = z_channel
        self.h_channel = h_channel
        self.u_channel = u_channel
        self.n_layer = n_layer
        self.scale = scale
        self.stride = stride

        # Representation
        r_channel = x_channel + v_dim + 2 + 64
        self.query_gen = nn.Sequential(
            nn.Conv2d(h_channel, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
        )
        self.attention = ScaledDotProduct2DAttention()

        # Distributions (variational posterior / prior)
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        self.posterior = nn.Conv2d(h_channel, z_channel * 2, **kwargs)
        self.prior = nn.Conv2d(h_channel, z_channel * 2, **kwargs)

        # Top layer
        self.read_head = nn.Conv2d(
            x_channel, x_channel, kernel_size=scale, stride=scale * stride,
            bias=False)
        self.write_head = nn.ConvTranspose2d(
            h_channel, u_channel, kernel_size=stride, stride=stride,
            bias=False)

        # Recurrent encoder/decoder models
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        self.encoder = Conv2dLSTMCell(
            v_dim + r_channel + x_channel + h_channel, h_channel, **kwargs)
        self.decoder = Conv2dLSTMCell(
            v_dim + r_channel + z_channel, h_channel, **kwargs)

        # Final layer to convert u -> canvas
        kwargs = dict(kernel_size=scale, stride=scale, padding=0)
        self.observation = nn.ConvTranspose2d(u_channel, x_channel, **kwargs)

    def forward(self, x: Tensor, v: Tensor, key: Tensor, value: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Inferences given query pair (x, v) and representation r.

        Args:
            x (torch.Tensor): True queried iamges `x_q`, size `(b, c, h, w)`.
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            key (torch.Tensor): Attention key, size `(d*l, 64, 8, 8)`.
            value (torch.Tensor): Attention value, size
                `(d*l, c+v+2+64, 8, 8)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            r_stack (torch.Tensor): Stacked representations, size
                `(b, c+v+2+64, 8, 8)`.
            kl_loss (torch.Tensor): Calculated KL loss, size `(b,)`.
        """

        # Data size
        batch_size, _, h, w = x.size()
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Generator initial state
        h_dec = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Inference initial state
        h_enc = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_enc = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = x.new_zeros((batch_size, self.u_channel, h_scale * self.stride,
                         w_scale * self.stride))

        # Representations
        _, *val_dims = value.size()
        r_stack = x.new_zeros(batch_size, *val_dims)

        # KL loss value
        kl_loss = x.new_zeros((batch_size,))

        # Reshape: Downsample x, upsample v
        x = self.read_head(x)
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h_scale, w_scale)

        for _ in range(self.n_layer):
            # Prior factor
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)

            # Query representation by attention
            query = self.query_gen(h_enc)
            r = self.attention(query, key, value)
            r_stack += r

            # Inference state update
            h_enc, c_enc = self.encoder(torch.cat([h_dec, x, v, r], dim=1),
                                        (h_enc, c_enc))

            # Posterior factor
            q_mu, q_logvar = torch.chunk(self.posterior(h_enc), 2, dim=1)

            # Posterior sample
            z = q_mu + (0.5 * q_logvar).exp() * torch.randn_like(q_logvar)

            # Generator state update
            h_dec, c_dec = self.decoder(torch.cat([v, r, z], dim=1),
                                        (h_dec, c_dec))

            # Draw canvas
            u = u + self.write_head(h_dec)

            # Calculate loss
            _kl_tmp = kl_divergence_normal(q_mu, q_logvar.exp(), p_mu,
                                           p_logvar.exp(), reduce=False)
            kl_loss += _kl_tmp.sum([1, 2, 3])

        # Returned value
        canvas = self.observation(u)

        return canvas, r_stack, kl_loss

    def sample(self, v: Tensor, key: Tensor, value: Tensor,
               x_shape: Tuple[int, int] = (64, 64)) -> Tuple[Tensor, Tensor]:
        """Samples images from the prior given viewpoint and representation.

        Args:
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            key (torch.Tensor): Attention key, size `(d*l, 64, 8, 8)`.
            value (torch.Tensor): Attention value, size
                `(d*l, c+v+2+64, 8, 8)`.
            x_shape (tuple of int, optional): Sampled x shape.

        Returns:
            canvas (torch.Tensor): Sampled data, size `(b, c, h, w)`.
            r_stack (torch.Tensor): Stacked representations, size
                `(b, c+v+2+64, 8, 8)`.
        """

        batch_size = v.size(0)
        h, w = x_shape
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Hidden states
        h_dec = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = v.new_zeros((batch_size, self.u_channel, h_scale * self.stride,
                         w_scale * self.stride))

        # Representations
        _, *val_dims = value.size()
        r_stack = v.new_zeros(batch_size, *val_dims)

        # Upsample v and r
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h_scale, w_scale)

        for _ in range(self.n_layer):
            # Sample prior
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)
            z = p_mu + (0.5 * p_logvar).exp() * torch.randn_like(p_logvar)

            # Query representation by attention
            query = self.query_gen(h_dec)
            r = self.attention(query, key, value)
            r_stack += r

            # Decode
            h_dec, c_dec = self.decoder(torch.cat([z, v, r], dim=1),
                                        (h_dec, c_dec))

            # Draw canvas
            u = u + self.write_head(h_dec)

        canvas = self.observation(u)

        return canvas, r_stack
