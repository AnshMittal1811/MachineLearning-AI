
"""Generator network.

Generator network is similar to Convolutional DRAW [K. Gregor et al., 2016],
[K. Gregor et al., 2016]. The original GQN paper [S. M. Ali Eslami et al.,
2018] does not mention it, but the successive paper [A. Kumar et al., (2018)]
explicitly uses Convolutional DRAW as a part of generator.

(Reference)

* A. Kumar et al., "Consistent Generative Query Networks" (2018).
  http://arxiv.org/abs/1807.02033

* K. Gregor et al., "DRAW: A Recurrent Neural Network For Image Generation"
  (2015)
  http://arxiv.org/abs/1502.04623

* K. Gregor et al., "Towards conceptual compression" (2016).
  http://arxiv.org/abs/1604.08772

(Reference code)

https://github.com/wohlert/generative-query-network-pytorch/blob/master/draw/draw.py
"""

from typing import Tuple

import torch
from torch import nn, Tensor

from .utils import kl_divergence_normal


class Conv2dLSTMCell(nn.Module):
    """2D Convolutional long short-term memory (LSTM) cell.

    Args:
        in_channels (int): Number of input channel.
        out_channels (int): Number of output channel.
        kernel_size (int): Size of image kernel.
        stride (int): Length of kernel stride.
        padding (int): Number of pixels to pad with.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int) -> None:
        super().__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

        self.transform = nn.Conv2d(out_channels, in_channels, **kwargs)

    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]
                ) -> Tuple[Tensor, Tensor]:
        """Forward through cell.

        Args:
            x (torch.Tensor): Input to send through.
            states (tuple of torch.Tensor): (hidden, cell) pair of internal
                state.

        Returns:
            next_states (tuple of torch.Tensor): (hidden, cell) pair of
                internal next state.
        """

        hidden, cell = states
        x = x + self.transform(hidden)

        forget_gate = torch.sigmoid(self.forget(x))
        input_gate = torch.sigmoid(self.input(x))
        output_gate = torch.sigmoid(self.output(x))
        state_gate = torch.tanh(self.state(x))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class InferenceCore(nn.Module):
    """Inference core module.

    Args:
        x_channel (int): Number of channel in input images.
        v_dim (int): Dimensions of viewpoints.
        r_dim (int): Dimensions of representations.
        u_channel (int): Number of channel in hidden layer for canvas.
        h_channel (int): Number of channel in hidden states.
    """

    def __init__(self, x_channel: int, v_dim: int, r_dim: int, u_channel: int,
                 h_channel: int) -> None:
        super().__init__()

        self.downsample_x = nn.Conv2d(
            x_channel, x_channel, kernel_size=4, stride=4, padding=0,
            bias=False)
        self.upsample_v = nn.ConvTranspose2d(
            v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(
            r_dim, r_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(
            u_channel, u_channel, kernel_size=4, stride=4, padding=0,
            bias=False)
        self.core = Conv2dLSTMCell(
            x_channel + v_dim + r_dim + u_channel + h_channel, h_channel,
            kernel_size=5, stride=1, padding=2)

    def forward(self, x: Tensor, v: Tensor, r: Tensor, u: Tensor,
                h_dec: Tensor, h_enc: Tensor, c_enc: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Inferences.

        Args:
            x (torch.Tensor): True queried iamges `x_q`, size `(b, c, h, w)`.
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.
            u (torch.Tensor): Sampled observation, size `(b, u, h, w)`.
            h_dec (torch.Tensor): Hidden state of decoder, size `(b, c, h, w)`.
            h_enc (torch.Tensor): Hidden state of encoder, size `(b, c, h, w)`.
            c_enc (torch.Tensor): Hidden state of encoder, size `(b, c, h, w)`.
        """

        # Convert size
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(*v.size(), 1, 1))
        if r.size(2) != h_enc.size(2):
            r = self.upsample_r(r)
        u = self.downsample_u(u)

        # Inference
        h_enc, c_enc = self.core(torch.cat([x, v, r, u, h_dec], dim=1),
                                 (h_enc, c_enc))

        return h_enc, c_enc


class GenerationCore(nn.Module):
    """Generation core module.

    Args:
        v_dim (int): Dimensions of viewpoints.
        r_dim (int): Dimensions of representations.
        z_channel (int): Number of channel in latent variable.
        u_channel (int): Number of channel in hidden layer for canvas.
        h_channel (int): Number of channel in hidden states.
        scale (int): Scale of image generation process.
    """

    def __init__(self, v_dim: int, r_dim: int, z_channel: int, u_channel: int,
                 h_channel: int, scale: int) -> None:
        super().__init__()

        self.upsample_v = nn.ConvTranspose2d(
            v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(
            r_dim, r_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(
            v_dim + r_dim + z_channel, h_channel, kernel_size=5, stride=1,
            padding=2)

        self.write_head = nn.ConvTranspose2d(
            h_channel, u_channel, kernel_size=scale, stride=scale, padding=0,
            bias=False)

    def forward(self, v: Tensor, r: Tensor, u: Tensor, z: Tensor,
                h_dec: Tensor, c_dec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Generates sample u.

        Args:
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.
            u (torch.Tensor): Sampled observation, size `(b, u, h, w)`.
            z (torch.Tensor): Stochastic latent, size `(b, c, h, w)`.
            h_dec (torch.Tensor): Hidden state of decoder, size `(b, c, h, w)`.
            c_dec (torch.Tensor): Hidden state of decoder, size `(b, c, h, w)`.
        """

        # Convert size
        v = self.upsample_v(v.view(*v.size(), 1, 1))
        if r.size(2) != h_dec.size(2):
            r = self.upsample_r(r)

        # Inference
        h_dec, c_dec = self.core(torch.cat([v, r, z], dim=1), (h_dec, c_dec))
        u = self.write_head(h_dec) + u

        return h_dec, c_dec, u


class ConvolutionalDRAW(nn.Module):
    """Convolutional DRAW (Deep Recurrent Attentive Writer).

    Args:
        x_channel (int, optional): Number of channel in input images.
        v_dim (int, optional): Dimensions of viewpoints.
        r_dim (int, optional): Dimensions of representations.
        z_channel (int, optional): Number of channel in latent variable.
        h_channel (int, optional): Number of channel in hidden states.
        u_channel (int, optional): Number of channel in hidden layer for
            canvas.
        n_layer (int, optional): Number of recurrent layers.
        scale (int, optional): Scale of image generation process.
        stride (int, optional): Kernel size of transposed conv. layer.
    """

    def __init__(self, x_channel: int = 3, v_dim: int = 7, r_dim: int = 256,
                 z_channel: int = 64, h_channel: int = 128,
                 u_channel: int = 128, n_layer: int = 8, scale: int = 4,
                 stride: int = 1) -> None:
        super().__init__()

        self.h_channel = h_channel
        self.z_channel = z_channel
        self.u_channel = u_channel
        self.n_layer = n_layer
        self.scale = scale
        self.stride = stride

        # Distributions (variational posterior / prior)
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        self.posterior = nn.Conv2d(h_channel, z_channel * 2, **kwargs)
        self.prior = nn.Conv2d(h_channel, z_channel * 2, **kwargs)

        # Recurrent encoder/decoder models
        self.encoder = InferenceCore(
            x_channel, v_dim, r_dim, u_channel, h_channel)
        self.decoder = GenerationCore(
            v_dim, r_dim, z_channel, u_channel, h_channel, scale)

        # Final layer to convert u -> canvas
        kwargs = dict(kernel_size=stride, stride=stride, padding=0)
        self.observation = nn.ConvTranspose2d(u_channel, x_channel, **kwargs)

    def forward(self, x: Tensor, v: Tensor, r: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Inferences given query pair (x, v) and representation r.

        Args:
            x (torch.Tensor): True queried iamges `x_q`, size `(b, c, h, w)`.
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
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
        u = x.new_zeros((batch_size, self.u_channel, h, w))

        # KL loss value
        kl_loss = x.new_zeros((batch_size,))

        for _ in range(self.n_layer):
            # Prior factor (eta_pi)
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)

            # Inference state update
            h_enc, c_enc = self.encoder(x, v, r, u, h_dec, h_enc, c_enc)

            # Posterior factor (eta_e)
            q_mu, q_logvar = torch.chunk(self.posterior(h_enc), 2, dim=1)

            # Posterior sample
            z = q_mu + (0.5 * q_logvar).exp() * torch.randn_like(q_logvar)

            # Generator state update
            h_dec, c_dec, u = self.decoder(v, r, u, z, h_dec, c_dec)

            # Calculate loss
            _kl_tmp = kl_divergence_normal(q_mu, q_logvar.exp(), p_mu,
                                           p_logvar.exp(), reduce=False)
            kl_loss += _kl_tmp.sum([1, 2, 3])

        # Returned values
        canvas = self.observation(u)

        return canvas, kl_loss

    def sample(self, v: Tensor, r: Tensor, x_shape: Tuple[int, int] = (64, 64)
               ) -> Tensor:
        """Samples images from the prior given viewpoint and representation.

        Args:
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.
            x_shape (tuple of int, optional): Sampled x shape.

        Returns:
            canvas (torch.Tensor): Sampled data, size `(b, c, h, w)`.
        """

        batch_size = v.size(0)
        h, w = x_shape
        h_scale = h // (self.scale * self.stride)
        w_scale = w // (self.scale * self.stride)

        # Hidden states
        h_dec = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = v.new_zeros((batch_size, self.u_channel, h, w))

        for _ in range(self.n_layer):
            # Sample prior
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)
            z = p_mu + (0.5 * p_logvar).exp() * torch.randn_like(p_logvar)

            # Decode
            h_dec, c_dec, u = self.decoder(v, r, u, z, h_dec, c_dec)

        canvas = self.observation(u)

        return canvas
