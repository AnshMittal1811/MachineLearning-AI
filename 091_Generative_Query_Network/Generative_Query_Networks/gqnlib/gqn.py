
"""Generative Query Network."""

from typing import Dict, Tuple, Optional

import math

import torch
from torch import Tensor

from .base import BaseGQN
from .generation import ConvolutionalDRAW
from .representation import Tower
from .utils import nll_normal


class GenerativeQueryNetwork(BaseGQN):
    """Generative Query Network class.

    Args:
        representation_params (dict, optional): Parameters of representation
            network.
        generator_params (dict, optional): Parameters of generator network.
    """

    def __init__(self, representation_params: Optional[dict] = None,
                 generator_params: Optional[dict] = None) -> None:
        super().__init__()

        rep_kwargs = representation_params if representation_params else {}
        gen_kwargs = generator_params if generator_params else {}

        self.representation = Tower(**rep_kwargs)
        self.generator = ConvolutionalDRAW(**gen_kwargs)

    def inference(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0, beta: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inference.

        Input tensor size should be `(batch, num_points, *dims)`.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            var (float, optional): Variance of observations normal dist.
            beta (float, optional): Coefficient of KL divergence.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size
                `(b, n, c, h, w)`.
            r (torch.Tensor): Representations, size `(b, n, r, x, y)`.
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses
                with size `(b, n)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *x_dims = x_c.size()
        _, _, *v_dims = v_c.size()

        x_c = x_c.view(-1, *x_dims)
        v_c = v_c.view(-1, *v_dims)

        n = x_q.size(1)
        x_q = x_q.view(-1, *x_dims)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context.
        r = self.representation(x_c, v_c)
        _, *r_dims = r.size()
        r = r.view(b, m, *r_dims)

        # Sum over representations, and repeat n times: (b*n, c, x, y)
        r = r.sum(1)
        r = r.repeat_interleave(n, dim=0)

        # Query images by v_q, i.e. reconstruct
        canvas, kl_loss = self.generator(x_q, v_q, r)
        kl_loss = kl_loss * beta

        # Reconstruction loss
        nll_loss = nll_normal(x_q, canvas, x_q.new_ones((1,)) * var,
                              reduce=False)
        nll_loss = nll_loss.sum([1, 2, 3])

        # Reshape loss
        nll_loss = nll_loss.view(b, n)
        kl_loss = kl_loss.view(b, n)

        # Returned loss
        loss_dict = {"loss": nll_loss + kl_loss, "nll_loss": nll_loss,
                     "kl_loss": kl_loss}

        # Restore origina shape
        canvas = canvas.view(b, n, *x_dims)
        r = r.view(b, n, *r_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return (canvas, r), loss_dict

    def sample(self, x_c: Tensor, v_c: Tensor, v_q: Tensor) -> Tensor:
        """Samples images `x_q` by context pair `(x, v)` and query viewpoint
        `v_q`.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size
                `(b, n, c, h, w)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *x_dims = x_c.size()
        _, _, *v_dims = v_c.size()

        x_c = x_c.view(-1, *x_dims)
        v_c = v_c.view(-1, *v_dims)

        n = v_q.size(1)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context.
        r = self.representation(x_c, v_c)
        _, *r_dims = r.size()
        r = r.view(b, m, *r_dims)

        # Sum over representations: (b, c, x, y)
        r = r.sum(1)
        r = r.repeat_interleave(n, dim=0)

        # Sample query images
        canvas = self.generator.sample(v_q, r)

        # Restore original shape
        canvas = canvas.view(b, n, *x_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return canvas
