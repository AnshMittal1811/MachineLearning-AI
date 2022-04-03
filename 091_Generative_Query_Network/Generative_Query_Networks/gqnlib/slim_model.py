
"""SLIM (Spatial Language Integrating Model)."""

from typing import Tuple, Dict, Optional

from torch import Tensor

from .base import BaseGQN
from .embedding import RepresentationNetwork
from .slim_generator import SlimGenerator
from .utils import nll_normal


class SlimGQN(BaseGQN):
    """Captioned Generative Query Network.

    Implementation of SLIM (Spatial Language Integrating Model).

    Args:
        vocab_dim (int, optional): Vocabulary size of caption data.
        representation_params (dict, optional): Parameters of representation
            network.
        generator_params (dict, optional): Parameters of generator network.
    """

    def __init__(self, vocab_dim: int = 5000,
                 representation_params: Optional[dict] = None,
                 generator_params: Optional[dict] = None) -> None:
        super().__init__()

        rep_kwargs = representation_params if representation_params else {}
        gen_kwargs = generator_params if generator_params else {}

        self.representation = RepresentationNetwork(
            vocab_dim=vocab_dim, **rep_kwargs)
        self.generator = SlimGenerator(**gen_kwargs)

    def inference(self, d_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0, beta: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inference.

        Args:
            d_c (torch.Tensor): Context captions, size `(b, m, l)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            var (float, optional): Variance of observations normal dist.
            beta (float, optional): Coefficient of KL divergence.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            r_c (torch.Tensor): Representations of context, size
                `(b, r, x, y)`.
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses
                with size `(b, n)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *d_dims = d_c.size()
        _, _, *v_dims = v_c.size()
        _, n, *x_dims = x_q.size()

        d_c = d_c.view(-1, *d_dims)
        v_c = v_c.view(-1, *v_dims)

        x_q = x_q.view(-1, *x_dims)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context
        r_c = self.representation(d_c, v_c)
        _, *r_dims = r_c.size()
        r_c = r_c.view(b, m, *r_dims)
        r_c = r_c.sum(1)

        # Copy representations for query
        r_c = r_c.repeat_interleave(n, dim=0)

        # Query images by v_q, i.e. reconstruct
        canvas, kl_loss = self.generator(x_q, v_q, r_c)
        kl_loss = kl_loss * beta

        # Reconstruction loss
        nll_loss = nll_normal(x_q, canvas, x_q.new_ones((1,)) * var,
                              reduce=False)
        nll_loss = nll_loss.sum([1, 2, 3])

        # Returned loss
        nll_loss = nll_loss.view(b, n)
        kl_loss = kl_loss.view(b, n)
        loss_dict = {"loss": nll_loss + kl_loss, "nll_loss": nll_loss,
                     "kl_loss": kl_loss}

        # Restore original shape
        canvas = canvas.view(b, n, *x_dims)
        r_c = r_c.view(b, n, *r_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return (canvas, r_c), loss_dict

    def sample(self, d_c: Tensor, v_c: Tensor, v_q: Tensor) -> Tensor:
        """Samples images `x_q` by context pair `(d, v)` and query viewpoint
        `v_q`.

        Args:
            d_c (torch.Tensor): Context captions, size `(b, m, l)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size
                `(b, n, c, h, w)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *d_dims = d_c.size()
        _, _, *v_dims = v_c.size()

        d_c = d_c.view(-1, *d_dims)
        v_c = v_c.view(-1, *v_dims)

        n = v_q.size(1)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context.
        r_c = self.representation(d_c, v_c)
        _, *r_dims = r_c.size()
        r_c = r_c.view(b, m, *r_dims)

        # Sum over representations: (b, c, h, w)
        r_c = r_c.sum(1)
        r_c = r_c.repeat_interleave(n, dim=0)

        # Sample query images
        canvas = self.generator.sample(v_q, r_c)

        # Restore origina shape
        _, *x_dims = canvas.size()
        canvas = canvas.view(b, n, *x_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return canvas
