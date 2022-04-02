
"""Attention GQN.

D. Rosenbaum et al., "Learning models for visual 3D localization with implicit
mapping", http://arxiv.org/abs/1807.03149
"""

from typing import Dict, Tuple, Optional

from torch import Tensor

from .attention_layer import DictionaryEncoder, AttentionGenerator
from .base import BaseGQN
from .utils import nll_normal


class AttentionGQN(BaseGQN):
    """Attention Generative Query Network.

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

        self.representation = DictionaryEncoder(**rep_kwargs)
        self.generator = AttentionGenerator(**gen_kwargs)

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
            key (torch.Tensor): Attention key, size `(b, d*l, 64, 8, 8)`.
            value (torch.Tensor): Attention value, size
                `(b, d*l, c+v+2+64, 8, 8)`.
            r_stack (torch.Tensor): Stacked representations, size
                `(b, n, c+v+2+64, 8, 8)`.
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

        # Attention (key, value) pairs from context
        key, value = self.representation(x_c, v_c)

        # Inference
        canvas, r_stack, kl_loss = self.generator(x_q, v_q, key, value)
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

        _, *k_dims = key.size()
        key = key.view(b, -1, *k_dims)

        _, *v_dims = value.size()
        value = value.view(b, -1, *v_dims)

        _, *r_dims = r_stack.size()
        r_stack = r_stack.view(b, -1, *r_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return (canvas, key, value, r_stack), loss_dict

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

        # Attention (key, value) pairs from context
        key, value = self.representation(x_c, v_c)

        # Sample
        canvas, _ = self.generator.sample(v_q, key, value)

        # Restore origina shape
        canvas = canvas.view(b, n, *x_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return canvas
