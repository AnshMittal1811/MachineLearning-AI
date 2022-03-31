import torch
import torch.optim as optim
import math


class UniformAdam(optim.Optimizer):
  def __init__(
    self,
    params,
    lr=5e-3,
    betas=(0.9, 0.999),
    diffusion:float = 1e-5,
    eps=1e-5,
  ):
    assert(lr > 0), "Must assign a learning rate greater than 0"
    defaults = dict(
      lr=lr,
      betas=betas,
      diffusion=diffusion,
      eps=eps,
    )

    super().__init__(params, defaults)
  def step(self, closure=None):
    loss = None
    if closure is not None: loss = closure()

    for group in self.param_groups:
      beta_1, beta_2 = group["betas"]
      diff = group["diffusion"]
      step_size = group["lr"]
      eps = group["eps"]

      for p in group["params"]:
        if p.grad is None: continue
        dphidx = p.grad.data
        assert(not dphidx.is_sparse), "Does not support sparse grads yet"
        state = self.state[p]
        if len(state) == 0:
          state["step"] = 0
          state["moving_avg_1"] = torch.zeros_like(p.grad)
          state["moving_avg_2"] = torch.zeros_like(p.grad)
          state["g"] = torch.zeros_like(p.data)
          state["u"] = torch.zeros_like(p.data)
        state["step"] += 1

        # sketchy ass laplacian matrix, what might be a better operator for neural parameters?
        # TODO do we need to flatten here? In some cases we may be able to get away with not
        # flattening.
        data = p.data[None]
        if len(data.shape) == 2: data.unsqueeze_(-1)
        L = torch.cdist(data, data).squeeze(0)
        L.clamp_(min=1e-3).reciprocal_()
        assert(len(L.shape) == 2)
        assert(L.shape[0] == L.shape[1])
        N = L.shape[0]
        L[range(N), range(N)] = 0
        L.neg_()
        # set the distance with itself to be the sum of distances with other elements
        L[range(N), range(N)] = -L.sum(dim=-1)
        L.mul_(diff)
        L[range(N), range(N)] += 1

        g = state["g"]
        m_1, m_2 = state["moving_avg_1"], state["moving_avg_2"]

        torch.linalg.solve(L, dphidx, out=g)
        m_1.mul_(beta_1).add_(g, alpha=(1-beta_1))
        m_2.mul_(beta_2).addcmul_(g,g,value=(1-beta_2))

        n_step = state["step"]

        bias_correction_1 = 1 - (beta_1**n_step)
        bias_correction_2 = 1 - (beta_2**n_step)
        step_size = step_size * math.sqrt(bias_correction_2)/bias_correction_1

        u = state["u"]
        torch.matmul(L, p.data, out=u)
        u.addcdiv_(
          m_1,
          torch.linalg.vector_norm(m_2,ord=float("inf"),dim=-1,keepdim=True).sqrt_().add_(eps),
          value=-step_size
        )
        with torch.no_grad():
          torch.linalg.solve(L, u, out=p)
    return loss


