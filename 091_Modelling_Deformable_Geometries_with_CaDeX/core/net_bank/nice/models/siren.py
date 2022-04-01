import torch
from torch import nn
import numpy as np


class SineLayer(nn.Module):
    def __init__(self, in_dims, out_dims, bias=True, is_first=False, omega_0=30):
        super().__init__()

        self.omega_0 = omega_0
        self.in_dims = in_dims

        # If is_first=True, omega_0 is a frequency factor which simply multiplies
        # the activations before the nonlinearity. Different signals may require
        # different omega_0 in the first layer - this is a hyperparameter.
        # If is_first=False, then the weights will be divided by omega_0 so as to
        # keep the magnitude of activations constant, but boost gradients to the
        # weight matrix (see supplement Sec. 1.5)
        self.is_first = is_first

        self.linear = nn.Linear(in_dims, out_dims, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_dims,
                                             1 / self.in_dims)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_dims) / self.omega_0,
                                             np.sqrt(6 / self.in_dims) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_dims, hidden_dims, hidden_layers, out_dims, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_dims, hidden_dims,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_dims, hidden_dims,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_dims, out_dims)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_dims) / hidden_omega_0,
                                              np.sqrt(6 / hidden_dims) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_dims, out_dims,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
