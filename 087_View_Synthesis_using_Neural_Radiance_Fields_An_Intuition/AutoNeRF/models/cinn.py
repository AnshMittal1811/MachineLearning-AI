"""Flow consisting of ActNorm, DoubleVectorCouplingBlock and Shuffle. Additionally, powerful conditioning encodings are
learned."""
import torch
import torch.nn as nn
import numpy as np


from AutoNeRF.models.blocks import ConditionalFlow




class SimpleEmbedder(nn.Module):
    """
    This module is the (simple) replacement for the conditioning function H(x) := x
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x



class ConditionalTransformer(nn.Module):
    """
    Conditional Invertible Neural Network.
    Can be conditioned both on input with spatial dimension (i.e. a tensor of shape BxCxHxW) and a flat input
    (i.e. a tensor of shape BxC)
    Our activation function is a LRELU
    """
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, 
        n_flows, conditioning_option="none", activation="lrelu"):
        """
            in_channels: the size of the input. This should be divisible by 2.
            cond_channels: the dimension of the hidden layers for s and t.
            hidden_dim: the dimension of the hidden layers for s and t.
            hidden_depth: number of hidden layers / depth in s_theta and t_theta respectively
            n_flows: number of cINN blocks in our final network
            conditioning_option: ----
            activation: ----
        """

        # ** experimental **
        #import torch.backends.cudnn as cudnn
        #cudnn.benchmark = True
        super().__init__()


        self.flow = ConditionalFlow(in_channels=in_channels, cond_channels=cond_channels, hidden_dim=hidden_dim,
                                    hidden_depth=hidden_depth, n_flows=n_flows, conditioning_option=conditioning_option,
                                    activation=activation)

        self.embedder = SimpleEmbedder()


    def embed(self, conditioning):
        # embed y via embedding layer H(.)

        embedding = self.embedder(conditioning)
        return embedding

    def sample(self, shape, conditioning):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, self.embed(conditioning))
        return sample

    def forward(self, input, conditioning, train=False):
        embedding = self.embed(conditioning)
        out, logdet = self.flow(input, embedding)
        if train:
            #  TODO: understand the difference between last_out and out ????
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, conditioning):
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)