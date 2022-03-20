import torch
import torch.nn as nn




class BasicFullyConnectedNet(nn.Module):
    """
        This class implements the architectures used in s and t.
    """
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        """
            dim: The input dimension of our network
            depth: The depth of our network
            hidden_dim: The hidden dimension of our network
            use_tanh: boolean to specify if we want a boolean at the end.
            use_bn: boolean to specify if we want batchnorm layers in between.
            out_dim: the output dimension of the network
        """
        super(BasicFullyConnectedNet, self).__init__()

        # we define our layers
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))

        # Add batch norm layer
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())

        # Add layers
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))

        # Add optional tanh activation at the end
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        
        z = x
        for layer in self.main:
     
            z = layer(z)

        return z
        #return self.main(x)



#--------------------------------------------------------------------------------------------------------------------
class ConditionalDoubleVectorCouplingBlock(nn.Module):
    """
        This implements s_theta and t_theta as defined in the paper.
    """
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        """
            in_channels: the size of the input. This should be divisible by 2.
            cond_channels: the size of the conditional H(y).
        """
        
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        
        
        # since we split the input into two halves, we only feed s and t respectively 
        # the input size in_channels // 2 + cond_channels.
        
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=True,
                                   out_dim=in_channels // 2) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False,
                                   out_dim=in_channels // 2) for _ in range(2)])

    def forward(self, x, xc, reverse=False):
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:

            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)

                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_
            return x[:, :, None, None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:, :, None, None]



#--------------------------------------------------------------------------------------------------------------------
class ActNorm(nn.Module):
    """
        Normalize the input using the mean and std we calculate from the first batch. 
    """

    def __init__(self, num_features, logdet=False):
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h




#--------------------------------------------------------------------------------------------------------------------
class ConditionalFlatDoubleCouplingFlowBlock(nn.Module):
    """
        Implementaion of the actual double coupling in the cINN.
        This works as follows:
            * normalize input
            * apply activation function (lrelu)
            * apply our coupling
            * shuffle the input
    """
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="lrelu"):
        super().__init__()
        #__possible_activations = {"lrelu": InvLeakyRelu
                                  #"none": IgnoreLeakyRelu
        #                          }
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = ConditionalDoubleVectorCouplingBlock(in_channels,
                                                             cond_channels,
                                                             hidden_dim,
                                                             hidden_depth)
        self.activation = InvLeakyRelu() #__possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h, xcond)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)



#--------------------------------------------------------------------------------------------------------------------
class Shuffle(nn.Module):
    """
        TODO: Annotate
    """
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]


#--------------------------------------------------------------------------------------------------------------------
class InvLeakyRelu(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        scaling = (input >= 0).to(input) + (input < 0).to(input) * self.alpha
        h = input * scaling
        return h, 0.0

    def reverse(self, input):
        scaling = (input >= 0).to(input) + (input < 0).to(input) * self.alpha
        h = input / scaling
        return h



#--------------------------------------------------------------------------------------------------------------------
class ConditionalFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""

    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels                  # the size of the input. This should be divisible by 2.
        self.cond_channels = cond_channels              # the size of the conditional H(y).
        self.mid_channels = hidden_dim                  # the dimension of the hidden layers for s and t.
        self.num_blocks = hidden_depth                  # number of hidden layers / depth in s_theta and t_theta respectively
        self.n_flows = n_flows                          # number of cINN blocks in our final network
        self.conditioning_option = conditioning_option  # how the conditioning y is handled. Possible values: none, sequential, parallel

        self.sub_layers = nn.ModuleList()               

        for flow in range(self.n_flows):
            self.sub_layers.append(
                ConditionalFlatDoubleCouplingFlowBlock(
                    self.in_channels, self.cond_channels, self.mid_channels,
                    self.num_blocks, activation=activation)
            )

    def forward(self, x, embedding, reverse=False):
        hconds = list()
        hcond = embedding[:, :, None, None]
        self.last_outs = []
        self.last_logdets = []
        for i in range(self.n_flows):
            hcond = embedding
            hconds.append(hcond)
            
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x, hconds[i])
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)