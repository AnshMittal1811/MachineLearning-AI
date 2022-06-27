from typing import List
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .utils import Timer
from .harmonic_embedding import HarmonicEmbedding

def _xavier_init(param):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    nn.init.xavier_uniform_(param.data)

def _uniform_init(param):
    dim = param.shape[-1]
    std = 1 / math.sqrt(dim)
    param.data.uniform_(-std, std)

# The modules are the revision from those in implicit_function.py

class NerfExperts(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        n_experts: int = 100,
        append_xyz: List[int] = (5,),
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp_xyz = MLPWithInputSkips(
            n_experts,
            n_layers_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            input_skips=append_xyz,
        )

        self.intermediate_linear = Experts(
            n_experts,
            n_hidden_neurons_xyz, 
            n_hidden_neurons_xyz
        )

        self.alpha_layer = Experts(n_experts, n_hidden_neurons_xyz, 1)
        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.alpha_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        self.color_layer = nn.ModuleList([
            Experts(n_experts, n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir),
            Experts(n_experts, n_hidden_neurons_dir, 3),
        ])


    def _get_colors(
            self, 
            features: torch.Tensor,
            directions: torch.Tensor,
            index: torch.Tensor,
        ):
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = torch.cat(
            (
                self.harmonic_embedding_dir(directions),
                directions,
            ),
            dim=-1,
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        intermediate_feat = self.intermediate_linear(features, index)
        color_layer_input = torch.cat(
            [intermediate_feat, rays_embedding], dim=-1
        )
        color = F.relu(self.color_layer[0](color_layer_input, index))
        color = self.color_layer[1](color, index)
        return color

    def forward(
        self,
        points, 
        directions,
        index,
        **kwargs,
    ):
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = torch.cat(
            (self.harmonic_embedding_xyz(points), points),
            dim=-1,
        )
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz, index)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        alpha = self.alpha_layer(features, index)
        # rays_densities.shape = [minibatch x ... x 1] in [0-1]

        colors = self._get_colors(features, directions, index)
        # rays_colors.shape = [minibatch x ... x 3] in [0-1]

        output = torch.cat([colors, alpha], dim=-1)
        output = torch.sigmoid(output)
        return output

class MLPWithInputSkips(nn.Module):
    def __init__(
        self,
        n_experts: int,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: List[int] = (),
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = output_dim
            experts = Experts(n_experts, dimin, dimout)
            layers.append(experts)
        self.layers = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)
    
    def forward(self, x, z, index):
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        out = x
        for i, layer in enumerate(self.layers):
            if i in self._input_skips:
                out = torch.cat([out, z], dim=-1)
            out = layer(out, index)
            out = F.relu(out)
        return out

class Experts(nn.Module):
    '''
    Single MLP layer with multiple experts
    '''
    def __init__(
        self,
        n_experts: int,
        in_features: int,
        out_features: int,    
    ):
        super().__init__()
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.empty(n_experts, in_features, out_features))
        self.bias  = nn.Parameter(torch.empty(n_experts, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_experts):
            _xavier_init(self.weight[i])

        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward_forloop(self, inputs, index):
        outputs = torch.zeros(inputs.shape[0], self.bias.shape[1], device=inputs.device)
        for i in range(self.n_experts):
            idx_i = index == i
            in_i = inputs[idx_i] 
            out_i = in_i @ self.weight[i] + self.bias[i]
            outputs[idx_i] = out_i
        return outputs

    def forward(self, inputs, index):
        '''
        Args
            inputs: (b, in_feat)
            index: (b, ), max < n_experts
        Return 
            outputs: (b, out_feat)
        '''
        weight_sample = self.weight[index] #(b, in_feat, out_feat)
        bias_sample = self.bias[index]
        # prod = torch.bmm(inputs.unsqueeze(1), weight_sample).squeeze(1)
        prod = torch.einsum('...i,...io->...o', inputs, weight_sample)
        outputs = prod + bias_sample
        return outputs
