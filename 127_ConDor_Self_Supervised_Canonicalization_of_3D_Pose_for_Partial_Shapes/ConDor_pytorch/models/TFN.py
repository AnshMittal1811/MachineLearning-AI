import torch
from utils.group_points import GroupPoints
from spherical_harmonics.spherical_cnn import torch_fibonnacci_sphere_sampling, SphericalHarmonicsEval, SphericalHarmonicsCoeffs
from spherical_harmonics.kernels import SphericalHarmonicsGaussianKernels, ShGaussianKernelConv
from models.layers import MLP, MLP_layer, set_sphere_weights, apply_layers
from utils.pooling import kd_pooling_1d

class TFN(torch.nn.Module):
    """
    TFN layer for prediction in Pytorch
    """
    def __init__(self, sphere_samples = 64, bn_momentum = 0.75, mlp_units = [[32, 32], [64, 64], [128, 256]], l_max = [3, 3, 3], l_max_out = [3, 3, 3], num_shells = [3, 3, 3]):
        super(TFN, self).__init__()
        self.l_max = l_max
        self.l_max_out = l_max_out
        self.num_shells = num_shells
        self.gaussian_scale = []
        for i in range(len(self.num_shells)):
            self.gaussian_scale.append(0.69314718056 * ((self.num_shells[i]) ** 2))
        self.radius = [0.2, 0.40, 0.8]
        self.bounded = [True, True, True]

        self.S2 = torch_fibonnacci_sphere_sampling(sphere_samples)

        self.num_points = [1024, 256, 64, 16]
        self.patch_size = [32, 32, 32]

        self.spacing = [0, 0, 0]
        self.equivariant_units = [32, 64, 128]
        self.in_equivariant_channels = [[6, 13, 12, 9], [387, 874, 1065, 966], [771, 1738, 2121, 1926]]

        self.mlp_units = mlp_units
        self.in_mlp_units = [32, 64, 128]
        self.bn_momentum = bn_momentum

        self.grouping_layers = []
        self.kernel_layers = []
        self.conv_layers = []
        self.eval = []
        self.coeffs = []

        for i in range(len(self.radius)):
            gi = GroupPoints(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            self.grouping_layers.append(gi)

            ki = SphericalHarmonicsGaussianKernels(l_max=self.l_max[i],
                                                   gaussian_scale=self.gaussian_scale[i],
                                                   num_shells=self.num_shells[i],
                                                   bound=self.bounded[i])
            ci = ShGaussianKernelConv(l_max=self.l_max[i], l_max_out=self.l_max_out[i])

            self.kernel_layers.append(ki)
            self.conv_layers.append(ci)

        self.conv_layers = torch.nn.Sequential(*self.conv_layers)
        self.kernel_layers = torch.nn.Sequential(*self.kernel_layers)
        self.mlp = []
        self.equivariant_weights = []
        self.bn = []

        for i in range(len(self.radius)):
            self.bn.append(torch.nn.BatchNorm2d(self.equivariant_units[i], momentum=self.bn_momentum))
            types = [str(l) for l in range(self.l_max_out[i] + 1)]
            self.equivariant_weights.append(set_sphere_weights(self.in_equivariant_channels[i], self.equivariant_units[i], types=types))
            self.mlp.append(MLP_layer(self.in_mlp_units[i], self.mlp_units[i], bn_momentum = self.bn_momentum))

        self.mlp = torch.nn.Sequential(*self.mlp)
        self.bn = torch.nn.Sequential(*self.bn)
        self.equivariant_weights = torch.nn.Sequential(*self.equivariant_weights)


        
    def forward(self, x):
        """
        Input:
            x - [B, N, 3] - Point cloud with batch dim as B and num points as N
        Returns:
            TFN features - F
        """

        points = [x]
        grouped_points = []
        kernels = []

        num_points_ = self.num_points        
        num_points_[0] = x.shape[1]

        for i in range(len(self.radius)):
            # Down sampling points for different resolutions
            pi = kd_pooling_1d(points[-1], int(num_points_[i] / num_points_[i + 1]))
            points.append(pi)

        yzx = []
        for i in range(len(points)):
            yzx_i = torch.stack([points[i][..., 1], points[i][..., 2], points[i][..., 0]], dim=-1)
            yzx.append(yzx_i.unsqueeze(-1))

        for i in range(len(self.radius)):
            
            # Finding nearest neighbors of each point to compute graph features
            gi = self.grouping_layers[i]({"source points": points[i], "target points": points[i + 1]})
            # gi["patches source"] - nearest neighbor points - B, N, K, 3
            # gi["patches dist source"] - distance to nearest neighbors - B, N, K
            
            # Computing kernels for patch neighbors
            ki = self.kernel_layers[i]({"patches": gi["patches source"], "patches dist": gi["patches dist source"]})

            # Storing outputs
            grouped_points.append(gi)
            kernels.append(ki)

        y = {'0': torch.ones((x.shape[0], x.shape[1], 1, 1)).type_as(x)}
        
        for i in range(len(self.radius)):
            y["source points"] = points[i]
            y["target points"] = points[i + 1]
            y["patches idx"] = grouped_points[i]["patches idx source"]
            y["patches dist source"] = grouped_points[i]["patches dist source"]
            y["kernels"] = kernels[i]



            if '1' in y:
                y['1'] = torch.cat([y['1'], yzx[i]], dim=-1)
            else:
                y['1'] = yzx[i]

            y = self.conv_layers[i](y)

            if '1' in y:
                y['1'] = torch.cat([y['1'], yzx[i + 1]], dim=-1)
            else:
                y['1'] = yzx[i + 1]

            # print("y[1]: before", y["1"].shape)
            y = apply_layers(y, self.equivariant_weights[i]) # B, d, 2*l + 1, C
            # print("y[1]: after", y["1"].shape)
            
            # print("Shape after equivariant layers:")
            # for key in y:
            #     print(key, " ", y[key].shape) 

            # Inverse Spherical Harmonic Transform
            y = SphericalHarmonicsEval(l_max=self.l_max_out[i], base=self.S2).compute(y)
            # print(y.shape)
            y = y.permute(0, 3, 1, 2)
            y = self.bn[i](y)
            y = torch.nn.ReLU(True)(y)
            y = y.permute(0, 2, 3, 1)
            y = self.mlp[i](y)
            # print(y.shape, "MLP")
            if i < len(self.radius) - 1:
                # Spherical Harmonic Transform
                y = SphericalHarmonicsCoeffs(l_max=self.l_max_out[i], base=self.S2).compute(y)

        
        # print(y.shape) # B, 16, 64, 256
        # 64 sphere samples
        F = torch.max(y, dim=1, keepdims=False).values # B, samples, feature_dim

        return F


if __name__ == "__main__":

    x = torch.randn((2, 1024, 3)).cuda()

    model = TFN().cuda()
    F = model(x)
    print(F.shape)