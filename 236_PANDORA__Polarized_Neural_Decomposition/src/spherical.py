# From https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/spherical_harmonics.py
# From https://github.com/sxyu/svox2/blob/master/svox2/utils.py#L115

from os import makedirs
import torch
import sys

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np


# SH

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10
def eval_sh_bases(L : int, m : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """

    if L == 0:
        return SH_C0 * torch.ones(dirs.shape[:-1], device=dirs.device, dtype=dirs.dtype)
    if L == 1:
        if (L == 1):
            if (m == -1):
                return -SH_C1 * dirs[...,1];
            if (m == 0):
                return SH_C1 * dirs[...,2];
            if (m == 1):
                return -SH_C1 * dirs[...,0];
    if L == 2:
        # x, y, z = dirs.unbind(-1)
        # xx, yy, zz = x * x, y * y, z * z
        # xy, yz, xz = x * y, y * z, x * z
        if (m == -2):
            return SH_C2[0] * (dirs[...,0]*dirs[...,1]);
        if (m == -1):
            return SH_C2[1] * (dirs[...,1]*dirs[...,2]);
        if (m == 0):
            return SH_C2[2] * (2.0 * dirs[...,2]*dirs[...,2] - dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == 1):
            return SH_C2[3] * (dirs[...,0]*dirs[...,2]);
        if (m == 2):
            return SH_C2[4] * (dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);

    if L == 3:
        # x, y, z = dirs.unbind(-1)
        # xx, yy, zz = x * x, y * y, z * z
        # xy, yz, xz = x * y, y * z, x * z
        if (m == -3):
            return SH_C3[0] * dirs[...,1] * (3 * dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == -2):
            return SH_C3[1] * dirs[...,0]*dirs[...,1] * dirs[...,2];
        if (m == -1):
            return SH_C3[2] * dirs[...,1] * (4 * dirs[...,2]*dirs[...,2] - dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == 0):
            return SH_C3[3] * dirs[...,2] * (2 * dirs[...,2]*dirs[...,2] - 3 * dirs[...,0]*dirs[...,0] - 3 * dirs[...,1]*dirs[...,1]);
        if (m == 1):
            return SH_C3[4] * dirs[...,0] * (4 * dirs[...,2]*dirs[...,2] - dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == 2):
            return SH_C3[5] * dirs[...,2] * (dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == 3):
            return SH_C3[6] * dirs[...,0] * (dirs[...,0]*dirs[...,0] - 3 * dirs[...,1]*dirs[...,1]);

    if L == 4:
        # x, y, z = dirs.unbind(-1)
        # xx, yy, zz = x * x, y * y, z * z
        # xy, yz, xz = x * y, y * z, x * z
        if (m == -4):
            return SH_C4[0] * dirs[...,0]*dirs[...,1] * (dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == -3):
            return SH_C4[1] * dirs[...,1]*dirs[...,2] * (3 * dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]);
        if (m == -2):
            return SH_C4[2] * dirs[...,0]*dirs[...,1] * (7 * dirs[...,2]*dirs[...,2] - 1);
        if (m == -1):
            return SH_C4[3] * dirs[...,1]*dirs[...,2] * (7 * dirs[...,2]*dirs[...,2] - 3);
        if (m == 0):
            return SH_C4[4] * (dirs[...,2]*dirs[...,2] * (35 * dirs[...,2]*dirs[...,2] - 30) + 3);
        if (m == 1):
            return SH_C4[5] * dirs[...,0]*dirs[...,2] * (7 * dirs[...,2]*dirs[...,2] - 3);
        if (m == 2):
            return SH_C4[6] * (dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]) * (7 * dirs[...,2]*dirs[...,2] - 1);
        if (m == 3):
            return SH_C4[7] * dirs[...,0]*dirs[...,2] * (dirs[...,0]*dirs[...,0] - 3 * dirs[...,1]*dirs[...,1]);
        if (m == 4):
            return SH_C4[8] * (dirs[...,0]*dirs[...,0] * (dirs[...,0]*dirs[...,0] - 3 * dirs[...,1]*dirs[...,1]) - dirs[...,1]*dirs[...,1] * (3 * dirs[...,0]*dirs[...,0] - dirs[...,1]*dirs[...,1]));





# from https://scipython.com/book/chapter-8-scipy/examples/visualizing-the-spherical-harmonics/
def plot_sph_harm(l_range,
                  sph_func, 
                  label=''):
    theta = torch.linspace(0, np.pi, 100)
    phi = torch.linspace(0, 2*np.pi, 100)
    theta, phi = torch.meshgrid(theta, phi)

    # The Cartesian coordinates of the unit sphere
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    xyz = torch.cat((x[...,None], y[...,None], z[...,None]), dim=-1)

    fig = plt.figure()

    sub_ctr = 1

    # all_sh = eval_sh_bases(basis_dim=25, dirs=xyz)

    for l_ind, l in enumerate(l_range):
        if l < 3:
            m_range = range(0, l+1)
        else:
            m_range = [0, l//2, l]
        for m_ind,m in enumerate(m_range):
            # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
            # fcolors = sph_func(l, m, theta, phi) # 100 x 100

            fcolors = sph_func(l, m, xyz)

            # fmax, fmin = fcolors.max(), fcolors.min()
            fmax, fmin = 0.75, -0.75
            fcolors = np.clip((fcolors - fmin)/(fmax - fmin),0,1)

            ax = fig.add_subplot(len(l_range),3, sub_ctr, projection='3d')
            ax.plot_surface(x.numpy(), y.numpy(), z.numpy(),  
                            rstride=1, cstride=1, 
                            facecolors=cm.RdBu(fcolors.numpy()))
            ax.set_xlim(np.array([-1,1])*.6)
            ax.set_ylim(np.array([-1,1])*.6)
            ax.set_zlim(np.array([-1,1])*.6)
            # Turn off the axis planes
            ax.set_axis_off()
            ax.set_box_aspect([1,1,1])
            ax.set_title(f'l={l}, m={m}')
            # ax.set_title(f'{fmin:.2f}, {fmax:.2f}')
            sub_ctr += 1
        # add empyt plots
        while m_ind < 2:
            sub_ctr += 1
            m_ind += 1
            
    # fig.suptitle(label)
    plt.tight_layout()
    makedirs(f'viz/sph_harm', exist_ok=True)
    plt.savefig(f'viz/sph_harm/{label}.png',dpi=200)


def get_ide_element(l, m, dirs, roughness):
    return torch.exp(-roughness*l*(l+1)/2)*eval_sh_bases(l, m, dirs)

def get_ide(dirs, roughness, L=2):
    """
    Integrated directional encodings as in Ref Nerf
    https://arxiv.org/abs/2112.03907
    Args:
        theta: Elevation angle torch tensor
        phi: Azimuth angle  torch tensor
        roughness: 1/K torch scalar (0-dim tensor)
    Note: Theta, phi should have same shape 
    Returns:
        tensor of shape [*theta.shape, 2**L + L -1]
    """

    return torch.cat([ torch.stack([ get_ide_element(2**l_exp, m, dirs, roughness) \
                                     for m in range(-2**l_exp, 2**l_exp +1) 
                                    ], dim = -1)
                        for l_exp in range(0,L+1)
                        ], dim=-1)


if __name__ == "__main__": 
    theta = torch.tensor([2.5,2.5,3.01,3.01])
    phi = torch.tensor([-1.4,1.4,2.3,-2.3])

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    dirs = torch.cat((x[...,None], y[...,None], z[...,None]),dim=-1)

    # L = 1

    # ide_test = get_ide(dirs=dirs,roughness=torch.tensor(0.01),L=L)
    # est_size = 2**(L+2) + L - 1
    # print(ide_test, ide_test.shape, est_size)

    # For reproducing Fig. 3 in Ref NeRF https://arxiv.org/abs/2112.03907
    plot_sph_harm(l_range=[1,2,4],
                  sph_func=eval_sh_bases,
                  label='pytorch_no_roughness')
    
    # Plot after expectation with different roughness values
    plot_sph_harm(l_range=[1,2,4],
                  sph_func=lambda l, m, dirs \
                                  :get_ide_element(l, m, dirs, 
                                                    torch.tensor(0.05)),
                  label='pytorch_alpha_0p05')

    plot_sph_harm(l_range=[1,2,4],
                  sph_func=lambda l, m, dirs \
                                  :get_ide_element(l, m, dirs, 
                                                    torch.tensor(0.12)),
                  label='pytorch_alpha_0p12')

    plot_sph_harm(l_range=[1,2,4],
                  sph_func=lambda l, m, dirs \
                                  :get_ide_element(l, m, dirs, 
                                                    torch.tensor(0.20)),
                  label='pytorch_alpha_0p20')