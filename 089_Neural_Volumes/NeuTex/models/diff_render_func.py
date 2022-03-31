import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def find_render_function(name):
    if name in "diffuse":
        return diffuse_render
    if name in ["microfacet", "specular"]:
        return specular_pipeline_render
    elif name == "nonmetallic":
        return nonmetallic_pipeline_render
    elif name == "metallic":
        return metallic_pipeline_render
    elif name == "radiance":
        return radiance_render
    raise RuntimeError("Unknown render function: " + name)


def find_render_function_dim(name):
    if name in "diffuse":
        return 6  # 3 + 3
    if name in ["microfacet", "specular"]:
        return 10  # 3 + 3 + 1 + 3
    elif name == "nonmetallic":
        return 8  # 3 + 3 + 1 + 1
    elif name == "metallic":
        return 9  # 3 + 3 + 1 + 1 + 1
    elif name == "radiance":
        return 3
    raise RuntimeError("Unknown render function: " + name)


def find_blend_function(name):
    if name == "alpha":
        return alpha_blend
    elif name == "alpha2":
        return alpha2_blend

    raise RuntimeError("Unknown blend function: " + name)


def alpha_blend(opacity, acc_transmission):
    return opacity * acc_transmission


def alpha2_blend(opacity, acc_transmission):
    """
    Consider a light collocated with the camera,
    multiply the transmission twice to simulate the light in a round trip
    """
    return opacity * acc_transmission * acc_transmission


def radiance_render(ray_feature, *args):
    return ray_feature[..., 1:4]


def metallic_pipeline_render(
    ray_feature, ray_pos, ray_dir, light_dir, light_intensity, clamp=False
):
    """
    Args:
        ray_feature: :math:`(*, F)`
        ray_pos: :math:`(*, 3)`
        ray_dir: :math:`(*, 3)`
        light_dir: :math:`(*, 3)`
        light_intensity: :math:`(*, 3)`
        All arguments should have the same shape until the last dimension, broadcasting 1 is allowed
    """
    assert ray_feature.shape[-1] == 10

    # sigma = ray_feature[..., 0]
    base_color = ray_feature[..., 1:4]
    normal = ray_feature[..., 4:7]
    roughness = ray_feature[..., [7, 7, 7]]
    specular = ray_feature[..., [8, 8, 8]]
    metallic = ray_feature[..., [9, 9, 9]]

    albedo = base_color * (1 - metallic)
    fresnel = specular * (1 - metallic) + base_color * metallic

    L = F.normalize(-1.0 * light_dir, dim=-1)
    V = F.normalize(-1.0 * ray_dir, dim=-1)
    H = F.normalize((L + V) / 2.0, dim=-1)
    N = normal

    NoV = torch.sum(N * V, dim=-1, keepdim=True)
    N = N * NoV.sign()

    NoL = (N * L).sum(-1, keepdim=True).clamp_(1e-6, 1)
    NoV = (N * V).sum(-1, keepdim=True).clamp_(1e-6, 1)
    NoH = (N * H).sum(-1, keepdim=True).clamp_(1e-6, 1)
    VoH = (V * H).sum(-1, keepdim=True).clamp_(1e-6, 1)

    alpha = roughness * roughness
    alpha2 = alpha * alpha
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 5.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)
    frac = frac0 * alpha2
    nom0 = NoH * NoH * (alpha2 - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k) + k
    nom = (4 * np.pi * nom0 * nom0 * nom1 * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom

    radiance = light_intensity

    NoV = torch.sum(N * V, dim=-1, keepdim=True)
    NoV = torch.gt(NoV, 0).float()

    color = (albedo / np.pi + spec) * NoL * radiance  # * NoV
    if clamp:
        color = color.clamp(min=0.0, max=1.0)
    else:
        color = color.clamp(min=0.0)

    return color


def diffuse_render(
    ray_feature, ray_pos, ray_dir, lightdir, light_intensity, clamp=False
):
    """
    Args:
        ray_pos: :math:`(N,Rays,Samples,3)`
        ray_dir: :math:`(N,Rays/1,Samples/1,3)
        ray_feature: :math:`(N,Rays,Samples,features)`
        lightdir: :math:`(N,Rays/1,Samples/1,3)
        light_intensity: :math:`(N,Rays/1,Samples/1,3)
    """

    num_channels = ray_feature.shape[-1]
    assert num_channels == 7
    albedo = ray_feature[..., 1:4].clamp(0.0, 1.0)
    normal = F.normalize(ray_feature[..., 4:7], dim=-1)

    L = F.normalize(-1.0 * lightdir, dim=-1)
    N = normal

    NoL = (N * L).sum(dim=-1, keepdim=True).abs().clamp_(1e-6, 1)
    # NoL = torch.sum(N * L, dim=-1, keepdim=True).clamp_(1e-6, 1)

    radiance = light_intensity

    color = (albedo / np.pi) * NoL * radiance
    if clamp:
        color = color.clamp(min=0.0, max=1.0)
    else:
        color = color.clamp(min=0.0)

    return color


def nonmetallic_pipeline_render(
    ray_feature, ray_pos, ray_dir, lightdir, light_intensity, clamp=False
):
    """
    Args:
        ray_pos: :math:`(N,Rays,Samples,3)`
        ray_dir: :math:`(N,Rays/1,Samples/1,3)
        ray_feature: :math:`(N,Rays,Samples,features)`
        lightdir: :math:`(N,Rays/1,Samples/1,3)
        light_intensity: :math:`(N,Rays/1,Samples/1,3)
    """

    assert ray_feature.shape[-1] == 9
    albedo = ray_feature[..., 1:4].clamp(0.0, 1.0)
    normal = F.normalize(ray_feature[..., 4:7], dim=-1)

    roughness = ray_feature[..., [7, 7, 7]].clamp(0.0, 1.0)
    fresnel = ray_feature[..., [8, 8, 8]].clamp(0.0, 1.0)

    L = F.normalize(-1.0 * lightdir, dim=-1)
    V = F.normalize(-1.0 * ray_dir, dim=-1)
    H = F.normalize((L + V) / 2.0, dim=-1)
    N = normal

    NoV = torch.sum(N * V, dim=-1, keepdim=True)
    N = N * NoV.sign()

    NoL = torch.sum(N * L, dim=-1, keepdim=True).clamp_(1e-6, 1)
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)
    NoH = torch.sum(N * H, dim=-1, keepdim=True).clamp_(1e-6, 1)
    VoH = torch.sum(V * H, dim=-1, keepdim=True).clamp_(1e-6, 1)

    alpha = roughness * roughness
    alpha2 = alpha * alpha
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 5.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)
    frac = frac0 * alpha2
    nom0 = NoH * NoH * (alpha2 - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k) + k
    nom = (4 * np.pi * nom0 * nom0 * nom1 * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom

    radiance = light_intensity

    NoV = torch.sum(N * V, dim=-1, keepdim=True)
    NoV = torch.gt(NoV, 0).float()

    color = (albedo / np.pi + spec) * NoL * radiance  # * NoV
    if clamp:
        color = color.clamp(min=0.0, max=1.0)
    else:
        color = color.clamp(min=0.0)

    return color


def specular_pipeline_render(
    ray_feature,
    ray_pos,
    ray_dir,
    lightdir,
    light_intensity,
    roughness=None,
    fresnel=None,
    clamp=False,
):
    """
    Args:
        ray_pos: :math:`(N,Rays,Samples,3)`
        ray_dir: :math:`(N,Rays/1,Samples/1,3)
        ray_feature: :math:`(N,Rays,Samples,features)`
        lightdir: :math:`(N,Rays/1,Samples/1,3)
        light_intensity: :math:`(N,Rays/1,Samples/1,3)
    """

    num_channels = ray_feature.shape[-1]
    assert num_channels in [7, 8, 11]  # roughness
    albedo = ray_feature[..., 1:4].clamp(0.0, 1.0)
    normal = F.normalize(ray_feature[..., 4:7], dim=-1)

    if roughness is None:
        if num_channels >= 8:
            roughness = ray_feature[..., [7, 7, 7]].clamp(0.0, 1.0)
        else:
            roughness = torch.ones_like(normal)

    if num_channels >= 11:
        fresnel = ray_feature[..., 8:11]
    elif fresnel is None:
        fresnel = 0.05

    L = F.normalize(-1.0 * lightdir, dim=-1)
    V = F.normalize(-1.0 * ray_dir, dim=-1)
    H = F.normalize((L + V) / 2.0, dim=-1)
    N = normal

    NoV = torch.sum(N * V, dim=-1, keepdim=True)
    N = N * NoV.sign()

    NoL = torch.sum(N * L, dim=-1, keepdim=True).clamp_(1e-6, 1)
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)
    NoH = torch.sum(N * H, dim=-1, keepdim=True).clamp_(1e-6, 1)
    VoH = torch.sum(V * H, dim=-1, keepdim=True).clamp_(1e-6, 1)

    alpha = roughness * roughness
    alpha2 = alpha * alpha
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 5.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)
    frac = frac0 * alpha2
    nom0 = NoH * NoH * (alpha2 - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k) + k
    nom = (4 * np.pi * nom0 * nom0 * nom1 * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom

    radiance = light_intensity

    NoV = torch.sum(N * V, dim=-1, keepdim=True)
    NoV = torch.gt(NoV, 0).float()

    color = (albedo / np.pi + spec) * NoL * radiance  # * NoV
    if clamp:
        color = color.clamp(min=0.0, max=1.0)
    else:
        color = color.clamp(min=0.0)

    return color


def simple_tone_map(color, gamma=2.2, exposure=1):
    return torch.pow(color * exposure + 1e-5, 1 / gamma).clamp_(0, 1)
