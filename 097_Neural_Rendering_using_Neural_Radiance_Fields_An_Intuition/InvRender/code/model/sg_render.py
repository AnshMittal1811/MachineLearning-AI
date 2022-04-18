import torch
import torch.nn.functional as F
import numpy as np

TINY_NUMBER = 1e-6


def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # same convetion as blender    
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), 
                            torch.sin(theta) * torch.sin(phi), 
                            torch.cos(phi)], dim=-1)    # [H, W, 3]
                            
    rgb = render_envmap_sg(lgtSGs, viewdirs)
    envmap = rgb.reshape((H, W, 3))
    return envmap


def render_envmap_sg(lgtSGs, viewdirs):
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
        (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    return rgb


def render_envmap(envmap, viewdirs):
    H, W = envmap.shape[:2]
    envmap = envmap.permute(2, 0, 1).unsqueeze(0)

    phi = torch.arccos(viewdirs[:, 2]).reshape(-1) - TINY_NUMBER
    theta = torch.atan2(viewdirs[:, 1], viewdirs[:, 0]).reshape(-1)

    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
    
    rgb = F.grid_sample(envmap, grid, align_corners=True)
    rgb = rgb.squeeze().permute(1, 0)
    return rgb


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    
    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    # for insurance
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)
    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


def norm_axis(x):
    return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)


def get_diffuse_visibility(points, normals, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=8):
    ########################################
    # sample dirs according to the light SG
    ########################################

    n_lobe = lgtSGLobes.shape[0]
    n_points = points.shape[0]
    light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)

    # add samples from SG lobes
    z_axis = torch.zeros_like(light_dirs).cuda()
    z_axis[:, :, 2] = 1

    light_dirs = norm_axis(light_dirs) #[num_lobes, 1, 3]
    U = norm_axis(torch.cross(z_axis, light_dirs))
    V = norm_axis(torch.cross(light_dirs, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sg_range = torch.zeros_like(sharpness).cuda()
    sg_range[:, :] = sharpness.min()
    r_phi_range = torch.arccos((-1.95 * sg_range) / sharpness + 1)
    r_theta = torch.rand(n_lobe, nsamp).cuda() * 2 * np.pi
    r_phi = torch.rand(n_lobe, nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + light_dirs * torch.cos(r_phi) # [num_lobe, num_sample, 3]
    sample_dir = sample_dir.reshape(-1, 3)

    ########################################
    # visibility
    ########################################
    input_dir = sample_dir.unsqueeze(0).expand(n_points, -1, 3)
    input_p = points.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
    normals = normals.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
    # vis = 0 if cos(n, w_i) < 0
    cos_term = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER

    # batch forward
    batch_size = 100000
    n_mask_dir = input_p[cos_term].shape[0]
    pred_vis = torch.zeros(n_mask_dir, 2).cuda()
    with torch.no_grad():
        for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
            pred_vis[indx] = VisModel(input_p[cos_term][indx], input_dir[cos_term][indx])

    _, pred_vis = torch.max(pred_vis, dim=-1)
    vis = torch.zeros(n_points, n_lobe * nsamp).cuda()
    vis[cos_term] = pred_vis.float()
    vis = vis.reshape(n_points, n_lobe, nsamp).permute(1, 2, 0)
    
    sample_dir = sample_dir.reshape(-1, nsamp, 3)
    weight_vis = torch.exp(lgtSGLambdas * (torch.sum(sample_dir * light_dirs, dim=-1, keepdim=True) - 1.))
    
    vis = torch.sum(vis * weight_vis, dim=1) / (torch.sum(weight_vis, dim=1) + TINY_NUMBER)

    # for debugging
    if torch.isnan(vis).sum() > 0:
        import ipdb; ipdb.set_trace()
        
    return vis


def get_specular_visibility(points, normals, viewdirs, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=24):
    ########################################
    # sample dirs according to the BRDF SG
    ########################################

    light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)

    n_dot_v = torch.sum(normals * viewdirs, dim=-1, keepdim=True)
    n_dot_v = torch.clamp(n_dot_v, min=0.)
    ref_dir = -viewdirs + 2 * n_dot_v * normals
    ref_dir = ref_dir.unsqueeze(1)
    
    # add samples from BRDF SG lobes
    z_axis = torch.zeros_like(ref_dir).cuda()
    z_axis[:, :, 2] = 1

    U = norm_axis(torch.cross(z_axis, ref_dir))
    V = norm_axis(torch.cross(ref_dir, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sharpness = torch.clip(sharpness, min=0.1, max=50)
    sg_range = torch.zeros_like(sharpness).cuda()
    sg_range[:, :] = sharpness.min()
    r_phi_range = torch.arccos((-1.90 * sg_range) / sharpness + 1)
    r_theta = torch.rand(ref_dir.shape[0], nsamp).cuda() * 2 * np.pi
    r_phi = torch.rand(ref_dir.shape[0], nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + ref_dir * torch.cos(r_phi)

    batch_size = 100000
    input_p = points.unsqueeze(1).expand(-1, nsamp, 3)
    input_dir = sample_dir
    normals = normals.unsqueeze(1).expand(-1, nsamp, 3)
    cos_term = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER
    n_mask_dir = input_p[cos_term].shape[0]
    pred_vis = torch.zeros(n_mask_dir, 2).cuda()
    with torch.no_grad():
        for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
            pred_vis[indx] = VisModel(input_p[cos_term][indx], input_dir[cos_term][indx])

    _, pred_vis = torch.max(pred_vis, dim=-1)
    vis = torch.zeros(points.shape[0], nsamp).cuda()
    vis[cos_term] = pred_vis.float()

    weight_vis = torch.exp(sharpness * (torch.sum(sample_dir * light_dirs, dim=-1) - 1.))
    inf_idx = torch.isinf(torch.sum(weight_vis, dim=-1))
    inf_sample = weight_vis[inf_idx]

    reset_inf = inf_sample.clone()
    reset_inf[torch.isinf(inf_sample)] = 1.0
    reset_inf[~torch.isinf(inf_sample)] = 0.0
    weight_vis[inf_idx] = reset_inf

    vis = torch.sum(vis * weight_vis, dim=-1) / (torch.sum(weight_vis, dim=-1) + TINY_NUMBER)

    # for debugging
    if torch.isnan(vis).sum() > 0:
        import ipdb; ipdb.set_trace()

    return vis


def render_with_all_sg(points, normal, viewdirs, lgtSGs, 
                       specular_reflectance, roughness, diffuse_albedo,
                       indir_lgtSGs=None,  VisModel=None):

    M = lgtSGs.shape[0]
    dots_shape = list(normal.shape[:-1])

    # direct light
    lgtSGs = lgtSGs.unsqueeze(0).expand(dots_shape + [M, 7])  # [dots_shape, M, 7]
    ret = render_with_sg(points, normal, viewdirs, lgtSGs, 
                         specular_reflectance, roughness, diffuse_albedo, 
                         comp_vis=True, VisModel=VisModel)

    # indirct light
    indir_rgb = torch.zeros_like(points).cuda()
    if indir_lgtSGs is not None:
        indir_rgb = render_with_sg(points, normal, viewdirs, indir_lgtSGs, 
                                   specular_reflectance, roughness, diffuse_albedo,
                                   comp_vis=False)['sg_rgb']
    ret.update({'indir_rgb': indir_rgb})
    return ret

    
#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg(points, normal, viewdirs, 
                   lgtSGs, specular_reflectance, roughness, diffuse_albedo,
                   comp_vis=True, VisModel=None):
    '''
    :param points: [batch_size, 3]
    :param normal: [batch_size, 3]; ----> camera; must have unit norm
    :param viewdirs: [batch_size, 3]; ----> camera; must have unit norm
    :param lgtSGs: [batch_size, M, 7]
    :param specular_reflectance: [1, 1]; 
    :param roughness: [batch_size, 1]; values must be positive
    :param diffuse_albedo: [batch_size, 3]; values must lie in [0,1]
    '''

    M = lgtSGs.shape[1]
    dots_shape = list(normal.shape[:-1])

    ########################################
    # light
    ########################################

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) # sharpness
    origin_lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    
    ########################################
    # specular color
    ########################################
    normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
    viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3]).detach()  # [dots_shape, M, 3]
    
    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 2. / (roughness * roughness * roughness * roughness)  # [dots_shape, 1]
    brdfSGLambdas = inv_roughness_pow4.unsqueeze(1).expand(dots_shape + [M, 1])
    mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape + [3])  # [dots_shape, 1] ---> [dots_shape, 3]
    brdfSGMus = mu_val.unsqueeze(1).expand(dots_shape + [M, 3])

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus  # [..., M, 3]

    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)

    specular_reflectance = specular_reflectance.unsqueeze(1).expand(dots_shape + [M, 3])
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    k = k.unsqueeze(1).expand(dots_shape + [M, 1])
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    vis_shadow = torch.zeros(dots_shape[0], 3).cuda()
    if comp_vis:
        # light SG visibility
        light_vis = get_diffuse_visibility(points, normal[:, 0, :], VisModel,
                                           lgtSGLobes[0], lgtSGLambdas[0], nsamp=32)
        light_vis = light_vis.permute(1, 0).unsqueeze(-1).expand(dots_shape +[M, 3])

        # BRDF SG visibility
        brdf_vis = get_specular_visibility(points, normal[:, 0, :], viewdirs[:, 0, :], 
                                           VisModel, warpBrdfSGLobes[:, 0], warpBrdfSGLambdas[:, 0], nsamp=16)
        brdf_vis = brdf_vis.unsqueeze(-1).unsqueeze(-1).expand(dots_shape + [M, 3])

        # using brdf vis if sharper
        # vis_brdf_mask = (warpBrdfSGLambdas > lgtSGLambdas).expand(dots_shape + [M, 3])
        # spec_vis = torch.zeros(dots_shape + [M, 3]).cuda()
        # spec_vis[vis_brdf_mask] = brdf_vis[vis_brdf_mask]
        # spec_vis[~vis_brdf_mask] = light_vis[~vis_brdf_mask]
        # vis_shadow = torch.mean(spec_vis, axis=1).squeeze()
        lgtSGMus = origin_lgtSGMus * brdf_vis
        vis_shadow = torch.mean(light_vis, axis=1).squeeze()
    else:
        lgtSGMus = origin_lgtSGMus

    # multiply with light sg
    final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                         warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    # [..., M, K, 3]
    specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    specular_rgb = specular_rgb.sum(dim=-2)
    specular_rgb = torch.clamp(specular_rgb, min=0.)


    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    # diffuse visibility
    if comp_vis:
        lgtSGMus = origin_lgtSGMus * light_vis
    else:
        lgtSGMus = origin_lgtSGMus 

    diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).expand(dots_shape + [M, 3])
    # multiply with light sg
    final_lobes = lgtSGLobes
    final_lambdas = lgtSGLambdas
    final_mus = lgtSGMus * diffuse

    # now multiply with clamped cosine, and perform hemisphere integral
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
                    final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    diffuse_rgb = diffuse_rgb.sum(dim=-2)
    diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    # for debugging
    if torch.isnan(specular_rgb).sum() > 0:
        import ipdb; ipdb.set_trace()
    if torch.isnan(diffuse_rgb).sum() > 0:
        import ipdb; ipdb.set_trace()

    # combine diffue and specular rgb
    rgb = specular_rgb + diffuse_rgb
    ret = {'sg_rgb': rgb,
           'sg_specular_rgb': specular_rgb,
           'sg_diffuse_rgb': diffuse_rgb,
           'vis_shadow': vis_shadow}

    return ret
