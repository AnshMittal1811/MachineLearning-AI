"""
Sphere tracer.
"""
import time
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils.math_utils as math_utils
import utils.math_utils_torch as mut
import modules
from modules_sdf import SDFIBRNet
import utils.diff_operators as diff_operators
import utils.common_utils as common_utils
from torchmeta.modules.utils import get_subdict


# Considered good enough for surface.
# Based on: https://arxiv.org/abs/2003.09852 [Lipman 2020]
SDF_THRESHOLD = 5e-5
SDF_THRESHOLD_RELAXED = SDF_THRESHOLD * 1e2


def _sdf(decoder: SDFIBRNet, coords: torch.Tensor, times: torch.Tensor, batch_size: int, params=None):
    """
    Shortcut method to compute SDF.
    """
    output = modules.batch_decode(
        decoder.decoder_sdf,                    # Only use sdf decoder - more efficient.
        {'coords': coords[None, ...], 'time': times[None, ...]},
        batch_size, out_feature_slice=slice(0, 1),
        return_inputs=False, params=get_subdict(params, 'decoder_sdf'))
    return output['model_out'][0, ...]


def get_rays_all(resolution: torch.Tensor,
                 model_matrix: torch.Tensor,
                 view_matrix: torch.Tensor,
                 projection_matrix: torch.Tensor):
    """
    Builds rays for all pixels of a given camera.
    """
    ndc = get_pixels_ndc(resolution)
    return get_rays(ndc, model_matrix, view_matrix, projection_matrix)


def get_pixels_ndc(resolution: torch.Tensor):
    """
    Gets pixel NDC coords.
    """
    W, H = resolution

    # Target the centers of pixels => 0.5px shift in the -1,1 NDC space.
    W_offset, H_offset = 2 * 0.5 / W.float(), 2 * 0.5 / H.float()
    # W_offset, H_offset = 0, 0

    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1 + W_offset, 1 - W_offset, W, device=W.device),
                          torch.linspace(1 - H_offset, -1 + H_offset, H, device=W.device))

    return torch.stack([x.T.flatten(), y.T.flatten()], -1)


def pixels_to_ndc(px: torch.Tensor, resolution: torch.Tensor):
    """
    Convert pixels to NDC coords.
    """

    # Map to 0-1.
    px_rel = px.float() / (resolution.float() - 1)

    # Target the centers of pixels => 0.5px shift in the -1,1 NDC space.
    offset = 2 * 0.5 / resolution.float()

    # Map to [-1+offset, 1-offset]
    new_min = -1 + offset
    new_max = 1 - offset
    return px_rel * (new_max - new_min) + new_min


def get_rays(pixels_ndc: torch.Tensor,
             model_matrix: torch.Tensor,
             view_matrix: torch.Tensor,
             projection_matrix: torch.Tensor):
    """
    Builds rays for given camera.
    """

    # https://stackoverflow.com/questions/2354821/raycasting-how-to-properly-apply-a-projection-matrix
    # Width x Height x 4 ([x, y, 0, 1] for each pixel).
    if pixels_ndc.shape[-1] != 4:
        pixels_ndc = torch.cat((
            pixels_ndc,
            torch.zeros_like(pixels_ndc[..., :1]),
            torch.ones_like(pixels_ndc[..., :1])
        ), -1)

    # Inverse transform from NDC pixels to camera space.
    screen_coords_in_cam_space = mut.transform_vectors(torch.inverse(projection_matrix), pixels_ndc)

    # Build ray directions in camera space.
    rays_d_cam = mut.normalize_vecs(screen_coords_in_cam_space[..., :3])

    # Apply camera rotation to get world coordinates.
    model_view_matrix = view_matrix @ model_matrix
    rays_d = mut.transform_vectors(torch.inverse(model_view_matrix)[..., :3, :3], rays_d_cam)
    rays_d = mut.normalize_vecs(rays_d)

    # Determine ray origin from camera position.
    cam_pos = torch.inverse(model_view_matrix)[..., :3, 3]
    rays_o = cam_pos.expand_as(rays_d)
    return rays_o, rays_d


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor):
    """
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    # NDC bounds.
    bb_min = [-1, -1, -1]
    bb_max = [1, 1, 1]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    rays_o = rays_o.detach()
    rays_d = rays_d.detach()

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(-1, 1), tmax.reshape(-1, 1)


def get_ray_limits_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor):
    """
    Intersects rays with the [-1, 1] unit sphere.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    """

    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)
    radius = 1
    radius2 = radius**2

    # geometric solution
    L = -rays_o.detach()
    tca = mut.torch_dot(L, rays_d.detach())
    is_valid[tca < 0] = False
    d2 = mut.torch_dot(L, L) - tca * tca
    is_valid[d2 > radius2] = False
    thc = (radius2 - d2).sqrt()
    tmin = tca - thc
    tmax = tca + thc

    # If we are inside, start immediately.
    tmin = torch.clamp_min(tmin, 0)

    # If we are past, mark invalid.
    is_valid[tmax < 0] = False

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(-1, 1), tmax.reshape(-1, 1)


def _raytrace_from_t(decoder: SDFIBRNet,
                     rays_o: torch.Tensor,
                     rays_d: torch.Tensor,
                     times: torch.Tensor,
                     t_0: torch.Tensor,
                     t_min: torch.Tensor,
                     t_max: torch.Tensor,
                     step_factor: float,
                     batch_size: int,
                     debug=False,
                     debug_gui=False,
                     params=None):
    """
    One-directional sphere tracing from a given starting t0.
    Returns argmin t and min |sdf|.
    """
    # Cloning here is really important!
    current_t = t_0.clone().detach()

    # Remember last SDF with correct sign.
    # This improves stability for the secant algorithm if the SDF sign oscilaltes.
    last_good_t = current_t.clone().detach()
    #last_good_sign = torch.zeros_like(current_t).to(decoder.device).requires_grad_(False)

    # Stop converged and diverged rays.
    is_ray_active = torch.ones((current_t.shape[0],)).bool().to(decoder.device).requires_grad_(False)

    pos_z_vals = []
    sdf_vals = []
    step_factor_tensor = torch.Tensor([step_factor]).float().to(decoder.device)

    sdf = None
    for i in range(decoder.opt.rt_num_steps):

        # Query the network.
        coords = rays_o + current_t * rays_d
        if i == 0:
            # Compute SDF for all rays.
            sdf = _sdf(decoder, coords, times, batch_size, params=params)
        else:
            # Update SDF only for not done rays.
            sdf[is_ray_active, :] = _sdf(decoder, coords[is_ray_active, ...],
                                         times[is_ray_active, ...], batch_size,
                                         params=params)

        # Update last good t
        sign_t = torch.sign(sdf)[..., 0]
        is_sign_t_good = sign_t * torch.sign(step_factor_tensor) >= 0
        is_good_update = torch.logical_and(is_sign_t_good, is_ray_active)
        last_good_t[is_good_update] = current_t[is_good_update]

        # Stop if sdf < thr.
        is_converged = torch.abs(sdf) < SDF_THRESHOLD
        is_ray_active[is_converged[..., 0]] = False

        # Use SDF to update position.
        # Update only non-finished rays.
        current_t[is_ray_active, ...] = current_t[is_ray_active, ...] + sdf[is_ray_active, ...] * step_factor

        # Stop if outside NDC.
        is_outside_ndc = torch.logical_or(current_t < t_min, current_t > t_max)
        is_ray_active[is_outside_ndc[..., 0]] = False

        if debug:
            dbg_index = current_t.shape[0] // 2
            dbg_t = current_t[dbg_index].item()
            dbg_coords = coords[dbg_index].cpu().numpy()
            dbg_sdf = sdf[dbg_index].item()
            rt_type = 'forward' if step_factor >= 0 else 'backward'

            num_active = is_ray_active.sum().item()
            num_all = is_ray_active.shape[0]
            # print((f'...RT Step {rt_type} #{i+1} '
            #        + f' for {num_active} of {num_all} rays ({num_active / num_all * 100:.2f}%): '
            #        + f't = {dbg_t:.3f} | Pos = {dbg_coords} | SDF = {dbg_sdf:.3f}.'))

            pos_z_vals += [dbg_coords[2]]
            sdf_vals += [dbg_sdf]
            # if debug_gui:
            #     sdf_2d = sdf.reshape(dbg_res[1], dbg_res[0]).cpu()
            #     common_utils.make_contour_plot(sdf_2d)
            #     plt.show()

    if debug_gui:
        plt.plot(pos_z_vals)
        plt.plot(sdf_vals)
        plt.legend(['Z', 'SDF'])
        plt.show()

    # return current_t
    return last_good_t


def _raytrace_sectioning(decoder: SDFIBRNet,
                         rays_o: torch.Tensor,
                         rays_d: torch.Tensor,
                         times: torch.Tensor,
                         t_min: torch.Tensor,
                         t_max: torch.Tensor,
                         num_steps: int,
                         batch_size: int,
                         debug: bool = False,
                         params=None):
    """
    Step the ray between two t's and find the first SDF crossing.
    Returns t before and after crossing.
    """

    # The last t_i before crossing
    t_i = t_min.clone().detach()
    is_ray_active = torch.ones((t_min.shape[0],)).bool().to(decoder.device)

    # We will never go beyond MAX_DISTANCE.
    MAX_DISTANCE = 1.0

    # Step length.
    t_offsets = np.linspace(0, 1, num_steps)**2 * MAX_DISTANCE
    t_i1 = t_i + t_offsets[1]

    sdf = None
    sdf_prev = None
    total_crossings = 0
    for i in range(len(t_offsets)):

        # Query the network.
        current_t = t_min + t_offsets[i]
        coords = rays_o + current_t * rays_d
        if i == 0:
            # Compute SDF for all rays.
            sdf = _sdf(decoder, coords, times, batch_size, params=params)
        else:
            # Update SDF only for not done rays.
            sdf[is_ray_active, :] = _sdf(decoder, coords[is_ray_active, ...],
                                         times[is_ray_active, ...], batch_size,
                                         params=params)

        if i == 0:
            # First step.
            sdf_prev = sdf.clone()
            # Mark those already inside as done.
            is_ray_active[sdf_prev[..., 0] < 0] = False
            continue

        # Update those not yet updated.
        sign_prev = torch.sign(sdf_prev)
        sign_now = torch.sign(sdf)
        is_crossing = sign_prev != sign_now
        mark_first_crossing = torch.logical_and(is_ray_active[:, None], is_crossing)
        t_i[mark_first_crossing] = current_t[mark_first_crossing] - t_offsets[i]
        t_i1[mark_first_crossing] = current_t[mark_first_crossing]
        is_ray_active[mark_first_crossing[..., 0]] = False

        if debug and (i % 10 == 0 or mark_first_crossing.sum().item() > 0):
            total_crossings += mark_first_crossing.sum().item()
            num_active = is_ray_active.sum().item()
            num_all = is_ray_active.shape[0]
            print((f'...Sectioning step #{i+1} '
                   + f' for {num_active} of {num_all} rays ({num_active / num_all * 100:.2f}%):'
                   + f' Marked {mark_first_crossing.sum().item()} ray crossings.'))

        # Remember last SDF.
        sdf_prev[:] = sdf[:]

        # Terminate rays that went too far.
        is_too_far = current_t > t_max
        is_ray_active[is_too_far[..., 0]] = False

    if debug:
        print(f'Total crossings = {total_crossings}')

    return t_i, t_i1


@torch.no_grad()
def _raytrace_secant(decoder: SDFIBRNet,
                     rays_o: torch.Tensor,
                     rays_d: torch.Tensor,
                     times: torch.Tensor,
                     t_0: torch.Tensor,
                     t_1: torch.Tensor,
                     num_steps: int,
                     batch_size: int,
                     params=None):
    """
    Does bisection to find exact SDF crossing.
    Returns argmin t.
    https://www.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/
    """

    # Step length.
    mid_t = t_0.clone().detach()

    # Remember min SDF.
    # This is useful for highly non-linear SDFs.
    argmin_t = t_0.clone().detach()
    argmin_t_sdf = torch.zeros_like(t_0).fill_(1e20).to(decoder.device).requires_grad_(False)

    def sdf_t(t):
        return _sdf(decoder, rays_o + t * rays_d, times, batch_size, params=params)
    sdf_0 = sdf_t(t_0)
    sdf_1 = sdf_t(t_1)

    for i in range(num_steps):
        # Choose mid point.
        denom = (sdf_1 - sdf_0)
        mid_t = t_0 - sdf_0 * (t_1 - t_0) / (sdf_1 - sdf_0)

        # Replace NaN by average of t0 and t1
        nan_mask = denom.abs() < 1e-5
        mid_t[nan_mask] = (t_0[nan_mask] + t_1[nan_mask]) * 0.5

        # Compute mid sdf.
        sdf_mid = sdf_t(mid_t)

        # Remember minimum.
        is_new_min = torch.abs(sdf_mid) < torch.abs(argmin_t_sdf)
        argmin_t[is_new_min] = mid_t[is_new_min]
        argmin_t_sdf[is_new_min] = sdf_mid[is_new_min]

        # Left side.
        is_left = sdf_0 * sdf_mid < 0
        t_1[is_left] = mid_t[is_left]

        # Right side.
        is_right = sdf_1 * sdf_mid < 0
        t_0[is_right] = mid_t[is_right]

    return argmin_t


def _raytrace_surface(decoder: SDFIBRNet,
                      rays_o: torch.Tensor,
                      rays_d: torch.Tensor,
                      times: torch.Tensor,
                      t_min: torch.Tensor,
                      t_max: torch.Tensor,
                      batch_size: int,
                      debug=False,
                      debug_gui=False,
                      params=None):
    # mode = 'sinesdf'
    mode = 'idr'
    if mode == 'sinesdf':
        return _raytrace_surface_sinesdf(
            decoder=decoder,
            rays_o=rays_o,
            rays_d=rays_d,
            times=times,
            t_min=t_min,
            t_max=t_max,
            batch_size=batch_size,
            debug=debug,
            debug_gui=debug_gui,
            params=params)
    else:
        return _raytrace_surface_idr(
            decoder=decoder,
            rays_o=rays_o,
            rays_d=rays_d,
            times=times,
            t_min=t_min,
            t_max=t_max,
            batch_size=batch_size,
            debug=debug,
            debug_gui=debug_gui,
            params=params)


def _raytrace_surface_sinesdf(decoder: SDFIBRNet,
                              rays_o: torch.Tensor,
                              rays_d: torch.Tensor,
                              times: torch.Tensor,
                              t_min: torch.Tensor,
                              t_max: torch.Tensor,
                              batch_size: int,
                              debug=False,
                              debug_gui=False,
                              params=None):
    """
    Returns surface points and their validity.
    Code from NLR.
    """

    # Raytrace foward.
    if debug:
        print(f'Forward tracing for {rays_o.shape[0]} rays.')
    argmin_t_0 = _raytrace_from_t(
        decoder, rays_o, rays_d, times,
        t_min, t_min, t_max,
        decoder.opt.rt_step_alpha * 1,
        batch_size, debug, debug_gui,
        params=params)

    # Check which rays need 2nd pass
    min_sdf_0 = _sdf(decoder, rays_o + argmin_t_0 * rays_d, times, batch_size, params=params)
    needs_2nd_pass = torch.abs(min_sdf_0) >= SDF_THRESHOLD
    # If the did not leave the volume, and did not converge, then trace it again.
    needs_2nd_pass = torch.logical_and(needs_2nd_pass, torch.logical_and(argmin_t_0 >= t_min, argmin_t_0 <= t_max))
    needs_2nd_pass = needs_2nd_pass[..., 0]

    if decoder.opt.rt_bidirectional and needs_2nd_pass.any().item():

        # Try to resolve the divergents.
        rays_o_div = rays_o[needs_2nd_pass, ...]
        rays_d_div = rays_d[needs_2nd_pass, ...]
        times_div = times[needs_2nd_pass, ...]
        argmin_t_0_div = argmin_t_0[needs_2nd_pass, ...]
        t_min_div = t_min[needs_2nd_pass, ...]
        t_max_div = t_max[needs_2nd_pass, ...]

        # Raytrace backward.
        if debug:
            print(
                f'Backward tracing for {rays_o_div.shape[0]} rays ({rays_o_div.shape[0] / rays_o.shape[0] * 100:.1f}%).')
        argmin_t_1 = _raytrace_from_t(
            decoder, rays_o_div, rays_d_div, times_div,
            t_max_div, t_min_div, t_max_div,
            decoder.opt.rt_step_alpha * -1,
            batch_size, debug, debug_gui,
            params=params)

        # Remove rays that are too much crossed.
        t_diff = argmin_t_1 - argmin_t_0_div
        # These rays are completely wrong
        is_lost = t_diff < -1.0
        # Mark their t_diff zero so that they require close to no work further down.
        argmin_t_1[is_lost[..., 0]] = argmin_t_0_div[is_lost[..., 0]]

        # Sectioning.
        if debug:
            print(
                f'Running sectioning for {rays_o_div.shape[0]} rays ({rays_o_div.shape[0] / rays_o.shape[0] * 100:.1f}%).')
        # Resolve cross-switch
        sec_min_t = torch.min(argmin_t_0_div, argmin_t_1)
        sec_max_t = torch.max(argmin_t_0_div, argmin_t_1)
        sec_t0, sec_t1 = _raytrace_sectioning(decoder,
                                              rays_o_div, rays_d_div, times_div,
                                              sec_min_t, sec_max_t,
                                              decoder.opt.rt_num_section_steps,
                                              batch_size,
                                              debug=False,
                                              params=params)

        # Secant algorithm to find in between.
        if debug:
            print(
                f'Running secant algorithm for {rays_o_div.shape[0]} rays ({rays_o_div.shape[0] / rays_o.shape[0] * 100:.1f}%).')
        argmin_t_div = _raytrace_secant(decoder,
                                        rays_o_div, rays_d_div, times_div,
                                        sec_t0, sec_t1,
                                        decoder.opt.rt_num_secant_steps, batch_size,
                                        params=params)

        # Merge convergent and divergent
        argmin_t = argmin_t_0
        argmin_t[needs_2nd_pass] = argmin_t_div

        # Final coords
        coords = rays_o + argmin_t * rays_d

        if torch.isnan(argmin_t).any().item():
            print('We have NAN in raytracer!!!!')
            import pdb
            pdb.set_trace()
            raise Exception('NaN in raytracer!')

        # Check if finally converged
        min_sdf = _sdf(decoder, coords, times, batch_size, params=params)

    else:
        # Use the forward pass.
        argmin_t = argmin_t_0
        min_sdf = min_sdf_0
        coords = rays_o + argmin_t * rays_d

    # Relaxed convergence check.
    is_converged = torch.logical_and(
        torch.abs(min_sdf) < SDF_THRESHOLD_RELAXED,
        torch.logical_and(argmin_t >= t_min, argmin_t <= t_max)
    )[..., 0]

    return {
        'coords': coords,
        't': argmin_t,
        'sdf': min_sdf,
        'is_valid': is_converged,
    }


def _raytrace_surface_idr(decoder: SDFIBRNet,
                          rays_o: torch.Tensor,
                          rays_d: torch.Tensor,
                          times: torch.Tensor,
                          t_min: torch.Tensor,
                          t_max: torch.Tensor,
                          batch_size: int,
                          debug=False,
                          debug_gui=False,
                          params=None):
    """
    Alternative sphere-tracer using IDR code.
    """
    from utils.idr.idr_ray_tracing import IDRRayTracing
    ray_tracer = IDRRayTracing()
    ray_tracer.device = decoder.device
    ray_tracer.eval()

    def idr_sdf(x): return _sdf(decoder, x, times, batch_size, params=params)[..., 0]
    object_mask = torch.ones((rays_d.shape[0],)).bool().to(decoder.device)
    points, network_object_mask, dists = ray_tracer(
        sdf=idr_sdf, cam_loc=rays_o[None, ...], object_mask=object_mask, ray_directions=rays_d[None, ...])
    coords = points
    argmin_t = dists[..., None]
    min_sdf = _sdf(decoder, coords, times, batch_size, params=params)
    is_converged = network_object_mask

    return {
        'coords': coords,
        't': argmin_t,
        'sdf': min_sdf,
        'is_valid': is_converged,
    }


def _raytrace_surface_differentiable(decoder: SDFIBRNet,
                                     rays_o: torch.Tensor,
                                     rays_d: torch.Tensor,
                                     times: torch.Tensor,
                                     t_min: torch.Tensor,
                                     t_max: torch.Tensor,
                                     batch_size: int,
                                     debug=False,
                                     debug_gui=False,
                                     params=None):
    """
    Differentiable sphere tracing based on:
    https://arxiv.org/abs/2003.09852
    """
    # First run standard non-diff sphere tracing.
    with torch.no_grad():
        surface = _raytrace_surface(decoder,
                                    rays_o, rays_d, times,
                                    t_min, t_max,
                                    batch_size, debug, debug_gui,
                                    params=params)

    # Compute grad f_0
    gradient_0, output_0 = _compute_normals(decoder, surface['coords'], times, params=params)
    sdf_0 = output_0[0, ..., :1]

    # Compute Lemma 1 to get differentiable surface coords
    denom = mut.torch_dot(gradient_0, rays_d)
    denom_sign = torch.sign(denom)
    denom_sign[denom_sign == 0] = 1
    denom = denom_sign * torch.clamp_min(denom.abs(), 1e-2)
    coef = -1.0 / denom.detach()
    t_0 = surface['t']
    coords = rays_o + rays_d * (t_0 + coef[..., None] * sdf_0)

    # Output
    return {
        'coords': coords,
        'is_valid': surface['is_valid'],
        'sdf': sdf_0,
        't_min': surface['t']
    }


def _raytrace_differentiable_mask(decoder: SDFIBRNet,
                                  rays_o: torch.Tensor,
                                  rays_d: torch.Tensor,
                                  times: torch.Tensor,
                                  t_min: torch.Tensor,
                                  t_max: torch.Tensor,
                                  batch_size: int,
                                  params=None,
                                  t_min_computed=None,
                                  is_valid_mask=None):
    """
    Trace the closest t for the "Masked rendering"
    See Sec. 3.3 in https://arxiv.org/abs/2003.09852
    Also see supplement A.3.
    Uses 100 uniform steps to find the argmin.
    Note that the sdf is not abs here.
    """
    # Remember the argmin t and its sdf
    argmin_t = torch.zeros_like(t_min).to(decoder.device).requires_grad_(False)
    argmin_t_sdf = torch.zeros_like(t_min).to(decoder.device).fill_(1e20).requires_grad_(False)

    # The sampling part is not-differentiable.
    with torch.no_grad():
        # Uniform step size.
        dt = (t_max - t_min) / (decoder.opt.rt_num_mask_steps - 1)
        for i in range(decoder.opt.rt_num_mask_steps):
            # Query the network.
            current_t = t_min + i * dt
            coords = rays_o + current_t * rays_d
            sdf = _sdf(decoder, coords, times, batch_size, params=params)

            # Update argmin t
            argmin_t_sdf_new = torch.min(argmin_t_sdf, sdf)  # No abs()
            mask_updated = argmin_t_sdf_new != argmin_t_sdf
            argmin_t[mask_updated, ...] = current_t[mask_updated, ...]
            argmin_t_sdf = argmin_t_sdf_new

    # argmin_t[is_valid_mask] = t_min_computed[is_valid_mask]
    # argmin_t[torch.logical_not(is_valid_mask)] = t_min_computed[is_valid_mask].mean(0, keepdim=True)

    # Evaluate min sdf
    coords = rays_o + argmin_t * rays_d
    min_sdf = _sdf(decoder, coords, times, batch_size, params=params)

    # Compute soft mask (Eq. 7)
    mask = torch.sigmoid(-decoder.opt.rt_mask_alpha * min_sdf)

    return {
        'coords': coords,
        'sdf': min_sdf,
        'mask': mask,
    }


@torch.enable_grad()
def _compute_normals(decoder: SDFIBRNet, coords: torch.Tensor, times: torch.Tensor = None,
                     params=None):
    """
    Computes differentiable normals for the SDF.
    Will apply flow (T->T0) if needed and will differentiate wrt original coords at T.
    That means that the normals will be in coordinates of the current frame.
    """
    # Evaluate SDF
    if not coords.requires_grad:
        coords.requires_grad_(True)
    # If times are provided, the coords are local and we need to convert to key.
    output = decoder.decoder_sdf({'coords': coords[None, ...]}, params=get_subdict(params, 'decoder_sdf'))
    sdf = output['model_out'][..., :1]

    # Compute grad f_0
    gradient = diff_operators.gradient(sdf, coords)
    return gradient, output['model_out']


#######################
# Non-diff test code
#########################


def _compute_depth(coords: torch.Tensor, view_matrix: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes linear depth for given 3D points.
     https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    """
    matrix_vp = projection_matrix @ view_matrix
    ones = torch.ones((*coords.shape[:-1], 1), dtype=coords.dtype, device=coords.device)
    coords_w = torch.cat((coords, ones), -1)
    coords_ndc = torch.matmul(coords_w, matrix_vp.T)

    # Relative depth [0,1]
    gl_depth = coords_ndc[..., 2] / coords_ndc[..., 3]

    # [0,1] -> [-1,1]
    clip_space_depth = 2.0 * gl_depth - 1.0

    # glDepth -> linear
    try:
        params = math_utils.decompose_projection_matrix(projection_matrix)
        near = params['n']
        far = params['f']
    except:
        near = 0.1
        far = 10
    linear_depth = 2.0 * near * far / (far + near - clip_space_depth * (far - near))
    return linear_depth


def _batched_normals(decoder: SDFIBRNet, coords: torch.Tensor, times: torch.Tensor, batch_size: int,
                     params=None):
    """
    Computes normals for the surface. Uses batching and is not differentiable.   
    Will apply flow (T->T0) if needed and will differentiate wrt original coords at T.
    That means that the normals will be in coordinates of the current frame.
    """
    normals = torch.zeros_like(coords[..., :3])

    def append_normals(batch_index, input, output, callback_args):
        """
        Append batched normals.
        """
        batch_normals = diff_operators.gradient(output['model_out'][..., :1], output['model_in'])[0, ...].detach()

        # Normalize.
        batch_normals = batch_normals / torch.norm(batch_normals, dim=1, keepdim=True)

        # Collect.
        batch_slice = slice(batch_index * batch_size, min((batch_index + 1) * batch_size, normals.shape[0]))
        normals[batch_slice] = batch_normals

        # Suppress the output to save memory.
        return None

    # Coords must be differentiable.
    if not coords.requires_grad:
        coords = coords.detach().requires_grad_(True)

    # Query the network. Capture outputs using callback.
    model_input = {'coords': coords[None, ...], 'time': times[None, ...]}
    with torch.enable_grad():
        modules.batch_decode(decoder.decoder_sdf, model_input, batch_size, callback=append_normals,
                             params=get_subdict(params, 'decoder_sdf'))

    return normals


def render_view_proj_differentiable(decoder: SDFIBRNet,
                                    resolution: torch.Tensor,
                                    model_matrix: torch.Tensor,
                                    view_matrix: torch.Tensor,
                                    projection_matrix: torch.Tensor,
                                    timestamp: float,
                                    batch_size: int,
                                    debug_gui: bool = False,
                                    params=None):
    """
    Renders preview of SDF from a given view with gradients.
    """
    resolution = resolution.squeeze()
    model_matrix = model_matrix.squeeze()
    view_matrix = view_matrix.squeeze()
    projection_matrix = projection_matrix.squeeze()

    w, h = resolution.cpu().numpy()

    # Ray trace to find surface.
    t_start = time.time()
    # Build rays.
    rays_o, rays_d = get_rays_all(resolution, model_matrix, view_matrix, projection_matrix)
    # Determine start and end limits inside the [-1, 1] NDC volume.
    t_min, t_max = get_ray_limits_sphere(rays_o, rays_d)
    # Prepare time.
    times = torch.zeros_like(t_min).to(t_min.device).fill_(timestamp)

    # Ray-trace.
    rt_res = _raytrace_surface_differentiable(decoder, rays_o, rays_d, times, t_min, t_max,
                                              batch_size, debug=False, debug_gui=debug_gui,
                                              params=params)
    # print(f'Ray tracing took {time.time() - t_start:.3f} seconds.')
    coords = rt_res['coords']
    is_valid = rt_res['is_valid']
    t_min_comp = rt_res['t_min']

    sdf = rt_res['sdf']
    sdf_raytraced = sdf.reshape(h, w)

    # To device (if not already).
    view_matrix = view_matrix.to(decoder.device)
    projection_matrix = projection_matrix.to(decoder.device)

    # Position map.
    pos_map = coords.reshape(h, w, coords.shape[-1])

    # Mask.
    mask = is_valid.reshape(h, w)

    # Build linear depth image.
    depth_map = _compute_depth(coords, view_matrix, projection_matrix)
    valid_depth_values = depth_map[is_valid]
    max_depth = depth_map[0] if len(valid_depth_values) == 0 else depth_map[is_valid].max()
    # depth_map = torch.where(is_valid, depth_map, max_depth)
    depth_map = torch.clamp(depth_map, 0, float(max_depth.detach().cpu().numpy()))
    depth_map = depth_map.reshape(h, w)

    mask_res = _raytrace_differentiable_mask(decoder,
                                             rays_o, rays_d, times,
                                             t_min, t_max, batch_size,
                                             params=params,
                                             t_min_computed=t_min_comp, is_valid_mask=is_valid)
    soft_mask = mask_res['mask'].reshape(h, w)

    return {
        'mask': mask,
        'pos': pos_map,
        'depth': depth_map,
        'sdf_raytraced': sdf_raytraced,
        'softmask': soft_mask,
    }


@torch.no_grad()
def render_view_proj(decoder: SDFIBRNet,
                     resolution: torch.Tensor,
                     model_matrix: torch.Tensor,
                     view_matrix: torch.Tensor,
                     projection_matrix: torch.Tensor,
                     timestamp: float,
                     batch_size: int,
                     debug_gui: bool = False,
                     params=None,
                     normals=False,
                     vid_frame=0):
    """
    Renders preview of SDF from a given view without gradients.
    """
    resolution = resolution.squeeze()
    model_matrix = model_matrix.squeeze()
    view_matrix = view_matrix.squeeze()
    projection_matrix = projection_matrix.squeeze()

    w, h = resolution.cpu().numpy()

    # Ray trace to find surface.
    t_start = time.time()
    # Build rays.
    rays_o, rays_d = get_rays_all(resolution, model_matrix, view_matrix, projection_matrix)
    # Determine start and end limits inside the [-1, 1] NDC volume.
    t_min, t_max = get_ray_limits_sphere(rays_o, rays_d)
    # Prepare time.
    times = torch.zeros_like(t_min).to(t_min.device).fill_(timestamp)

    # Ray-trace.
    rt_res = _raytrace_surface(decoder, rays_o, rays_d, times, t_min, t_max,
                               batch_size, debug=False, debug_gui=debug_gui,
                               params=params)
    # print(f'Ray tracing took {time.time() - t_start:.3f} seconds.')
    coords = rt_res['coords']
    is_valid = rt_res['is_valid']

    # To device (if not already).
    view_matrix = view_matrix.to(decoder.device)
    projection_matrix = projection_matrix.to(decoder.device)

    # Position map.
    pos_map = coords.reshape(h, w, coords.shape[-1])

    # Mask.
    mask = is_valid.reshape(h, w)

    # Build linear depth image.
    depth_map = _compute_depth(coords, view_matrix, projection_matrix)
    valid_depth_values = depth_map[is_valid]
    max_depth = depth_map[0] if len(valid_depth_values) == 0 else depth_map[is_valid].max()
    # depth_map = torch.where(is_valid, depth_map, max_depth)
    depth_map = torch.clamp(depth_map, 0, max_depth)
    depth_map = depth_map.reshape(h, w)

    # Normals in local frame coords.
    normals_local = None
    if normals:
        normals_local = _batched_normals(decoder, coords, times, batch_size, params=params)
        normals_local = normals_local.reshape(h, w, 3)

    return {
        'mask': mask,
        'pos': pos_map,
        'depth': depth_map,
        'normals': normals_local,
    }


def ndc_to_standard(ndc_coords, resolution):
    """
    Convert ndc coordinates to [-1,1]^2 image coordinates as used by PyTorch.
    """
    resolution = resolution.to(ndc_coords.device)
    return ndc_coords / ((1 - (1 / resolution)) * torch.Tensor([1, -1]).to(ndc_coords.device))
