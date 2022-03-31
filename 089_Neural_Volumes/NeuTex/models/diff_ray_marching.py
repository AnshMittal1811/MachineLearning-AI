import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .diff_render_func import alpha_blend


def find_ray_generation_method(name):
    assert isinstance(name, str), "ray generation method name must be string"
    if name == "cube":
        return cube_ray_generation
    raise RuntimeError("No such ray generation method: " + name)


def find_refined_ray_generation_method(name):
    assert isinstance(name, str), "ray generation method name must be string"
    if name == "cube":
        return refine_cube_ray_generation
    raise RuntimeError("No such refined ray generation method: " + name)


def sample_pdf(in_bins, in_weights, n_samples, det=False):
    # bins: N x R x S x 1
    # weights: N x R x s x 1
    in_shape = in_bins.shape
    device = in_weights.device

    bins = in_bins.data.cpu().numpy().reshape([-1, in_shape[2]])
    bins = 0.5 * (bins[..., 1:] + bins[..., :-1])
    # bins: [NR x (S-1)]

    weights = in_weights.data.cpu().numpy().reshape([-1, in_shape[2]])
    weights = weights[..., 1:-1]
    # weights: [NR x (S-2)]

    weights += 1e-5
    pdf = weights / np.sum(weights, axis=-1, keepdims=True)
    cdf = np.cumsum(pdf, axis=-1)
    cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], -1)
    # cdf: [NR x (S-1)]

    if det:
        ur = np.broadcast_to(
            np.linspace(0, 1, n_samples, dtype=np.float32), (cdf.shape[0], n_samples)
        )
    else:
        ur = np.random.rand(cdf.shape[0], n_samples).astype(np.float32)
    # u: [NR x S2]

    inds = np.stack([np.searchsorted(a, i, side="right") for a, i in zip(cdf, ur)])
    below = np.maximum(0, inds - 1)
    above = np.minimum(cdf.shape[-1] - 1, inds)
    cdf_below = np.take_along_axis(cdf, below, 1)
    cdf_above = np.take_along_axis(cdf, above, 1)
    bins_below = np.take_along_axis(bins, below, 1)
    bins_above = np.take_along_axis(bins, above, 1)
    denom = cdf_above - cdf_below
    denom = np.where(denom < 1e-5, np.ones_like(denom), denom)
    t = (ur - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    samples = (
        torch.from_numpy(samples)
        .view((in_shape[0], in_shape[1], n_samples, 1))
        .to(device)
    )
    samples = torch.cat([samples, in_bins.detach()], dim=-2)
    samples, _ = torch.sort(samples, dim=-2)
    samples = samples.detach()

    return samples


# def sample_pdf(in_bins, in_weights, n_samples, det=False):
#     # bins: N x R x S x 1
#     # weights: N x R x S x 1
#     import tensorflow as tf
#     tf.config.set_visible_devices([], 'GPU')

#     ori_shape = in_bins.shape
#     device = in_weights.device

#     # bins: (N*R, S)
#     bins = tf.convert_to_tensor(in_bins.data.cpu().numpy().reshape((-1, in_bins.shape[-2])))
#     weights = tf.convert_to_tensor(in_weights.data.cpu().numpy().reshape((-1, in_weights.shape[-2])))

#     bins = 0.5 * (bins[..., 1:] + bins[..., :-1])
#     weights = weights[..., 1:-1]

#     # Get pdf
#     weights += 1e-5  # prevent nans
#     pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
#     cdf = tf.cumsum(pdf, -1)
#     cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

#     # Take uniform samples
#     if det:
#         u = tf.linspace(0., 1., n_samples)
#         u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [n_samples])
#     else:
#         u = tf.random.uniform(list(cdf.shape[:-1]) + [n_samples])

#     # Invert CDF
#     inds = tf.searchsorted(cdf, u, side='right')
#     below = tf.maximum(0, inds - 1)
#     above = tf.minimum(cdf.shape[-1] - 1, inds)
#     inds_g = tf.stack([below, above], -1)
#     cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)
#     bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)

#     denom = (cdf_g[..., 1] - cdf_g[..., 0])
#     denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
#     t = (u - cdf_g[..., 0]) / denom
#     samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

#     # N x R x N_samples x 1
#     samples = torch.from_numpy(samples.numpy()).view(
#         (in_bins.shape[0], in_bins.shape[1], n_samples, 1)).to(in_bins.device)

#     #  print(samples[0,0,:, 0])
#     #  print(in_bins[0,0,:, 0])
#     # N x R x (N_samples + S) x 1
#     samples = torch.cat([samples, in_bins.detach()], dim=-2)
#     #  samples = torch.cat([samples, in_bins.data], dim=-2)
#     samples, _ = torch.sort(samples, dim=-2)
#     samples = samples.detach()

#     return samples


def cube_ray_generation(campos, raydir, point_count, domain_size=1.0, jitter=0.0):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    with torch.no_grad():
        t1 = (-domain_size - campos[:, None, :]) / raydir
        t2 = (domain_size - campos[:, None, :]) / raydir
        tmin = torch.max(
            torch.min(t1[..., 0], t2[..., 0]),
            torch.max(
                torch.min(t1[..., 1], t2[..., 1]), torch.min(t1[..., 2], t2[..., 2])
            ),
        )
        tmax = torch.min(
            torch.max(t1[..., 0], t2[..., 0]),
            torch.min(
                torch.max(t1[..., 1], t2[..., 1]), torch.max(t1[..., 2], t2[..., 2])
            ),
        )
        intersections = tmin < tmax
        t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.0)
        tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
        tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        dt = domain_size * 2 / point_count

        # segment_length is the effective ray length for each ray point
        segment_length = dt + dt * jitter * (
            torch.rand(
                (raydir.shape[0], raydir.shape[1], point_count), device=campos.device
            )
            - 0.5
        )

        end_point_ts = torch.cumsum(segment_length, dim=2)
        end_point_ts = torch.cat(
            [
                torch.zeros(
                    (end_point_ts.shape[0], end_point_ts.shape[1], 1),
                    device=end_point_ts.device,
                ),
                end_point_ts,
            ],
            dim=2,
        )
        end_point_ts = t[:, :, None] + end_point_ts

        middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
        raypos = (
            campos[:, None, None, :]
            + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
        )
        valid = torch.prod(
            torch.gt(raypos, -domain_size) * torch.lt(raypos, domain_size), dim=-1
        ).byte()

    return raypos, segment_length, valid, middle_point_ts


def refine_cube_ray_generation(
    campos, raydir, point_count, prev_ts, prev_weights, domain_size=1.0, jitter=0
):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # point_count: int
    # prev_ts: N x Rays x PrevSamples
    # prev_weights: N x Rays x PrevSamples
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    with torch.no_grad():
        end_point_ts = sample_pdf(
            prev_ts[..., None], prev_weights[..., None], point_count + 1, jitter <= 0
        )
        end_point_ts = end_point_ts.view(end_point_ts.shape[:-1])
        segment_length = end_point_ts[:, :, 1:] - end_point_ts[:, :, :-1]
        middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
        raypos = (
            campos[:, None, None, :]
            + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
        )
        valid = torch.prod(
            torch.gt(raypos, -domain_size) * torch.lt(raypos, domain_size), dim=-1
        ).byte()

    return raypos, segment_length, valid, middle_point_ts


def ray_march(
    raydir,
    raypos,
    ray_dist,
    ray_valid,
    ray_features,
    lightpos,
    light_intensity,
    render_func,
    blend_func,
):
    """
    Args:
        raydir: :math:`(N,Rays,3)`
        raypos: :math:`(N,Rays,Samples,3)`
        ray_dist: :math:`(N,Rays,Samples)`
        ray_valid: :math:`(N,Rays,Samples)`
        ray_features: :math:`(N,Rays,Samples,Features)`
        lightpos: :math:`(N,3)`
        light_intensity: :math:`(N,3)`
    Return:
        ray_color: :math:`(N,Rays,3)`
        point_color: :math:`(N,Rays,Samples,3)`
        point_opacity: :math:`(N,Rays,Samples)`
        acc_transmission: :math:`(N,Rays,Samples)`
        blend_weight: :math:`(N,Rays,Samples)`
        background_transmission: :math:`(N,Rays)`
        background_blend_weight: :math:`(N,Rays)`
    """

    if lightpos is not None:
        lightdir = raypos - lightpos[:, None, None, :]
        dist2 = torch.sum(lightdir * lightdir, dim=3, keepdim=True)
        point_color = render_func(
            ray_features,
            raypos,
            raydir[:, :, None, :],
            lightdir,
            light_intensity[:, None, None, :] / dist2,
        )
    else:
        point_color = render_func(
            ray_features, raypos, raydir[:, :, None, :], None, None
        )

    # we are essentially predicting predict 1 - e^-sigma
    sigma = ray_features[..., 0] * ray_valid.float()
    opacity = 1 - torch.exp(-sigma * ray_dist)

    # cumprod exclusive
    acc_transmission = torch.cumprod(1.0 - opacity + 1e-10, dim=-1)
    temp = torch.ones(opacity.shape[0:2] + (1,)).to(opacity.device).float()  # N x R x 1

    background_transmission = acc_transmission[:, :, -1]
    acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)

    blend_weight = blend_func(opacity, acc_transmission)
    ray_color = torch.sum(point_color * blend_weight[..., None], dim=-2, keepdim=False)

    background_blend_weight = alpha_blend(
        1, background_transmission
    )  # background is always alpha blend

    return (
        ray_color,
        point_color,
        opacity,
        acc_transmission,
        blend_weight,
        background_transmission,
        background_blend_weight,
    )


def alpha_ray_march(raydir, raypos, ray_dist, ray_valid, ray_features, blend_func):
    sigma = ray_features[..., 0] * ray_valid.float()
    opacity = 1 - torch.exp(-sigma * ray_dist)

    acc_transmission = torch.cumprod(1.0 - opacity + 1e-10, dim=-1)
    temp = torch.ones(opacity.shape[0:2] + (1,)).to(opacity.device).float()  # N x R x 1
    background_transmission = acc_transmission[:, :, -1]
    acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)

    blend_weight = blend_func(opacity, acc_transmission)
    background_blend_weight = alpha_blend(1, background_transmission)

    return (
        opacity,
        acc_transmission,
        blend_weight,
        background_transmission,
        background_blend_weight,
    )


def cube_ray_generation_with_end(
    campos, raydir, end, point_count, domain_size=1.0, jitter=0.0
):
    """
    Args
        campos: :math:`(N, 3)`
        raydir: :math:`(N, Rays, 3)`
        end: :math:`(N, Rays, 3)`
    Return
        raypos: :math:`(N, Rays, Samples, 3)`
        segment_length: :math:`(N, Rays, Samples)`
        valid: :math:`(N, Rays, Samples)`
        ts: :math:`(N, Rays, Samples)`
    """
    with torch.no_grad():
        t_end = torch.min(
            (end - campos[:, None, :]) / raydir, dim=-1
        ).values  # N x Rays

        t1 = (-domain_size - campos[:, None, :]) / raydir
        t2 = (domain_size - campos[:, None, :]) / raydir
        tmin = torch.max(
            torch.min(t1[..., 0], t2[..., 0]),
            torch.max(
                torch.min(t1[..., 1], t2[..., 1]), torch.min(t1[..., 2], t2[..., 2])
            ),
        )
        tmax = torch.min(
            torch.max(t1[..., 0], t2[..., 0]),
            torch.min(
                torch.max(t1[..., 1], t2[..., 1]), torch.max(t1[..., 2], t2[..., 2])
            ),
        )
        intersections = tmin < tmax
        t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.0)
        tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
        tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        dt = domain_size * 2 / point_count

        # segment_length is the effective ray length for each ray point
        segment_length = dt + dt * jitter * (
            torch.rand(
                (raydir.shape[0], raydir.shape[1], point_count), device=campos.device
            )
            - 0.5
        )

        end_point_ts = torch.cumsum(segment_length, dim=2)
        end_point_ts = torch.cat(
            [
                torch.zeros(
                    (end_point_ts.shape[0], end_point_ts.shape[1], 1),
                    device=end_point_ts.device,
                ),
                end_point_ts,
            ],
            dim=2,
        )
        end_point_ts = t[:, :, None] + end_point_ts
        middle_point_ts = (
            end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]
        ) / 2  # N x Rays x Samples

        raypos = (
            campos[:, None, None, :]
            + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
        )
        valid = torch.prod(
            torch.gt(raypos, -domain_size)
            * torch.lt(raypos, domain_size)
            * torch.lt(middle_point_ts, t_end[:, :, None]),
            dim=-1,
        ).byte()

    return raypos, segment_length, valid, middle_point_ts


# campos = torch.from_numpy(np.array([[1.1, 0, 0]])).float()
# raydir = torch.from_numpy(
#     np.array([[[-1, 0, 0], [-0.89442719, 0., 0.4472136], [-0.89442719, 0., -0.4472136],
#                [-0.89442719, 0.4472136, 0]]])).float()
# raypos, seg_length, ray_valid, ray_ts = cube_ray_generation(campos, raydir, 10, 1, 0.)
# print(raydir.shape, raypos.shape, seg_length.shape, ray_valid.shape, ray_ts.shape)
# print(raypos)

# import matplotlib.pyplot as plt
# plt.scatter(ray_ts[0, 0], np.ones_like(ray_ts[0, 0]) * 0)
# plt.scatter(ray_ts[0, 1], np.ones_like(ray_ts[0, 0]) * 1)
# plt.scatter(ray_ts[0, 2], np.ones_like(ray_ts[0, 0]) * 2)

# weights = torch.ones_like(ray_ts)
# weights[0, 0] = torch.FloatTensor([1, 1, 1, 1, 1, 6, 6, 6, 6, 6])
# weights = weights[..., None]
# print(weights.shape)

# print(campos.shape, raydir.shape, ray_ts.shape, weights.shape)

# raypos, seg_length, ray_valid, ray_ts = refine_cube_ray_generation(campos, raydir, 20, ray_ts, weights)

# import matplotlib.pyplot as plt
# plt.scatter(ray_ts[0, 0], np.ones_like(ray_ts[0, 0]) * 3)
# plt.scatter(ray_ts[0, 1], np.ones_like(ray_ts[0, 0]) * 4)
# plt.scatter(ray_ts[0, 2], np.ones_like(ray_ts[0, 0]) * 5)
# plt.show()

# ray_features = torch.zeros((1, 4, 30, 11))

# lightpos = campos
# light_intensity = torch.ones((1, 3))
# from diff_render_func import microfacet_render, alpha2_blend

# ray_color, point_color, opacity, acc_transmission, blend_weight = ray_march(raydir, raypos, seg_length,
#                                                                             ray_valid, ray_features, lightpos,
#                                                                             light_intensity,
#                                                                             microfacet_render, alpha2_blend)

# print(ray_color.shape, point_color.shape, opacity.shape, acc_transmission.shape, blend_weight.shape)
