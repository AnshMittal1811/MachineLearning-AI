import numpy as np
import torch
import torchvision
from PIL import Image

from utils import rend_util


tonemap_img = lambda x: torch.pow(x, 1./2.2)
clip_img = lambda x: torch.clamp(x, min=0., max=1.)


def plot_idr(model_outputs, pose, rgb_gt, path, iters, img_res):
    ''' write idr result when train geometry and radiance field '''
    
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)

    normal = model_outputs['normals']
    normal = normal.reshape(batch_size, num_samples, 3)

    idr_rgb = model_outputs['idr_rgb']
    idr_rgb = idr_rgb.reshape(batch_size, num_samples, 3)
    plot_idr_rgb(normal, idr_rgb, rgb_gt, path, iters, img_res)

    depth = torch.ones(batch_size * num_samples).cuda().float()
    if network_object_mask.sum() > 0:
        depth_valid = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
        depth[network_object_mask] = depth_valid
        depth[~network_object_mask] = 0.98 * depth_valid.min()
    depth = depth.reshape(batch_size, num_samples, 1)

    # plot depth maps
    plot_depth_maps(depth, path, iters, img_res)


def plot_illum(model_outputs, rgb_gt, path, iters, img_res):
    ''' write tracing result when train indirect illumination and visibility field '''
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']

    pred_radiance = model_outputs['pred_radiance']
    pred_radiance = pred_radiance.reshape(batch_size, num_samples, 3)

    traced_radiance = model_outputs['traced_radiance']
    traced_radiance = traced_radiance.reshape(batch_size, num_samples, 3)

    pred_vis = model_outputs['pred_vis']
    pred_vis = pred_vis.reshape(batch_size, num_samples, 1).expand(-1, -1, 3)

    gt_vis = model_outputs['gt_vis']
    gt_vis = gt_vis.reshape(batch_size, num_samples, 1).expand(-1, -1, 3)

    pred_radiance = clip_img(tonemap_img(pred_radiance))
    traced_radiance = clip_img(tonemap_img(traced_radiance))
    rgb_gt = clip_img(tonemap_img(rgb_gt.cuda()))

    output_vs_gt = torch.cat((pred_vis, gt_vis, 
            pred_radiance, traced_radiance, rgb_gt), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=1).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    print('saving render img to {0}/rendering_{1}.png'.format(path, iters))
    img.save('{0}/rendering_{1}.png'.format(path, iters))


def plot_mat(model_outputs, rgb_gt, path, iters, img_res):
    ''' write inverse rendering result '''
    
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)

    normal = model_outputs['normals']
    normal = normal.reshape(batch_size, num_samples, 3)

    specular_rgb = model_outputs['sg_specular_rgb']
    specular_rgb = specular_rgb.reshape(batch_size, num_samples, 3)

    sg_rgb = model_outputs['sg_rgb']
    sg_rgb = sg_rgb.reshape(batch_size, num_samples, 3)

    indir_rgb = model_outputs['indir_rgb']
    indir_rgb = indir_rgb.reshape(batch_size, num_samples, 3)

    roughness = model_outputs['roughness'].reshape(batch_size, num_samples, 3)
    diffuse_albedo = model_outputs['diffuse_albedo'].reshape(batch_size, num_samples, 3)
    visibility = model_outputs['vis_shadow'].reshape(batch_size, num_samples, 3)

    # plot rendered images
    plot_materials(normal, rgb_gt, 
                visibility, diffuse_albedo, roughness, specular_rgb, 
                indir_rgb, sg_rgb, path, iters, img_res)


def plot_idr_rgb(normal, idr_rgb, ground_true, path, iters, img_res):

    normal = clip_img((normal + 1.) / 2.)
    idr_rgb = clip_img(tonemap_img(idr_rgb))
    ground_true = clip_img(tonemap_img(ground_true.cuda()))

    output_vs_gt = torch.cat((normal, idr_rgb, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=1).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    print('saving render img to {0}/rendering_{1}.png'.format(path, iters))
    img.save('{0}/rendering_{1}.png'.format(path, iters))


def plot_materials(normal, ground_true,
                visibility, diffuse_albedo, roughness, specular_rgb, 
                indir_rgb, sg_rgb, path, iters, img_res):

    normal = clip_img((normal + 1.) / 2.)
    specular_rgb = clip_img(tonemap_img(specular_rgb))
    indir_rgb = clip_img(tonemap_img(indir_rgb))
    sg_rgb = clip_img(tonemap_img(sg_rgb))
    diffuse_albedo = clip_img(tonemap_img(diffuse_albedo))
    ground_true = clip_img(tonemap_img(ground_true.cuda()))

    output_vs_gt = torch.cat((normal, visibility, diffuse_albedo, roughness, 
                        specular_rgb, indir_rgb, sg_rgb, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=1).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    print('saving render img to {0}/rendering_{1}.png'.format(path, iters))
    img.save('{0}/rendering_{1}.png'.format(path, iters))


def plot_depth_maps(depth_maps, path, iters, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=1).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, iters))


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

