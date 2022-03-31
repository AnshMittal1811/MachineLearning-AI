import torch
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

import utils.common_utils as common_utils
import sdf_meshing

from modules_sdf import SDFIBRNet
import utils.error_metrics as error_metrics

import utils.utils_sdf as utils_sdf
import flow_vis


@torch.no_grad()
def _write_gtvspred_img(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    trgt_img = torch.clamp(model_output['target_img'], 0, 1)
    gt_img = gt['rays_colors'].view(1, trgt_img.shape[-2], trgt_img.shape[-1], 3).permute(0, 3, 1, 2)
    mask = model_output['rays_mask'][-1:].unsqueeze(0).expand_as(trgt_img)

    # if opt.im_scale != 1.0:
    #     trgt_img = F.interpolate(trgt_img, scale_factor=opt.im_scale, mode='bilinear', align_corners=True)
    #     gt_img = F.interpolate(gt_img, scale_factor=opt.im_scale, mode='bilinear', align_corners=True)

    trgt_vs_gt = torch.cat([trgt_img, gt_img], 0)
    writer.add_images(prefix + 'trgt_vs_gt', trgt_vs_gt, total_steps)
    writer.add_images(prefix + 'trgt_vs_gt_masked', trgt_vs_gt * mask, total_steps)

    PSNR = 10 * torch.log10(1 / torch.mean(mask*(gt_img - trgt_img)**2))
    writer.add_scalar(prefix + 'image_psnr_masked', PSNR, total_steps)


@torch.no_grad()
def _write_all_masks(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Target is the last one
    source_imgs = model_output['source_images'].float()
    frustum_masks = model_output['frustum_mask'].float().unsqueeze(1)
    occlusion_weights = model_output['occlusion_weights'].float().unsqueeze(1)
    rays_mask = model_output['rays_mask'].float().unsqueeze(1)
    proj_rays_mask = model_output['proj_rays_mask'].float().unsqueeze(1)

    # Resize masks for writing
    if opt.im_scale != 1.0:
        frustum_masks = F.interpolate(frustum_masks, scale_factor=opt.im_scale, mode='nearest')
        occlusion_weights = F.interpolate(occlusion_weights, scale_factor=opt.im_scale, mode='nearest')
        rays_mask = F.interpolate(rays_mask, scale_factor=opt.im_scale, mode='nearest')
        source_imgs = F.interpolate(source_imgs, scale_factor=opt.im_scale, mode='bilinear', align_corners=True)
        proj_rays_mask = F.interpolate(proj_rays_mask, scale_factor=opt.im_scale, mode='nearest')

    # writer.add_images(prefix + 'mask_frustum', frustum_masks, total_steps)
    # writer.add_images(prefix + 'mask_rays_orig', rays_mask, total_steps)
    # writer.add_images(prefix + 'mask_rays_proj', proj_rays_mask, total_steps)

    occ_x_frus = frustum_masks * occlusion_weights
    occ_x_frus_x_rays = occ_x_frus * proj_rays_mask

    # writer.add_images(prefix + 'mask_occ_x_frus', occ_x_frus, total_steps)
    # writer.add_images(prefix + 'mask_occ_x_frus_x_rays', occ_x_frus_x_rays, total_steps)

    # Aggregation weights.
    if model_output['aggragation_weights'] is not None:
        aggregation_weights = model_output['aggragation_weights'].float().unsqueeze(1)
        aggregation_weights_masked = aggregation_weights * model_output['rays_mask'][-1:].float().unsqueeze(1)

        if opt.im_scale != 1.0:
            aggregation_weights = F.interpolate(aggregation_weights, scale_factor=opt.im_scale, mode='nearest')
            aggregation_weights_masked = F.interpolate(aggregation_weights_masked, scale_factor=opt.im_scale, mode='nearest')

        source_imgs_short = source_imgs[:-1, ...]
        gt_img = source_imgs[-1:, ...]
        agg_mix = torch.cat([gt_img.expand_as(source_imgs_short), source_imgs_short,
                             aggregation_weights.expand_as(source_imgs_short),
                             aggregation_weights_masked.expand_as(source_imgs_short)], axis=0)
        agg_mix_grid = make_grid(agg_mix, nrow=source_imgs_short.shape[0])
        writer.add_image(prefix + 'aggregation_weights', agg_mix_grid, total_steps)

    gt_img_rgb = source_imgs[-1:, ...].expand_as(source_imgs)
    rays_mask_rgb = rays_mask.expand_as(source_imgs)
    occ_x_frus_rgb = occ_x_frus.expand_as(source_imgs)
    occ_x_frus_x_rays_rgb = occ_x_frus_x_rays.expand_as(source_imgs)

    all_imgs = torch.cat([gt_img_rgb, source_imgs, rays_mask_rgb,
                          occ_x_frus_rgb, occ_x_frus_x_rays_rgb], axis=0)

    # Write a full summary of masks and images
    all_img_grid = make_grid(all_imgs, nrow=rays_mask_rgb.shape[0])
    writer.add_image(prefix + 'mask_all', all_img_grid, total_steps)


@torch.no_grad()
def _write_feature_summary(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    occlusion_weights = model_output['occlusion_weights'].float().unsqueeze(1)
    if model_output['aggragation_weights'] is not None:
        aggregation_weights = model_output['aggragation_weights'].float().unsqueeze(1)
        agg_x_occ = aggregation_weights * occlusion_weights[:-1]
    source_imgs = model_output['source_images'].float()

    if opt.im_scale != 1.0:
        if model_output['aggragation_weights'] is not None:
            aggregation_weights = F.interpolate(aggregation_weights, scale_factor=opt.im_scale, mode='nearest')
            agg_x_occ = F.interpolate(agg_x_occ, scale_factor=opt.im_scale, mode='nearest')
        occlusion_weights = F.interpolate(occlusion_weights, scale_factor=opt.im_scale, mode='nearest')
        source_imgs = F.interpolate(source_imgs, scale_factor=opt.im_scale, mode='bilinear', align_corners=True)

    gt_img_rgb = source_imgs[-1:, ...].expand_as(source_imgs)
    if model_output['aggragation_weights'] is not None:
        agg_weights_rgb = torch.cat([aggregation_weights, torch.zeros_like(aggregation_weights[:1])],
                                    dim=0).expand_as(source_imgs)
        agg_x_occ_rgb = torch.cat([agg_x_occ, torch.zeros_like(agg_x_occ[:1])],
                                  dim=0).expand_as(source_imgs)
    occlusion_weights_rgb = occlusion_weights.expand_as(source_imgs)

    if model_output['aggragation_weights'] is not None:
        all_imgs = torch.cat([gt_img_rgb, source_imgs, agg_weights_rgb,
                              occlusion_weights_rgb, agg_x_occ_rgb], axis=0)
    else:
        all_imgs = torch.cat([gt_img_rgb, source_imgs,
                              occlusion_weights_rgb], axis=0)

    all_img_grid = make_grid(all_imgs, nrow=gt_img_rgb.shape[0],
                             normalize=True, scale_each=True)
    writer.add_image(prefix + 'feature_weighting', all_img_grid, total_steps)


@ torch.no_grad()
def _write_summary_ray_trace(opt, dataset, model: SDFIBRNet, model_input, gt, model_output, writer, total_steps,
                             prefix, last_frame: bool = False, params=None):
    """
    Ray traces SDF.
    """
    im_size = (dataset.resolution * opt.im_scale + 0.5).astype(int)
    timestamp = common_utils.KEY_FRAME_TIME
    render_softmask = total_steps >= 100
    suffix = ''
    frame_index = 0
    if last_frame:
        # Last frame of video
        timestamp = 1.0
        render_softmask = False
        suffix = '_last'
        frame_index = dataset.dataset_img.num_frames - 1

    # Ray-trace.
    res = sdf_meshing.raytrace_sdf_ibr(model,
                                       resolution=im_size,
                                       projection_matrix=dataset.projection_matrix,
                                       view_matrix=dataset.view_matrix,
                                       model_matrix=dataset.model_matrix,
                                       timestamp=timestamp,
                                       render_softmask=render_softmask,
                                       build_pcd=False,
                                       batch_size=opt.batch_size,
                                       params=params)

    # Measure PCD error if possible.
    coords = dataset.dataset_pcd.get_frame_coords(frame_index)
    if coords.shape[0] > 0:
        # Compute 3D errror.
        error = error_metrics.compute_pcd_error(coords, res)
        if error >= 0:
            writer.add_scalar(f"pcd_error{suffix}", error, total_steps)

    # Write predictions.
    for name, im in res['viz'].items():
        im_tb = torch.from_numpy(im).permute(2, 0, 1)
        writer.add_image(prefix + f'rt_{name}{suffix}', im_tb, global_step=total_steps)


@ torch.no_grad()
def write_sdf_color_summary(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    """
    Summarizes training.
    """

    # Compare GT versus predicted image.
    _write_gtvspred_img(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix)

    # Summary of masks and features
    _write_all_masks(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix)
    _write_feature_summary(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix)

    # 2D SDF slices summary.
    utils_sdf._write_summary_sdf_slices(opt, model, None, None, None, writer, total_steps, prefix)

    # Write summary of shape, pcd error, normals, etc... from the reference view.
    if total_steps % 400 == 0:
        _write_summary_ray_trace(opt, dataset, model, None, None, None, writer, total_steps, prefix, False)


@torch.no_grad()
def write_sdf_color_summary_mult(opt, dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    model_out_single = model_output['trgt_outputs'][model_output['dense_idx']]
    model_out_single['source_images'] = model_output['source_images']

    # Compare GT versus predicted image.
    _write_gtvspred_img(opt, None, None, None, gt, model_out_single, writer, total_steps, prefix)

    # Summary of masks and features
    _write_all_masks(opt, None, None, None, None, model_out_single, writer, total_steps, prefix)
    _write_feature_summary(opt, None, None, None, None, model_out_single, writer, total_steps, prefix)

    # 2D SDF slices summary.
    utils_sdf._write_summary_sdf_slices(opt, model, None, None, None, writer, total_steps, prefix)

    # Write summary of shape, pcd error, normals, etc... from the reference view.
    if total_steps % 400 == 0:
        _write_summary_ray_trace(opt, dataset, model, None, None, None, writer, total_steps, prefix, False)


@ torch.no_grad()
def write_pretrain_features_summary(opt, dataset, model, model_input, model_output, writer, total_steps, prefix='train_'):
    """
    Summarizes training
    """
    # Write grid of warped features versus regular features
    img0 = model_input['img0'][0]
    warped_img0 = model_output['image_warped'][0]
    decoded_img0 = model_output['decoded_image'][0]

    img1 = model_input['img1'][0]
    decoded_img1 = model_output['decoded_image_warped'][0]

    flow10 = model_input['flow10'][0].permute(1, 2, 0).cpu().numpy()
    flow10 = torch.from_numpy(flow_vis.flow_to_color(flow10, convert_to_bgr=False)).float().to(opt.device).permute(2, 0, 1)

    flow_warp_grid = torch.stack([img0, warped_img0, img1, flow10], axis=0)
    flow_warp_grid = make_grid(flow_warp_grid, nrow=4, normalize=True, scale_each=True)
    writer.add_image(prefix+'warping_summary', flow_warp_grid, total_steps)

    all_imgs = torch.stack([img0, decoded_img0, warped_img0,
                            img1, decoded_img1, flow10], axis=0)

    all_img_grid = make_grid(all_imgs, nrow=5, normalize=True, scale_each=True)
    writer.add_image(prefix + 'image_summary', all_img_grid, total_steps)

