import matplotlib.pyplot as plt
import numpy as np
import torch
import utils.utils_ibr as utils_ibr
from torchmeta.modules.utils import get_subdict


@ torch.no_grad()
def write_meta_convergence_summary(meta_model_output, writer, total_steps, prefix='train_'):
    loss_values = meta_model_output['intermed_predictions']['loss']
    iter_nums = list(range(len(loss_values)))

    fig = plt.figure(figsize=(4,4))
    plt.plot(iter_nums, loss_values)
    plt.xlabel('iteration')
    plt.ylabel('total loss')
    writer.add_figure(prefix + 'meta_loss_plot', fig, global_step=total_steps)


@torch.no_grad()
def write_meta_ibr_summary(opt, meta_dataset, meta_model, meta_batch, meta_model_output, writer, total_steps, prefix='train_'):
    """
    Summarizes SDF and colors for IBR model during meta-training.
    """

    SDF_dataset = meta_dataset.get_SDFDataset(int(meta_batch['dataset_number'].cpu().numpy()))
    fast_params = meta_model_output['fast_params']
    mid_params = meta_model_output['intermed_predictions']['mid_params']
    init_params = meta_model_output['intermed_predictions']['init_params']

    utils_ibr._write_summary_ray_trace(opt, SDF_dataset, meta_model.hypo_module, None, None, None,
                                       writer, total_steps, prefix, False, params=fast_params)

    write_meta_convergence_summary(meta_model_output, writer, total_steps, prefix)

    _write_gtvspred_img_meta(opt, SDF_dataset, meta_model.hypo_module, meta_batch,
                             meta_model_output, writer, total_steps, prefix, params=fast_params)

    # Write ray trace results for intermediate parameters
    utils_ibr._write_summary_ray_trace(opt, SDF_dataset, meta_model.hypo_module, None, None, None,
                             writer, total_steps, prefix+'meta_midpoint_', False, params=mid_params)
    utils_ibr._write_summary_ray_trace(opt, SDF_dataset, meta_model.hypo_module, None, None, None,
                             writer, total_steps, prefix+'meta_initialization_', False, params=init_params)


@torch.no_grad()
def write_meta_ibr_summary_mult(opt, meta_dataset, meta_model, meta_batch, meta_model_output, writer, total_steps, prefix='train_'):
    SDF_dataset = meta_dataset.get_SDFDataset(int(meta_batch['dataset_number'].cpu().numpy()))
    fast_params = meta_model_output['fast_params']
    # mid_params = meta_model_output['intermed_predictions']['mid_params']
    init_params = meta_model_output['intermed_predictions']['init_params']

    utils_ibr._write_summary_ray_trace(opt, SDF_dataset, meta_model.hypo_module, None, None, None,
                                       writer, total_steps, prefix, False, params=fast_params)
    write_meta_convergence_summary(meta_model_output, writer, total_steps, prefix)

    utils_ibr._write_summary_ray_trace(opt, SDF_dataset, meta_model.hypo_module, None, None, None,
                             writer, total_steps, prefix+'meta_initialization_', False, params=init_params)

    _write_gtvspred_img_meta(opt, SDF_dataset, meta_model.hypo_module, meta_batch,
                             meta_model_output, writer, total_steps, prefix, params=fast_params)

@torch.no_grad()
def _write_gtvspred_img_meta(opt, dataset, model, meta_batch, meta_model_output, writer, total_steps, prefix='train_', params=None):
    model_out_single = meta_model_output['model_out']['trgt_outputs'][meta_model_output['model_out']['dense_idx']]
    model_out_single['source_images'] = meta_model_output['model_out']['source_images']

    trgt_img = torch.clamp(model_out_single['target_img'], 0, 1)
    gt_img = meta_batch['query']['gt']['rays_colors'].view(1, trgt_img.shape[-2], trgt_img.shape[-1], 3).permute(0, 3, 1, 2)
    mask = model_out_single['rays_mask'][-1:].unsqueeze(0).expand_as(trgt_img)

    # if opt.im_scale != 1.0:
    #     trgt_img = F.interpolate(trgt_img, scale_factor=opt.im_scale, mode='bilinear', align_corners=True)
    #     gt_img = F.interpolate(gt_img, scale_factor=opt.im_scale, mode='bilinear', align_corners=True)

    trgt_vs_gt = torch.cat([trgt_img, gt_img], 0)
    writer.add_images(prefix + 'trgt_vs_gt', trgt_vs_gt, total_steps)
    writer.add_images(prefix + 'trgt_vs_gt_masked', trgt_vs_gt * mask, total_steps)

    PSNR = 10 * torch.log10(1 / torch.mean(mask*(gt_img - trgt_img)**2))
    writer.add_scalar(prefix + 'image_psnr_masked', PSNR, total_steps)

    # mid_trgt_img = meta_model_output['intermed_predictions'].get('mid_image', None)
    # if mid_trgt_img is not None:
    #     mid_trgt_img = torch.clamp(mid_trgt_img, 0, 1)
    #
    #     trgt_vs_gt = torch.cat([mid_trgt_img, gt_img], 0)
    #     writer.add_images(prefix + 'mid_trgt_vs_gt_masked', trgt_vs_gt * mask, total_steps)
