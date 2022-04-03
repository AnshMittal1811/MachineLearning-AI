import torch
import forward_models


def function_mse(model_output, gt):
    idx = model_output['model_in']['idx'].long().squeeze()
    loss = (model_output['model_out']['output'][:, idx] - gt['func'][:, idx]) ** 2
    return {'func_loss': loss.mean()}


def tomography2D(model_output, gt):
    gt_radon_integral = gt['radon_integral']
    eval_phi_samples_on_lines = model_output['model_out']['output']  # [batch_size, mc_resolution, 1]
    line_integrals = gt['ray_len'].squeeze() * torch.mean(eval_phi_samples_on_lines, dim=1, keepdim=False).squeeze()

    integral_loss = (line_integrals - gt_radon_integral)**2
    return {'integral': integral_loss.mean()}


def tomo_radiance_sigma_rgb_loss(model_outputs, gt, dataloader=None,
                                 num_cuts=32,
                                 use_piecewise_model=False,
                                 tv_regularization=False,
                                 lambda_tv=1e-2,
                                 use_mask=False,
                                 lambda_mask=1e-2):
    # Predicted
    if 'combined' in model_outputs:
        pred_sigma = model_outputs['combined']['model_out']['output'][..., -1:]
        pred_rgb = model_outputs['combined']['model_out']['output'][..., :-1]
    else:
        pred_sigma = model_outputs['sigma']['model_out']['output']
        pred_rgb = model_outputs['rgb']['model_out']['output']

    # Pass through the forward models
    t_intervals = model_outputs['sigma']['model_in']['t_intervals']
    if use_piecewise_model:
        pred_weights = forward_models.compute_transmittance_weights_piecewise(pred_sigma, t_intervals, num_cuts)
        pred_pixel_samples = forward_models.compute_tomo_radiance_piecewise(pred_weights, pred_rgb, num_cuts)
    else:
        pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_intervals)
        pred_pixel_samples = forward_models.compute_tomo_radiance(pred_weights, pred_rgb)

    # Target Ground truth
    target_pixel_samples = gt['pixel_samples'][..., :3]  # rgba -> rgb

    # Loss
    tomo_loss = (pred_pixel_samples - target_pixel_samples)**2
    loss = {'tomo_rad_loss': tomo_loss.mean()}

    if tv_regularization:
        reg_outputs_sigma = model_outputs['sigma']['model_out']['reg_output']
        loss['tv'] = lambda_tv * torch.norm(reg_outputs_sigma, p=2, dim=-1).mean()

    if use_mask:
        pred_sigma_ray_sum = torch.sum(torch.abs(pred_weights), dim=-2)
        loss['mask'] = lambda_mask * torch.where(gt['mask_samples'] == 1,
                                                 torch.abs(pred_sigma_ray_sum),
                                                 torch.zeros_like(pred_sigma_ray_sum))
        loss['mask'] = loss['mask'].sum() / torch.sum(loss['mask'] > 0).float()

    return loss
