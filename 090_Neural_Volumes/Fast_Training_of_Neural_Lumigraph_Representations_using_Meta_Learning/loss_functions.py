import torch
import torch.nn.functional as F

import utils.diff_operators as diff_operators


def _generalized_color_loss(opt, pred, gt, **kwargs) -> torch.Tensor:
    """
    Computes color loss based on opts.
    """
    if opt.color_loss == 'l2':
        return F.mse_loss(pred, gt, **kwargs)
    elif opt.color_loss == 'smooth_l1':
        return F.smooth_l1_loss(pred, gt, **kwargs)
    else:
        return F.l1_loss(pred, gt, **kwargs)


def loss_shape(opt, model_output, gt) -> dict:
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']['coords']
    pred_sdf = model_output['sdf_out'][..., :1]

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    is_on_surface = gt_sdf >= 0
    # On-surface SDF == 0
    sdf_constraint = torch.where(is_on_surface, pred_sdf, torch.zeros_like(pred_sdf))
    # Off-surface |SDF| > 0
    inter_constraint = torch.where(is_on_surface, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    # On-surface normal == GT
    normal_is_valid = torch.any(gt_normals != 0, dim=-1, keepdim=True)
    normal_constraint = torch.where(normal_is_valid, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    # ||Grad SDF|| = 1
    grad_constraint = gradient[:, torch.logical_not(is_on_surface.reshape(-1)), :].norm(dim=-1) - 1
    if opt.loss_eikonal_metric == 'l2':
        grad_constraint = grad_constraint**2
    else:
        grad_constraint = torch.abs(grad_constraint)

    # Direct supervision of SDF.
    direct_loss = F.l1_loss(pred_sdf, gt_sdf)

    # Exp      # Lapl
    # -----------------
    losses = {}  # enabled, disabled

    weight_eikonal = opt.opt_sdf_eikonal_w
    if opt.opt_sdf_eikonal_w <= 0:
        weight_eikonal = 1.0  # Preview only, not optimized.
    if opt.model == 'idr':
        weight_eikonal = 0.1

    losses['surface_sdf_0'] = [torch.abs(sdf_constraint).mean() * 60, opt.opt_sdf_onsurface > 0]
    losses['off_surface_sdf_large'] = [inter_constraint.mean() * 2, opt.opt_sdf_offsurface > 0]
    losses['normal_constraint'] = [normal_constraint.mean() * 2, opt.opt_sdf_normal > 0]
    losses['eikonal_constraint'] = [grad_constraint.mean() * weight_eikonal, opt.opt_sdf_eikonal_w > 0]
    losses['sdf'] = [torch.abs(direct_loss).mean() * 1, opt.opt_sdf_direct > 0]
    return losses


def _compute_weight_reg_loss(parameters, multiplier: float, opt) -> torch.Tensor:
    """
    Computes regularization loss.
    """
    if multiplier == 0:
        return torch.Tensor([0.0]).to(opt.device)
    loss = 0
    for _, weights in parameters.items():
        loss += torch.norm(weights, 2)
    return loss / len(parameters) * multiplier


def loss_rendered_image(opt, model_output, gt) -> dict:
    """
    Computes loss on rendered image versus ground truth image
    """
    image_pred = model_output['target_img'].view(1, 3, -1).permute(0, 2, 1)
    image_gt = gt['rays_colors']

    # Apply mask
    mask = model_output['rays_mask'][-1].view(1, -1, 1).float()
    loss = _generalized_color_loss(opt, mask*image_pred, mask*image_gt)

    weight = 1
    return {'image_loss': [loss.mean() * weight, opt.ibr_dataset]}


def loss_rendered_images(opt, model_output, gt) -> dict:
    image_outputs = model_output['trgt_outputs']

    image_losses = []
    for trgt_idx, single_out in image_outputs.items():
        image_pred = single_out['target_img'].view(1, 3, -1).permute(0, 2, 1)
        image_gt = model_output['source_images'][trgt_idx].view(1, 3, -1).permute(0, 2, 1)

        # Apply mask
        mask = single_out['rays_mask'][trgt_idx].view(1, -1, 1).float()

        loss = _generalized_color_loss(opt, mask*image_pred, mask*image_gt)
        image_losses.append(loss.mean())

    total_loss = sum(image_losses)

    return {'image_loss': [total_loss, opt.ibr_dataset]}


def loss_rays_mask_mult(opt, model_output, gt) -> dict:
    image_outputs = model_output['trgt_outputs']

    mask_losses = []
    for trgt_idx, single_out in image_outputs.items():
        is_rays_valid = single_out['trgt_rays_is_valid']
        softmask = single_out['softmask']
        gt_mask = single_out['rays_mask'][trgt_idx]

        loss = F.binary_cross_entropy(softmask, gt_mask.float(), reduction='none')

        is_P_in = torch.logical_and(gt_mask, is_rays_valid)
        loss_mask = torch.where(is_P_in, torch.zeros_like(loss), loss)

        mask_losses.append(loss_mask)

    # weight = opt.rt_mask_loss_weight
    weight = 1e2
    weight /= opt.rt_mask_alpha

    total_loss = sum(mask_losses) * weight

    return {'rays_softmask_2': [total_loss, weight > 0]}


def loss_sdf_ibr(opt, model_output, gt) -> dict:
    """
    All losses for the IBR case.
    """
    losses = {}

    # Image error
    losses.update(loss_rendered_image(opt, model_output, gt))

    losses.update(loss_shape(opt, model_output, gt))

    return losses


def loss_sdf_ibr_mult(opt, model_output, gt) -> dict:
    """
    All losses for IBR case, when we use multiple target views per iteration.
    """
    losses = {}

    if not model_output['model_in']['train_shape']:
        losses.update(loss_rendered_images(opt, model_output, gt))
        return losses

    losses.update(loss_rendered_images(opt, model_output, gt))
    losses.update(loss_shape(opt, model_output, gt))

    losses.update(loss_rays_mask_mult(opt, model_output, gt))
    # losses.update(loss_regularize_weights_ibr(opt, model_output, gt))

    return losses


def loss_pretrain_features(opt, model_output, model_input) -> dict:
    """
    Loss on image reconstruction pre-training task
    """
    losses = {}

    # MSE on the images
    warped_img_loss = _generalized_color_loss(opt, model_output['decoded_image_warped'], model_input['img1'])
    losses.update({'warped_image_loss': [warped_img_loss, 1]})

    img_loss = _generalized_color_loss(opt, model_output['decoded_image'], model_input['img0'])
    losses.update({'image_loss': [img_loss, opt.input_image_loss]})

    return losses


def loss_regularize_weights_ibr(opt, model_output, gt, multiplier=None) -> dict:
    """
    Regularize weights.
    """
    losses = {}

    if multiplier is None:
        multiplier_sdf = opt.regularize_weights_sdf
    else:
        multiplier_sdf = multiplier

    reg_loss_sdf = _compute_weight_reg_loss(model_output['weights_sdf'], multiplier_sdf, opt)

    losses['reg_weights_sdf'] = [reg_loss_sdf, opt.regularize_weights_sdf > 0]
    return losses
