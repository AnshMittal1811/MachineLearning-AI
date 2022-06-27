import torch 
import torch.nn.functional as F
from mnh.utils import *
from mnh.model_experts import ModelExperts
from mnh.utils_model import *
from mnh.utils_camera import *
from omegaconf import DictConfig

def get_model_from_config(cfg:DictConfig):
    model = ModelExperts(
        n_plane=cfg.model.n_plane,
        image_size=cfg.data.image_size,
        # # Radiance field 
        n_harmonic_functions_pos=cfg.model.mlp_experts.n_harmonic_functions_pos,
        n_harmonic_functions_dir=cfg.model.mlp_experts.n_harmonic_functions_dir,
        n_hidden_neurons_pos=cfg.model.mlp_experts.n_hidden_neurons_pos,
        n_hidden_neurons_dir=cfg.model.mlp_experts.n_hidden_neurons_dir,
        n_layers=cfg.model.mlp_experts.n_layers,
        # train & test
        n_train_sample=cfg.model.n_train_sample,
        n_infer_sample=cfg.model.n_infer_sample,
        anti_aliasing=cfg.model.anti_aliasing,
        premultiply_alpha=cfg.model.premultiply_alpha,
        # accelerate
        n_bake_sample=cfg.model.accelerate.n_bake_sample,
        bake_res=cfg.model.accelerate.bake_res, 
        filter_thresh=cfg.model.accelerate.thresh,
        white_bg=cfg.data.white_bg
    )
    return model 

def forward_pass(
    data, 
    model,
    device,
    cfg, # config object
    optimizer=None,
    training:bool=False,
    **kwargs,
):
    camera   = data['camera'].to(device)
    color_gt = data['color'].to(device)
    points_dense = data['points'].to(device)

    timer = Timer(cuda_sync= not training)
    if training:
        model.train()
        out = model(camera)
        sample_idx = out['sample_idx']
        color_gt = color_gt.view(-1, 3)[sample_idx]
    else:
        with torch.no_grad():
            model.eval()
            out = model(camera)
    time = timer.get_time()
    depth_pred = out['depth']
    color_pred = out['color']
    # points_pred = out['points']
    # points = torch.cat([points_dense, points_pred])

    mse_color = F.mse_loss(color_pred, color_gt)
    loss_geo = model.compute_geometry_loss(points_dense)
    loss_point2plane = loss_geo['loss_point2plane']

    if training: #training
        optimizer.zero_grad()
        loss =  mse_color * cfg.loss_weight.color
        loss += loss_point2plane * cfg.loss_weight.point2plane
        loss.backward()
        optimizer.step()
    
    psnr = compute_psnr(color_gt, color_pred)
    ssim = compute_ssim(color_gt, color_pred) if not training else 0

    stats = {
        'mse_color': mse_color.detach().cpu().item(),
        # 'mse_point2plane': loss_point2plane.detach().cpu().item(),
        'psnr': psnr,
        'ssim': ssim,
        'FPS': 1/time,
        'time': time
    }
    images = {
        'depth_pred': depth_pred,
        'color_gt': color_gt,
        'color_pred': color_pred 
    }
    return stats, images

def learn_from_teacher(
    data, 
    model,
    teacher,
    device,
    cfg, # config object
    optimizer
):
    camera   = data['camera'].to(device)
    points, planes_idx = model.plane_geo.sample_planes_points(cfg.model.n_train_sample)
    directions = get_normalized_direction(camera, points)

    timer = Timer(cuda_sync=False)
    model_rgba = model.plane_radiance_field(points, directions, planes_idx)
    with torch.no_grad():
        teacher_rgba = teacher.radiance_field(points, directions)

    loss = F.mse_loss(model_rgba, teacher_rgba)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    time = timer.get_time()
    stats = {
        'loss_teacher': loss.detach().cpu().item(),
        'FPS': 1/time,
    }
    return stats