import os
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra
import copy
from mnh.dataset import load_datasets
from mnh.stats import StatsLogger
from mnh.utils import *
from mnh.utils_model import freeze_model
from experts_forward import *
import mnh_cuda
import time
import math

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs')
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')

def serialize_model_params(model, num_networks):
    '''
    # fused kernel expects IxO matrix instead of OxI matrix (Important!!!)
    '''
    def process_weight(w):
        return w.reshape(num_networks, -1)
    serialized_params = []
    nerf_experts = model.plane_radiance_field
    for l in nerf_experts.mlp_xyz.layers:
        serialized_params += [l.bias, process_weight(l.weight)]
    alpha_layer = nerf_experts.alpha_layer
    inter_layer = nerf_experts.intermediate_linear
    serialized_params += [alpha_layer.bias, process_weight(alpha_layer.weight)]
    serialized_params += [inter_layer.bias, process_weight(inter_layer.weight)]
    for l in nerf_experts.color_layer:
        serialized_params += [l.bias, process_weight(l.weight)]
    serialized_params = torch.cat(serialized_params, dim=1).contiguous()
    return serialized_params

def serialize_plane_params(model):
    planes_basis = model.plane_geo.basis()          # (plane_n, 3, 3)
    planes_x = planes_basis[:, :, 0].contiguous()   # (plane_n, 3)
    planes_y = planes_basis[:, :, 1].contiguous()   # (plane_n, 3)
    planes_center = model.plane_geo.position()      # (plane_n, 3)
    planes_size = model.plane_geo.size()            # (plane_n, 2)
    planes_w = planes_size[:, 0].unsqueeze(-1).contiguous()
    planes_h = planes_size[:, 1].unsqueeze(-1).contiguous()
    planes_params = torch.cat([planes_x, planes_y, planes_center, planes_w, planes_h], dim=1).contiguous()
    return planes_params

@hydra.main(config_path=CONFIG_DIR)
def main(cfg: DictConfig):
    # Set random seed for reproduction
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set device for training
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cfg.cuda))
    else:
        device = torch.device('cpu')
        
    # set DataLoader objects
    train_dataset, valid_dataset = load_datasets(os.path.join(CURRENT_DIR, cfg.data.path), cfg)
    datasets = {
        'train': train_dataset,
        'valid': valid_dataset
    }
    options = 'valid'
    dataset = datasets[options]

    model = get_model_from_config(cfg)
    model.to(device)

    # load checkpoints    
    checkpoint_experts = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.experts)
    if os.path.isfile(checkpoint_experts):
        print('Loading from checkpoint: {}'.format(checkpoint_experts))
        loaded_data = torch.load(checkpoint_experts, map_location=device)
        model.load_state_dict(loaded_data['model'])
    else:
        print('[Error] experts checkpoint does not exist...')
        return

    output_dir = os.path.join(CURRENT_DIR, 'output_images', cfg.name, 'experts_cuda')
    os.makedirs(output_dir, exist_ok=True)

    H, W = cfg.data.image_size

    num_networks = int(cfg.model.n_plane)
    model_params = serialize_model_params(model, num_networks)

    # Ndc grid for ray sampling
    ndc_points = get_ndc_grid(cfg.data.image_size).to(device)
    ndc_points = ndc_points.view(-1, 3)

    # Pre-baked alpha maps for each plane
    if cfg.model.accelerate.bake == True:
        model.bake_planes_alpha()
        transmittance_threshold = float(cfg.model.accelerate.thresh)
        bake_res = int(cfg.model.accelerate.bake_res)
        planes_alpha = model.planes_alpha.squeeze()
        planes_alpha = planes_alpha.view(num_networks, -1)

    # Attributes for each plane (xy-basis, center, width, height)
    planes_params = serialize_plane_params(model)

    # Get 3 out of 4 corner points from each plane
    planes_vertices = model.plane_geo.planes_vertices()
    planes_vertices = planes_vertices[:, :-1, :].contiguous()

    # Choose BATCH_NUM for each dataset
    BATCH_NUM = 512 * 512
    if 'replica' in cfg.data.path:
        BATCH_NUM = 512 * 512
    elif 'tat' in cfg.data.path:
        BATCH_NUM = 720 * 720

    total_time = 0.0
    total_eval_num = 0
    total_eval_num_after_filter = 0
    total_time_ray_plane_inter = 0.0
    total_time_pre_process = 0.0
    total_time_network_eval = 0.0
    total_time_integrate = 0.0

    for i in range(len(dataset)):

        data = dataset[i]
        camera = data['camera'].to(device)
        camera_center = camera.get_camera_center()[0]
        c2w = get_cam2world(camera)[:3,:3].T
        k = get_camera_k(camera)
        fx, fy = k[0, 0], k[1, 1]
        px, py = k[2, 0], k[2, 1]
        torch.cuda.synchronize()

        global_start_time = time.time()

        start_time = time.time()
        rgb_map = torch.zeros([H, W, 3], dtype=torch.float, device=device)
        acc_map = torch.zeros([H, W], dtype=torch.float, device=device)
        transmittance = torch.ones([H, W], dtype=torch.float, device=device)

        # Flatten
        rgb_map = rgb_map.view(-1, 3)
        acc_map = acc_map.view(-1)
        transmittance = transmittance.view(-1)

        '''
        Note: camera matrix R is world2camera matrix but in transpose (ex: points @ w2c)
        '''
        # Get rays (world coordinates)
        root_num_blocks, root_num_threads = 64, 16
        norm_dirs = mnh_cuda.get_rays_d(H, W, px, py, fx, fy, c2w.contiguous(), ndc_points, root_num_blocks, root_num_threads)
        norm_dirs = norm_dirs.view(-1, 3)
        torch.cuda.synchronize()
        total_time_ray_plane_inter += time.time() - start_time

        # Batching rays to avoid out-of-memory
        TOTAL_NUM = H * W
        N_FOLDS = math.ceil(TOTAL_NUM / BATCH_NUM)

        for N in range(N_FOLDS):

            start_time = time.time()

            start_idx = N * BATCH_NUM
            end_idx = (N + 1) * BATCH_NUM if (N + 1) * BATCH_NUM <= TOTAL_NUM else TOTAL_NUM
            num_rays = end_idx - start_idx

            norm_dirs_split = norm_dirs[start_idx:end_idx].contiguous()
            rgb_map_split = rgb_map[start_idx:end_idx].contiguous()
            acc_map_split = acc_map[start_idx:end_idx].contiguous()
            transmittance_split = transmittance[start_idx:end_idx].contiguous()

            # Ray-plane intersection (first pass) -> compute intersection info.
            hit = torch.empty([num_networks, num_rays], dtype=torch.bool, device=norm_dirs.device)
            num_blocks, num_threads = num_networks, 256
            mnh_cuda.compute_ray_plane_intersection_mt(planes_vertices, norm_dirs_split, camera_center, hit, num_blocks, num_threads)

            assigned_indices = hit.nonzero(as_tuple=True)
            assigned_networks = assigned_indices[0].to(torch.int32)
            assigned_rays = assigned_indices[1].to(torch.int32)

            # Compute starting and ending index for each plane segment
            total_points = int(assigned_networks.shape[0])
            total_eval_num += total_points
            contained_networks, batch_size_per_network = torch.unique_consecutive(assigned_networks, return_counts=True)
            batch_size_per_network_full = torch.zeros([num_networks], dtype=torch.int64, device=hit.device)
            contained_networks = contained_networks.to(torch.int64)
            batch_size_per_network_full[contained_networks] = batch_size_per_network
            ends_plane = batch_size_per_network_full.cumsum(0).to(torch.int32)
            starts_plane = ends_plane - batch_size_per_network_full.to(torch.int32)

            '''
            hit_mask            : plane-wise allocation (p1, p1, p1, p2, p2......, pn, pn)
            hit_mask_backordered: ray-wise allocation (r1, r1, r1, r1, r2, r2......, rn)
            '''
            # memory allocation
            active_points = torch.empty([total_points, 3], dtype=torch.float, device=hit.device)
            active_view_dirs = torch.empty([total_points, 3], dtype=torch.float, device=hit.device)
            active_depth = torch.empty([total_points], dtype=torch.float, device=hit.device)
            active_alphas = torch.zeros([total_points], dtype=torch.float, device=hit.device)
            hit_mask = torch.empty([total_points], dtype=torch.bool, device=hit.device)
            hit_mask_backordered = torch.full([total_points], False, dtype=torch.bool, device=hit.device)
            output_backordered = torch.zeros([total_points, 4], dtype=torch.float, device=hit.device)

            # Ray-plane intersection (second pass) -> store intersection info.
            num_blocks, num_threads = num_networks, 256
            mnh_cuda.store_ray_plane_intersection_mt(planes_vertices, norm_dirs_split, assigned_rays, camera_center, starts_plane, ends_plane,
                                                active_points, active_view_dirs, active_depth, num_blocks, num_threads)
            torch.cuda.synchronize()
            total_time_ray_plane_inter += time.time() - start_time

            # Sorting with depth
            start_time = time.time()
            reorder_indices_by_depth = torch.arange(active_depth.size(0), dtype=torch.int64, device=active_depth.device)
            mnh_cuda.sort_by_key_float32_int64(active_depth, reorder_indices_by_depth)

            assigned_rays = assigned_rays[reorder_indices_by_depth]   # make rays index follow the order of depth

            # Sorting with rays indices
            reorder_indices_by_rays = torch.arange(assigned_rays.size(0), dtype=torch.int64, device=assigned_rays.device)
            mnh_cuda.sort_by_key_int32_int64(assigned_rays, reorder_indices_by_rays)

            # Sample from pre-baked alpha
            num_blocks, num_threads = num_networks, 256
            mnh_cuda.sample_from_planes_alpha(active_points, active_alphas, planes_params, planes_alpha, bake_res, starts_plane, ends_plane, num_blocks, num_threads)    

            reorder_indices = reorder_indices_by_depth[reorder_indices_by_rays]  # merge two reorder indices into one (preserve depth order for each ray)
            active_alphas_backordered = active_alphas[reorder_indices]           # from plane-wise to ray-wise allocation

            # Compute starting and ending index for each ray segment
            contained_rays, batch_size_per_ray = torch.unique_consecutive(assigned_rays, return_counts=True)
            batch_size_per_ray_full = torch.zeros([num_rays], dtype=torch.int64, device=hit.device)
            contained_rays = contained_rays.to(torch.int64)
            batch_size_per_ray_full[contained_rays] = batch_size_per_ray
            ends_ray = batch_size_per_ray_full.cumsum(0).to(torch.int32)
            starts_ray = ends_ray - batch_size_per_ray_full.to(torch.int32)

            # Early ray filtering
            num_blocks, num_threads = 64, 512
            mnh_cuda.early_ray_filtering(active_alphas_backordered, hit_mask_backordered, starts_ray, ends_ray, transmittance_threshold, num_rays, num_blocks, num_threads)

            hit_mask[reorder_indices] = hit_mask_backordered
            active_points = active_points[hit_mask]
            active_view_dirs = active_view_dirs[hit_mask]
            assigned_networks = assigned_networks[hit_mask]

            total_eval_num_after_filter += int(active_points.shape[0])   # profiling eval number...

            # Compute new starting and ending index for each plane segment after filtering
            contained_nets, batch_size_per_network = torch.unique_consecutive(assigned_networks, return_counts=True)
            ends_plane_new = batch_size_per_network.cumsum(0).to(torch.int32)
            starts_plane_new = ends_plane_new - batch_size_per_network.to(torch.int32)
            torch.cuda.synchronize()
            total_time_pre_process += time.time() - start_time

            contained_nets_num = int(contained_nets.shape[0])
            if contained_nets_num == 0:
                continue

            # Parallel inference
            start_time = time.time()
            num_blocks, num_threads = contained_nets_num, 256
            output_1d = mnh_cuda.mlp_eval_1d_filter(active_points, active_view_dirs, model_params, starts_plane_new, ends_plane_new, contained_nets, num_blocks, num_threads)
            torch.cuda.synchronize()
            total_time_network_eval += time.time() - start_time

            start_time = time.time()
            output_backordered[hit_mask] = output_1d
            output_backordered = output_backordered[reorder_indices]
            # active_depth = active_depth[reorder_indices_by_rays]
        
            # Integrate color & alpha along each ray for final rendering
            integrate_num_blocks, integrate_num_threads = 64, 512
            mnh_cuda.integrate(output_backordered, rgb_map_split, acc_map_split, transmittance_split, transmittance_threshold, starts_ray, ends_ray,
                    num_rays, integrate_num_blocks, integrate_num_threads)
            torch.cuda.synchronize()
            total_time_integrate += time.time() - start_time
            
        start_time = time.time()

        rgb_map = rgb_map.view(H, W, 3)
        acc_map = acc_map.view(H, W)
        transmittance = transmittance.view(H, W)

        # Replace background color in region with low alpha value
        if 'replica' not in cfg.data.path:
            background_color = torch.ones([3], dtype=torch.float, device=rgb_map.device)
            mnh_cuda.replace_transparency_by_background_color(rgb_map, acc_map, background_color, integrate_num_blocks, integrate_num_threads)

        torch.cuda.synchronize()
        total_time_integrate += time.time() - start_time
        total_time += time.time() - global_start_time

        folder_path = os.path.join(output_dir, 'color', options)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        # Save rendering images
        image_path = os.path.join(folder_path, '{:0>5}-pred.png'.format(i))
        image = tensor2Image(rgb_map)
        image.save(image_path)
        # Save ground truth images
        gt_image_path = os.path.join(folder_path, '{:0>5}-gt.png'.format(i))
        gt_image = data['color']
        gt_image = tensor2Image(gt_image)
        gt_image.save(gt_image_path)

        torch.cuda.synchronize()

    num_data = len(dataset)
    print("--- [%s] Render %d samples in %f sec. (FPS = %f) ---" % (cfg.name, num_data, total_time, num_data / total_time))

    print("-------- Avg #points intersected          : %d " % (total_eval_num / num_data))
    print("-------- Avg #points evaluated            : %d " % (total_eval_num_after_filter / num_data))
    print("-------- Avg filter ratio                 : %.1f %%" % ((1 - total_eval_num_after_filter / total_eval_num) * 100))
    print("-------- Avg time (ray-plane intersection): %.6f sec" % (total_time_ray_plane_inter / num_data))
    print("-------- Avg time (pre-processing)        : %.6f sec" % (total_time_pre_process / num_data))
    print("-------- Avg time (network eval)          : %.6f sec" % (total_time_network_eval / num_data))
    print("-------- Avg time (integrate)             : %.6f sec" % (total_time_integrate / num_data))
    print("-------- Avg time (total)                 : %.6f sec" % (total_time / num_data))

                
if __name__ == '__main__':
    main()