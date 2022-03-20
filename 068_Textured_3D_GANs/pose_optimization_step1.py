import os
# Workaround for PyTorch spawning too many threads
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import argparse
import sys
import math
import time
import datetime
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import cv2
cv2.setNumThreads(0)

from packaging import version
from tqdm import tqdm

from rendering.parallel_rasterizer import ParallelKaolinRasterizer
from rendering.utils import qrot, qmul
from rendering.mesh_template import MeshTemplate
from utils.losses import mean_iou_noreduce, geodesic_distance, evaluate_geodesic_distance, agreement_score

from cmr_data.custom import CustomDataset
from data.image_dataset import ImageDataset, AdjustedBatchSampler
from data.definitions import dataset_to_class_name

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--mesh_path', type=str, default='autodetect')
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--image_resolution', type=int, default=256)
parser.add_argument('--symmetric', type=bool, default=True)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gpu_ids', type=str, default='0', help='comma-separated')

parser.add_argument('--no_prune', action='store_true', help='do not prune proposals in multi-template mode (slow)')
parser.add_argument('--mode', type=str, required=True, help='single or multiple templates (singletpl|multitpl)')
parser.add_argument('--camera_optim_steps', type=int, default=100, help='camera optimization iterations per image')
parser.add_argument('--camera_lr_decay_after', type=int, default=80, help='decay learning rate after this step')
parser.add_argument('--v_agr_threshold', type=float, default=0.3, help='agreement score cutoff')

args = parser.parse_args()

gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
print('Using {} GPUs: {}'.format(len(gpu_ids), gpu_ids))
torch.cuda.set_device(min(gpu_ids))

assert args.mode in ['multitpl', 'singletpl']
multi_template = args.mode == 'multitpl'

if args.batch_size == -1:
    # Default: 2 images per GPU (it's usually enough to saturate it due to the multiple camera hypotheses)
    args.batch_size = 2*len(gpu_ids)
    
assert args.batch_size % len(gpu_ids) == 0, 'Batch size must be divisible by # gpus'

template_dir = f'cache/remeshed_templates/{args.mode}'

if args.mesh_path == 'autodetect':
    args.mesh_path = 'mesh_templates/uvsphere_31rings.obj'
    print('Using initial mesh topology', args.mesh_path)

mesh_template = MeshTemplate(args.mesh_path, is_symmetric=args.symmetric)


canonical_category = dataset_to_class_name[args.dataset]
remeshed_meshes = torch.load(f'{template_dir}/{canonical_category}_templates.pth')
assert mesh_template.mesh.vertices.shape == remeshed_meshes.shape[1:]

if multi_template:
    print(f'Using {remeshed_meshes.shape[0]} templates')
    assert remeshed_meshes.shape[0] > 1
else:
    print('Using single template')
    assert remeshed_meshes.shape[0] == 1

renderer_res = args.image_resolution

renderer = nn.DataParallel(ParallelKaolinRasterizer(renderer_res), gpu_ids)

cmr_dataset = CustomDataset(False, args.image_resolution, args.dataset,
                            poses_dir=None, enable_seg=False)
mesh_ds_train = ImageDataset(cmr_dataset, args.image_resolution)

batch_size = args.batch_size
sampler = AdjustedBatchSampler(torch.utils.data.SequentialSampler(mesh_ds_train), batch_size=batch_size, drop_last=False)
train_loader_seq = torch.utils.data.DataLoader(mesh_ds_train, num_workers=args.num_workers, pin_memory=True,
                                               batch_sampler=sampler)

if cmr_dataset.gt_available:
    print('Ground-truth poses are available for evaluation purposes. Loading...')
    all_gt_R = []
    for i, (_, _, _, _, gt_rot, _, _, _, _) in enumerate(tqdm(train_loader_seq)):
        all_gt_R.append(gt_rot.clone())
    all_gt_R = torch.cat(all_gt_R, dim=0)
    available_poses = (all_gt_R[..., 0] != -1000).sum().item()
    print(f'The geodesic distance (GD) will be evaluated on {available_poses} matching images (out of {all_gt_R.shape[0]}).')
else:
    all_gt_R = None
    print('Ground-truth poses are not available. The geodesic distance (GD) metric will not be evaluated.')

# Camera proposals initialization
def quantize_views(num_views_azimuth=8, num_views_elevation=5, num_views_roll=1):
    with torch.no_grad():
        views = []
        for azimuth in np.linspace(0, 360, num_views_azimuth, endpoint=False):
            for elevation in np.linspace(90-45, 90+45, num_views_elevation, endpoint=True):
                if num_views_roll == 1:
                    rolls = [-90]
                else:
                    rolls = np.linspace(-90-30, -90+30, num_views_roll, endpoint=True)
                for roll in rolls:
                    rad = roll / 180 * np.pi
                    q0 = torch.Tensor([np.cos(-rad/2), 0, 0, np.sin(-rad/2)])
                    rad = elevation / 180 * np.pi
                    q1 = torch.Tensor([np.cos(-rad/2), 0, np.sin(-rad/2), 0])
                    q0 = qmul(q0, q1)
                    rad = azimuth / 180 * np.pi
                    q = torch.Tensor([np.cos(-rad/2), 0, 0, np.sin(-rad/2)])
                    q = qmul(q0, q)
                    views.append(q)
        views = torch.stack(views, dim=0)
    return views

def transform_vertices(vtx, w, scale, translation, rotation, z0):
    # vtx (bs, nt, v, 3)
    # w (bs, nt)
    vtx = (vtx * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    translation = torch.cat((translation, torch.zeros_like(translation[..., :1])), dim=-1)
    vtx = vtx.expand(scale.shape[0], -1, -1)
    vtx = qrot(rotation, scale.unsqueeze(1)*vtx) + translation.unsqueeze(1)
    
    z0 = 1 + z0.unsqueeze(-1).exp()
    z = vtx[..., 2:]
    factor = (z0 - z/2)/(z0 + z/2)
    vtx = torch.cat((vtx[..., :2]*factor, z), dim=-1)
    
    vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)
    return vtx

def to_grid(x):
    return torchvision.utils.make_grid((x[:16, :3]+1)/2, nrow=4)

def render_mesh_kaolin(vtx, tex, image_size, **kwargs):
    if 'bs' in kwargs and bs % len(gpu_ids) != 0:
        # Last batch not divisible by num_gpus. Render one by one
        return mesh_template.forward_renderer(renderer.module, vtx, tex, num_gpus=1,
                                       return_hardmask=False, image_size=image_size, **kwargs)
    else:
        return mesh_template.forward_renderer(renderer, vtx, tex, num_gpus=len(gpu_ids),
                                       return_hardmask=False, image_size=image_size, **kwargs)

def render_mesh(vtx, tex, image_size=None, **kwargs):
    return render_mesh_kaolin(vtx, tex, image_size, **kwargs)


initial_views = quantize_views().cuda()
criterion = nn.MSELoss()


# Adam with batched full-matrix preconditioning
class AdamFull(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamFull, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamFull, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = []
            first = False
            state0 = None
            for p in group['params']:
                assert p.grad is not None
                state = self.state[p]
                if state0 is None:
                    state0 = state
                if len(state) == 0:
                    first = True
                    state['exp_avg'] = state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['step'] = 0
                    
                grads.append(p.grad)

            grads = torch.cat(grads, dim=-1)
            grads_sq = grads.unsqueeze(-1) @ grads.unsqueeze(-2)
            
            if first:
                # Exponential moving average of squared gradient values
                state0['exp_avg_sq'] = torch.zeros_like(grads_sq, memory_format=torch.preserve_format)
            exp_avg_sq = state0['exp_avg_sq']
                
            beta1, beta2 = group['betas']
            exp_avg_sq.mul_(beta2).add_(grads_sq, alpha=1 - beta2)
            M = (exp_avg_sq + group['eps']*torch.eye(exp_avg_sq.shape[-1], device=exp_avg_sq.device)).inverse()
            if version.parse(torch.__version__) < version.parse('1.8'):
                # As of version 1.8, PyTorch switched to a faster CUDA SVD routine. (CUDA >= 10.1 required)
                # Prior to that, the CPU version was much faster
                M = M.cpu()
            _, e, V = M.svd()
            e = e.to(exp_avg_sq.device)
            V = V.to(exp_avg_sq.device)
            Msqrtinv = V*e.clamp(min=0).sqrt().unsqueeze(-2) @ V.transpose(-1, -2)
            
            exp_avg_all = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_all.append(exp_avg)
                
            exp_avg_all = torch.cat(exp_avg_all, dim=-1)
            delta = (Msqrtinv @ exp_avg_all.unsqueeze(-1)).squeeze(-1) * math.sqrt(bias_correction2)
            param_ptr = 0
            for p in group['params']:
                state = self.state[p]
                grad = p.grad
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1
                delta_ = delta[..., param_ptr:param_ptr+grad.shape[-1]]
                param_ptr += grad.shape[-1]
                p.add_(delta_, alpha=-step_size)
            assert param_ptr == delta.shape[-1]

        return loss  
        
class PruningWrapper:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def zero_grad(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = None
                
    def step(self, closure=None):
        return self.optimizer.step(closure)
        
    @torch.no_grad()
    def prune(self, kept_indices):
        new_param_list = []
        for group in self.optimizer.param_groups:
            group_list = []
            for p in group['params']:
                assert p.grad is None, 'zero_grad() must be called before pruning'
                state = self.optimizer.state[p]
                new_param = p[kept_indices].detach().requires_grad_()
                group_list.append(new_param)
                del self.optimizer.state[p]
                self.optimizer.state[new_param] = state
                for k, v in state.items():
                    if torch.is_tensor(v) and v.shape[0] == p.shape[0]:
                        state[k] = v[kept_indices]
                
            group['params'] = group_list
            new_param_list += group_list
        return new_param_list


all_R_proposal = []
all_s_proposal = []
all_t_proposal = []
all_z0_proposal = []
all_w_proposal = []
all_iou = []

renderer.module.set_mode('alpha')

total_done = 0
start_time = time.time()
total_iou = 0
total_iou_best = 0
total_mse = 0
print('Optimizing poses...')
schedule = [args.camera_optim_steps*3//10, args.camera_optim_steps*6//10]
if (not args.no_prune) and multi_template:
    print('Pruning camera hypotheses and increasing resolution at steps', schedule)
else:
    print('Increasing camera resolution at steps', schedule)

if all_gt_R is not None:
    # Initialize running stats for evaluation
    gd1_sum = 0
    gd1_count = 0
    gdr_sum = 0
    gdr_count = 0
    recall_sum = 0
    recall_count = 0

for i, (X_real, X_seg, _, _, gt_rot, _, _, _, gt_idx) in enumerate(train_loader_seq):
    X_alpha = X_real[:, 3:4].cuda()
    gt_rot = gt_rot.cuda()
    gt_idx = gt_idx.squeeze(1)

    num_hypotheses = initial_views.shape[0] # Number of camera hypotheses
    bs = gt_idx.shape[0] # Batch size
    nt = remeshed_meshes.shape[0] # Number of mesh templates
    
    with torch.no_grad():
        R_proposal = initial_views.clone().unsqueeze(0).expand(bs, -1, -1)
        s_proposal = torch.ones((bs, num_hypotheses, 1), device=R_proposal.device)
        t_proposal = torch.zeros((bs, num_hypotheses, 2), device=R_proposal.device)
        z0_proposal = torch.full((bs, num_hypotheses, 1), 2, device=R_proposal.device, dtype=torch.float32) # Initialize to a distant value
        w_proposal = torch.zeros((bs, num_hypotheses, nt), device=R_proposal.device)

        # Replicate hypotheses according to # of mesh templates
        R_proposal = R_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        s_proposal = s_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        t_proposal = t_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        z0_proposal = z0_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        w_proposal = w_proposal.unsqueeze(2).expand(-1, -1, nt, -1).contiguous()
        
        # One-hot vector
        for k in range(nt):
            w_proposal[:, :, k, k] = 1
        w_proposal = w_proposal.flatten(1, 2)
        num_hypotheses *= nt

        R_proposal = R_proposal.flatten(0, 1)
        s_proposal = s_proposal.flatten(0, 1)
        t_proposal = t_proposal.flatten(0, 1)
        z0_proposal = z0_proposal.flatten(0, 1)
        w_proposal = w_proposal.flatten(0, 1)

        raw_vtx_all = remeshed_meshes.unsqueeze(0).cuda()

        # Set initial scale to maximize camera area (this is done in vertex space without rendering)
        raw_vtx_nograd = raw_vtx_all.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, F.normalize(R_proposal, dim=-1), z0_proposal)
        scale_max = vtx[..., :2].abs().max(dim=-1).values.max(dim=-1).values
        s_proposal.data[:] = 0.9 * (1/scale_max.view(s_proposal.shape)) # Matches cmr margin

    # Enable gradients
    R_proposal.requires_grad_()
    s_proposal.requires_grad_()
    t_proposal.requires_grad_()
    z0_proposal.requires_grad_()

    camera_optim_lr = 0.1
    camera_optim_momentum = 0.9

    optim_params = [R_proposal, s_proposal, t_proposal, z0_proposal]
    camera_optim = AdamFull(optim_params, lr=camera_optim_lr, betas=(camera_optim_momentum, 0.95))
    camera_optim = PruningWrapper(camera_optim)

    num_camera_steps = args.camera_optim_steps
    render_size = renderer_res // 2

    target_cache = {} # Copy targets only in the first iteration, to avoid too much cross-GPU data movement
    target_cache_lock = threading.Lock()
    for k in range(num_camera_steps):
        renderer.module.set_sigma_mul(10**(-(k/(num_camera_steps-1))/num_camera_steps)) # from 1.0 to 0.1
        raw_vtx_nograd = raw_vtx_all.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, F.normalize(R_proposal, dim=-1), z0_proposal)

        # Closure for parallel loss computation inside each GPU with nn.DataParallel (avoids unnecessary data aggregation/exchange)
        def closure(image_pred, alpha_pred, **kwargs):
            device = alpha_pred.device
            with target_cache_lock:
                if device not in target_cache:
                    alpha_target = kwargs['target']
                    target_cache[device] = alpha_target
                else:
                    assert kwargs['target'] is None
                    alpha_target = target_cache[device]
            alpha_pred = alpha_pred.view(-1, num_hypotheses, *alpha_pred.shape[1:])
            alpha_target = alpha_target.unsqueeze(1).expand(-1, num_hypotheses, -1, -1, -1)
            silhouette_loss = criterion(alpha_pred, alpha_target)
            # Aggregate using sum instead of mean -> independent of batch size (gradients are disjoint)
            camera_loss = silhouette_loss*alpha_pred.shape[0]

            with torch.no_grad():
                iteration_ious = mean_iou_noreduce(alpha_pred, alpha_target).squeeze(2)

            return camera_loss.unsqueeze(0), iteration_ious

        if k == 0 or (k-1) in schedule:
            if render_size == renderer_res:
                X_alpha_target = X_alpha
            else:
                X_alpha_target = F.interpolate(X_alpha, size=render_size, mode='bilinear', align_corners=False)
        else:
            # Already copied
            X_alpha_target = None

        camera_loss, iteration_ious = render_mesh(vtx, None, image_size=render_size,
                                                  closure=closure, target=X_alpha_target, bs=bs)

        # Add brick wall loss (prevent object from going outside the camera frustum)
        brick_wall_loss = F.relu(vtx[:, :, :2].abs() - 1)**2
        brick_wall_loss = brick_wall_loss.mean(dim=[1, 2])
        camera_loss = camera_loss.sum() + brick_wall_loss.sum()

        # Update cameras
        camera_loss.backward()
        camera_optim.step()
        camera_optim.zero_grad()

        R_proposal.data[:] = F.normalize(R_proposal.data, dim=-1) # Renormalize rotation
        s_proposal.data.abs_() # Scale is always positive
        z0_proposal.data.clamp_(-4, 4) # Avoid extreme values which might lead to numerical instability

        with torch.no_grad():
            if k in schedule:
                if render_size < renderer_res:
                    render_size += 64 # 128 -> 192 -> 256
                    target_cache = {} # Reset target cache
                if (not args.no_prune) and multi_template:
                    # Prune proposals (delete bottom 50% according to IoU)
                    indices = iteration_ious.topk(num_hypotheses//2, dim=1, sorted=False)[1].flatten()
                    indices_base = torch.arange(bs, device=indices.device).unsqueeze(-1)                                                 .expand(-1, num_hypotheses//2).contiguous().flatten() * num_hypotheses
                    indices += indices_base
                    new_param_list = camera_optim.prune(indices)
                    
                    # Update tensors
                    w_proposal = w_proposal[indices]
                    R_proposal, s_proposal, t_proposal, z0_proposal = new_param_list

                    num_hypotheses //= 2
                    target_cache = {} # Reset target cache


        if k == args.camera_lr_decay_after: # Lower learning rate after this threshold
            for param_group in camera_optim.optimizer.param_groups:
                param_group['lr'] *= 0.1
            camera_optim_lr *= 0.1
    camera_optim = None # Delete optimizer

    # Evaluate and store final camera hypotheses
    with torch.no_grad():
        raw_vtx_nograd = raw_vtx_all.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, R_proposal, z0_proposal)

        def closure(image_pred, alpha_pred, **kwargs):
            alpha_pred = alpha_pred.view(-1, num_hypotheses, *alpha_pred.shape[1:])
            device = alpha_pred.device
            alpha_target = target_cache[device].unsqueeze(1).expand(-1, num_hypotheses, -1, -1, -1)
            camera_iou = mean_iou_noreduce(alpha_pred, alpha_target).squeeze(2)
            silhouette_loss = criterion(alpha_pred, alpha_target)*alpha_pred.shape[0]
            
            return camera_iou, silhouette_loss.unsqueeze(0)

        camera_iou, camera_loss = render_mesh(vtx, None,
                                             closure=closure,
                                             bs=bs)
        camera_loss = camera_loss.sum()


        R_proposal = R_proposal.view(bs, num_hypotheses, 4)
        s_proposal = s_proposal.view(bs, num_hypotheses, 1)
        t_proposal = t_proposal.view(bs, num_hypotheses, 2)
        z0_proposal = z0_proposal.view(bs, num_hypotheses, 1)
        w_proposal = w_proposal.view(bs, num_hypotheses, nt)
        
        if all_gt_R is not None:
            # Evaluate geodesic distance if ground-truth is available
            v_agr_iou = agreement_score(camera_iou, R_proposal)
            valid_indices = (v_agr_iou < args.v_agr_threshold)
            recall_sum += valid_indices.float().sum().item()
            recall_count += len(valid_indices)
            best_R_proposal = R_proposal.gather(1, camera_iou.argmax(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
            gd_at_1, c1 = geodesic_distance(best_R_proposal, gt_rot, return_count=True)
            gd_at_recall, cr = geodesic_distance(best_R_proposal[valid_indices], gt_rot[valid_indices], return_count=True)
            
            gd1_sum += gd_at_1*c1
            gd1_count += c1
            
            gdr_sum += gd_at_recall*cr
            gdr_count += cr
            
            if gd1_count > 0:
                gd = gd1_sum / gd1_count
                gd_str = f', gd@1 {gd:.5f}'
            else:
                gd_str = ', gd@1 n/a'
            r = recall_sum / recall_count
            if gdr_count > 0:
                gd = gdr_sum / gdr_count
                gd_str += f', gd@{r:.3f} {gd:.5f}'
            else:
                gd_str += f', gd@{r:.3f} n/a'
        else:
            gd_str = ''

    all_R_proposal.append(R_proposal.detach().cpu())
    all_s_proposal.append(s_proposal.detach().cpu())
    all_t_proposal.append(t_proposal.detach().cpu())
    all_z0_proposal.append(z0_proposal.detach().cpu())
    all_w_proposal.append(w_proposal.detach().cpu())
    all_iou.append(camera_iou.detach().cpu())
    
    # Print progress
    total_done += bs
    total_mse += camera_loss.item()
    total_iou += camera_iou.mean(dim=1).sum(dim=0).item()
    total_iou_best += camera_iou.max(dim=1).values.sum(dim=0).item()
    end_time = time.time()
    im_s = total_done/(end_time - start_time)
    im_remaining = len(mesh_ds_train) - total_done
    eta = str(datetime.timedelta(seconds=int(im_remaining / im_s)))
    print(f'[{total_done}/{len(mesh_ds_train)}] {im_s:.03f} im/s, time_remaining {eta}, '
          f'avg_mse {total_mse/total_done:.05f}, avg_iou {total_iou/total_done:.05f}, '
          f'avg_best_iou {total_iou_best/total_done:.05f}' + gd_str)


all_R_proposal = torch.cat(all_R_proposal, dim=0)
all_s_proposal = torch.cat(all_s_proposal, dim=0)
all_t_proposal = torch.cat(all_t_proposal, dim=0)
all_z0_proposal = torch.cat(all_z0_proposal, dim=0)
all_w_proposal = torch.cat(all_w_proposal, dim=0)
all_iou = torch.cat(all_iou, dim=0)

fname = f'cache/{args.dataset}/camera_hypotheses_silhouette_{args.mode}.bin'
torch.save({
    'R': all_R_proposal,
    's': all_s_proposal,
    't': all_t_proposal,
    'z0': all_z0_proposal,
    'w': all_w_proposal,
    'iou': all_iou,
}, fname)
print('Saved to', fname)

if all_gt_R is not None:
    v_agr_iou = agreement_score(all_iou, all_R_proposal)
    best_R_proposal = all_R_proposal.gather(1, all_iou.argmax(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
    valid_indices = (v_agr_iou < args.v_agr_threshold)
    evaluate_geodesic_distance(f'Silhouette GD @ Recall', best_R_proposal[valid_indices], all_gt_R[valid_indices], len(valid_indices))
    evaluate_geodesic_distance(f'Silhouette GD @ Recall=1', best_R_proposal, all_gt_R)