import os
# Workaround for PyTorch spawning too many threads
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import cv2
cv2.setNumThreads(0)

from tqdm import tqdm
from rendering.parallel_rasterizer import ParallelKaolinRasterizer
from rendering.utils import qrot, qmul
from rendering.mesh_template import MeshTemplate
from utils.losses import mean_iou_noreduce, mean_miou_noreduce, agreement_score, geodesic_distance, evaluate_geodesic_distance

from cmr_data.custom import CustomDataset
from data.image_dataset import ImageDataset, AdjustedBatchSampler
from data.definitions import dataset_to_class_name
from utils.misc import random_color_palette

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--mesh_path', type=str, default='autodetect')
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--image_resolution', type=int, default=256)
parser.add_argument('--symmetric', type=bool, default=True)

parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu_ids', type=str, default='0', help='comma-separated')

parser.add_argument('--mode', type=str, required=True, help='single or multiple templates (singletpl|multitpl)')
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
                            poses_dir=None, enable_seg=True)
mesh_ds_train = ImageDataset(cmr_dataset, args.image_resolution)

num_parts = cmr_dataset.num_parts + 1
color_palette = random_color_palette(num_parts)
color_palette_cuda = color_palette.t().cuda().unsqueeze(-1).unsqueeze(-1)


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


def render_mesh_kaolin(vtx, tex, image_size, hard=False, **kwargs):
    if 'bs' in kwargs and bs % len(gpu_ids) != 0:
        # Last batch non divisible by num_gpus. Render one by one
        return mesh_template.forward_renderer(renderer.module, vtx, tex, num_gpus=1,
                                       return_hardmask=False, image_size=image_size, **kwargs)
    else:
        return mesh_template.forward_renderer(renderer, vtx, tex, num_gpus=len(gpu_ids),
                                       return_hardmask=False, image_size=image_size, **kwargs)

def render_mesh(vtx, tex, image_size=None, hard=False, **kwargs):
    return render_mesh_kaolin(vtx, tex, image_size, hard, **kwargs)

criterion = nn.MSELoss()

# Load camera hypotheses from step 1 (silhouette optimization)
hypotheses_filename = f'cache/{args.dataset}/camera_hypotheses_silhouette_{args.mode}.bin'
chk = torch.load(hypotheses_filename)
all_R_proposal = chk['R']
all_s_proposal = chk['s']
all_t_proposal = chk['t']
all_z0_proposal = chk['z0']
all_w_proposal = chk['w']
all_iou = chk['iou']


def complementary_pose(q):
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    return torch.stack((y, -z, w, -x), dim=-1)



# Compute how close the pose is to the left/right side,
# which does not have a complementary pose and is thus unambiguous
complementary_score = torch.sum(all_R_proposal * complementary_pose(all_R_proposal), dim=-1)**2
complementary_score_max = complementary_score.gather(1, all_iou.argmax(dim=1, keepdim=True)).squeeze(1)

v_agr_iou = agreement_score(all_iou, all_R_proposal)

nt = all_w_proposal.shape[-1] # Number of mesh templates
assert nt == remeshed_meshes.shape[0]

if all_gt_R is not None:
    # Evaluate silhouette geodesic distance and recall (step 1)
    best_R_proposal = all_R_proposal.gather(1, all_iou.argmax(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
    valid_indices = (v_agr_iou < args.v_agr_threshold)
    evaluate_geodesic_distance(f'Silhouette GD @ Recall', best_R_proposal[valid_indices], all_gt_R[valid_indices], len(valid_indices))
    evaluate_geodesic_distance(f'Silhouette GD @ Recall=1', best_R_proposal, all_gt_R)

# Get top k images for each template
best_proposal_w = all_w_proposal.argmax(dim=-1).gather(1, all_iou.argmax(dim=1).unsqueeze(-1)).squeeze(1)
target_indices = []
for k in range(nt):
    valid_k = ((best_proposal_w == k) & (v_agr_iou < args.v_agr_threshold))
    valid_k &= complementary_score_max > 0.5
    values, indices = (all_iou.max(dim=1).values * valid_k.float()).topk(100) # k = 100 (at most)
    target_indices.append(indices[values > 0])
target_indices = torch.cat(target_indices, dim=0)
        
    
target_proposal_indices = all_iou[target_indices].argmax(dim=1)
all_R_proposal = all_R_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
all_s_proposal = all_s_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)
all_t_proposal = all_t_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
all_z0_proposal = all_z0_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)

all_w_proposal = all_w_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nt)).squeeze(1)
    

all_iou_proposal = all_iou[target_indices].unsqueeze(-1).gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)

print('Template distribution for semantic inference:', all_w_proposal.sum(dim=0))

if all_gt_R is not None:
    evaluate_geodesic_distance(f'Top samples GD', all_R_proposal, all_gt_R[target_indices])
    
mesh_ds_subset = torch.utils.data.Subset(mesh_ds_train, target_indices)
train_loader_semantic = torch.utils.data.DataLoader(mesh_ds_subset, batch_size=batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

semantic_template = torch.zeros(1, nt, mesh_template.mesh.vertices.shape[0], num_parts).cuda()

renderer.module.set_sigma_mul(0.1)
renderer.module.set_mode('vc') # Vertex color mode
    
print(f'Computing semantic template(s) for dataset {args.dataset}...')
for i, (_, X_seg, _, _, _, _, _, _, _) in enumerate(tqdm(train_loader_semantic)):
    X_seg = X_seg.cuda()

    with torch.no_grad():
        R = all_R_proposal[i*batch_size : (i+1)*batch_size].cuda()
        s = all_s_proposal[i*batch_size : (i+1)*batch_size].cuda()
        t = all_t_proposal[i*batch_size : (i+1)*batch_size].cuda()
        z0 = all_z0_proposal[i*batch_size : (i+1)*batch_size].cuda()
        w = all_w_proposal[i*batch_size : (i+1)*batch_size].cuda()

        seg_target = X_seg[:, 1:]
        raw_vtx_nograd = remeshed_meshes.unsqueeze(0).cuda()
        vtx = transform_vertices(raw_vtx_nograd, w, s, t, R, z0)


    # For parallel loss computation
    def closure(image_pred, alpha_pred, **kwargs):
        semantic_target = kwargs['target']
        return nn.MSELoss(reduction='sum')(image_pred, semantic_target).unsqueeze(0)

    # Dummy tensor whose content is not important. It is just used to accumulate gradients.
    dummy_template = torch.zeros_like(semantic_template).requires_grad_()
    dummy_template_expanded = dummy_template.expand(R.shape[0], -1, -1, -1)
    dummy_template_expanded = (dummy_template_expanded*w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    semantic_loss = render_mesh(vtx, dummy_template_expanded,
                                hard=False, closure=closure, target=seg_target)

    semantic_loss = semantic_loss.sum()
    semantic_loss.backward()
    semantic_template.data -= dummy_template.grad # Accumulate
    dummy_template.grad = None


# Compute final semantic template (distribution over object parts)
epsilon = 1e-6 # Additive smoothing
aggr_ind = 3
semantic_template_avg = (semantic_template + epsilon) / (semantic_template.sum(dim=aggr_ind, keepdim=True) + num_parts*epsilon)

if args.symmetric:
    # Symmetrize by averaging left and right
    avg_lr = (semantic_template_avg[:, :,  mesh_template.pos_indices] + semantic_template_avg[:, :, mesh_template.neg_indices])/2
    semantic_template_avg[:, :, mesh_template.pos_indices] = avg_lr
    semantic_template_avg[:, :, mesh_template.neg_indices] = avg_lr



# Reload hypotheses because we have overwritten them earlier
all_R_proposal = chk['R']
all_s_proposal = chk['s']
all_t_proposal = chk['t']
all_z0_proposal = chk['z0']
all_w_proposal = chk['w']

# Resolve ambiguities and obtain final poses
all_iou = []
all_miou = []
batch_ptr = 0
print('Resolving ambiguities using semantics...')
for i, (X_real, X_seg, _, _, _, _, _, _, _) in enumerate(tqdm(train_loader_seq)):
    X_seg = X_seg.cuda()
    X_alpha = X_real[:, 3:4].cuda()

    bs = X_seg.shape[0]
    num_hypotheses = all_R_proposal.shape[1]

    with torch.no_grad():
        R_proposal = all_R_proposal[batch_ptr:batch_ptr+bs].cuda().flatten(0, 1)
        s_proposal = all_s_proposal[batch_ptr:batch_ptr+bs].cuda().flatten(0, 1)
        t_proposal = all_t_proposal[batch_ptr:batch_ptr+bs].cuda().flatten(0, 1)
        z0_proposal = all_z0_proposal[batch_ptr:batch_ptr+bs].cuda().flatten(0, 1)
        w_proposal = all_w_proposal[batch_ptr:batch_ptr+bs].cuda().flatten(0, 1)
        batch_ptr += bs

        X_seg_target = X_seg[:, 1:]
        alpha_target = X_alpha
        tex_nograd = semantic_template_avg.clone()
        tex_nograd = (tex_nograd*w_proposal.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        raw_vtx = remeshed_meshes.unsqueeze(0).cuda()
        raw_vtx_nograd = raw_vtx.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, R_proposal, z0_proposal)

        def closure(image_pred, alpha_pred, **kwargs):
            image_pred = image_pred.view(-1, num_hypotheses, *image_pred.shape[1:])
            alpha_pred = alpha_pred.view(-1, num_hypotheses, *alpha_pred.shape[1:])

            alpha_target = kwargs['alpha_target'].unsqueeze(1).expand(-1, num_hypotheses, -1, -1, -1)
            seg_target = kwargs['seg_target'].unsqueeze(1).expand(-1, num_hypotheses, -1, -1, -1)

            camera_iou = mean_iou_noreduce(alpha_pred, alpha_target).squeeze(2)
            camera_miou = mean_miou_noreduce(image_pred, seg_target)

            return camera_iou, camera_miou

        camera_iou, camera_miou = render_mesh(vtx, tex_nograd, hard=True,
                                             closure=closure,
                                             alpha_target=X_alpha,
                                             seg_target=X_seg_target,
                                             bs=bs,
                                             )

        R_proposal = R_proposal.view(bs, num_hypotheses, 4)
        s_proposal = s_proposal.view(bs, num_hypotheses, 1)
        t_proposal = t_proposal.view(bs, num_hypotheses, 2)
        z0_proposal = z0_proposal.view(bs, num_hypotheses, 1)

        all_iou.append(camera_iou.cpu())
        all_miou.append(camera_miou.cpu())

all_iou = torch.cat(all_iou, dim=0)
all_miou = torch.cat(all_miou, dim=0)

# Compute new agreement score v_agr using mIoU instead of IoU
v_agr_miou = agreement_score(all_miou, all_R_proposal)

best_proposal_indices = all_miou.argmax(dim=1)
best_proposal_miou = all_miou.max(dim=1).values
best_proposal_iou = all_iou.gather(1, best_proposal_indices.unsqueeze(-1)).squeeze(1)

# Discard bottom 10%
thresh_miou = torch.kthvalue(best_proposal_miou, int(0.1*best_proposal_miou.shape[0]), dim=0).values
valid = (v_agr_miou < args.v_agr_threshold) & (best_proposal_miou > thresh_miou)

target_indices = valid
target_proposal_indices = best_proposal_indices[valid]
all_R_proposal2 = all_R_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
all_R_proposal2_recall1 = all_R_proposal.gather(1, best_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
all_s_proposal2 = all_s_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)
all_t_proposal2 = all_t_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
all_z0_proposal2 = all_z0_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)

    
nt = all_w_proposal.shape[-1]
all_w_proposal2 = all_w_proposal[target_indices].gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nt)).squeeze(1)

all_iou_proposal2 = all_iou[target_indices].unsqueeze(-1).gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)

all_miou_proposal2 = all_miou[target_indices].unsqueeze(-1).gather(1, target_proposal_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1)).squeeze(1)

if all_gt_R is not None:
    evaluate_geodesic_distance(f'Semantic GD @ Recall', all_R_proposal2, all_gt_R[np.where(target_indices)[0]], len(valid))
    evaluate_geodesic_distance(f'Semantic GD @ Recall=1', all_R_proposal2_recall1, all_gt_R)

    
fname = f'cache/{args.dataset}/poses_estimated_{args.mode}.bin'
torch.save({
    'indices': torch.where(target_indices)[0],
    'R': all_R_proposal2,
    's': all_s_proposal2,
    't': all_t_proposal2,
    'z0': all_z0_proposal2,
    'w': all_w_proposal2,
    'iou': all_iou_proposal2,
    'miou': all_miou_proposal2,
}, fname)
print('Saved to', fname)