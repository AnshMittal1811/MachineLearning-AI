import os
# Workaround for PyTorch spawning too many threads
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from packaging import version

try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. Install it to see the progress bar.')
    def tqdm(x): return x

import cv2
cv2.setNumThreads(0) # Prevent opencv from spawning too many threads in the data loaders

import kaolin as kal
from rendering.parallel_rasterizer import ParallelKaolinRasterizer
from rendering.renderer import Renderer
from rendering.utils import qrot, qmul, circpad, symmetrize_texture, adjust_poles
from rendering.mesh_template import MeshTemplate
from utils.losses import loss_flat

from models.reconstruction import ReconstructionNetwork
from data.definitions import class_names as all_categories
from data.definitions import dataset_to_class_name, default_cache_directories
from cmr_data.custom import CustomDataset
from data.image_dataset import ImageDataset
from utils.misc import random_color_palette

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='name of the experiment')
parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
parser.add_argument('--mode', type=str, default='autodetect', help='single or multiple templates (autodetect|singletpl|multitpl)')
parser.add_argument('--mesh_path', type=str, default='autodetect', help='path to initial mesh topology')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_resolution', type=int, default=256)
parser.add_argument('--symmetric', type=bool, default=True)
parser.add_argument('--texture_resolution', type=int, default=128)
parser.add_argument('--mesh_resolution', type=int, default=32)
parser.add_argument('--loss', type=str, default='mse', help='(mse|l1)')
parser.add_argument('--norm_g', type=str, default='syncbatch', help='(syncbatch|batch|instance|none)')

parser.add_argument('--checkpoint_freq', type=int, default=100) # Epochs
parser.add_argument('--save_freq', type=int, default=10) # Epochs

parser.add_argument('--tensorboard', action='store_true') # Epochs
parser.add_argument('--image_freq', type=int, default=50) # Epochs

parser.add_argument('--generate_pseudogt', action='store_true')
parser.add_argument('--pseudogt_path', type=str, required=False, help='override output path for pseudo ground-truth')
parser.add_argument('--pseudogt_resolution', type=int, default=512) # Output texture resolution
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--continue_train', action='store_true') # Resume from checkpoint
parser.add_argument('--which_epoch', type=str, default='latest') # Epoch from which to resume (or evaluate)
parser.add_argument('--mesh_regularization', type=float, default=0.00005)
parser.add_argument('--prediction_type', type=str, default='both', help='(none|texture|semantics|both)')
parser.add_argument('--semi_fraction', type=float, default=0.1, help='fraction of top images to use for semi-supervision')

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--iters', type=int, default=-1, help='mutually exclusive with --epochs')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay_every', type=int, default=250)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu_ids', type=str, default='0', help='comma-separated')
parser.add_argument('--no_prefetch', action='store_true', help='do not load full dataset into memory (for large datasets)')

args = parser.parse_args()

gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
print('Using {} GPUs: {}'.format(len(gpu_ids), gpu_ids))
torch.cuda.set_device(min(gpu_ids))

# Set random seed
torch.manual_seed(1)

if args.mesh_path == 'autodetect':
    args.mesh_path = 'mesh_templates/uvsphere_31rings.obj'
    print('Using initial mesh topology', args.mesh_path)

mesh_template = MeshTemplate(args.mesh_path, is_symmetric=args.symmetric)

assert args.mode in ['autodetect', 'singletpl', 'multitpl'], 'Invalid mode'
if args.mode == 'autodetect':
    if 'singletpl' in args.name:
        print('Autodetected single-template setting')
        args.mode = 'singletpl'
        multi_template = False
    elif 'multitpl' in args.name:
        print('Autodetected multi-template setting')
        args.mode = 'multitpl'
        multi_template = True
    else:
        print('Unable to autodetect setting (single-template or multi-template) from the experiment name. '
              'Please put either "singletpl" or "multitpl" in the name, or specify --mode manually.')
        sys.exit(1)
else:
    multi_template = args.mode == 'multitpl'
    

template_dir = f'cache/remeshed_templates/{args.mode}'

remeshed_meshes = []
template_offset = []
prev_num_shapes = 0
if args.dataset == 'all':
    for category in all_categories:
        retargeted_mesh = torch.load(f'{template_dir}/{category}_templates.pth')
        assert mesh_template.mesh.vertices.shape == retargeted_mesh.shape[1:]
        if not multi_template:
            assert retargeted_mesh.shape[0] == 1
        remeshed_meshes.append(retargeted_mesh)
        template_offset.append(prev_num_shapes)
        prev_num_shapes += retargeted_mesh.shape[0]

    remeshed_meshes = torch.cat(remeshed_meshes, dim=0)
else:
    canonical_category = dataset_to_class_name[args.dataset]
    remeshed_meshes = torch.load(f'{template_dir}/{canonical_category}_templates.pth')
    template_offset.append(0)

print('Total mesh templates:', remeshed_meshes.shape[0])
remeshed_meshes = remeshed_meshes.cuda()


if args.generate_pseudogt:
    # Ideally, the renderer should run at a higher resolution than the input image,
    # or a sufficiently high resolution at the very least.
    renderer_res = max(1024, 2*args.pseudogt_resolution)
else:
    # Match neural network input resolution
    renderer_res = args.image_resolution
    
renderer = ParallelKaolinRasterizer(renderer_res, mode='texture')
if len(gpu_ids) > 1:
    renderer = nn.DataParallel(renderer, gpu_ids)


class CategoryMerger(torch.utils.data.Dataset):
    def __init__(self, categories, is_train, dataloader_resolution, unfiltered=False, enable_seg=False):
        
        self.datasets = []
        self.total_len = 0
        self.category_offset = [] # Needed for balancing
        for category in categories:
            print(f'Loading class {category}...')
            poses_dir = f'cache/{category}/poses_estimated_{args.mode}.bin'
            
            dataset = CustomDataset(is_train, dataloader_resolution, category, poses_dir,
                                    unfiltered=unfiltered, enable_seg=enable_seg, rasterize_argmax=True,
                                    semi_fraction=args.semi_fraction)
            self.datasets.append(ImageDataset(dataset, dataloader_resolution))
            self.total_len += len(dataset)
            self.category_offset.append(self.total_len)
        
        global_part_remapper = {}
        switching_matrix = [] # num_classes x max_parts_per_class x num_total_categories
        self.num_classes = len(self.datasets)
        for i in range(self.num_classes):
            parts = self.datasets[i].cmr_dataset.part_id_remapper
            dataset_remapper = []
            for i, (k, v) in enumerate(parts.items()):
                assert i == v
                if k not in global_part_remapper:
                    global_part_remapper[k] = len(global_part_remapper)
                dataset_remapper.append(global_part_remapper[k])
            switching_matrix.append(dataset_remapper)
    
        max_num_parts = max([len(x) for x in switching_matrix])
        total_num_categories = max([max(x) for x in switching_matrix]) + 1
        M = torch.zeros((self.num_classes, max_num_parts+1, total_num_categories+1))
        M_mask = torch.zeros((self.num_classes, max_num_parts+1))
        M[:, 0, 0] = 1 # Background class
        M_mask[:, 0] = 1
        for i, line in enumerate(switching_matrix):
            M_mask[i, torch.arange(len(line))+1] = 1
            for j, x in enumerate(line):
                M[i, j+1, x+1] = 1
                
        self.M = M
        self.M_mask = M_mask
        
        # Compute sample weights for balancing
        self.class_weights = torch.FloatTensor([len(x) for x in self.datasets])
        self.class_weights = self.class_weights.sum() / (self.class_weights * len(self.class_weights))
        self.image_weights = []
        for i, weight in enumerate(self.class_weights):
            weight = weight.item()
            self.image_weights += [weight]*len(self.datasets[i])
        assert len(self.image_weights) == self.total_len
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # Select correct dataset
        prev = 0
        for category_idx, offset in enumerate(self.category_offset):
            if idx >= prev and idx < offset:
                dataset_idx = idx - prev
                break
            prev = offset
        dataset = self.datasets[category_idx]
        sample = list(dataset[dataset_idx]) # Tuple to list
        
        seg = sample[1]
        w = sample[-3]
        # Note: background (channel 0) must be properly handled
        if (seg.shape[0] - 1) < self.M_mask.shape[-1]:
            pad_amount = self.M_mask.shape[-1] - (seg.shape[0] - 1)
            seg = torch.cat((seg, torch.zeros(pad_amount, *seg.shape[1:])), dim=0)
        sample[1] = seg # Overwrite
        
        w_new = torch.zeros(remeshed_meshes.shape[0])
        offset = template_offset[category_idx]
        w_new[offset:offset+w.shape[-1]] = w
        sample[-3] = w_new # Replace
        sample[-1] = idx # Replace index
        
        return (*sample, category_idx, self.M[category_idx], self.M_mask[category_idx])
        
    
class Prefetcher(torch.utils.data.Dataset):
    def __init__(self, dataloader):
        self.cache = []
        print('Prefetching...')
        for data in tqdm(dataloader):
            bs = len(data[0])
            for i in range(bs):
                unrolled_item = [item[i].clone() for item in data]
                self.cache.append(unrolled_item)
        assert len(self.cache) == len(dataloader.dataset), f'Got {len(self.cache)}, expected {len(dataloader.dataset)}'
        
    def __len__(self):
        return len(self.cache)
            
    def __getitem__(self, idx):
        return self.cache[idx]


if not args.generate_pseudogt:
    dataloader_resolution = args.image_resolution
    dataloader_resolution_val = args.image_resolution
else:
    # We need images at different scales
    inception_resolution = 299
    dataloader_resolution = [args.image_resolution, inception_resolution, renderer_res]
    dataloader_resolution_val = inception_resolution

if args.dataset == 'all':
    cats = list(default_cache_directories.values())
else:
    cats = [args.dataset]
enable_seg = not args.generate_pseudogt and args.prediction_type in ['semantics', 'both']
mesh_ds_train = CategoryMerger(cats, False, dataloader_resolution, enable_seg=enable_seg)

if args.iters != -1:
    assert args.epochs == 1000 # Default value
    print('Iterations specified:', args.iters)
    iters_per_epoch = int(len(mesh_ds_train) / args.batch_size)
    print('Iterations per epoch:', iters_per_epoch)
    args.epochs = int(args.iters / iters_per_epoch)
    evaluate_freq = int(args.epochs / 30)
    args.checkpoint_freq = evaluate_freq
    args.image_freq = evaluate_freq
    args.lr_decay_every = int(args.epochs / 4)
    print(f'*** Training for {args.epochs} epochs, saving/evaluating every {evaluate_freq} epochs ***')
    print(f'*** Decaying lr every {args.lr_decay_every} epochs ***')

if args.generate_pseudogt:
    mesh_ds_inception = CategoryMerger(cats, False, dataloader_resolution, unfiltered=True, enable_seg=False)

batch_size = args.batch_size
shuffle = not (args.generate_pseudogt or args.evaluate)

if not args.generate_pseudogt:
    # Train mode
    balance = True
    if args.dataset == 'all' and balance:
        sampler = torch.utils.data.WeightedRandomSampler(mesh_ds_train.image_weights, len(mesh_ds_train))
    
    if args.no_prefetch:
        # Disable prefetching
        if args.dataset == 'all' and balance:
            train_loader = torch.utils.data.DataLoader(mesh_ds_train, batch_size=batch_size, sampler=sampler,
                                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(mesh_ds_train, batch_size=batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
    else:
        # Prefetch data to speed up training
        # Note: we set shuffle to False because the dataset will be loaded by the prefetcher
        # The prefetcher dataloader (i.e. the one used by the training script) of course has shuffle=True
        train_loader = torch.utils.data.DataLoader(mesh_ds_train, batch_size=batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=args.generate_pseudogt,
                                                   drop_last=False)
        if args.dataset == 'all' and balance:
            train_loader = torch.utils.data.DataLoader(Prefetcher(train_loader), batch_size=batch_size, sampler=sampler,
                                                   num_workers=2, pin_memory=True, drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(Prefetcher(train_loader), batch_size=batch_size, shuffle=True,
                                                   num_workers=2, pin_memory=True, drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(mesh_ds_train, batch_size=batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True,
                                           drop_last=False)
    inception_loader = torch.utils.data.DataLoader(mesh_ds_inception, batch_size=batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True, drop_last=False)


def mean_iou(alpha_pred, alpha_real):
    alpha_pred = alpha_pred > 0.5
    alpha_real = alpha_real > 0.5
    intersection = (alpha_pred & alpha_real).float().sum(dim=[1, 2])
    union = (alpha_pred | alpha_real).float().sum(dim=[1, 2])
    iou = intersection / union
    return torch.mean(iou)


def to_grid(x, nrow=4, limit=16):
    return torchvision.utils.make_grid((x[:limit, :3]+1)/2, nrow=nrow)

def transform_vertices(vtx, gt_scale, gt_translation, gt_rot, gt_idx, gt_z0):
    vtx = qrot(gt_rot, gt_scale.unsqueeze(-1)*vtx) + gt_translation.unsqueeze(1)
    vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)
    
    # Apply perspective distortion
    z0 = 1 + torch.exp(gt_z0)
    z = vtx[:, :, 2:]
    factor = (z0 + z/2)/(z0 - z/2)
    vtx = torch.cat((vtx[:, :, :2]*factor, z), dim=2)
    
    return vtx


import pathlib
import os
if args.tensorboard:
    import shutil
    from torch.utils.tensorboard import SummaryWriter
    import torchvision

num_classes = mesh_ds_train.num_classes
num_parts = mesh_ds_train.M.shape[-1]
color_palette = random_color_palette(num_parts)
color_palette_cuda = color_palette.t().cuda().unsqueeze(-1).unsqueeze(-1)

generator = ReconstructionNetwork(num_classes,
                                  test_mode=args.generate_pseudogt,
                                  symmetric=args.symmetric,
                                  texture_res=args.texture_resolution,
                                  mesh_res=args.mesh_resolution,
                                  prediction_type=args.prediction_type,
                                  num_parts=num_parts,
                                  norm_g=args.norm_g,
                                 )

if args.norm_g == 'syncbatch':
    # Import library for synchronized batch normalization
    from sync_batchnorm import DataParallelWithCallback
    dataparallel = DataParallelWithCallback
    print('Using SyncBN')
else:
    dataparallel = nn.DataParallel

generator = dataparallel(generator, gpu_ids).cuda()

optimizer = optim.Adam(generator.parameters(), lr=args.lr)

criteria = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
}

criterion = criteria[args.loss]
g_curve = []
total_it = 0
epoch = 0

ampl_factor = 0.00005/args.mesh_regularization
flat_warmup = 10 * ampl_factor
semi_coeff = 1
    
checkpoint_dir = 'checkpoints_recon/' + args.name
if args.evaluate or args.generate_pseudogt:
    # Load last checkpoint
    chk = torch.load(os.path.join(checkpoint_dir, f'checkpoint_{args.which_epoch}.pth'), map_location=lambda storage, loc: storage)
    if 'epoch' in chk:
        epoch = chk['epoch']
        total_it = chk['iteration']
        flat_warmup = chk['flat_warmup']
        semi_coeff = chk['semi_coeff']
    generator.load_state_dict(chk['generator'])
        
    if args.continue_train:
        optimizer.load_state_dict(chk['optimizer'])
        
        print(f'Resuming from epoch {epoch}')
    else:
        if 'epoch' in chk:
            print(f'Evaluating epoch {epoch}')
        args.epochs = -1 # Disable training
    chk = None

if args.tensorboard and not (args.evaluate or args.generate_pseudogt):
    log_dir = 'tensorboard_recon/' + args.name
    shutil.rmtree(log_dir, ignore_errors=True) # Delete logs
    writer = SummaryWriter(log_dir)
else:
    writer = None

if not (args.generate_pseudogt or args.evaluate):
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(os.path.join(checkpoint_dir, 'log.txt'), 'a' if args.continue_train else 'w', buffering=1) # Line buffering
    print(' '.join(sys.argv), file=log_file)
else:
    log_file = None

def log(text):
    if log_file is not None:
        print(text, file=log_file)
    print(text)
    
if args.tensorboard and not (args.evaluate or args.generate_pseudogt):
    import imageio
    tex_wireframe = imageio.imread(f'mesh_templates/wireframe_{mesh_template.rings}rings.png')
    tex_wireframe = torch.Tensor(tex_wireframe).unsqueeze(0).permute(0, 3, 1, 2).cuda()/255*2 - 1
    
def visualize(vtx, M, X_real, pred_seg, pred_tex, it):
    with torch.no_grad():
        gt_viz = X_real[:, :3] + (1 - X_real[:, 3:4])
        stacked_imgs = [gt_viz.cpu()]

        show_argmax = False

        cat_input = []
        if args.prediction_type in ['semantics', 'both']:
            pred_seg_shape = pred_seg.shape
            pred_seg_selection = torch.bmm(M, pred_seg.flatten(-2, -1))
            pred_seg_selection = pred_seg_selection.view(*pred_seg_selection.shape[:2], *pred_seg_shape[2:])
            cat_input.append(pred_seg_selection)
        if args.prediction_type in ['texture', 'both']:
            cat_input.append(pred_tex)
        cat_input = torch.cat(cat_input, dim=1)

        # Render
        color, alpha = mesh_template.forward_renderer(renderer, vtx, cat_input,
                                                      num_gpus=len(gpu_ids),
                                                      return_hardmask=True,
                                                     )

        viz_seg, viz_tex = None, None
        if args.prediction_type in ['texture', 'both']:
            viz_texture = color[:, -3:]
            viz_texture += 1-alpha
            stacked_imgs.append(viz_texture.cpu())

        if args.prediction_type in ['semantics', 'both']:
            viz_seg = color[:, :pred_seg_selection.shape[1]]
            viz_seg_gt = seg_real
            if show_argmax:
                viz_seg = F.softmax(viz_seg*100, dim=1)
                viz_seg_gt = F.softmax(viz_seg_gt*100, dim=1)

            viz_seg_gt = F.conv2d(seg[:, 1:], color_palette_cuda) + (seg[:, :1])
            stacked_imgs.append(viz_seg_gt.cpu())

            viz_seg = F.conv2d(viz_seg, color_palette_cuda)
            viz_seg += 1-alpha
            stacked_imgs.append(viz_seg.cpu())


        # Render wireframe
        viz_wireframe, alpha = mesh_template.forward_renderer(renderer, vtx, tex_wireframe.expand(X_real.shape[0], -1, -1, -1),
                                                      num_gpus=len(gpu_ids),
                                                      return_hardmask=True,
                                                     )
        viz_wireframe += 1-alpha
        stacked_imgs.append(viz_wireframe.cpu())

        nrow = len(stacked_imgs)
        stacked_imgs = torch.stack(stacked_imgs, dim=1).flatten(0, 1)
        result = to_grid(stacked_imgs, nrow=1*nrow, limit=nrow*10)
        
    if args.tensorboard:
        writer.add_image('image_train/grid', result, it)
        
    return result

# Training loop
try:
    while epoch < args.epochs:
        generator.train()

        start_time = time.time()
        for i, (X, seg, gt_scale, gt_translation, gt_rot, gt_z0, semi_w_batch, semi_mask_batch,
                gt_idx, C, M, M_mask) in enumerate(train_loader):
            
            seg = seg.cuda()
            X_real = X.cuda()
            M = M.cuda()
            M_mask = M_mask.cuda()
            seg_real = seg[:, 1:].clone()
            seg_real += seg[:, :1] / M_mask.sum(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1) # Background = uniform distribution
            seg_real *= M_mask.unsqueeze(-1).unsqueeze(-1)
            gt_scale = gt_scale.cuda()
            gt_translation = gt_translation.cuda()
            gt_rot = gt_rot.cuda()
            gt_z0 = gt_z0.cuda().unsqueeze(-1)
            
            C = C.cuda()
            semi_w_batch = semi_w_batch.cuda()
            semi_mask_batch = semi_mask_batch.cuda()
            semi_mask_batch = (semi_mask_batch + semi_coeff).clamp_(max=1)
            if len(gt_idx.shape) == 2:
                gt_idx = gt_idx.squeeze(-1).cuda()
            else:
                gt_idx = None

            optimizer.zero_grad()

            pred_tex, mesh_map, pred_seg = generator(X_real, C, M)
            raw_vtx = mesh_template.get_vertex_positions(mesh_map)
            raw_vtx_semi = (semi_w_batch.unsqueeze(-1).unsqueeze(-1) * remeshed_meshes.unsqueeze(0)).sum(dim=1)
            vtx = transform_vertices(raw_vtx, gt_scale, gt_translation, gt_rot, gt_idx, gt_z0)

            def closure(color_pred, alpha_pred, **kwargs):
                target = kwargs['target']
                alpha_target = kwargs['alpha_target']
                color_loss = criterion(color_pred, target)
                alpha_loss = criterion(alpha_pred, alpha_target)
                with torch.no_grad():
                    miou = mean_iou(alpha_pred[:, 0], alpha_target[:, 0]) # Done on alpha channel
                return color_loss, alpha_loss, miou
                
            cat_input = []
            if args.prediction_type in ['semantics', 'both']:
                pred_seg_shape = pred_seg.shape
                pred_seg_selection = torch.bmm(M, pred_seg.flatten(-2, -1))
                pred_seg_selection = pred_seg_selection.view(*pred_seg_selection.shape[:2], *pred_seg_shape[2:])
                cat_input.append(pred_seg_selection)
            if args.prediction_type in ['texture', 'both']:
                cat_input.append(pred_tex)
            cat_input = torch.cat(cat_input, dim=1)
            
            cat_target = []
            if args.prediction_type in ['semantics', 'both']:
                cat_target.append(seg_real)
            if args.prediction_type in ['texture', 'both']:
                cat_target.append(X_real[:, :3])
            cat_target = torch.cat(cat_target, dim=1)
            
            color_loss, alpha_loss, miou = mesh_template.forward_renderer(renderer, vtx, cat_input,
                                                                  num_gpus=len(gpu_ids),
                                                                  closure=closure,
                                                                  target=cat_target,
                                                                  alpha_target=X_real[:, 3:4],
                                                                 )
            color_loss = color_loss.mean()
            alpha_loss = alpha_loss.mean()
            miou = miou.mean()
            
            semi_loss = semi_mask_batch * (torch.norm(raw_vtx - raw_vtx_semi, dim=-1)**2).mean(dim=-1)
            semi_loss = semi_loss.mean()
            recon_loss = 10*semi_loss + 10*color_loss + alpha_loss
            
            flat_loss = loss_flat(mesh_template.mesh, mesh_template.compute_normals(raw_vtx))
            
            # Penalize symmetry violations / avoid mesh intersections
            symm_penalty = F.relu(-raw_vtx[:, mesh_template.pos_indices, 0])
            symm_penalty = symm_penalty.mean()

            flat_coeff = args.mesh_regularization*flat_warmup
            flat_warmup = max(flat_warmup - 0.1*ampl_factor, 1)
            loss = recon_loss + flat_coeff*flat_loss + symm_penalty

            loss.backward()
            optimizer.step()
            g_curve.append(loss.item())


            if total_it % 10 == 0:
                log('[{}] epoch {}, {}/{}, recon_loss {:.5f} flat_loss {:.5f} total {:.5f} iou {:.5f}'.format(
                                                                        total_it, epoch, i, len(train_loader),
                                                                        recon_loss.item(), flat_loss.item(),
                                                                        loss.item(), miou.item(),
                                                                        ))

            if args.tensorboard:
                writer.add_scalar(args.loss + '/train', recon_loss.item(), total_it)
                writer.add_scalar('flat/train', flat_loss.item(), total_it)
                writer.add_scalar('iou/train', miou.item(), total_it)


            total_it += 1
            semi_coeff *= 0.99

        epoch += 1

        log('Time per epoch: {:.3f} s'.format(time.time() - start_time))


        if epoch % args.lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        def save_checkpoint(it):
            torch.save({
                'optimizer': optimizer.state_dict(),
                'generator': generator.state_dict(),
                'epoch': epoch,
                'iteration': total_it,
                'semi_coeff': semi_coeff,
                'flat_warmup': flat_warmup,
                'args': vars(args),
            }, os.path.join(checkpoint_dir, f'checkpoint_{it}.pth'))

        if epoch % args.save_freq == 0:
            save_checkpoint('latest')
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(str(epoch))
        if args.tensorboard and epoch % args.image_freq == 0:
            visualize(vtx, M, X_real, pred_seg, pred_tex, total_it)
        
except KeyboardInterrupt:
    print('Aborted.')
        
if not (args.generate_pseudogt or args.evaluate):
    save_checkpoint('latest')
elif args.evaluate:
    evaluate_all(val_loader, writer, total_it)
elif args.generate_pseudogt:
    from utils.fid import calculate_stats, init_inception, forward_inception_batch
    
    inception_model = init_inception().cuda().eval()
    
    print('Exporting pseudo-ground-truth data...')
    shrink = True # Treat background as transparent/masked
    
    class InverseRenderer(nn.Module):
        def __init__(self, mesh, res_h, res_w):
            super().__init__()

            self.res = (res_h, res_w)
            self.inverse_renderer = Renderer(res_h, res_w)
            self.mesh = mesh

        def forward(self, predicted_vertices, target):
            with torch.no_grad():
                tex = target # The texture is the target image
                uvs = (predicted_vertices[..., :2] + 1)/2
                vertices = self.mesh.uvs.unsqueeze(0)*2 - 1
                vertices = torch.cat((vertices, torch.zeros_like(vertices[..., :1])), dim=-1)
                image_pred, alpha_pred = self.inverse_renderer(points=[vertices.expand(target.shape[0], -1, -1),
                          self.mesh.face_textures],
                          uv_bxpx2=uvs,
                          texture_bx3xthxtw=tex,
                          ft_fx3=self.mesh.faces,
                          return_hardmask=True,
                         )
            return image_pred, alpha_pred

    inverse_renderer = InverseRenderer(mesh_template.mesh, args.pseudogt_resolution, args.pseudogt_resolution)

    cache_dir = os.path.join('cache', args.dataset)
    pseudogt_dir = os.path.join(cache_dir, f'pseudogt_{args.pseudogt_resolution}x{args.pseudogt_resolution}_{args.mode}')
    pathlib.Path(pseudogt_dir).mkdir(parents=True, exist_ok=True)
    
    all_gt_scale = []
    all_gt_translation = []
    all_gt_rotation = []
    all_gt_z0 = []
    all_C = []
    all_inception_activation = []
    
    generator.eval()
    for net_image, seg, inception_image, hd_image, gt_scale, gt_translation, gt_rot, gt_z0, semi_w_batch, semi_mask_batch, \
        indices, C, M, M_mask in tqdm(train_loader):
            
        # Compute visibility mask
        with torch.no_grad():    
            net_image = net_image.cuda()
            gt_scale = gt_scale.cuda()
            gt_translation = gt_translation.cuda()
            gt_rot = gt_rot.cuda()
            gt_z0 = gt_z0.cuda()
            C = C.cuda()
            if len(indices.shape) == 2:
                gt_idx = indices.squeeze(-1).cuda()
            else:
                gt_idx = indices.cuda()

            pred_tex, mesh_map, pred_seg = generator(net_image, C, M)
            pred_seg_saved = pred_seg
            raw_vtx = mesh_template.get_vertex_positions(mesh_map)

            vtx = transform_vertices(raw_vtx, gt_scale, gt_translation, gt_rot, gt_idx, gt_z0.unsqueeze(-1))
            if pred_tex.shape[2] > renderer_res//8:
                # To reduce aliasing in the gradients from the renderer,
                # the rendering resolution must be much higher than the texture resolution.
                # As a rule of thumb, we came up with render_res >= 8*texture_res
                # This is already ensured by the default hyperparameters (1024 and 128).
                # If not, the texture is resized.
                pred_tex = F.interpolate(pred_tex, size=(renderer_res//8, renderer_res//8),
                                         mode='bilinear', align_corners=False)
        
        pred_tex.requires_grad_()
        image_pred, alpha_pred = mesh_template.forward_renderer(renderer, vtx, pred_tex, num_gpus=len(gpu_ids))

        # Compute gradient
        visibility_mask, = torch.autograd.grad(image_pred, pred_tex, torch.ones_like(image_pred))
        
        with torch.no_grad():
            # Compute inception activations
            all_inception_activation.append(forward_inception_batch(inception_model, inception_image[:, :3].cuda()/2 + 0.5))
            
            # Project ground-truth image onto the UV map
            if not shrink:
                hd_image = hd_image[:, :3]
            inverse_tex, inverse_alpha = inverse_renderer(vtx, hd_image.cuda())

            # Mask projection using the visibility mask
            mask = F.interpolate(visibility_mask, args.pseudogt_resolution,
                                 mode='bilinear', align_corners=False).permute(0, 2, 3, 1).cuda()
            mask = (mask > 0).any(dim=3, keepdim=True).float()
            inverse_tex *= mask
            inverse_alpha *= mask
            if shrink:
                inverse_alpha *= inverse_tex[..., 3:4]
                inverse_tex = inverse_tex[..., :3]
            
            inverse_tex = inverse_tex.permute(0, 3, 1, 2)
            inverse_alpha = inverse_alpha.permute(0, 3, 1, 2)
            
            # Convert to half to save disk space
            inverse_tex = inverse_tex.half().cpu()
            inverse_alpha = inverse_alpha.half().cpu()
            
            all_gt_scale.append(gt_scale.cpu().clone())
            all_gt_translation.append(gt_translation.cpu().clone())
            all_gt_rotation.append(gt_rot.cpu().clone())
            all_gt_z0.append(gt_z0.cpu().clone())
            all_C.append(C.cpu().clone())
            for i, idx in enumerate(indices):
                idx = idx.item()
                
                pseudogt = {
                    'mesh': mesh_map[i].cpu().clone(),
                    'seg': pred_seg_saved[i].cpu().clone(),
                    'texture': inverse_tex[i].clone(),
                    'texture_alpha': inverse_alpha[i].clone(),
                    'image': inception_image[i].half().clone(),
                    'category': C[i].item(),
                }
                
                np.savez_compressed(os.path.join(pseudogt_dir, f'{idx}'),
                                    data=pseudogt,
                                   )
    
    print('Saving pose metadata...')
    all_paths = sum([x.paths for x in mesh_ds_train.datasets], [])
    poses_metadata = {
        'scale': torch.cat(all_gt_scale, dim=0),
        'translation': torch.cat(all_gt_translation, dim=0),
        'rotation': torch.cat(all_gt_rotation, dim=0),
        'z0': torch.cat(all_gt_z0, dim=0),
        'category': torch.cat(all_C, dim=0),
        'path': all_paths,
    }
    np.savez_compressed(os.path.join(cache_dir, f'poses_metadata_{args.mode}'),
                        data=poses_metadata,
                       )
    
    if 'p3d' not in args.dataset and args.dataset != 'cub':
        print('Saving precomputed FID statistics...')
        with torch.no_grad():
            all_inception_activation = []

            for net_image, seg, inception_image, hd_image, gt_scale, gt_translation, gt_rot, gt_z0, \
                semi_w_batch, semi_mask_batch, indices, C, M, M_mask in tqdm(inception_loader):
                    # Compute inception activations
                    all_inception_activation.append(forward_inception_batch(inception_model, inception_image[:, :3].cuda()/2 + 0.5))

            all_inception_activation = np.concatenate(all_inception_activation, axis=0)

            m_real, s_real = calculate_stats(all_inception_activation)
            fid_save_path = os.path.join(cache_dir, f'precomputed_fid_{inception_resolution}x{inception_resolution}')
            np.savez_compressed(fid_save_path, 
                                stats_m=m_real,
                                stats_s=np.tril(s_real.astype(np.float32)), # The matrix is symmetric, we only store half of it
                                num_images=len(all_inception_activation),
                                resolution=inception_resolution,
                               )
            print(len(all_inception_activation), 'images')
    
    print('Done.')

if writer is not None:
    writer.close()