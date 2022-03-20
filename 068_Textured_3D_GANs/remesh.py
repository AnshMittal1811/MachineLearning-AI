import os
# Workaround for PyTorch spawning too many threads
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
import math

from rendering.parallel_rasterizer import ParallelKaolinRasterizer
from rendering.mesh_template import MeshTemplate
from utils.losses import loss_flat, LaplacianLoss
from rendering.utils import qrot, qmul
from data.definitions import class_names
from skimage.segmentation import flood_fill

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, default='autodetect', help='path to initial mesh template')
parser.add_argument('--image_resolution', type=int, default=256)
parser.add_argument('--symmetric', type=bool, default=True)
parser.add_argument('--gpu_ids', type=str, default='0', help='comma-separated')
parser.add_argument('--mode', type=str, required=True, help='single or multiple templates (singletpl|multitpl)')
parser.add_argument('--classes', type=str, default='all', help='all (default), or comma-separated list')

args = parser.parse_args()

gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
print('Using {} GPUs: {}'.format(len(gpu_ids), gpu_ids))
torch.cuda.set_device(min(gpu_ids))

assert args.mode in ['multitpl', 'singletpl']
multi_template = args.mode == 'multitpl'

if args.mesh_path == 'autodetect':
    args.mesh_path = 'mesh_templates/uvsphere_31rings.obj'
    print('Using initial mesh topology', args.mesh_path)

    
def render_views(mesh, raw_vtx, rot, hardmask=False, closure=None, **kwargs):
    assert raw_vtx.shape[0] == rot.shape[0]
    assert len(raw_vtx.shape) == 3
    assert len(rot.shape) == 3
    assert raw_vtx.shape[-1] == 3
    assert rot.shape[-1] == 4
    
    bs = rot.shape[0]
    num_views = rot.shape[1]
    rot = rot.expand(raw_vtx.shape[0], -1, -1).flatten(0, 1)
    raw_vtx = raw_vtx.unsqueeze(1).expand(-1, num_views, -1, -1).flatten(0, 1)

    vtx = qrot(rot, raw_vtx) / math.sqrt(2)
    vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)
    tex = None
    ret = mesh.forward_renderer(renderer, vtx, tex, return_hardmask=hardmask, num_gpus=len(gpu_ids),
                                closure=closure, **kwargs)
    if closure is None:
        pred_rgb, pred_alpha = ret
        pred_alpha = pred_alpha.view(bs, num_views, *pred_alpha.shape[1:])
        return pred_alpha
    else:
        return ret


renderer = nn.DataParallel(ParallelKaolinRasterizer(args.image_resolution, mode='alpha'), gpu_ids)

if args.classes == 'all':
    selected_classes = class_names
else:
    selected_classes = args.classes.split(',')
    for cl in selected_classes:
        assert cl in class_names, f'Invalid class {cl}'

classes = {}
class_is_aligned = {}
for cl in selected_classes:
    classes[cl] = []
    class_is_aligned[cl] = False
    
# The mesh templates of animals are already pre-aligned, there is no need to find optimal alignment
aligned_classes = ['bird', 'sheep', 'elephant', 'zebra', 'horse', 'cow', 'bear', 'giraffe']
for cl in aligned_classes:
    if cl in class_is_aligned:
        class_is_aligned[cl] = True


# Load mesh templates for each class
for cl in classes.keys():
    for suf in range(1, 100):
        fname = f'mesh_templates/classes/{cl}{suf}.obj'
        if os.path.isfile(fname):
            classes[cl].append(fname)
            if not multi_template:
                # Load only first template
                break
        else:
            break
            
# Print summary
print('----------- Summary of selected classes -----------')
for k, v in classes.items():
    print(f'{k}: loaded {len(v)} template(s)')
print('---------------------------------------------------')


output_dir = f'cache/remeshed_templates/{args.mode}'
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

print('Multi-template setting:', multi_template)
print('Output dir:', output_dir)

for selected_class in classes.keys():
    print('Processing', selected_class)
    mesh_paths = classes[selected_class]

    torch.manual_seed(1)

    target = []
    initial_scaling = None
    num_views = 64
    is_aligned = class_is_aligned[selected_class]
    target_rots = F.normalize(torch.randn(len(mesh_paths), num_views, 4), dim=-1).cuda() # Random viewpoints

    templates = []
    bbox_min = None
    bbox_max = None
    for i, mesh_path in enumerate(mesh_paths):
        source_mesh = MeshTemplate(mesh_path, is_symmetric=False)
        templates.append(source_mesh)
        vertices = source_mesh.mesh.vertices
        with torch.no_grad():
            bbox_lower = vertices.min(dim=0, keepdim=True)[0]
            bbox_higher = vertices.max(dim=0, keepdim=True)[0]
            if i == 0:
                bbox_min = bbox_lower
                bbox_max = bbox_higher
            else:
                bbox_min = torch.min(bbox_lower, bbox_min)
                bbox_max = torch.max(bbox_lower, bbox_max)

    for i, source_mesh in enumerate(templates):

        # Add backfaces
        source_mesh.mesh.faces = torch.cat((source_mesh.mesh.faces, source_mesh.mesh.faces[..., [2, 1, 0]]), dim=0)

        # Dummy UVs
        source_mesh.mesh.face_textures = torch.zeros_like(source_mesh.mesh.faces)
        source_mesh.mesh.uvs = torch.zeros((1, 2), device=source_mesh.mesh.face_textures.device)

        # Normalize source mesh
        def normalize_vertices(vertices):
            with torch.no_grad():
                if is_aligned:
                    bbox_lower = bbox_min
                    bbox_higher = bbox_max
                else:
                    bbox_lower = vertices.min(dim=0, keepdim=True)[0]
                    bbox_higher = vertices.max(dim=0, keepdim=True)[0]
                center = (bbox_lower + bbox_higher)/2
                center[..., 0] = 0 # No left-right shift
                vertices -= center
                vertices /= vertices.abs().max()

                return vertices.abs().max(dim=0)[0]

        scaling = normalize_vertices(source_mesh.mesh.vertices)
        if initial_scaling is None:
            initial_scaling = scaling
        else:
            initial_scaling = torch.max(initial_scaling, scaling)

        with torch.no_grad():
            mesh_targets = render_views(source_mesh, source_mesh.mesh.vertices.unsqueeze(0), target_rots[i:i+1], hardmask=True)
        target.append(mesh_targets)


    target = torch.cat(target, dim=0)
    
    # Fill holes/gaps that might mess up the result
    images = []
    for im in target.cpu().flatten(0, 1).numpy():
        images.append(torch.FloatTensor(flood_fill(im[0], (0, 0), 1)))
    target = target + (1 - torch.stack(images, dim=0).view(target.shape).cuda())


    mesh_template = MeshTemplate(args.mesh_path, is_symmetric=True)

    def pdist(vertices):
        # Sparse L2 mode
        dists = (vertices.unsqueeze(0) - vertices.unsqueeze(1)).norm(dim=-1)
        return dists.mean()


    # Mesh to optimize
    source = mesh_template.mesh.vertices.clone().unsqueeze(0).expand(target.shape[0], -1, -1).contiguous().requires_grad_()
    print(source.shape, target.shape)

    alignment_t = torch.zeros(source.shape[0], 1, 3).cuda().requires_grad_()
    alignment_s1 = torch.ones(1, 1, 3).cuda().requires_grad_()
    alignment_s2 = torch.ones(source.shape[0], 1, 1).cuda().requires_grad_()
    alignment_s1.data *= initial_scaling

    pdist_t = torch.zeros(source.shape[0], 1, 3).cuda().requires_grad_()
    pdist_s = torch.ones(source.shape[0], 1, 1).cuda().requires_grad_()

    # Find optimal rigid alignment between meshes before actually optimizing individual vertices
    # (helps with local minima)
    lr = 0.0001
    optimizer = optim.SGD([alignment_t, alignment_s1, alignment_s2, pdist_t, pdist_s], lr=lr, momentum=0.9)
    criterion = nn.L1Loss()

    pdist_coeff = 0.001 if multi_template else 0
    print('Computing alignment...')
    for i in range(1000):
        optimizer.zero_grad()

        renderer.module.set_sigma_mul(1.0)

        source_translated = alignment_s1*alignment_s2*source.detach() + alignment_t
        pred = render_views(mesh_template, source_translated, target_rots)
        recon_loss = criterion(pred, target)

        pdist_loss = pdist(source_translated*pdist_s + pdist_t)

        loss = recon_loss + pdist_coeff*pdist_loss
        loss.backward()

        alignment_t.grad /= alignment_t.grad.norm(dim=-1, keepdim=True) + 1e-6
        alignment_s1.grad /= alignment_s1.grad.norm(dim=-1, keepdim=True) + 1e-6
        alignment_s2.grad /= alignment_s2.grad.norm(dim=-1, keepdim=True) + 1e-6

        if source.shape[0] > 1 and pdist_coeff > 0 and multi_template:
            pdist_t.grad /= pdist_t.grad.norm(dim=-1, keepdim=True) + 1e-6
            pdist_s.grad /= pdist_s.grad.norm() + 1e-6

        optimizer.step()

        # Reproject to enforce symmetry
        with torch.no_grad():
            alignment_t.data[..., 0] = 0
            pdist_t.data[..., 0] = 0
            if multi_template and not is_aligned:
                pdist_s.data /= pdist_s.data.max()
                pdist_s.data.clamp_(min=0.8) # Avoid extreme scales
                pdist_t.data -= pdist_t.data.mean(dim=0, keepdim=True) # Re-center
            else:
                pdist_s.data[:] = 1
                pdist_t.data[:] = 0
            if is_aligned or not multi_template:
                alignment_s2.data[:] = 1

        if i % 100 == 0:
            print('[{}] lr {:.5f} recon {:.5f} pdist {:.5f}'.format(i, lr,
                recon_loss.item(), pdist_loss.item()))

    print(loss.item())

    # Perform alignment
    with torch.no_grad():
        source.data[:] = source.data * alignment_s1.data * alignment_s2.data + alignment_t.data
        alignment_s1.data[:] = 1
        alignment_s2.data[:] = 1
        alignment_t.data[:] = 0

    # Reset
    alignment_t = torch.zeros(source.shape[0], 1, 3).cuda().requires_grad_()
    alignment_s = torch.ones(source.shape[0], 1, 1).cuda().requires_grad_()


    # Optimize vertices
    lr = 0.0001
    optimizer = optim.SGD([source, alignment_t, alignment_s, pdist_t, pdist_s], lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    grid_laplacian, uv_connectivity = mesh_template.compute_grid_laplacian()
    lap_regularizer = LaplacianLoss(grid_laplacian).cuda()

    def length_regularizer(faces, vertices):
        grid_positions = source[:, uv_connectivity]
        tv_y = (grid_positions[:, 1:, :] - grid_positions[:, :-1, :]).abs()
        tv_x = (grid_positions[:, :, 1:] - grid_positions[:, :, :-1]).abs()
        return tv_x.mean() + tv_y.mean()


    inv_mask = torch.FloatTensor([-1, 1, 1]).to(source.device) # Symmetry mask

    loss_curve = []

    lap_coeff = 0.003
    len_coeff = 0.01
    pdist_coeff = 0.001 if multi_template else 0
    sigma_mul = 1

    lr_warmup = True
    lr_warmup_stop = 0.0005

    print('Optimizing vertices...')
    for i in range(100000):
        optimizer.zero_grad()

        renderer.module.set_sigma_mul(sigma_mul)

        source_translated = alignment_s*source + alignment_t
        pred = render_views(mesh_template, source_translated, target_rots)
        recon_loss = criterion(pred, target)

        flat_loss = loss_flat(mesh_template.mesh, mesh_template.compute_normals(source))
        length_loss = length_regularizer(mesh_template.mesh.faces, source_translated)
        laplacian_loss = lap_regularizer(source_translated).mean()
        pdist_loss = pdist(source_translated*pdist_s + pdist_t)

        loss = recon_loss + 0.00001*flat_loss + len_coeff*length_loss + pdist_coeff*pdist_loss + lap_coeff*laplacian_loss
        loss.backward()

        source.grad /= source.grad.norm(dim=2, keepdim=True) + 1e-6

        alignment_t.grad /= alignment_t.grad.norm(dim=-1, keepdim=True) + 1e-6
        alignment_s.grad /= alignment_s.grad.norm(dim=-1, keepdim=True) + 1e-6

        if source.shape[0] > 1 and pdist_coeff > 0 and multi_template:
            pdist_t.grad /= pdist_t.grad.norm(dim=-1, keepdim=True) + 1e-6
            pdist_s.grad /= pdist_s.grad.norm() + 1e-6

        optimizer.step()

        # Reproject to enforce symmetry
        with torch.no_grad():
            if args.symmetric:
                avg_lr = (source[:, mesh_template.pos_indices] + source[:, mesh_template.neg_indices]*inv_mask)/2
                avg_lr[avg_lr[..., 0] < 0] *= inv_mask # Avoid violations across symmetry axis
                source.data[:, mesh_template.pos_indices] = avg_lr
                source.data[:, mesh_template.neg_indices] = avg_lr*inv_mask
                source.data *= mesh_template.symmetry_mask
            alignment_t.data[..., 0] = 0
            pdist_t.data[..., 0] = 0
            if multi_template and not is_aligned:
                pdist_s.data /= pdist_s.data.max()
                pdist_s.data.clamp_(min=0.8) # Avoid extreme scales
                pdist_t.data -= pdist_t.data.mean(dim=0, keepdim=True) # Re-center
            else:
                pdist_s.data[:] = 1
                pdist_t.data[:] = 0

        if i % 100 == 0:
            print('[{}] lr {:.5f} recon {:.5f} flat {:.5f} lap {:.5f} len {:.5f} pdist {:.5f}'.format(i, lr,
                recon_loss.item(), flat_loss.item(), laplacian_loss.item(), length_loss.item(), pdist_loss.item()))

        if not lr_warmup:
            decay_rate = 0.9999
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
            lr *= decay_rate

            lap_coeff *= decay_rate
            sigma_mul *= decay_rate

        else:
            lr_delta = 0.000001
            for param_group in optimizer.param_groups:
                param_group['lr'] += lr_delta
            lr += lr_delta
            if lr >= lr_warmup_stop:
                lr_warmup = False

        if lr < 1e-4:
            break

    print(loss.item())

    # Perform alignment
    with torch.no_grad():
        source.data[:] = source.data * alignment_s.data + alignment_t.data
        alignment_s.data[:] = 1
        alignment_t.data[:] = 0

    # Align different templates (has an effect only in multi-template setting)
    with torch.no_grad():
        source.data[:] = source.data * pdist_s.data + pdist_t.data
        pdist_s.data[:] = 1
        pdist_t.data[:] = 0

    # Post-normalization: ensure that longest side is 1 & re-center.
    source_post = source.detach().clone()
    bbox_lower = source_post.flatten(0, 1).min(dim=0, keepdim=True)[0]
    bbox_higher = source_post.flatten(0, 1).max(dim=0, keepdim=True)[0]
    center = (bbox_lower + bbox_higher)/2
    source_post -= center
    source_post /= source_post.abs().max()

    # Save result as PyTorch tensor
    torch.save(source_post.cpu(), f'{output_dir}/{selected_class}_templates.pth')
    # Save result as .obj (not used in practice, but useful for debugging)
    mesh_template.export_obj(f'{output_dir}/{selected_class}_templates', source_post.detach().cpu(), texture=None)
    print('Saved.')

print('Done.')