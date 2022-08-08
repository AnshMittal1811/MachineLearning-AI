"""
YouTubeVOS has a label structure that is more complicated than DAVIS 
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set

Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

Validation/test set.
"""


import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import PCVOS
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from util.tensor_util import unpad
from inference_core_yv import InferenceCore_Per_Clip

from progressbar import progressbar
import matplotlib.pyplot as plt
from dataset.range_transform import inv_im_trans
import pdb
import time

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/s03_pcvos_ytvos.pth')
parser.add_argument('--yv_path', default='./data/YouTube')

parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')
parser.add_argument('--predict_all', help=
"""
We will use all the frames if this is set to true.
""", action='store_true')

parser.add_argument('--output')
parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
# Set the clip length
parser.add_argument('--clip_length', default=5, type=int)
# Intra-Clip Refinement (ICR)
parser.add_argument('--refine_clip', default='ICR', type=str, choices=('None', 'ICR'),
                    help='refine features of a clip')
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--value_dim', default=512, type=int, 
                    help="Size of the value embeddings (output dim of the swin-transformer)")
parser.add_argument('--T_window', help='Temporal Window size', default=2, type=int, nargs='*')
parser.add_argument('--S_window', help='Spatial Window size', default=7, type=int, nargs='*')
parser.add_argument('--shared_proj', help='Whether shared proj for q / k', action='store_true')
# Progressive Memory Matching Mechanism (PMM)
parser.add_argument('--memory_read', default='PMM', type=str, choices=('parallel', 'PMM'), 
                    help='type of memory read mechanism')
# Measure FPS
parser.add_argument('--time', action='store_true')


args = parser.parse_args()

yv_path = args.yv_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Load the json if we have to
if not args.output_all:
    with open(path.join(yv_path, args.split, 'meta.json')) as f:
        meta = json.load(f)['videos']

# Setup Dataset
test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = args.top
prop_model = PCVOS(vars(args)).cuda().eval()
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0
# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb']
        msk = data['gt'][0]
        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])
        gt_obj = info['gt_obj']
        size = info['size']

        # print('Processing', name, '...')
        # Load the required set of frames (if we don't need all)
        req_frames = None
        if not args.output_all:
            req_frames = []
            objects = meta[name]['objects']
            for key, value in objects.items():
                req_frames.extend(value['frames'])

            # Map the frame names to indices
            req_frames_names = set(req_frames)
            req_frames = []
            for fi in range(rgb.shape[1]):
                frame_name = info['frames'][fi][0][:-4]
                if frame_name in req_frames_names:
                    req_frames.append(fi)
            req_frames = sorted(req_frames)

        req_frames_pred = None if args.predict_all else req_frames

        if args.time:
            torch.cuda.synchronize()
            process_begin = time.time()

        # Frames with labels, but they are not exhaustively labeled
        frames_with_gt = sorted(list(gt_obj.keys()))

        processor = InferenceCore_Per_Clip(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                    mem_every=args.clip_length, include_last=args.include_last, 
                                    req_frames=req_frames_pred, clip_length=args.clip_length)
        # min_idx tells us the starting point of propagation
        # Propagating before there are labels is not useful
        min_idx = 99999
        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            # Note that there might be more than one label per frame
            obj_idx = gt_obj[frame_idx][0].tolist()
            # Map the possibly non-continuous labels into a continuous scheme
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            # Append the background label
            with_bg_msk = torch.cat([
                1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                msk[:,frame_idx],
            ], 0).cuda()

            # We perform propagation from the current frame to the next frame with label
            if i == len(frames_with_gt) - 1:
                processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
            else:
                processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

        # Do unpad -> upsample to original size (we made it 480p)
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        if args.time:
            torch.cuda.synchronize()
            total_process_time += time.time() - process_begin
            if args.predict_all:
                total_frames += out_masks.shape[0]
            else:
                total_frames += len(req_frames_pred)
            # total_frames += out_masks.shape[0]

        # Remap the indices to the original domain
        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects+1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks==i] = backward_idx
        
        # Save the results
        this_out_path = path.join(out_path, 'Annotations', name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            if f >= min_idx:
                if args.output_all or (f in req_frames):
                    img_E = Image.fromarray(idx_masks[f])
                    img_E.putpalette(palette)
                    img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

        del rgb
        del msk
        del processor

if args.time:
    print('Total processing time: ', total_process_time)
    print('Total processed frames: ', total_frames)
    print('FPS: ', total_frames / total_process_time)