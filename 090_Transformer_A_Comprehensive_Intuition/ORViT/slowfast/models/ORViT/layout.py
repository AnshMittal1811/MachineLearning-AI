#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import numpy as np

"""
Functions for computing image layouts from object vectors, bounding boxes,
and segmentation masks. These are used to compute course scene layouts which
are then fed as input to the cascaded refinement network.
"""


def boxes_to_layout(vecs, boxes, H, W=None, pooling='sum'):
    """
    Inputs:
        - vecs: Tensor of shape (O, C) giving vectors
        - boxes: Tensor of shape (O, 4) giving bounding boxes in the format [x0, y0, x1, y1] in the [0, 1] coordinate space
        - H, W: Size of the output

    Returns:
        - out: Tensor of shape (N, D, H, W)
    """

    # Remove illegal boxes [0, 0, 0, 0]
    legal_boxes = (boxes != 0).any(dim=-1)
    boxes = boxes[legal_boxes]
    vecs = vecs[legal_boxes]

    O, D = vecs.size()
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W) # [O, H, W, 2]

    # If we don't add extra spatial dimensions here then out-of-bounds
    # elements won't be automatically set to 0
    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in.float(), grid.float(), align_corners=True)  # (O, D, H, W)

    # Explicitly masking makes everything quite a bit slower.
    # If we rely on implicit masking the interpolated boxes end up
    # blurred around the edges, but it should be fine.
    # mask = ((X < 0) + (X > 1) + (Y < 0) + (Y > 1)).clamp(max=1)
    # sampled[mask[:, None]] = 0

    out = _pool_samples(sampled, pooling=pooling)

    return out


def masks_to_layout(vecs, boxes, masks, H, W=None, pooling='sum', test_mode=False):
    """
  Inputs:
  - vecs: Tensor of shape (O, D) giving vectors
  - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
    [x0, y0, x1, y1] in the [0, 1] coordinate space
  - masks: Tensor of shape (O, M, M) giving binary masks for each object
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - H, W: Size of the output image.

  Returns:
  - out: Tensor of shape (N, D, H, W)
  """
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid, align_corners=True)

    if test_mode:
        clean_mask_sampled = F.grid_sample(masks.float().view(O, 1, M, M), grid, align_corners=True)
    else:
        clean_mask_sampled = None
    out = _pool_mask_samples(sampled, clean_mask_sampled, pooling=pooling)
    return out


def _boxes_to_grid(boxes, H, W):
    """
  Input:
  - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
    format in the [0, 1] coordinate space
  - H, W: Scalars giving size of output

  Returns:
  - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
  """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def boxes_to_mask(boxes, H, W):
    """
  Input:
  - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, w, h]
    format in the [0, 1] coordinate space
  - H, W: Scalars giving size of output

  Returns:
  - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
  """

    # Remove illegal boxes [0, 0, 0, 0]
    legal_boxes = (boxes != 0).any(dim=-1)
    boxes = boxes[legal_boxes]

    O = boxes.size(0)
    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    heat_maps = torch.zeros((O, H, W)).to(boxes)
    for i in range(O):
        x0_, y0_ = int(x0[i] * W), int(y0[i] * H)
        ww_, hh_ = int(ww[i] * W), int(hh[i] * H)
        x1_, y1_ = x0_ + ww_, y0_ + hh_
        heat_maps[i][y0_:y1_, x0_: x1_] = 1
    return torch.max(heat_maps, dim=0)[0]


def _pool_mask_samples(samples, clean_mask_sampled, pooling='sum'):
    """
    Input:
    - samples: FloatTensor of shape (O, D, H, W)
    - obj_to_img: LongTensor of shape (O,) with each element in the range
      [0, N) mapping elements of samples to output images

    Output:
    - pooled: FloatTensor of shape (N, D, H, W)
    """
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    obj_to_img = torch.LongTensor([0]).repeat(O).to(device)
    N = obj_to_img.data.max().item() + 1

    if clean_mask_sampled is None:
        # Use scatter_add to sum the sampled outputs for each image
        out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
        idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
        out = out.scatter_add(0, idx, samples)
    else:
        _, d, h, w = samples.shape
        all_out = []
        for i in range(N):
            mass = [torch.sum(samples[j, :, :, :]).item() for j in range(O)]
            argsort = np.argsort(mass)
            result = torch.zeros((d, h, w), device=samples.device, dtype=samples.dtype)
            result_clean = torch.zeros((h, w), device=samples.device, dtype=samples.dtype)
            for j in argsort:
                masked_mask = (result_clean == 0).float() * (clean_mask_sampled[j, 0] > 0.5).float()
                result_clean += masked_mask
                result += samples[j] * masked_mask
            all_out.append(result)
        out = torch.stack(all_out)

    if pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out


def _pool_samples(samples, pooling='sum'):
    """
  Input:
  - samples: FloatTensor of shape (O, D, H, W)
  - obj_to_img: LongTensor of shape (O,) with each element in the range
    [0, N) mapping elements of samples to output images

  Output:
  - pooled: FloatTensor of shape (N, D, H, W)
  """
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    obj_to_img = torch.LongTensor([0]).repeat(O).to(device)
    N = 1

    # Use scatter_add to sum the sampled outputs for each image
    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        # Divide each output mask by the number of objects; use scatter_add again
        # to count the number of objects per image.
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out

