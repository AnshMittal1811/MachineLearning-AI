from torch.utils.data import DataLoader, ConcatDataset
import argparse
from pathlib import Path
from src.util.misc import collate_fn
from src.datasets import build_dataset
import random
import numpy as np
import os

import cv2
import torch
import src.util.misc as utils
from matplotlib.patches import Polygon
from src.util.box_ops import box_cxcywh_to_xyxy

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


from main import get_args_parser, get_cfg_defaults


def un_normalize(tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor


def process_bbx(boxes, tgt_size):
    boxes = box_cxcywh_to_xyxy(boxes)
    img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct
    return boxes

def process_centroid(centroids, tgt_size):
    img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
    scale_fct = torch.stack([img_w, img_h], dim=1)
    centroids = centroids * scale_fct
    return centroids


def imshow_det_bboxes(img, targets, cmap,
                      min_th=0.6,
                      class_names=None,
                      thickness=2,
                      font_size=8,
                      win_name='COCO visualization',
                      out_file=None):

    text_color = (1, 1, 1)
    bbox_color = (1, 0, 0)


    img = un_normalize(img)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    EPS = 1e-2
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    color = []


    for idx, (mask, bbox, label, valid) in enumerate(zip(targets[0], targets[1], targets[2], targets[3])):
        polygons = []
        # if not valid:
        #     assert mask.sum() == 0
        #     continue

        bbox = process_bbx(bbox, (height, width))
        bbox_int = bbox.numpy().astype(np.int32)[0]
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))

        polygons.append(Polygon(np_poly))
        text_coords = (bbox_int[0], bbox_int[1])

        color.append(bbox_color)

        mask = mask.numpy()[0]
        label = label.item()

        label_text = class_names[label] if class_names is not None else f'class {label}'
        ax.text(text_coords[0],
            text_coords[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')

        color_mask = cmap[idx]
        img[mask != 0] = img[mask != 0] * 0.3 + color_mask * 0.7

        p = PatchCollection(
            polygons, facecolor='none', edgecolors=[(1, 0, 0)], linewidths=thickness)
        ax.add_collection(p)

    plt.imshow(img)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')

    cv2.imwrite(out_file, img)

    plt.close()


def create_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def parse_targets_per_frame(num_annots, num_frames, targets):
    num_gt = num_annots // num_frames

    targets["masks"] = targets["masks"].reshape(num_gt, num_frames, targets["masks"].shape[-2], targets["masks"].shape[-1])
    targets["boxes"] = targets["boxes"].reshape(num_gt, num_frames, targets["boxes"].shape[-1])

    targets["labels"] = targets["labels"].reshape(num_gt, num_frames)
    targets["valid"] = targets["valid"].reshape(num_gt, num_frames)


    return targets


def visualize_transformed_images(idx, clip_images,  targets, out_path, class_name):
    targets = targets[0]
    clip_images = clip_images.tensors.split(1, dim=0)
    clip_images = [image.squeeze(0) for image in clip_images]
    num_images = len(clip_images)
    targets = parse_targets_per_frame(targets["boxes"].shape[0], num_images, targets)

    cmap = create_color_map()
    out_path = os.path.join(out_path, f"idx_{idx}")
    os.makedirs(out_path, exist_ok=True)
    num_tgt = targets["masks"].shape[0]
    if num_tgt == 0:
        return
    for idx, image in enumerate(clip_images):
        frame_results = (targets["masks"][:, idx].chunk(num_tgt),
                         targets["boxes"][:, idx].chunk(num_tgt),
                         targets["labels"][:, idx].chunk(num_tgt),
                         targets["valid"][:, idx].chunk(num_tgt))

        out_image_path = os.path.join(out_path, f"img_{idx:04d}.jpg")
        imshow_det_bboxes(image, frame_results, cmap=cmap, class_names=class_name, out_file=out_image_path)


def main(args, cfg):
    seed = cfg.SEED + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset, num_classes = build_dataset(image_set="TRAIN", cfg=cfg)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 1, drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=0,
                                   worker_init_fn=utils.seed_worker, generator=g)

    num_samples = len(train_dataset)
    categories = {}
    vis_dataset = train_dataset
    if isinstance(train_dataset, ConcatDataset):
        vis_dataset = train_dataset.datasets[1]

    for id_, info in vis_dataset.ytvos.cats.items():
        categories[int(id_) - 1] = info["name"]

    for idx, sample in enumerate(data_loader_train):
        clip_images, targets = sample
        visualize_transformed_images(idx, clip_images, targets, cfg.OUTPUT_DIR, categories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DeVIS training and evaluation script',
                                     parents=[get_args_parser()])
    args_ = parser.parse_args()

    cfg_ = get_cfg_defaults()
    cfg_.merge_from_file(args_.config_file)
    cfg_.merge_from_list(args_.opts)
    cfg_.freeze()
    if cfg_.OUTPUT_DIR:
        Path(cfg_.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    main(args_, cfg_)
