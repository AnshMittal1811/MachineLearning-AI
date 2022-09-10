import os
from typing import Union

import numpy as np
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from torchvision import transforms as T

from lib.structures import BoxList, DepthMap

from . import io, utils

__imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
TransFunc = T.ToPILImage()
NormalizeFunc = T.Normalize(mean=[-mean / std for mean, std in zip(__imagenet_stats['mean'], __imagenet_stats['std'])],
                            std=[1.0 / std for std in __imagenet_stats['std']])


def write_detection_image(image: Union[np.array, torch.Tensor], detections: BoxList, output_file: os.PathLike) -> None:
    bounding_box = detections.bbox
    label = detections.get_field("label")
    masks = detections.get_field("mask2d")

    # TODO: clean up
    image = overlay_detections(image, bounding_box,
                      np.stack([mask.get_mask_tensor().numpy() for mask in masks]),
                      label.numpy(), np.ones_like(label))

    io.write_image(image, output_file)


def overlay_detections(image: np.array, boxes: np.array, masks: np.array, class_ids: np.array) -> np.array:
    # Number of instances
    if boxes is not None:
        num_instances = boxes.shape[0]
    elif masks is not None:
        num_instances = masks.shape[0]
    else:
        num_instances = 0

    if num_instances == 0:
        return image

    # Generate random colors
    colors = utils.random_colors(num_instances)

    fig = Figure()
    ax = fig.gca()
    ax.axis("off")
    fig.tight_layout(pad=0)
    ax.margins(0)  # To remove the huge white borders

    masked_image = image.astype(np.uint32).copy()

    for i in range(num_instances):
        color = colors[i]

        # Bounding box
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i]
            patch = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle="dashed",
                                      edgecolor=color, facecolor="none")
            ax.add_patch(patch)

            # TODO: Label
            # if class_ids is not None:
            #     class_id = class_ids[i]
            #     label = class_names[int(class_id)]
            #     caption = f"{label}"
            #     ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")

        # Mask
        if masks is not None:
            mask = masks[i, :, :]
            masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask

    fig.canvas.draw()
    ax.imshow(masked_image.astype(np.uint8))
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image_from_plot


def apply_mask(image: np.array, mask: np.array, color, alpha=0.5):

    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def write_rgb_image(image: Union[np.array, torch.Tensor], output_file: os.PathLike) -> None:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.array(TransFunc(NormalizeFunc(image)))

    io.write_image(image, output_file)


def write_depth(depth_map: Union[DepthMap, np.array, torch.Tensor], output_file: os.PathLike) -> None:
    if isinstance(depth_map, DepthMap):
        depth_map = depth_map.get_tensor()

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    io.write_image(depth_map, output_file, kwargs={"cmap": "rainbow", "vmin": 1.0, "vmax": 6.0})
