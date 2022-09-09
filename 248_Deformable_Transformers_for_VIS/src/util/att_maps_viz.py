import numpy as np
import os
import cv2
import torch
import pycocotools.mask as mask_tools

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
import torch.nn.functional as F
from . import box_ops
from .viz_utils import create_color_map, get_most_left_coordinate


def imshow_det_masks_for_tracks(img, instance, cmap,
                                class_names=None,
                                thickness=2,
                                font_size=11,
                                win_name='COCO visualization',
                                out_file=None):
    cmap = cmap[1:]
    text_color = (1, 1, 1)
    if len(img.shape) > 3:
        img = torch.squeeze(img)

    # img = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
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

    if instance["mask"] is None:
        return
    if isinstance(instance["mask"], dict):
        mask = mask_tools.decode(instance["mask"])
    else:
        mask = instance["mask"]

    show_bbox = "bbox" in instance
    show_centroid = "centroid_point" in instance

    if show_bbox:
        polygons = []
        bbox = instance["bbox"]
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))

        polygons.append(Polygon(np_poly))
        text_coords = (bbox_int[0], bbox_int[1])
    else:
        text_coords = get_most_left_coordinate(mask)
        text_coords = (text_coords[1], text_coords[0])

    if show_centroid:
        centroids = []
        centroid = instance["centroid_point"]
        centroids.append(Circle((centroid[0], centroid[1]), radius=5, fill=True))

    label_text = class_names[
        instance["category_id"]] if class_names is not None else f'class {instance["category_id"]}'
    # label_text += f'|{instance["score"]:.02f}'

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
    color_mask = cmap[instance["instance_id"]]
    img[mask != 0] = img[mask != 0] * 0.3 + color_mask * 0.7

    plt.imshow(img)
    if show_bbox:
        p = PatchCollection(
            polygons, facecolor='none', edgecolors=[(1, 0, 0)], linewidths=thickness)
        ax.add_collection(p)

    if show_centroid:
        p = PatchCollection(
            centroids, edgecolors=[(0, 0, 1)])
        ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    cv2.imwrite(out_file, img)
    plt.close()


def read_image_for_axis(track, t, folder_images, images_path, cmap):
    image = cv2.imread(os.path.join(folder_images, images_path[t]))
    mask = track.masks[t]
    if mask is None:
        mask = np.zeros_like(image)
    else:
        mask = mask_tools.decode(mask)

    color_mask = cmap[track.get_id()]
    image[mask != 0] = image[mask != 0] * 0.65 + color_mask * 0.35

    return image


def create_masks(idx, folder_images, images_path, tracks, out_path, class_name):
    video_name = images_path[0].split("/")[0]
    cmap = create_color_map()[1:]

    out_folder = os.path.join(out_path, video_name, f"window_{idx}")
    os.makedirs(out_folder, exist_ok=True)

    for t, image_path in enumerate(images_path):

        for track_id, track in enumerate(tracks[:1]):
            mask = track.masks[t]
            mask = mask_tools.decode(mask)
            mask = (mask * 255).astype(np.uint8)
            hp = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            name = image_path.split("/")[1].split(".")[0]
            cv2.imwrite(os.path.join(out_folder, f"frame_{name}.png"), hp)


def plot_both_masks(idx, folder_images, images_path, tracks, out_path, class_name):
    video_name = images_path[0].split("/")[0]
    cmap = create_color_map()[1:]

    out_folder = os.path.join(out_path, video_name, f"window_{idx}")
    os.makedirs(out_folder, exist_ok=True)

    fig, axs = plt.subplots(1, len(images_path), figsize=(40, 30))
    for idx, image_path in enumerate(images_path):
        image = cv2.imread(os.path.join(folder_images, image_path))

        for track_id, track in enumerate(tracks):
            mask = track.masks[idx]
            if mask is None:
                mask = np.zeros_like(image)
            else:
                mask = mask_tools.decode(mask)

            bbox = track.boxes[idx]
            if isinstance(bbox, list):
                bbox = np.array(bbox)

            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))

            color_mask = cmap[track.get_id()]
            bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255),
                          float(color_mask[0] / 255))]

            p = PatchCollection(
                [Polygon(np_poly)], facecolor='none', edgecolors=bbx_color, linewidths=1)

            axs[idx].add_collection(p)

            # color_mask = cmap[track.get_id()]
            image[mask != 0] = image[mask != 0] * 0.65 + color_mask * 0.35

        image = image[..., ::-1].copy()
        axs[idx].imshow(image)
        axs[idx].axis('off')

    plt.savefig(os.path.join(out_folder, f"frame_{idx}.png"), bbox_inches='tight')


def process_ref_point(ref_point, image_shape):
    centroid_ref_point = ref_point[:2]
    centroid_ref_point = centroid_ref_point.cpu() * torch.tensor([image_shape[1], image_shape[0]])

    boxes = box_ops.box_cxcywh_to_xyxy(ref_point)
    boxes = boxes.cpu() * torch.tensor(
        [image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
    bbox_int = boxes.numpy().astype(np.int32)
    poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]], [bbox_int[2], bbox_int[3]],
            [bbox_int[2], bbox_int[1]]]
    np_poly = np.array(poly).reshape((4, 2))

    return np_poly, centroid_ref_point


def visualize_clips_with_att_maps_merged_res_v2(idx, folder_images, images_path, tracks, layer,
                                                merge_resolution, out_path, class_name):
    video_name = images_path[0].split("/")[0]
    # cmap = create_color_map()[1:]

    cmap = np.array([[227, 0, 227], [151, 208, 119]])

    window_idx = idx
    out_folder_track = os.path.join(out_path, video_name)
    os.makedirs(out_folder_track, exist_ok=True)

    border_color = (255/255, 128/255, 0/255)

    num_frames = len(images_path)
    target_width = None
    image_shape = None
    if target_width:
        image = cv2.imread(os.path.join(folder_images, images_path[0]))
        height, width = image.shape[:2]
        resize_factor = target_width / width
        image_shape = (int(height * resize_factor), int(width * resize_factor))

    # fig, axs = plt.subplots(ncols=num_frames, nrows=num_frames + 1, figsize=(40, 23))
    fig, axs = plt.subplots(ncols=num_frames, nrows=num_frames + 1, figsize=(38, 28))

    # First row images
    for i, t in enumerate(range(num_frames)):
        image = cv2.imread(os.path.join(folder_images, images_path[t]))
        if image_shape:
            image = cv2.resize(image, (image_shape[1], image_shape[0]))
        image_shape = image.shape[:2]
        for id_, track_to_put in enumerate(tracks):
            mask = track_to_put.masks[i]
            if mask is None:
                mask = np.zeros_like(image)
            else:
                mask = mask_tools.decode(mask)
            if target_width:
                mask = np.resize(mask, image_shape)
            bbox = track_to_put.boxes[i]
            if isinstance(bbox, list):
                bbox = np.array(bbox)

            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))

            color_mask = cmap[track_to_put.get_id()]
            bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255),
                          float(color_mask[0] / 255))]

            p = PatchCollection(
                [Polygon(np_poly)], facecolor='none', edgecolors=bbx_color, linewidths=4)

            axs[0, i].add_collection(p)

            image[mask != 0] = image[mask != 0] * 0.65 + color_mask * 0.35

        image = image[..., ::-1].copy()
        axs[0, i].imshow(image)
        # axs[0, i].axis('off')
        axs[0, i].set_yticklabels([])
        axs[0, i].set_xticklabels([])

        axs[0, i].axes.xaxis.set_visible(False)
        axs[0, i].axes.yaxis.set_visible(False)


        axs[0, i].patch.set_edgecolor(border_color)
        axs[0, i].patch.set_linewidth('13')

    for i, curr_frame in enumerate(range(num_frames)):

        temporal_frames = [t_1 for t_1 in range(-curr_frame, num_frames - curr_frame)]
        temporal_counter = 0

        for idx, temporal_frame in enumerate(temporal_frames):
            att_map_resolution = tracks[0].spatial_shapes[merge_resolution]
            feature_map = torch.zeros((att_map_resolution[0], att_map_resolution[1]),
                                      device=tracks[0].temporal_positions.device)

            if temporal_frame == 0:
                for track in tracks:
                    curr_frame_positions = track.curr_position[curr_frame].flatten(0, 2)
                    curr_frame_positions[:, 0].clamp_(0, att_map_resolution[1] - 1)
                    curr_frame_positions[:, 1].clamp_(0, att_map_resolution[0] - 1)
                    curr_frame_weights = track.curr_att_weights[curr_frame].flatten(0, 2)

                    for position, weight in zip(curr_frame_positions, curr_frame_weights):
                        feature_map[position[1], position[0]] += weight

                # feature_map = (feature_map / feature_map.max())

                feature_map = \
                    F.interpolate(feature_map[None, None], image_shape, mode="bilinear",
                                  align_corners=False).detach().to("cpu")[0, 0]

                axs[i + 1, idx].imshow(feature_map, cmap='cividis')

                for track in tracks:
                    color_track = cmap[track.get_id()]
                    bbx_track_color = [(float(color_track[2] / 255), float(color_track[1] / 255),
                                        float(color_track[0] / 255))]

                    ref_point = track.ref_point[curr_frame]

                    if ref_point.shape[-1] == 4:
                        np_poly, centroid_ref_point = process_ref_point(ref_point, image_shape)

                        p = PatchCollection(
                            [Polygon(np_poly)], facecolor='none', edgecolors=bbx_track_color,
                            linewidths=3.5, linestyles='solid')
                        axs[i + 1, idx].add_collection(p)

                    else:
                        centroid_ref_point = torch.round(
                            ref_point * torch.tensor([image_shape[1], image_shape[0]],
                                                     device=ref_point.device)).cpu()

                    mymarker = axs[i + 1, idx].scatter(centroid_ref_point[0],
                                                       centroid_ref_point[1], s=700,
                                                       color=bbx_track_color[0],
                                                       marker='x', linewidths=2.5)
                    axs[i + 1, idx].add_artist(mymarker)
                    # axs[i + 1, idx].axis('off')

                axs[i + 1, idx].set_yticklabels([])
                axs[i + 1, idx].set_xticklabels([])

                axs[i + 1, idx].axes.xaxis.set_visible(False)
                axs[i + 1, idx].axes.yaxis.set_visible(False)

                axs[i + 1, idx].patch.set_edgecolor(border_color)
                axs[i + 1, idx].patch.set_linewidth('13')

            else:
                for track in tracks:
                    temporal_positions = track.temporal_positions[curr_frame, :,
                                         temporal_counter].flatten(0, 2)
                    temporal_positions[:, 0].clamp_(0, att_map_resolution[1] - 1)
                    temporal_positions[:, 1].clamp_(0, att_map_resolution[0] - 1)
                    temporal_frame_weights = track.temporal_att_weights[curr_frame, :,
                                             temporal_counter].flatten(0, 2)

                    for position, weight in zip(temporal_positions, temporal_frame_weights):
                        feature_map[position[1], position[0]] += weight

                # feature_map = (feature_map / feature_map.max())
                feature_map = \
                    F.interpolate(feature_map[None, None], image_shape, mode="bilinear",
                                  align_corners=False).detach().to("cpu")[0, 0]
                # feature_map = feature_map.to("cpu")
                axs[i + 1, idx].imshow(feature_map, cmap='cividis')

                axs[i + 1, idx].set_yticklabels([])
                axs[i + 1, idx].set_xticklabels([])

                axs[i + 1, idx].axes.xaxis.set_visible(False)
                axs[i + 1, idx].axes.yaxis.set_visible(False)

                # axs[i + 1, idx].axis('off')

                for track in tracks:
                    color_track = cmap[track.get_id()]
                    bbx_track_color = [(float(color_track[2] / 255), float(color_track[1] / 255),
                                        float(color_track[0] / 255))]
                    # Allows visualizing instance aware decoder attention
                    if layer == 0:
                        ref_point_frame = curr_frame
                    else:
                        ref_point_frame = curr_frame + temporal_frame

                    ref_point = track.ref_point[ref_point_frame]

                    if ref_point.shape[-1] == 4:
                        np_poly, centroid_ref_point = process_ref_point(ref_point, image_shape)
                        p = PatchCollection(
                            [Polygon(np_poly)], facecolor='none', edgecolors=bbx_track_color,
                            linewidths=3.5, linestyles='dashed')

                        # linewidths=4, linestyles='dashed')

                        axs[i + 1, idx].add_collection(p)

                    else:
                        centroid_ref_point = torch.round(
                            ref_point * torch.tensor([image_shape[1], image_shape[0]],
                                                     device=ref_point.device)).cpu()

                    mymarker = axs[i + 1, idx].scatter(centroid_ref_point[0],
                                                       centroid_ref_point[1], s=1000,
                                                       color=bbx_track_color[0], marker='x',
                                                       linewidths=3, linestyles='dotted')
                    axs[i + 1, idx].add_artist(mymarker)

                temporal_counter += 1

        axs[-1, -1].patch.set_edgecolor(border_color)
        axs[-1, -1].patch.set_linewidth('13')

        # plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,   hspace=0, wspace=0.045)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        out_name = f"window_{window_idx}_res_" \
                   f"{track.spatial_shapes[merge_resolution][0]}-{track.spatial_shapes[merge_resolution][1]}.png"

        plt.savefig(os.path.join(out_folder_track, out_name), bbox_inches='tight', pad_inches=0.1)

    plt.close('all')


def visualize_clips_with_att_maps_merged_res(idx, folder_images, images_path, tracks, layer,
                                             merge_resolution, out_path, class_name):
    video_name = images_path[0].split("/")[0]
    cmap = create_color_map()[1:]
    window_idx = idx
    out_folder_track = os.path.join(out_path, video_name)
    os.makedirs(out_folder_track, exist_ok=True)

    num_frames = len(images_path)
    for track in tracks:
        track_id = track.get_id()
        track_score = track.mean_score()
        track_score_str = f"{track_score * 100:.2f}".replace(".", "").zfill(4)
        # out_folder_track = os.path.join(out_folder,
        #                                 f"{track_score_str}_track_{track_id}_id_{track.mask_id}")
        # os.makedirs(out_folder_track, exist_ok=True)

        # out_folder_att_maps = os.path.join(out_folder_track, "att_maps")
        # os.makedirs(out_folder_att_maps, exist_ok=True)

        # for lvl in range(track.spatial_shapes.shape[0]):
        #     if lvl not in res_lvls:
        #         continue

        fig, axs = plt.subplots(ncols=num_frames + 1, nrows=num_frames + 1, figsize=(34, 22))
        axs[0, 0].axis('off')

        # First row images
        for i, t in enumerate(range(num_frames)):
            image = cv2.imread(os.path.join(folder_images, images_path[t]))

            for id_, track_to_put in enumerate(tracks):
                mask = track_to_put.masks[i]
                if mask is None:
                    mask = np.zeros_like(image)
                else:
                    mask = mask_tools.decode(mask)

                bbox = track_to_put.boxes[i]
                if isinstance(bbox, list):
                    bbox = np.array(bbox)

                bbox_int = bbox.astype(np.int32)
                poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                        [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
                np_poly = np.array(poly).reshape((4, 2))

                color_mask = cmap[track_to_put.get_id()]
                bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255),
                              float(color_mask[0] / 255))]

                p = PatchCollection(
                    [Polygon(np_poly)], facecolor='none', edgecolors=bbx_color, linewidths=1)

                axs[0, i + 1].add_collection(p)

                image[mask != 0] = image[mask != 0] * 0.65 + color_mask * 0.35

            image = image[..., ::-1].copy()
            axs[0, i + 1].imshow(image)
            axs[0, i + 1].axis('off')

        for i, curr_frame in enumerate(range(num_frames)):
            image = read_image_for_axis(track, curr_frame, folder_images, images_path, cmap)
            image = image[..., ::-1].copy()
            image_shape = image.shape[:2]
            axs[i + 1, 0].imshow(image)

            bbox = track.boxes[curr_frame]
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))

            color_mask = cmap[track.get_id()]
            bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255),
                          float(color_mask[0] / 255))]
            p = PatchCollection(
                [Polygon(np_poly)], facecolor='none', edgecolors=bbx_color, linewidths=1)

            axs[i + 1, 0].add_collection(p)
            axs[i + 1, 0].axis('off')

            temporal_frames = [t_1 for t_1 in range(-curr_frame, num_frames - curr_frame)]
            temporal_counter = 0

            for idx, temporal_frame in enumerate(temporal_frames):
                att_map_resolution = track.spatial_shapes[merge_resolution]
                feature_map = torch.zeros((att_map_resolution[0], att_map_resolution[1]),
                                          device=track.temporal_positions.device)

                if temporal_frame == 0:
                    curr_frame_positions = track.curr_position[curr_frame].flatten(0, 2)
                    curr_frame_positions[:, 0].clamp_(0, att_map_resolution[1] - 1)
                    curr_frame_positions[:, 1].clamp_(0, att_map_resolution[0] - 1)
                    curr_frame_weights = track.curr_att_weights[curr_frame].flatten(0, 2)

                    for position, weight in zip(curr_frame_positions, curr_frame_weights):
                        feature_map[position[1], position[0]] += weight

                    # feature_map = (feature_map / feature_map.max())
                    feature_map = \
                        F.interpolate(feature_map[None, None], image_shape, mode="bilinear",
                                      align_corners=False).detach().to("cpu")[0, 0]
                    # feature_map = feature_map.to("cpu")[0, 0]
                    axs[i + 1, idx + 1].imshow(feature_map, cmap='cividis')

                    ref_point = track.ref_point[curr_frame]

                    if ref_point.shape[-1] == 4:
                        np_poly, centroid_ref_point = process_ref_point(ref_point, image_shape)

                        p = PatchCollection(
                            [Polygon(np_poly)], facecolor='none', edgecolors=[(1, 0, 0)],
                            linewidths=2)
                        axs[i + 1, idx + 1].add_collection(p)

                    else:
                        centroid_ref_point = torch.round(
                            ref_point * torch.tensor([image_shape[1], image_shape[0]],
                                                     device=ref_point.device)).cpu()

                    mymarker = axs[i + 1, idx + 1].scatter(centroid_ref_point[0],
                                                           centroid_ref_point[1], s=80, color='red',
                                                           marker='x', linewidths=1.4)
                    axs[i + 1, idx + 1].add_artist(mymarker)
                    axs[i + 1, idx + 1].axis('off')

                else:
                    temporal_positions = track.temporal_positions[curr_frame, :,
                                         temporal_counter].flatten(0, 2)
                    temporal_positions[:, 0].clamp_(0, att_map_resolution[1] - 1)
                    temporal_positions[:, 1].clamp_(0, att_map_resolution[0] - 1)
                    temporal_frame_weights = track.temporal_att_weights[curr_frame, :,
                                             temporal_counter].flatten(0, 2)

                    for position, weight in zip(temporal_positions, temporal_frame_weights):
                        feature_map[position[1], position[0]] += weight

                    # feature_map = (feature_map / feature_map.max())
                    feature_map = \
                        F.interpolate(feature_map[None, None], image_shape, mode="bilinear",
                                      align_corners=False).detach().to("cpu")[0, 0]
                    # feature_map = feature_map.to("cpu")
                    axs[i + 1, idx + 1].imshow(feature_map, cmap='cividis')
                    axs[i + 1, idx + 1].axis('off')

                    # Allows visualizing instance aware decoder attention
                    if layer == 0:
                        ref_point_frame = curr_frame
                    else:
                        ref_point_frame = curr_frame + temporal_frame

                    ref_point = track.ref_point[ref_point_frame]

                    if ref_point.shape[-1] == 4:
                        np_poly, centroid_ref_point = process_ref_point(ref_point, image_shape)
                        p = PatchCollection(
                            [Polygon(np_poly)], facecolor='none', edgecolors=[(0.5, 1, 0)],
                            linewidths=2)
                        axs[i + 1, idx + 1].add_collection(p)

                    else:
                        centroid_ref_point = torch.round(
                            ref_point * torch.tensor([image_shape[1], image_shape[0]],
                                                     device=ref_point.device)).cpu()

                    mymarker = axs[i + 1, idx + 1].scatter(centroid_ref_point[0],
                                                           centroid_ref_point[1], s=80,
                                                           color=(0.5, 1, 0), marker='x',
                                                           linewidths=1.4)
                    axs[i + 1, idx + 1].add_artist(mymarker)

                    temporal_counter += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.05, wspace=0.05)
            plt.margins(0, 0)

            out_name = f"window_{window_idx}_{track_score_str}_track_{track_id}_id_{track.mask_id}_res_" \
                       f"{track.spatial_shapes[merge_resolution][0]}-{track.spatial_shapes[merge_resolution][1]}.png"

            plt.savefig(os.path.join(out_folder_track, out_name), bbox_inches='tight')

    plt.close('all')


def visualize_clips_with_att_maps_per_reslvl(idx, folder_images, images_path, tracks, layer,
                                             res_lvls, out_path, class_name, ):
    video_name = images_path[0].split("/")[0]
    cmap = create_color_map()[1:]
    window_idx = idx
    out_folder_track = os.path.join(out_path, video_name)
    os.makedirs(out_folder_track, exist_ok=True)

    num_frames = len(images_path)
    for track in tracks:
        track_id = track.get_id()
        track_score = track.mean_score()
        track_score_str = f"{track_score * 100:.2f}".replace(".", "").zfill(4)
        # out_folder_track = os.path.join(out_folder,
        #                                 f"{track_score_str}_track_{track_id}_id_{track.mask_id}")
        # os.makedirs(out_folder_track, exist_ok=True)

        # out_folder_att_maps = os.path.join(out_folder_track, "att_maps")
        # os.makedirs(out_folder_att_maps, exist_ok=True)

        for lvl in range(track.spatial_shapes.shape[0]):
            if lvl != res_lvls:
                continue

            fig, axs = plt.subplots(ncols=num_frames + 1, nrows=num_frames + 1, figsize=(34, 22))
            axs[0, 0].axis('off')

            # First row images
            for i, t in enumerate(range(num_frames)):
                image = cv2.imread(os.path.join(folder_images, images_path[t]))

                for id_, track_to_put in enumerate(tracks):
                    mask = track_to_put.masks[i]
                    if mask is None:
                        mask = np.zeros_like(image)
                    else:
                        mask = mask_tools.decode(mask)

                    bbox = track_to_put.boxes[i]
                    if isinstance(bbox, list):
                        bbox = np.array(bbox)

                    bbox_int = bbox.astype(np.int32)
                    poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                            [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
                    np_poly = np.array(poly).reshape((4, 2))

                    color_mask = cmap[track_to_put.get_id()]
                    bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255),
                                  float(color_mask[0] / 255))]

                    p = PatchCollection(
                        [Polygon(np_poly)], facecolor='none', edgecolors=bbx_color, linewidths=1)

                    axs[0, i + 1].add_collection(p)

                    image[mask != 0] = image[mask != 0] * 0.65 + color_mask * 0.35

                image = image[..., ::-1].copy()
                axs[0, i + 1].imshow(image)
                axs[0, i + 1].axis('off')

            for i, curr_frame in enumerate(range(num_frames)):
                image = read_image_for_axis(track, curr_frame, folder_images, images_path, cmap)
                image = image[..., ::-1].copy()
                image_shape = image.shape[:2]
                axs[i + 1, 0].imshow(image)

                bbox = track.boxes[curr_frame]
                if isinstance(bbox, list):
                    bbox = np.array(bbox)
                bbox_int = bbox.astype(np.int32)
                poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                        [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
                np_poly = np.array(poly).reshape((4, 2))

                color_mask = cmap[track.get_id()]
                bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255),
                              float(color_mask[0] / 255))]
                p = PatchCollection(
                    [Polygon(np_poly)], facecolor='none', edgecolors=bbx_color, linewidths=1)

                axs[i + 1, 0].add_collection(p)
                axs[i + 1, 0].axis('off')

                temporal_frames = [t_1 for t_1 in range(-curr_frame, num_frames - curr_frame)]
                temporal_counter = 0

                for idx, temporal_frame in enumerate(temporal_frames):
                    att_map_resolution = track.spatial_shapes[lvl]
                    feature_map = torch.zeros((att_map_resolution[0], att_map_resolution[1]),
                                              device=track.temporal_positions.device)

                    if temporal_frame == 0:
                        curr_frame_positions = track.curr_position[curr_frame, :, lvl].flatten(0, 1)
                        curr_frame_positions[:, 0].clamp_(0, att_map_resolution[1] - 1)
                        curr_frame_positions[:, 1].clamp_(0, att_map_resolution[0] - 1)
                        curr_frame_weights = track.curr_att_weights[curr_frame, :, lvl].flatten(0,
                                                                                                1)

                        for position, weight in zip(curr_frame_positions, curr_frame_weights):
                            feature_map[position[1], position[0]] += weight

                        # feature_map = (feature_map / feature_map.max())
                        feature_map = \
                            F.interpolate(feature_map[None, None], image_shape, mode="bilinear",
                                          align_corners=False).detach().to("cpu")[0, 0]
                        # feature_map = feature_map.to("cpu")[0, 0]
                        axs[i + 1, idx + 1].imshow(feature_map, cmap='cividis')

                        ref_point = track.ref_point[curr_frame]

                        if ref_point.shape[-1] == 4:
                            np_poly, centroid_ref_point = process_ref_point(ref_point, image_shape)

                            p = PatchCollection(
                                [Polygon(np_poly)], facecolor='none', edgecolors=[(1, 0, 0)],
                                linewidths=2)
                            axs[i + 1, idx + 1].add_collection(p)

                        else:
                            centroid_ref_point = torch.round(
                                ref_point * torch.tensor([image_shape[1], image_shape[0]],
                                                         device=ref_point.device)).cpu()

                        mymarker = axs[i + 1, idx + 1].scatter(centroid_ref_point[0],
                                                               centroid_ref_point[1], s=80,
                                                               color='red',
                                                               marker='x', linewidths=1.4)
                        axs[i + 1, idx + 1].add_artist(mymarker)
                        axs[i + 1, idx + 1].axis('off')

                    else:
                        temporal_positions = track.temporal_positions[curr_frame, :,
                                             temporal_counter, lvl].flatten(0, 1)
                        temporal_positions[:, 0].clamp_(0, att_map_resolution[1] - 1)
                        temporal_positions[:, 1].clamp_(0, att_map_resolution[0] - 1)
                        temporal_frame_weights = track.temporal_att_weights[curr_frame, :,
                                                 temporal_counter, lvl].flatten(0, 1)

                        for position, weight in zip(temporal_positions, temporal_frame_weights):
                            feature_map[position[1], position[0]] += weight

                        # feature_map = (feature_map / feature_map.max())
                        feature_map = \
                            F.interpolate(feature_map[None, None], image_shape, mode="bilinear",
                                          align_corners=False).detach().to("cpu")[0, 0]
                        # feature_map = feature_map.to("cpu")
                        axs[i + 1, idx + 1].imshow(feature_map, cmap='cividis')
                        axs[i + 1, idx + 1].axis('off')

                        # Allows visualizing instance aware decoder attention
                        if layer == 0:
                            ref_point_frame = curr_frame
                        else:
                            ref_point_frame = curr_frame + temporal_frame

                        ref_point = track.ref_point[ref_point_frame]

                        if ref_point.shape[-1] == 4:
                            np_poly, centroid_ref_point = process_ref_point(ref_point, image_shape)
                            p = PatchCollection(
                                [Polygon(np_poly)], facecolor='none', edgecolors=[(0.5, 1, 0)],
                                linewidths=2)
                            axs[i + 1, idx + 1].add_collection(p)

                        else:
                            centroid_ref_point = torch.round(
                                ref_point * torch.tensor([image_shape[1], image_shape[0]],
                                                         device=ref_point.device)).cpu()

                        mymarker = axs[i + 1, idx + 1].scatter(centroid_ref_point[0],
                                                               centroid_ref_point[1], s=80,
                                                               color=(0.5, 1, 0), marker='x',
                                                               linewidths=1.4)
                        axs[i + 1, idx + 1].add_artist(mymarker)

                        temporal_counter += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.05, wspace=0.05)
            plt.margins(0, 0)

            out_name = f"window_{window_idx}_{track_score_str}_track_{track_id}_id_{track.mask_id}_res_" \
                       f"{track.spatial_shapes[lvl][0]}-{track.spatial_shapes[lvl][1]}.png"

            plt.savefig(os.path.join(out_folder_track, out_name), bbox_inches='tight')

        # for idx, image_path in enumerate(images_path):
        #     frame_name = image_path.split("/")[1]
        #     image = cv2.imread(os.path.join(folder_images, image_path))
        #     image = np.copy(image)
        #     out_image_path = os.path.join(out_folder_track, frame_name)
        #
        #     if track.masks[idx] is None:
        #         cv2.imwrite(out_image_path, image)
        #
        #     else:
        #         instance = {
        #             "mask": track.masks[idx],
        #             # "score": track.scores[idx],
        #             "category_id": track.categories[idx],
        #             # "bbox":  track.boxes[idx],
        #             "instance_id": track_id,
        #             # "centroid_point": track.centroid_points[idx]
        #         }
        #         imshow_det_masks_for_tracks(image, instance, cmap=cmap, class_names=class_name,
        #                                     out_file=out_image_path)

    plt.close('all')
