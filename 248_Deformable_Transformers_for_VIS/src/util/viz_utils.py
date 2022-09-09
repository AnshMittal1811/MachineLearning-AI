import numpy as np
import os
import cv2
import torch
import pycocotools.mask as mask_tools

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection


def get_most_left_coordinate(mask):
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    if horizontal_indicies.shape[0]:
        x1 = horizontal_indicies[0]
        y1 = np.where(mask[:, x1])[0]
        if y1.shape[0]:
            return y1[-1], x1

    return 0, 0


def process_centroid(centroids, tgt_size):
    img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
    scale_fct = torch.stack([img_w, img_h], dim=1)
    centroids = centroids * scale_fct
    return centroids


def imshow_det_bboxes_for_tracks(img, instance, cmap,
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
    # show_centroid = "centroid_point" in instance

    show_centroid = False
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

    label_text = class_names[instance["category_id"]] if class_names is not None else f'class {instance["category_id"]}'
    label_text += f'|{instance["score"]:.02f}'

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
            polygons, facecolor='none', edgecolors=[(0, 0, 1)], linewidths=thickness)
        ax.add_collection(p)

    if show_centroid:
        p = PatchCollection(
            centroids,  edgecolors=[(0, 0, 1)])
        ax.add_collection(p)

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


def visualize_clips_after_processing(idx, folder_images, images_path, tracks, out_path, class_name):
    video_name = images_path[0].split("/")[0]
    cmap = create_color_map()

    out_folder = os.path.join(out_path, video_name, "clip_results", f"window_{idx}")
    os.makedirs(out_folder, exist_ok=True)

    for track in tracks:
        track_id = track.get_id()
        track_score = track.mean_score()
        track_score_str = f"{track_score * 100:.2f}".replace(".", "").zfill(4)
        out_folder_track = os.path.join(out_folder, f"{track_score_str}_track_{track_id}_id_{track.mask_id}")
        os.makedirs(out_folder_track, exist_ok=True)
        for idx, image_path in enumerate(images_path):
            frame_name = image_path.split("/")[1]
            image = cv2.imread(os.path.join(folder_images, image_path))
            image = np.copy(image)
            out_image_path = os.path.join(out_folder_track, frame_name)

            if track.masks[idx] is None:
                cv2.imwrite(out_image_path, image)

            else:
                instance = {
                    "mask": track.masks[idx],
                    "score": track.scores[idx],
                    "category_id": track.categories[idx],
                    "bbox":  track.boxes[idx],
                    "instance_id": track_id,
                    # "centroid_point": track.centroid_points[idx]
                }
                imshow_det_bboxes_for_tracks(image, instance, cmap=cmap, class_names=class_name, out_file=out_image_path)


def visualize_tracks_independently(folder_images, images_path, video_results, final_class_policy, final_score_policy, out_path, class_name):
    all_files = []
    for clip_path in images_path:
        all_files.extend(clip_path)
    unique_files = list(set(all_files))
    unique_files.sort()
    video_name = unique_files[0].split("/")[0]
    cmap = create_color_map()
    out_folder = os.path.join(out_path, video_name, "tracks_results")
    os.makedirs(out_folder, exist_ok=True)

    for track in video_results:
        final_score = track.compute_final_score(final_score_policy)
        final_category_id = track.compute_final_category(final_class_policy)
        final_masks = []
        final_bbx = []
        final_centroids = []
        for valid, mask, box, centroid in zip(track.valid_frames, track.masks, track.boxes, track.centroid_points):
            if valid:
                final_masks.append(mask)
                final_bbx.append(box)
                final_centroids.append(centroid)
            else:
                final_masks.append(None)
                final_bbx.append(None)
                final_centroids.append(None)

        track_score_str = f"{final_score * 100:.2f}".replace(".", "").zfill(4)
        out_folder_track = os.path.join(out_folder, f"{track_score_str}_track_{track.get_id()}")
        os.makedirs(out_folder_track, exist_ok=True)

        with open(os.path.join(out_folder_track, "matches.txt"), 'w') as f:
            for idx, match in enumerate(track.matching_ids_record):
                f.write(f"Window_{idx} track_{match[0]} / Window_{idx+1} track_{match[1]}\n")

        for idx, image_path in enumerate(unique_files):
            frame_name = image_path.split("/")[1]
            image = cv2.imread(os.path.join(folder_images, image_path))
            image = np.copy(image)
            out_image_path = os.path.join(out_folder_track, frame_name)

            if final_masks[idx] is None:
                cv2.imwrite(out_image_path, image)

            else:
                frame_results = {
                    "mask": final_masks[idx],
                    "score": final_score,
                    "category_id": final_category_id,
                    "bbox": final_bbx[idx],
                    # "centroid_point": final_centroids[idx],
                    "instance_id": track.get_id(),

                }
                imshow_det_bboxes_for_tracks(image, frame_results, cmap=cmap, class_names=class_name, out_file=out_image_path)


def visualize_results_merged(folder_images, images_path, video_results, final_class_policy, final_score_policy, out_path, class_name):

    all_files = []
    for clip_path in images_path:
        all_files.extend(clip_path)
    unique_files = list(set(all_files))
    unique_files.sort()
    video_name = unique_files[0].split("/")[0]
    cmap = create_color_map(N=15)[1:]
    # cmap = np.array([[151, 208, 119], [227, 0, 227]])

    out_folder = os.path.join(out_path, video_name)
    os.makedirs(out_folder, exist_ok=True)

    class_names = None
    thickness = 2
    font_size = 15
    win_name = 'COCO visualization'
    out_file = None

    show_bbx = False
    show_cat = True

    for t, image_path  in enumerate(unique_files):
        frame_name = image_path.split("/")[1]
        image = cv2.imread(os.path.join(folder_images, image_path))

        out_image_path = os.path.join(out_folder, frame_name)
        image = np.copy(image)

        width, height = image.shape[1], image.shape[0]
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

        for track in video_results:
            cat = int(track.compute_final_category("most_common"))
            mask, bbox, valid = track.masks[t], track.boxes[t], track.valid_frames[t]
            if len(image.shape) > 3:
                image = torch.squeeze(image)
            # img = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
            width, height = image.shape[1], image.shape[0]
            image = np.ascontiguousarray(image)
            if not valid:
                continue
            if isinstance(mask, dict):
                mask = mask_tools.decode(mask)
            else:
                mask = mask

            polygons = []
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))

            polygons.append(Polygon(np_poly))

            text_coords = (bbox_int[0], bbox_int[1])
            label_text = class_name[cat] if class_name is not None else f'class {cat}'

            if show_cat:
                ax.text(text_coords[0],
                        text_coords[1],
                        f'{label_text}',
                        bbox={
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        },
                        color=(1, 1, 1),
                        fontsize=font_size,
                        verticalalignment='top',
                        horizontalalignment='left')

            color_mask = cmap[track.get_id()]
            image[mask != 0] = image[mask != 0] * 0.45 + color_mask * 0.55
            # bbx_color = [(float(color_mask[2] / 255), float(color_mask[1] / 255), float(color_mask[0] / 255))]
            bbx_color = [(float(color_mask[0] / 255), float(color_mask[1] / 255), float(color_mask[1] / 255))]

            p = PatchCollection(
                polygons, facecolor='none', edgecolors=bbx_color, linewidths=thickness)

            if show_bbx:
                ax.add_collection(p)

        plt.imshow(image)
        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        image = rgb.astype('uint8')
        cv2.imwrite(out_image_path, image)
        plt.close()



