"""
Tools for measuring quntitative errors.
"""

from collections import OrderedDict
import time
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import open3d as o3d
import imageio
import torch
import torch.nn.functional as F
import lpips

from sdf_meshing import mask_image, raytrace_sdf_ibr
from modules_sdf import SDFIBRNet
from data_processing.datasets.image_view import ImageView
from data_processing.datasets.dataio_sdf import DatasetSDF
import data_processing.components.conversions as conversions
import utils.math_utils as math_utils


def resize_image_area(im: np.array, size: np.array) -> np.array:
    """
    Resizes image with area filter.
    """
    im = torch.from_numpy(im).permute(2, 0, 1)
    im = F.interpolate(im[None, ...], tuple(size[::-1]), mode='area')[0, ...]
    im = im.permute(1, 2, 0).numpy()
    return im


def error_mse(im_pred: np.array, im_gt: np.array, mask: np.array = None):
    """
    Computes MSE metric. Optionally applies mask.
    """
    # Linearize.
    im_pred = im_pred[..., :3].reshape(-1, 3)
    im_gt = im_gt[..., :3].reshape(-1, 3)

    # Mask?
    if mask is not None:
        mask = mask.flatten()
        # im_pred = im_pred[mask, :]
        # im_gt = im_gt[mask, :]

        # Use multiplication method as described in paper
        im_pred = im_pred * mask[..., None]
        im_gt = im_gt * mask[..., None]

    mse = (im_pred - im_gt) ** 2
    return mse.mean()


def error_psnr(im_pred: np.array, im_gt: np.array, mask: np.array = None):
    """
    Computes PSNR metric. Optionally applies mask.
    Assumes floats [0,1].
    """
    mse = error_mse(im_pred, im_gt, mask)
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    return 20 * np.log10(1.0) - 10 * np.log10(mse)


def error_ssim(im_pred: np.array, im_gt: np.array, mask: np.array = None):
    """
    Computes SSIM metric. Optionally applies mask.
    """
    # RGB
    im_pred = im_pred[..., :3]
    im_gt = im_gt[..., :3]

    # Mask?
    if mask is not None:
        mask = mask.reshape(im_pred.shape[0], im_pred.shape[1], 1).repeat(3, axis=2)
        im_pred = im_pred * mask
        im_gt = im_gt * mask

    return structural_similarity(im_pred, im_gt, data_range=1.0, multichannel=True)


def error_lpips(im_pred: np.array, im_gt: np.array, mask: np.array = None, metric: lpips.LPIPS = None):
    """
    Computes LPIPS metric. Optionally applies mask.
    """
    # RGB
    im_pred = im_pred[..., :3]
    im_gt = im_gt[..., :3]

    # Mask?
    if mask is not None:
        mask = mask.reshape(im_pred.shape[0], im_pred.shape[1], 1).repeat(3, axis=2)
        im_pred = im_pred * mask
        im_gt = im_gt * mask

    if np.max(im_pred.shape) > 5000:
        # Too large (SCU dataset)
        new_size = np.array([im_pred.shape[1], im_pred.shape[0]], int) // 2
        im_pred = resize_image_area(im_pred, new_size)
        im_gt = resize_image_area(im_gt, new_size)

    # To torch.
    device = 'cuda'
    im_pred = torch.from_numpy(im_pred).permute(2, 0, 1)[None, ...].to(device)
    im_gt = torch.from_numpy(im_gt).permute(2, 0, 1)[None, ...].to(device)

    # Make metric.
    if metric is None:
        # best forward scores
        metric = lpips.LPIPS(net='alex').to(device)
        # # closer to "traditional" perceptual loss, when used for optimization
        # metric = lpips.LPIPS(net='vgg').to(device)

    # Compute metric.
    loss = metric(im_pred, im_gt)

    return loss.item()


def error_iou(mask_pred: np.array, mask_gt: np.array):
    """
    Measures area of intersection over union.
    """
    intersection = np.logical_and(mask_pred, mask_gt)
    union = np.logical_or(mask_pred, mask_gt)
    return intersection.sum() / union.sum()


def compute_pcd_error(coords_gt: np.array, rt_res: dict, m_to_original: np.array = np.eye(4, dtype=np.float32)):
    """
    Computes distance between two point clouds.
    Both clouds are in the normalized model coordinates (inside unit sphere).
    Adds error map into rt_res.
    """

    # Build ray-traced PCD
    pos = rt_res['raw']['pos'].cpu().numpy().reshape(-1, 3)
    mask = rt_res['raw']['mask'].cpu().numpy().reshape(-1)
    pos = pos[mask, ...]
    if pos.shape[0] == 0:
        # No valid points!
        return -1

    # Scale to reference coordinate system.
    coords_gt = math_utils.transform_points(m_to_original, coords_gt, return_euclidean=True)
    pos = math_utils.transform_points(m_to_original, pos, return_euclidean=True)

    # Build PCDs
    pcd_gt = o3d.geometry.PointCloud()
    pcd_rt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(coords_gt)
    pcd_rt.points = o3d.utility.Vector3dVector(pos)

    # Compute distance for each RT point
    distances = pcd_rt.compute_point_cloud_distance(pcd_gt)

    # Create error map.
    error = distances / np.percentile(distances, 85)
    error_map = np.zeros((rt_res['raw']['pos'].shape[0] * rt_res['raw']['pos'].shape[1]))
    error_map[mask] = error
    error_map = error_map.reshape(rt_res['raw']['pos'].shape[0], rt_res['raw']['pos'].shape[1])
    error_map = plt.get_cmap("viridis")(error_map)
    rt_res['viz']['pcd_error'] = error_map

    return np.mean(distances)


def measure_error_2d_view(decoder: SDFIBRNet, dataset: DatasetSDF,
                          view: ImageView, frame_index: int, view_index: int, batch_size: int, opt,
                          output_path: Path = None):
    """
    Measures error of 2D projection for single view.
    """
    # Raytrace view.
    start_time = time.time()
    resolution = (view.resolution * opt.im_scale + 0.5).astype(int)
    assert opt.ibr_dataset
    render = raytrace_sdf_ibr(decoder=decoder,
                              resolution=resolution,
                              projection_matrix=view.projection_matrix,
                              view_matrix=view.view_matrix,
                              model_matrix=dataset.model_matrix,
                              timestamp=dataset.frame_index_to_timestamp(frame_index),
                              build_pcd=False,
                              render_softmask=False,
                              batch_size=batch_size,
                              debug_gui=False,
                              vid_frame=view_index)

    render_ibr = decoder.forward_test(view_index)
    render['viz']['colors'] = render_ibr['target_img'].squeeze().permute(1,2,0).cpu().numpy()
    # render['viz']['mask'] = render_ibr['mask'].unsqueeze(-1).cpu().numpy()
    render['viz']['mask'] = render['raw']['mask'].cpu().numpy()[..., None].repeat(3, 2)
    render['viz']['color_masked'] = render['viz']['colors'] * render['viz']['mask']

    rt_time = time.time() - start_time

    # Test.
    im_pred = render['viz']['colors']
    mask_pred = render['viz']['mask'][..., 0] > 0.5

    # GT.
    im_gt = view.get_image_resized(resolution).permute(1, 2, 0).cpu().numpy()
    # mask_gt = view.get_mask_resized(resolution)[0, ...].cpu().numpy() > 0.5
    mask_gt = view.get_mask_resized(resolution)[0, ...].cpu().numpy() == 1

    error = ((im_gt - im_pred) ** 2).sum(-1)
    error_map = plt.get_cmap("viridis")(error)

    if output_path is not None:
        conversions.imwritef(output_path / f'f{frame_index:06d}_v{view_index:03d}_color_gt.png', im_gt)
        conversions.imwritef(output_path / f'f{frame_index:06d}_v{view_index:03d}_mask_gt.png', mask_gt)
        conversions.imwritef(output_path / f'f{frame_index:06d}_v{view_index:03d}_color_pred.png', im_pred)
        conversions.imwritef(output_path / f'f{frame_index:06d}_v{view_index:03d}_mask_pred.png', mask_pred)
        conversions.imwritef(output_path / f'f{frame_index:06d}_v{view_index:03d}_error_map.png', error_map)

        # Mask using GT mask and save for presentation purposes.
        masked_path = output_path / 'masked_gt'
        masked_path.mkdir(0o777, True, True)
        im_pred_masked = mask_image(im_pred, mask_gt, (1, 1, 1), close_hole_size=0)
        conversions.imwritef(masked_path / f'f{frame_index:06d}_v{view_index:03d}_masked.png', im_pred_masked)

        # Mask using RT mask and save for presentation purposes.
        masked_path = output_path / 'masked_rt'
        masked_path.mkdir(0o777, True, True)
        im_pred_masked = render['viz']['color_masked']
        conversions.imwritef(masked_path / f'f{frame_index:06d}_v{view_index:03d}_masked.png', im_pred_masked)
        error_masked = error_map * mask_gt[..., None]
        conversions.imwritef(masked_path / f'f{frame_index:06d}_v{view_index:03d}_error_masked.png', error_masked)

    # PCD error?
    pcd_error = -1
    # if view_index >= 0 and view_index < dataset.dataset_pcd.num_views and dataset.dataset_pcd.num_views == len(dataset.dataset_img.image_views):
    if dataset.coords.shape[0] > 0:
        # Compute 3D errror.
        #gt_coords = dataset.dataset_pcd.get_view_coords(view_index)
        print('\tComputing PCD error...')
        pcd_error = compute_pcd_error(dataset.coords, render, dataset.dataset_pcd.m_pcd_to_original)

    # Accounting.
    epoch = -1
    steps = -1
    loss = -1
    if opt.optimizer_state is not None:
        epoch = opt.optimizer_state['epoch']
        steps = opt.optimizer_state['step']
        loss = opt.optimizer_state['loss']

    # Error.
    return OrderedDict([
        ('epoch', epoch),
        ('steps', steps),
        ('render_time', rt_time),
        ('width', im_gt.shape[1]),
        ('height', im_gt.shape[0]),
        ('psnr_mask_pred', error_psnr(im_pred, im_gt, mask_pred)),
        ('lpips_mask_pred', error_lpips(im_pred, im_gt, mask_pred)),
        ('psnr_mask_gt', error_psnr(im_pred, im_gt, mask_gt)),
        ('ssim_mask_gt', error_ssim(im_pred, im_gt, mask_gt)),
        ('lpips_mask_gt', error_lpips(im_pred, im_gt, mask_gt)),
        ('mask_iou', error_iou(mask_pred, mask_gt)),
        ('pcd_error', pcd_error),
    ])


def measure_error(output_path_base: str, opt, decoder: SDFIBRNet, dataset: DatasetSDF):
    """
    Measures different metrics wrt GT.
    """
    out_dir = Path(output_path_base + '_benchmark')
    out_dir.mkdir(0o777, True, True)

    # 2D metrics
    res_2d = OrderedDict([('frame', []), ('view', []), ('split', [])])

    # 2D metrics.
    dataset_2d = dataset.dataset_img

    frame_list = [0]

    # Frame-by-frame.
    for frame_index in frame_list:
        # View-by-view.
        views = dataset_2d.frames[frame_index].image_views
        for view_index, view in enumerate(views):
            if view_index >= opt.total_number_source_views:
                break

            split = 'test'
            if not opt.benchark_all_views and view_index not in opt.WITHHELD_VIEWS:
                continue

            print(f'Benchmarking frame {frame_index} -- view {view_index} / {len(views)}....')
            res_view = measure_error_2d_view(decoder, dataset, view, frame_index, view_index,
                                             opt.batch_size, opt, out_dir)
            res_2d['frame'] += [frame_index]
            res_2d['view'] += [view_index]
            res_2d['split'] += [split]
            for k, v in res_view.items():
                if len(res_2d['view']) == 1:
                    res_2d[k] = []
                res_2d[k] += [v]

    # Write results to CSV.
    if output_path_base:
        output_file = output_path_base + f'_metrics_2d.csv'
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(res_2d.keys())
            for view_index in range(len(res_2d['view'])):
                writer.writerow([res_2d[k][view_index] for k in res_2d.keys()])

    return res_2d
