"""
Collection of tools for non-differentiable
evaluation of the SDF decoder.
For test only - not for training.
"""

import time
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import skimage.measure
import torch
import imageio
import open3d as o3d
import matplotlib.pyplot as plt
import skvideo.io
import cv2

import data_processing.datasets.dataio as dataio
import data_processing.components.conversions as conversions
import utils.common_utils as common_utils
from utils.common_utils import imwritef
import utils.diff_operators as diff_operators
import utils.math_utils as math_utils
import utils.math_utils_torch as mut
import sdf_rendering
import modules
from modules_sdf import SDFIBRNet
import utils.camera_visualization as cviz
from torchmeta.modules.utils import get_subdict


@torch.enable_grad()
def plot_sdf(decoder: SDFIBRNet, params=None):
    """
    Draws the SDF plot for the center-most vertex.
    front = min(z) (z axis points away)
    """

    samples_z = torch.linspace(-1, 1, 1024)
    samples = torch.stack((torch.zeros_like(samples_z), torch.zeros_like(samples_z), samples_z), 1)
    samples = samples.to(decoder.device)
    samples.requires_grad = True

    output = decoder.decoder_sdf({'coords': samples[None, ...]}, params=get_subdict(params, 'decoder_sdf'))
    y = output['model_out'][0, ..., :1]
    grad = diff_operators.gradient(y, output['model_in'])[0, ...]
    try:
        lapl = diff_operators.laplace(y, output['model_in'])[0, ...]
    except:
        lapl = torch.zeros_like(grad)

    def norm(x):
        return x / x.abs().max()

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(samples_z, norm(y.detach().cpu()))
    ax.plot(samples_z, norm(grad[..., 2].detach().cpu()))
    ax.plot(samples_z, np.clip(np.linalg.norm(grad.detach().cpu(), axis=1), -1.2, 1.2))
    ax.plot(samples_z, norm(lapl.detach().cpu()))
    ax.plot(samples_z, torch.zeros_like(samples_z))
    ax.legend(["SDF", "Grad Z", "Grad Magn", "Laplacian"])
    ax.set_ylim(-1.2, 1.2)
    return fig


@ torch.no_grad()
def _render_slice(decoder, resolution: int, axes: tuple, data_type: str, batch_size: int,
                  timestamp: float = common_utils.KEY_FRAME_TIME):
    """
    Renders one 2D slice throught the function domain.
    """
    # Coords
    slice_coords_2d = dataio.get_mgrid(resolution)
    slice_coords_3d = torch.zeros((slice_coords_2d.shape[0], 3), dtype=slice_coords_2d.dtype)
    if axes[0] < 3:
        slice_coords_3d[:, axes[0]] = slice_coords_2d[:, 0]
    if axes[1] < 3:
        slice_coords_3d[:, axes[1]] = slice_coords_2d[:, 1]

    if data_type == 'sdf':
        out_slice = slice(0, 1)
    else:
        out_slice = None

    model_in = {'coords': slice_coords_3d.to(decoder.device)[None, ...]}

    model_out = modules.batch_decode(decoder, model_in, batch_size=batch_size, out_feature_slice=out_slice)
    values = model_out['model_out']

    ax_names = ['x', 'y', 'z', 'time']

    # Make image.
    image = []
    values = values.cpu().reshape(resolution, resolution, -1)
    assert data_type == 'sdf'
    # SDF.
    fig = common_utils.make_contour_plot(values[..., 0], resolution=[resolution, resolution])
    plt.xlabel(ax_names[axes[0]])
    plt.ylabel(ax_names[axes[1]])
    image = torch.from_numpy(common_utils.figure_to_image(fig, True)).float() / 255
    plt.close('all')
    # plt.imshow(image.permute(1, 2, 0).numpy())
    # plt.show()

    return image, values


def _render_camera_axis(axes, decoder: SDFIBRNet, dataset, resolution: int, batch_size: int):
    """
    Renders camera poses projected over the SDF slice.
    """
    # Extract 4x4 view matrices.
    model_matrix = dataset.model_matrix.detach().cpu().numpy()
    poses_orig = np.array([view.meta_view_matrix @ model_matrix for view in dataset.image_views])
    poses_new = np.array([view.view_matrix.detach().cpu().numpy() @ model_matrix for view in dataset.image_views])

    # Camera object in camera space.
    cam_size = 0.15
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ], np.float32) * cam_size * model_matrix[0, 0]
    points = points[None, ...].repeat(poses_orig.shape[0], 0)

    # Tranform from camera space to world space.
    points_orig = math_utils.transform_points(np.linalg.inv(poses_orig), points, return_euclidean=True)
    points_new = math_utils.transform_points(np.linalg.inv(poses_new), points, return_euclidean=True)

    # Project to axes.
    points_orig = points_orig[..., axes]
    points_new = points_new[..., axes]

    # Define needed range.
    margin = 0.05
    range_min = np.minimum(-1, np.minimum(points_orig.min((0, 1)), points_new.min((0, 1)))) - margin
    range_max = np.maximum(1, np.maximum(points_orig.max((0, 1)), points_new.max((0, 1)))) + margin
    range_size = (range_max - range_min).max()
    range_mean = (range_max + range_min) / 2
    range_min = range_mean - range_size / 2
    range_max = range_mean + range_size / 2
    px_size = range_size / resolution

    # Compute needed slice resolution.
    slice_res = max(int(2 / px_size + 0.5), 2)

    # Render underlying slice.
    sdf = _render_slice(decoder.decoder_sdf, slice_res, axes[::-1], 'sdf', batch_size)[0]
    # Flip vertically to get it to flipped space.
    sdf = np.flipud(sdf.permute(1, 2, 0).numpy())

    # Splat slice to canvas.
    canvas = np.ones((resolution, resolution, 3), np.float32)
    sdf_a = ((-1 - range_min) / px_size + 0.5).astype(int)
    sdf_b = sdf_a + [sdf.shape[1], sdf.shape[0]]
    canvas[sdf_a[1]:sdf_b[1], sdf_a[0]:sdf_b[0], :] = sdf

    # Scale points to px.
    points_orig = ((points_orig - range_min) / (range_max - range_min) * resolution + 0.5).astype(int)
    points_new = ((points_new - range_min) / (range_max - range_min) * resolution + 0.5).astype(int)

    # Draw the cameras.
    def draw_camera(im, points, colored=False):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        gray = (127, 127, 127)
        for k in range(3):
            cv2.line(im, tuple(points[0]), tuple(points[1 + k]), colors[k]
                     if colored else gray, 2, lineType=cv2.LINE_AA)

    # OpenCV only does AA for 8bit images.
    canvas = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    # Draw originals.
    for i in range(points_orig.shape[0]):
        draw_camera(canvas, points_orig[i, ...], False)
    # Draw optimized.
    for i in range(points_new.shape[0]):
        draw_camera(canvas, points_new[i, ...], True)

    # Flip vertical.
    canvas = np.ascontiguousarray(np.flipud(canvas))

    # Draw labels
    for i in range(points_new.shape[0]):
        label = f'{i}'  # dataset.image_views[i].name
        pos = points_new[i, 0] + [5, -5]
        pos[1] = resolution - 1 - pos[1]
        cv2.putText(canvas, label, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return torch.from_numpy(canvas.astype(np.float32) / 255).permute(2, 0, 1)


def _render_cameras_3d(dataset, resolution: int):
    """
    Renders camera poses in 3D using the Pytorch3D
    utility and matplotlib.
    """
    # Extract 4x4 view matrices.
    model_matrix = dataset.model_matrix.detach().cpu().numpy().copy()
    poses_orig = np.array([view.meta_view_matrix for view in dataset.image_views])
    poses_new = np.array([view.view_matrix.detach().cpu().numpy() for view in dataset.image_views])

    fig = cviz.plot_camera_scene(poses_orig, poses_new, model_matrix, [resolution, resolution])
    image = torch.from_numpy(common_utils.figure_to_image(fig, True)).float() / 255
    plt.close('all')
    return image


def _render_cameras(decoder: SDFIBRNet, dataset, resolution: int, batch_size: int) -> OrderedDict:
    """
    Renders camera poses projected to X,Y and Z plane
    over the SDF slices.
    """
    dims = {
        "xy": (0, 1),
        "zy": (2, 1),
        "xz": (0, 2),
    }
    res = OrderedDict()
    for dim, axes in dims.items():
        res[dim] = _render_camera_axis(axes, decoder, dataset, resolution, batch_size)

    # 3D View.
    res['3d'] = _render_cameras_3d(dataset, resolution)

    return res


def render_cameras(decoder: SDFIBRNet, dataset, output_path_base: str, resolution: int, batch_size: int):
    """
    Renders camera poses projected to X,Y and Z plane
    over the SDF slices.
    """
    if len(dataset.image_views) == 0:
        return

    res = _render_cameras(decoder, dataset, resolution, batch_size)
    for name, im in res.items():
        im = im.permute(1, 2, 0).numpy()
        imageio.imwrite(output_path_base + f'_cameras_{name}.png',
                        (np.clip(im, 0, 1) * 255).astype(np.uint8))


def mask_image(image: np.array, mask: np.array, background_color=[1.0, 1.0, 1.0],
               close_hole_size=0) -> np.array:
    """
    Makes the background great again.
    """
    # Make mask 2D.
    if len(mask.shape) > 2:
        mask = mask[..., 0]

    # Close holes.
    if close_hole_size > 0:
        from scipy import ndimage
        struct1 = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_closing(mask, structure=struct1, iterations=close_hole_size)

    # Mask.
    maskf = mask.astype(np.float32)[..., None].repeat(3, -1)
    bg_color = np.array(background_color, dtype=image.dtype).reshape(1, 1, 3)
    return image * maskf + bg_color * (1 - maskf)


def remove_green_spill(img, weights=(1.0 / 3, 1.0 / 3, 1.0 / 3), linear=False):
    """
    Remove the green spill by replacing the green channel with an average of all channels.
    """
    # Make weights numpy array.
    weights = np.array(weights)
    if linear:
        # Convert image to linear.
        img_lin = (img.astype(np.float32) / 255.)**2.2
        # Calculate new green channel.
        new_green_channel = img_lin @ weights
        # Replace.
        img_lin[:, :, 1] = new_green_channel
        # Back to uint8, with gamma.
        return np.clip((img_lin ** (1 / 2.2)) * 255, 0, 255).astype(np.uint8)
    else:
        new_green_channel = img @ weights
        img[:, :, 1] = new_green_channel
        return img


def remove_green_spill_images(path: Path):
    """
    Processes video.
    """
    image_files = sorted([x for x in path.iterdir() if x.suffix in ['.png', '.jpg', '.jpeg']])
    for i, image_file in enumerate(image_files):
        print(f'[{i}/{len(image_files)}] Removing green from {image_file}...')
        im = imageio.imread(image_file).astype(np.float32) / 255
        im = remove_green_spill(im)
        imwritef(image_file, im)


def remove_green_spill_video(video_in: Path, video_out: Path):
    """
    Processes video.
    """

    # Prepare output.
    video_out.parent.mkdir(0o777, True, True)

    reader = skvideo.io.FFmpegReader(str(video_in))
    writer = skvideo.io.FFmpegWriter(
        str(video_out),
        inputdict={'-r': '30'},
        outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', '-r': '30'})

    print(f'Removing green {video_in} -> {video_out}.')

    for frame in reader.nextFrame():
        im = frame.astype(np.float32) / 255
        im = remove_green_spill(im)
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)

        # To video.
        writer.writeFrame(im)

    # Close.
    writer.close()
    reader.close()


def raytrace_sdf_ibr(decoder: SDFIBRNet,
                     resolution=(640, 640),
                     projection_matrix: torch.Tensor = None,
                     view_matrix: torch.Tensor = None,
                     model_matrix: torch.Tensor = None,
                     timestamp: float = common_utils.KEY_FRAME_TIME,
                     build_pcd=False,
                     render_softmask=True,
                     batch_size=131072,
                     debug_gui=False,
                     background_color=[1.0, 1.0, 1.0],
                     close_hole_size=1,
                     params=None,
                     vid_frame=0):
    """
    Raytrace the SDF.
    Generate vizualizations projected into image plane.
    Optionally also builds PCD.
    """
    if projection_matrix is None:
        aspect = resolution[0] / resolution[1]
        hfov = 60
        projection_matrix = torch.from_numpy(
            math_utils.getPerspectiveProjection(hfov, hfov / aspect)).to(decoder.device)
    if view_matrix is None:
        view_matrix = torch.from_numpy(np.eye(4, dtype=np.float32)).to(decoder.device)
        view_matrix[:3, 3] = [0, 0, -1.8]  # Camera is in Z+ and looks into Z-
    if model_matrix is None:
        model_matrix = torch.from_numpy(np.eye(4, dtype=np.float32)).to(decoder.device)

    # Render.
    output = sdf_rendering.render_view_proj(decoder,
                                            torch.from_numpy(resolution).to(decoder.device),
                                            model_matrix,
                                            view_matrix,
                                            projection_matrix,
                                            timestamp=timestamp,
                                            batch_size=batch_size,
                                            debug_gui=debug_gui,
                                            params=params,
                                            normals=True,
                                            vid_frame=vid_frame)

    res = {
        'raw': output,
        'viz': {}
    }

    # Validity mask.
    cmap = plt.cm.gray
    # res['viz']['mask'] = cmap(output['mask'].cpu().numpy().astype(np.float32))

    # # 3D positions.
    norm = plt.Normalize(vmin=-1, vmax=1, clip=True)
    res['viz']['pos'] = norm(output['pos'].cpu().numpy())
    mask = output['mask'].cpu().numpy()[..., None].repeat(3, 2)
    res['viz']['mask'] = mask

    # Normals.
    norm = plt.Normalize(vmin=-1, vmax=1, clip=True)
    res['viz']['normals'] = norm(output['normals'].cpu().numpy())

    # Shaded surface.
    normals = output['normals'].cpu().reshape(-1, 3)
    light_dir = torch.Tensor([0, -1, -1]).float()
    light_dir /= torch.norm(light_dir)
    ambient = torch.Tensor([0.0, 0.0, 0.0]).float()
    albedo = torch.Tensor([1.0, 1.0, 1.0]).float()
    diffusion = torch.clamp(mut.torch_dot(normals, -light_dir) * 0.5 + 0.5, 0, 1)[..., None]
    shaded = torch.clamp(ambient + albedo[None, :] * diffusion, 0, 1)
    shaded = shaded.reshape(output['normals'].shape[0], output['normals'].shape[1], -1)
    res['viz']['shaded'] = mask_image(shaded.numpy(), mask, (1, 1, 1), close_hole_size=close_hole_size)

    # Linear depth.
    depth_map = output['depth']
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    res['viz']['depth'] = cmap(depth_map.cpu().numpy())

    if build_pcd:
        # Build PCD.
        pcd_coords = output['pos'].cpu().numpy().reshape(-1, 3)
        pcd_normals = output['normals'].cpu().numpy().reshape(-1, 3)

        # Transform back to input coords using the model matrix.
        pcd_coords = math_utils.transform_points(model_matrix, pcd_coords, return_euclidean=True)
        pcd_normals = math_utils.transform_normals(model_matrix, pcd_normals)

        # Create pointcloud.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_coords)
        pcd.normals = o3d.utility.Vector3dVector(pcd_normals)

        res['pcd'] = pcd

    return res


def write_raytrace_results(res: dict, output_path_base: str, gui=True):
    """
    Writes out outputs of raytrace_sdf().
    """

    # Save all.
    for i, (name, im) in enumerate(res['viz'].items()):
        imageio.imwrite(output_path_base + f'_rt_{name}.png',
                        (np.clip(im, 0, 1) * 255).astype(np.uint8))

    # Plot.
    if gui:
        _, axs = plt.subplots(2, 3)
        axs = axs.flatten()
        for i, (name, im) in enumerate(res['viz'].items()):
            axs[i].imshow(im)
            axs[i].set_title(name)
        plt.show()

    # Save PCD.
    if 'pcd' in res:
        pcd = res['pcd']
        o3d.io.write_point_cloud(str(output_path_base + '_pcd.ply'), pcd)


def pad_video_frame(im: np.array):
    """
    Pads video frame to be safe for video.
    """
    if len(im.shape) == 2:
        im = im[..., None].repeat(3, axis=2)
    if im.shape[0] % 2 != 0:
        # Make Height multiple of 2
        im = np.concatenate((im, np.zeros((1, im.shape[1], im.shape[2]), dtype=np.dtype)), axis=0)
    if im.shape[1] % 2 != 0:
        # Make Width multiple of 2
        im = np.concatenate((im, np.zeros((im.shape[0], 1, im.shape[2]), dtype=np.dtype)), axis=1)
    return im


def raytrace_video_ibr(output_path_base: str,
                       decoder: SDFIBRNet,
                       resolution: np.array,
                       projection_matrix: np.array,
                       view_matrices: list,
                       model_matrix: np.array,
                       timestamps: list,
                       batch_size,
                       render_diffuse=False,  # Additional diffuse color.
                       save_frames=False,
                       debug_gui=False):
    """
    Ray-traces general space-time video frame afer frame into a video file.
    """
    video_folder = Path(output_path_base) / 'video'
    video_folder.mkdir(0o777, True, True)
    video_frames_folder = None
    if save_frames:
        video_frames_folder = Path(output_path_base) / 'frames'
        video_frames_folder.mkdir(0o777, True, True)
    frame_folders = {}
    video_writers = {}

    # Export path to JSON.
    export_poses_json(video_folder / 'video_poses.json', view_matrices, projection_matrix, resolution)

    # decoder.view_number = 0
    for i, view_matrix in enumerate(view_matrices):
        print(f'Video frame {i} / {len(view_matrices)}....')
        render = raytrace_sdf_ibr(decoder=decoder,
                                  resolution=resolution,
                                  projection_matrix=projection_matrix,
                                  view_matrix=view_matrix,
                                  model_matrix=model_matrix,
                                  timestamp=timestamps[0],
                                  build_pcd=False,
                                  render_softmask=False,
                                  batch_size=batch_size,
                                  debug_gui=debug_gui,
                                  background_color=(1, 1, 1),
                                  close_hole_size=3,
                                  vid_frame=i)

        render_ibr = decoder.forward_test_custom(model_matrix, view_matrix, projection_matrix, resolution, vid_frame=i)
        render['viz']['colors'] = render_ibr['target_img'].squeeze().permute(1,2,0).cpu().numpy()

        render['viz']['valid_mask'] = render_ibr['valid_mask'].squeeze().unsqueeze(-1).cpu().numpy()
        render['viz']['colors_masked'] = render['viz']['colors'] * render['viz']['valid_mask']

        render['viz']['colors_masked_processed'] = mask_image(render['viz']['colors'],
                                                              render['viz']['valid_mask'],
                                                              (1, 1, 1), close_hole_size=3)

        render['viz']['colors_masked_edited'] = mask_image(render['viz']['colors'],
                                                           render['viz']['valid_mask'],
                                                           (1, 1, 1), close_hole_size=3)
        render['viz']['colors_masked_edited'][..., 1] = (render['viz']['colors_masked_edited'][..., 0] +
                                                         render['viz']['colors_masked_edited'][..., 2]) / 2.

        if i == 0:
            # Initialize.
            for name, im in render['viz'].items():
                if save_frames:
                    # Frame folder.
                    frame_folders[name] = video_frames_folder / name
                    frame_folders[name].mkdir(0o777, True, True)

                # Video writer.
                video_writers[name] = skvideo.io.FFmpegWriter(
                    str(video_folder / f'{name}.mp4'),
                    inputdict={'-r': '30'},
                    outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', '-r': '30'})

        # Add frames
        for name, im in render['viz'].items():
            # To 8bit.
            im8 = (np.clip(im[..., :3], 0, 1) * 255).astype(np.uint8)
            if save_frames:
                # To file.
                imageio.imwrite(str(frame_folders[name] / f'{i:06d}.jpg'), im8)
            # To video.
            video_writers[name].writeFrame(pad_video_frame(im8))

    # Close video writers.
    for name, writer in video_writers.items():
        writer.close()


@ torch.no_grad()
def export_poses_json(output_filename: Path, view_matrices, projection_matrix: torch.tensor, resolution: torch.tensor):
    """
    Exports poses that allows external re-render.
    """
    print('Exporting poses to JSON.')

    views = []
    for i, view_matrix in enumerate(view_matrices):
        views += [
            OrderedDict([
                ('view', view_matrix.flatten().tolist()),
                ('projection', projection_matrix.flatten().tolist()),
                ('projection_params', math_utils.decompose_projection_matrix(projection_matrix.detach().cpu().numpy())),
                ('resolution', np.array(resolution).tolist()),
            ])
        ]

    with output_filename.open('w') as outfile:
        json.dump(views, outfile, indent=4, separators=(',', ': '))
