"""
Plots camera poses using matplotlib.
From Pytorch3D:
https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/docs/tutorials/utils/camera_visualization.py
Example:
https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/bundle_adjustment.ipynb

Rewritten to numpy by me.
"""

import matplotlib.pyplot as plt
import numpy as np
import utils.math_utils as math_utils
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import << MUST BE HERE!!!


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * np.array([-2, 1.5, 4])
    b = 0.5 * np.array([2, 1.5, 4])
    c = 0.5 * np.array([-2, -1.5, 4])
    d = 0.5 * np.array([2, -1.5, 4])
    C = np.zeros(3)
    F = np.array([0, 0, 3])
    camera_points = [a, b, d, c, a, C, b, d, C, c, C, F]
    lines = np.stack([x.astype(np.float32) for x in camera_points]) * scale
    lines[:, 2] *= -1
    lines *= 0.3
    return lines


def plot_cameras(ax, cameras: np.array, model_matrix: np.array, color: str = "blue", show_labels=False):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe() * model_matrix[0, 0]
    # Cam to world.
    cam_wires_trans = math_utils.transform_points(
        np.linalg.inv(cameras @ model_matrix),
        cam_wires_canonical, return_euclidean=True)
    plot_handles = []
    for i, wire in enumerate(cam_wires_trans):
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
        if show_labels:
            ax.text(x_[10], y_[10], z_[10], f'{i}')
    return plot_handles


def axisEqual3D(ax):
    """
    Makes the 3D axes keep aspect ratio.
    https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
    """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_camera_scene(view_original: np.array, view_optimized: np.array,
                      model_matrix: np.array, resolution=None, show_gui=False):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    Cameras are view_matrices 4x4.
    """
    dpi = 100
    if resolution is None:
        fig = plt.figure(figsize=(2.75, 2.75), dpi=dpi)
    else:
        fig = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi), dpi=dpi)
    ax = fig.gca(projection="3d")
    ax.clear()
    # ax.set_title(status)
    handle_cam_orig = plot_cameras(ax, view_original, model_matrix, color="#888888")
    handle_cam_opt = plot_cameras(ax, view_optimized, model_matrix, color="#FFA500", show_labels=True)
    # plot_radius = 0.5
    # ax.set_xlim3d([-plot_radius, plot_radius])
    # ax.set_ylim3d([-plot_radius, plot_radius])
    # ax.set_zlim3d([-plot_radius, plot_radius])
    axisEqual3D(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    # ax.axis('off')
    labels_handles = {
        "Original": handle_cam_orig[0],
        "Optimized": handle_cam_opt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper left",
        bbox_to_anchor=(0, 0),
    )

    # Left-handed -> Right-handed
    ax.invert_xaxis()

    if show_gui:
        plt.show()
    return fig
