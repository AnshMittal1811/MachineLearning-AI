import numpy as np
import os
from PIL import Image
from collections import OrderedDict
import time

# import open3d


def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (
        len(img_array.shape) == 3 and img_array.shape[2] in [3, 4]
    )
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)


def depth_to_pointcloud(depth_array, camera_position, ray_directions, mask=None):
    """
    Args:
        depth_array: :math:`[M]`
        camera_position: :math:`[M, 3]` or :math: `[3]`
        ray_directions: :math:`[M, 3]`
        mask: :math:`[M]` or None
    Return:
        points: :math:`[M', 3]` valid 3d points
    """
    assert len(depth_array.shape) == 1
    M = depth_array.shape[0]
    assert camera_position.shape in [(3,), (M, 3)]
    assert ray_directions.shape == (M, 3)
    assert mask is None or mask.shape == (M,)

    if mask is None:
        mask = np.ones_like(depth_array)

    ray_dir = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    points = camera_position + ray_dir * depth_array.reshape((M, 1))
    points = points[mask]
    return points


# TODO: implement a custom module for saving full PCD
def save_pointcloud(points, filename):
    assert len(points.shape) == 2 and points.shape[1] == 3

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        + "VERSION 0.7\n"
        + "FIELDS x y z\n"
        + "SIZE 4 4 4\n"
        + "TYPE F F F\n"
        + "COUNT 1 1 1\n"
        + "WIDTH {}\n".format(len(points))
        + "HEIGHT 1\n"
        + "VIEWPOINT 0 0 0 1 0 0 0\n"
        + "POINTS {}\n".format(len(points))
        + "DATA binary\n"
    )

    with open(filename, "wb") as f:
        f.write(bytearray(header, "ascii"))
        f.write(points.astype(np.float32).tobytes())


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.image_dir = os.path.join(opt.checkpoints_dir, opt.name, "images")

    def display_current_results(
        self, visuals, total_steps, camera_position=None, ray_directions=None
    ):
        for name, img in visuals.items():
            img = np.array(img)
            filename = "step-{:08d}-{}.png".format(total_steps, name)
            filepath = os.path.join(self.image_dir, filename)
            save_image(img, filepath)

        if camera_position is not None and ray_directions is not None:
            camera_position = np.array(camera_position)
            ray_directions = np.array(ray_directions)
            for name, img in visuals.items():
                if len(img.shape) == 2 and "depth" in name:
                    depth = np.array(img).reshape(-1)
                    filename = "step-{:08d}-{}.pcd".format(total_steps, name)
                    filepath = os.path.join(self.image_dir, filename)
                    pcd = depth_to_pointcloud(
                        depth, camera_position, ray_directions, depth != 0
                    )
                    save_pointcloud(pcd, filepath)

    def reset(self):
        self.start_time = time.time()
        self.acc_iterations = 0
        self.acc_losses = OrderedDict()

    def accumulate_losses(self, losses):
        self.acc_iterations += 1
        for k, v in losses.items():
            if k not in self.acc_losses:
                self.acc_losses[k] = 0
            self.acc_losses[k] += v

    def print_losses(self, total_steps):
        m = "End of iteration {} \t Number of batches {} \t Time taken: {:.2f}s\n".format(
            total_steps, self.acc_iterations, (time.time() - self.start_time)
        )
        m += "[Average Loss] "
        for k, v in self.acc_losses.items():
            m += "{}: {:.10f}   ".format(k, v / self.acc_iterations)

        filepath = os.path.join(self.log_dir, "log.txt")
        with open(filepath, "a") as f:
            f.write(m + "\n")
        print(m)
