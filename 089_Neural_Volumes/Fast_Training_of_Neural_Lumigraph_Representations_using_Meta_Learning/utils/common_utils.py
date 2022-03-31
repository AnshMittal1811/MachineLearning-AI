import platform
import os
from pathlib import Path


def system_has_display():
    """
    Checks if display available.
    """
    return bool(os.environ.get('DISPLAY', None)) or platform.system() == 'Windows'


# initialize Matplotlib early
import matplotlib
if not system_has_display():
    # No-display.
    matplotlib.use('Agg')  # set the backend before importing pyplot

import numpy as np
import torch
import imageio
import shutil
import skimage.measure
import matplotlib.pyplot as plt

# Relative timestamp [-1, 1] of the key frame.
KEY_FRAME_TIME = -1.0


def prepare_path(path: Path):
    """
    Creates dir recursively if does not exist.
    """
    path.mkdir(0o777, True, True)
    return path


def copy_exclusive(src: Path, dst: Path, ignore_missing=False):
    """
    Copies directory while deleting the target if exists.
    """
    if src.absolute() == dst.absolute():
        return dst

    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.is_file():
        dst.unlink()

    if not src.exists() and ignore_missing:
        return None

    if src.is_dir():
        return shutil.copytree(src, dst)
    else:
        return shutil.copy(src, dst)


def move_exclusive(src: Path, dst: Path, ignore_missing=False):
    """
    Copies directory while deleting the target if exists.
    """
    if src.absolute() == dst.absolute():
        return dst

    if not src.exists() and ignore_missing:
        return None

    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.is_file():
        dst.unlink()

    return shutil.move(src, dst)


def cond_mkdir(path):
    """
    Creates dir recursively if does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def cond_rmtree(path):
    """
    Removes dir tree if exists.
    """
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            print(f'ERROR: Could not remove {path}!')


def densely_sample_activations(model, num_dim=1, num_steps=int(1e6)):
    input = torch.linspace(-1., 1., steps=num_steps).float()

    if num_dim == 1:
        input = input[..., None]
    else:
        input = torch.stack(torch.meshgrid(*(input for _ in num_dim)), dim=-1).view(-1, num_dim)

    input = {'coords': input[None, :].to(model.device)}
    with torch.no_grad():
        activations = model.forward_with_activations(input)['activations']
    return activations


def make_contour_plot(array_2d, mode='lin', resolution=None):
    if resolution is None:
        fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(resolution[0] / 300, resolution[1] / 300), dpi=300)

    if(mode == 'log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels * 2 + 1))
    elif(mode == 'lin'):
        num_levels = 5
        levels = np.linspace(-1, 1, num=num_levels * 2 + 1)
        colors = plt.get_cmap("RdBu")(np.linspace(0., 1., num=num_levels * 2 + 1))

    sample = np.flipud(array_2d)
    #CS = ax.contourf(sample, levels=levels, colors=colors)
    #cbar = fig.colorbar(CS)

    def my_cmap(x):
        x = x * 10
        x = 1 / (1 + np.exp(-x))
        return plt.get_cmap("RdBu")(x)
    ax.imshow(my_cmap(sample))

    ax.contour(sample, levels=levels, colors='k', linewidths=0.3)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.9)
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    return fig


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)


# Functions for converting
def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.
    Note that this requires the ``matplotlib`` package.
    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure
    Returns:
        numpy.array: image in [CHW] order
    From: https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
    """
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as plt_backend_agg

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            print(fn)
            assert all(t is None or torch.all(~torch.isnan(t))
                       for t in grad_input), f"{fn} grad_input={grad_input} grad_output={grad_output}"
            assert all(t is None or torch.all(~torch.isnan(t))
                       for t in grad_output), f"{fn} grad_input={grad_input} grad_output={grad_output}"

            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)


def get_git_revision():
    """
    Returns current git HEAD.
    """
    return 'unknown'  # run_command.run_command('git rev-parse HEAD', timeout=1)[1].strip()


def parse_comma_int_args(values: str):
    """
    Parses comma separated integer arguments.
    """
    try:
        return [int(x) for x in values.split(',') if len(x) > 0]
    except:
        return []


def get_checkpoint_name(epoch: int) -> str:
    """
    Gets checkpoint name for an epoch.
    """
    if epoch < 0:
        return 'model_current'
    return f'model_epoch_{epoch:04d}'


def get_optim_name(epoch: int) -> str:
    """
    Gets optim name for an epoch.
    """
    if epoch < 0:
        return 'optim_current'
    return f'optim_{epoch:04d}'


def checkpoint_to_optim_name(checkpoint_name: str):
    return checkpoint_name.replace('model_', 'optim_').replace('epoch_', '')


def format_matrix(matrix: np.array) -> str:
    """
    Nice format matrix.
    """
    return np.array2string(matrix, precision=2, formatter={'float': lambda x: f'{x:.3f}'})


def imwritef(filename, im, format=None, **kwargs):
    """
    Saves float image.
    """
    imageio.imwrite(filename, (np.clip(im, 0, 1) * 255).astype(np.uint8), format=None, **kwargs)
