
"""Plot example."""

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torchvision.utils import make_grid

import gqnlib


def main() -> None:
    """Load pre-trained model and show reconstruction and samples."""

    # Settings
    root = "./data/shepard_metzler_5_parts_torch/train/"
    cp_path = "./logs/tmp/example.pt"
    show_single = False

    # Load pre-trained model
    model = gqnlib.GenerativeQueryNetwork()
    cp = torch.load(cp_path)
    model.load_state_dict(cp["model_state_dict"])

    # Data
    dataset = gqnlib.SceneDataset(root, 20)
    images, viewpoints = dataset[0][0]
    x_c, v_c, x_q, v_q = gqnlib.partition_scene(images, viewpoints)

    # Reconstruct and sample
    with torch.no_grad():
        recon = model.reconstruct(x_c, v_c, x_q, v_q)
        sample = model.sample(x_c, v_c, v_q)

    # Plot
    if show_single:
        plt.figure(figsize=(12, 8))

        plt.subplot(131)
        imshow(x_q[0])
        plt.title("Original")

        plt.subplot(132)
        imshow(recon[0])
        plt.title("Reconstructed")

        plt.subplot(133)
        imshow(sample[0])
        plt.title("Sampled")
    else:
        # Show grid
        plt.figure(figsize=(20, 12))

        plt.subplot(311)
        gridshow(x_q)
        plt.title("Original")

        plt.subplot(312)
        gridshow(recon)
        plt.title("Reconstructed")

        plt.subplot(313)
        gridshow(sample)
        plt.title("Sampled")

    plt.tight_layout()
    plt.show()


def imshow(img: Tensor) -> None:
    """Show single image.

    Args:
        img (torch.Tensor): (c, h, w) or (1, c, h, w).
    """

    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)
    elif img.dim() != 3:
        raise ValueError(f"Wrong image size: {img.size()}")

    # CHW -> HWC
    npimg = img.permute(1, 2, 0).numpy()
    plt.imshow(npimg, interpolation="nearest")


def gridshow(img: Tensor) -> None:
    """Show images in grid.

    Args:
        img (torch.Tensor): (b, c, h, w) or (b, 1, c, h, w).
    """

    if img.dim() == 5 and img.size(1) == 1:
        img = img.squeeze(1)
    elif img.dim() != 4:
        raise ValueError(f"Wrong image size: {img.size()}")

    grid = make_grid(img)
    npgrid = grid.permute(1, 2, 0).numpy()
    plt.imshow(npgrid, interpolation="nearest")


if __name__ == "__main__":
    main()
