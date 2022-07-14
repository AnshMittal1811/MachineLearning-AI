# MIT Licence

# Methods to predict the SSIM, taken from
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

import os
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Lambda
import numpy as np
import json
import uuid

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(
    img1, img2, window, window_size, channel, mask=None, size_average=True
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if not (mask is None):
        b = mask.size(0)
        ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask
        ssim_map = ssim_map.view(b, -1).sum(dim=1) / mask.view(b, -1).sum(
            dim=1
        ).clamp(min=1)
        return ssim_map

    import pdb

    pdb.set_trace

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, mask=None, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, mask, size_average)

# Classes for PSNR and SSIM, taken from SynSin

class SSIM(nn.Module):
    """The structural similarity index (SSIM) is a method for predicting the perceived quality of images."""
    def forward(self, pred_img, gt_img):
        """
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """
        return {"ssim": ssim(pred_img, gt_img)}

class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) is an expression for the ratio between the maximum possible value (power)
    of a signal and the power of distorting noise that affects the quality of its representation.
    """
    def forward(self, pred_img, gt_img):
        """
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """
        mse = F.mse_loss(pred_img, gt_img)
        psnr = 10 * (1 / mse).log10()
        return {"psnr": psnr.mean()}

def find_images(dir, key):
    images = []
    for file in os.listdir(dir):
        if key in file:
            images.append(os.path.join(dir, file))
    return sorted(images)

def calc_metrics(pred_images, gt_images):
    # objects for calculations
    psnr = PSNR()
    ssim = SSIM()
    to_tensor = Compose([
        Resize(192),
        ToTensor(),
        Lambda(lambda x: x.unsqueeze(0))
    ])

    # storage of all scores
    psnr_scores = []
    ssim_scores = []

    # calculate for all images independently (batch-size = 1)
    for i in range(len(pred_images)):
        # load images
        pred_image = to_tensor(Image.open(pred_images[i]))
        gt_image = to_tensor(Image.open(gt_images[i]))

        # calc psnr
        psnr_score = psnr(pred_image, gt_image)["psnr"].detach().cpu().numpy()
        psnr_scores.append(psnr_score)

        # calc ssim
        ssim_score = ssim(pred_image, gt_image)["ssim"].detach().cpu().numpy()
        ssim_scores.append(ssim_score)

    # mean scores across all images
    metrics = {
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores)
    }

    print(f"Calculated metrics: {metrics}")

    return metrics


def add_metadata(metrics, input_pred, input_gt, number_images):
    metrics["input_pred"] = input_pred
    metrics["input_gt"] = input_gt
    metrics["number_images"] = number_images

    print(f"Added metadata to metrics: {metrics}")

def save_metrics(metrics, output):
    id = str(uuid.uuid4())
    file = os.path.join(output, "metrics_" + id + ".json")
    with open(file, "w") as f:
        json.dump(str(metrics), f)
        print(f"Saved metrics under {file}")

def main(input_pred, input_gt, output):
    # find pred and gt images in folders
    pred_images = find_images(input_pred, "synthesized_image") # key from the pix2pix repo that gets added to predictions when running their test.py script
    gt_images = find_images(input_gt, ".png") # in the train_B folder all images are the gt images and thus ".png" is the only requirement

    # must have equal length for calculation to make sense
    gt_images = gt_images[:len(pred_images)]
    assert (len(pred_images) == len(gt_images))

    # calculate metrics
    # assume the prefix for pred and gt images are similar and thus the sorting groups i-th pred to i-th gt image
    metrics = calc_metrics(pred_images, gt_images)

    # add metadata for this metrics
    add_metadata(metrics, input_pred, input_gt, len(pred_images))

    # save metrics to file
    save_metrics(metrics, output)

    print("Completed calculation successfully.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate SSIM and PSNR metric on a folder of pix2pix seg2rgb pairs.')
    parser.add_argument('--input_pred', metavar='path', required=True,
                        help='path/to/image/folder where the predicted rgb images lie.')
    parser.add_argument('--input_gt', metavar='path', required=True,
                        help='path/to/image/folder where the gt rgb images lie.')
    parser.add_argument('--output', metavar='path', required=False, default=None,
                        help='path/to/output/directory. Where to store the metrics output as file. Optional. Default: input_pred directory')

    args = parser.parse_args()
    main(input_pred=args.input_pred,
         input_gt=args.input_gt,
         output=args.output if args.output is not None else args.input_pred)
