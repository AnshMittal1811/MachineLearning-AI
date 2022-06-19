import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import numpy as np
import scipy

def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()


class HSI_Calculator(nn.Module):
    def __init__(self):
        super(HSI_Calculator, self).__init__()

    def forward(self, image):
        image = transforms.ToTensor()(image)
        I = torch.mean(image)
        Sum = image.sum(0)
        Min = 3 * image.min(0)[0]
        S = (1 - Min.div(Sum.clamp(1e-6))).mean()
        numerator = (2 * image[0] - image[1] - image[2]) / 2
        denominator = ((image[0] - image[1]) ** 2 + (image[0] - image[2]) * (image[1] - image[2])).sqrt()
        theta = (numerator.div(denominator.clamp(1e-6))).clamp(-1 + 1e-6, 1 - 1e-6).acos()
        logistic_matrix = (image[1] - image[2]).ceil()
        H = (theta * logistic_matrix + (1 - logistic_matrix) * (360 - theta)).mean() / 360
        return H, S, I


def eval_steps_adaptive(var):
    return {
            400 * 100: 5000,
            400 * 500: 2000,
            400 * 1000: 1000,
    }.get(var, 1600)


def get_density_map_gaussian(N, M, points, adaptive_kernel=False, fixed_value=15):
    density_map = np.zeros([N, M], dtype=np.float32)
    h, w = density_map.shape[:2]
    h = h // 8
    w = w // 8
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_kernel:
        # referred from https://github.com/vlad3996/computing-density-maps/blob/master/make_ShanghaiTech.ipynb
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances = tree.query(points, k=4)[0]

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1] // 8), min(w-1, p[0] // 8)
        if num_gt > 1:
            if adaptive_kernel:
                sigma = int(np.sum(distances[idx][1:4]) // 3 * 0.3)
            else:
                sigma = fixed_value
        else:
            sigma = fixed_value  # np.average([h, w]) / 2. / 2.
        sigma = max(1, sigma)

        gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0] // 8, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(density_map.shape[1] // 8, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map