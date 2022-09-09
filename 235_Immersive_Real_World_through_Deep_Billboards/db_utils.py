import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
import imageio
from tqdm import tqdm
from copy import deepcopy
import svox


def xyzrpy2c2w(xyz, rpy):
    c2w = np.zeros([4,4])
    c2w[:3,:3] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    c2w[:,3] = np.concatenate([xyz, np.array([1.,])])
    return c2w


def show_c2w(c2w, ss=3, sv=0.5, figsize=(4,4), rot1=None, rot2=None):
    c2w = deepcopy(c2w)
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_xlim(-ss, ss); ax.set_ylim(-ss,ss); ax.set_zlim(-ss, ss)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.quiver(0, 0, 0, 1, 0, 0, length=sv*2, color="r", alpha=0.5)
    ax.quiver(0, 0, 0, 0, 1, 0, length=sv*2, color="g", alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, 1, length=sv*2, color="b", alpha=0.5)
    ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], [0,0,0,0,0], color="k", linestyle=":")

    if rot1:  # rot first
        rot1 = Rotation.from_euler('xyz', rot1, degrees=True).as_matrix()
        c2w[:3,:3] = c2w[:3,:3].dot(rot1)
    if rot2:  # c2w first
        rot2 = Rotation.from_euler('xyz', rot2, degrees=True).as_matrix()
        c2w[:3,:3] = rot2.dot(c2w[:3,:3])

    x, y, z = c2w[:3,3]
    ax.scatter(x, y, z, c='blue', s=0.1)
    x_c = np.array([[1,0,0]]).T; x_w = c2w[:3,:3].dot(x_c)
    dx, dy, dz = x_w[:,0]
    ax.quiver(x, y, z, dx, dy, dz, length=sv, color="r")
    # print("dx:", np.round(np.array([dx,dy,dz]), 3))
    x_c = np.array([[0,1,0]]).T; x_w = c2w[:3,:3].dot(x_c)
    dx, dy, dz = x_w[:,0]
    ax.quiver(x, y, z, dx, dy, dz, length=sv, color="g")
    # print("dy:", np.round(np.array([dx,dy,dz]), 3))
    x_c = np.array([[0,0,1]]).T; x_w = c2w[:3,:3].dot(x_c)
    dx, dy, dz = x_w[:,0]
    ax.quiver(x, y, z, dx, dy, dz, length=sv, color="b")
    # print("dz:", np.round(np.array([dx,dy,dz]), 3))

    fig.add_axes(ax)
    fig.canvas.draw(); im = np.array(fig.canvas.renderer.buffer_rgba()); plt.close()
    return Image.fromarray(im)


def render_and_show(r, c2w, w, h, focal, ss=3, sv=0.5, device=0):
    c2w = deepcopy(c2w)
    im1 = show_c2w(c2w, ss, sv)

    c2w = torch.from_numpy(c2w).float()
    fig = plt.figure(figsize=(4,4))
    with torch.no_grad():
        im = r.render_persp(c2w.to(device), w, h, fx=focal, fast=True)
    im = (im.clamp_(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)

    plt.imshow(im)
    fig.canvas.draw(); im2 = np.array(fig.canvas.renderer.buffer_rgba()); plt.close()
    return Image.fromarray(np.hstack([im1, im2])[:,:,:3])


def speedtest(r, dataset, save_path=None, device=0):
    w, h, focal = dataset.w, dataset.h, dataset.focal
    frames = []
    for idx in tqdm(range(dataset.size)):
        c2w = torch.from_numpy(dataset.camtoworlds[idx]).float().to(device)
        with torch.no_grad():
            im = r.render_persp(c2w, w, h, fx=focal, fast=True)
        im = (im.clamp_(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        if idx==0: plt.imshow(im); plt.show()
        if save_path: frames.append(im)
    if save_path: imageio.mimwrite(save_path, frames)


def lighten_tree(t, thresh, value=0):
    with torch.no_grad():
        t[
            (-thresh > t.corners[:,0].cpu().numpy()) | \
            (t.corners[:,0].cpu().numpy() > thresh) | \
            (-thresh > t.corners[:,1].cpu().numpy()) | \
            (t.corners[:,1].cpu().numpy() > thresh) | \
            (-thresh > t.corners[:,2].cpu().numpy()) | \
            (t.corners[:,2].cpu().numpy() > thresh),
            -1 
        ] = value
    return t