import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image

from lib.mls import mls_affine_deformation
from .labwidget import Property, Button, Label, Div
from .warpwidget import WarpWidget
from .show import show
from .renormalize import as_url, from_url


def warp(im, flow, alpha=1, interp=cv2.INTER_CUBIC):
    height, width, _ = flow.shape
    cart = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    pixel_map = (cart + alpha * flow).astype(np.float32)
    warped = cv2.remap(
        im,
        pixel_map[:, :, 0],
        pixel_map[:, :, 1],
        interp,
        borderMode=cv2.BORDER_REPLICATE)
    return warped


def warp_by_keypoints(image, keypoints, interp=cv2.INTER_CUBIC):
    is_pil = type(image) == Image.Image
    if is_pil:
        image = np.asarray(image)

    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vx, vy = np.meshgrid(gridX, gridY)

    keypoints = np.floor(np.array(keypoints, dtype=np.float32))
    src = keypoints[:, :2]
    tgt = keypoints[:, 2:]

    deform = mls_affine_deformation(vx, vy, src, tgt, alpha=1)
    deform = np.array(deform)
    idx_i, idx_j = tgt[:, 1].astype(int), tgt[:, 0].astype(int)
    deform[:, idx_i, idx_j] = src[:, ::-1].T

    dx, dy = deform[1], deform[0]
    warped = cv2.remap(image, dx, dy, interp, borderMode=cv2.BORDER_REFLECT)
    if is_pil:
        warped = Image.fromarray(warped)

    return warped


status = Property('Press the button to save the image.')


class WarpInterface:
    def __init__(self, model_pkl_path, truncation, seed, save_dir, crop_car=False, stylegan_repo='./models/networks/stylegan3'):
        self.trunc = truncation
        self.save_dir = save_dir
        self.crop_car = crop_car

        # import stylegan3 repo
        sys.path.append(stylegan_repo)
        import dnnlib
        import legacy

        # setup folders and subfolders for the dataset
        directories = ['originals', 'targets', 'latents', 'keypoints']
        for d in directories:
            os.makedirs(os.path.join(self.save_dir, d), exist_ok=True)

        # Load networks.
        print('Loading networks from "%s"...' % model_pkl_path)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(model_pkl_path) as fp:
            self.G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(self.device)  # type: ignore

        # sampling noises based on a fixed seed
        num_samples = 1000
        self.noises = np.random.RandomState(seed).randn(num_samples, self.G.z_dim)

        # setup the UI widgets for warping
        self.setup_widget()

    def display(self):
        show(self.layout)

    def setup_widget(self):
        size = self.G.img_resolution
        if self.crop_car:
            width, height = size, size * 3 // 4
        else:
            width, height = size, size
        wbd = width - 1
        hbd = height - 1
        fixed_keypts = [[0, 0, 0, 0], [0, hbd, 0, hbd], [wbd, 0, wbd, 0], [wbd, hbd, wbd, hbd]]

        self.V = WarpWidget(width=width, height=height, keypt_init=fixed_keypts)
        self.warped = None
        self.img_offset = 0

        self.V.on('keypt', self.changed)
        self.refresh()

        self.button0 = Button('refresh', style={'display': 'inline'})
        self.button0.on('click', self.refresh)

        self.button1 = Button('skip', style={'display': 'inline'})
        self.button1.on('click', self.skip)

        self.button2 = Button('save', style={'display': 'inline'})
        self.button2.on('click', self.save_info)

        self.button_div = Div(style={'margin': 'auto'})
        self.button_div.show([self.button0, self.button1, self.button2])

        self.label = Label(status)

        self.layout = [self.V, self.button_div, self.label]

    # warp the images
    def changed(self, c):
        if len(self.V.keypt) < 3:
            return
        warped = warp_by_keypoints(self.V.source, self.V.keypt)
        self.V.image_warped = as_url(warped)

    def refresh(self):
        f = open(self.save_dir + '/counter', "a+")
        f.seek(0)
        t = f.read()
        t = int(t) if t else 0
        f.close()
        img_pil = self.sample_image(t + self.img_offset, t)

        if self.crop_car:
            size = self.G.img_resolution
            img_pil = img_pil.crop((0, size // 8, size, size * 7 // 8))
        self.V.set_new_image(img_pil)

    def skip(self):
        self.img_offset += 1
        self.refresh()

    def save_info(self):
        f = open(self.save_dir + '/counter', "a+")
        f.seek(0)
        t = f.read()
        t = int(t) if t else 0
        f.seek(0)
        f.truncate()
        f.write(str(t + 1))
        f.truncate()
        f.close()

        save_im_path = os.path.join(self.save_dir, 'targets', f'{t}.png')
        target_im = from_url(self.V.image_warped, target='image')
        if self.crop_car:
            size = self.G.img_resolution
            canvas = Image.new("RGB", (size, size))
            canvas.paste(target_im, (0, size // 8, size, size * 7 // 8))
            target_im = canvas
        target_im.save(save_im_path)

        save_pt_path = os.path.join(self.save_dir, 'keypoints', f'{t}.npy')
        np.save(save_pt_path, self.V.keypt[4:])

        status.set(f'warped image saved at: {save_im_path}\n'
                   f'keypoints saved at: {save_pt_path}')

    @torch.no_grad()
    def sample_image(self, ind, save_ind):
        # Load image.
        z = torch.from_numpy(self.noises[[ind]]).to(self.device)
        img = self.G(z, None, truncation_psi=self.trunc)
        w_init = self.G.mapping(z, None, truncation_psi=self.trunc)

        img_pil = (img + 1) * (255 / 2)
        img_pil = img_pil.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        img_pil = Image.fromarray(img_pil)

        # Save the image and latents
        orig_path = os.path.join(self.save_dir, 'originals', f'{save_ind}_original.png')
        latent_path = os.path.join(self.save_dir, 'latents', f'{save_ind}_w.pth')
        img_pil.save(orig_path)
        torch.save(w_init.detach().cpu(), latent_path)

        return img_pil
