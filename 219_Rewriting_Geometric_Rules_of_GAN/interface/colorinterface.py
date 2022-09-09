import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image

from .labwidget import Property, Button, Label, Div
from .paintwidget import ColorPaintWidget
from .show import show, url_to_pil


def morphological_close(mask, kernel_size):
    k = np.ones((kernel_size, kernel_size), np.uint8)
    orig_dtype = mask.dtype
    dilated = cv2.dilate(mask.astype(np.uint8), k)
    closed = cv2.erode(dilated, k).astype(orig_dtype)
    return closed


def morphological_open(mask, kernel_size):
    k = np.ones((kernel_size, kernel_size), np.uint8)
    orig_dtype = mask.dtype
    eroded = cv2.erode(mask.astype(np.uint8), k)
    opened = cv2.dilate(eroded, k).astype(orig_dtype)
    return opened


def preprocess_mask(mask_np):
    color_np = mask_np[..., :3].copy()
    drawn = mask_np[..., -1] == 255
    # get background mask, clean up with morphological closing
    background_mask = np.all(color_np == 128, axis=-1).astype(np.uint8)
    background_mask = morphological_close(background_mask, 5)
    # get color mask, further clean with with morphological opening
    color_mask = drawn ^ (background_mask & drawn)
    color_mask = drawn & morphological_open(color_mask, 5)
    # only keep colors from the color mask
    color_np = color_np * color_mask[..., None]

    return {
        'color': color_np,
        'color_mask': color_mask,
        'background_mask': background_mask
    }


status = Property('Press the button to save the image.')


class ColorInterface:
    def __init__(self, model_pkl_path, truncation, seed, save_dir, stylegan_repo='./models/networks/stylegan3'):
        self.trunc = truncation
        self.save_dir = save_dir

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
        brushsize = 20

        self.V = ColorPaintWidget(width=size, height=size, brushsize=brushsize)
        self.prohibited_color = '#808080'
        self.prev_color = self.V.stroke_color
        self.prev_palette = self.V.gamut.palette

        self.img_offset = 0

        self.V.on('stroke_color', self.changed)
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

    def changed(self):
        global prev_color, prev_palette
        if self.V.stroke_color == self.prohibited_color:
            self.V.prop('stroke_color').value = prev_color  # hack to avoid infinite loop
            self.V.change_palette(prev_palette)
            status.set(f'Color {self.prohibited_color} is reserved for fixed background.')
        else:
            prev_color = self.V.stroke_color
            prev_palette = self.V.gamut.palette

    def refresh(self):
        f = open(self.save_dir + '/counter', "a+")
        f.seek(0)
        t = f.read()
        t = int(t) if t else 0
        f.close()
        img_pil = self.sample_image(t + self.img_offset, t)

        self.V.set_image(img_pil)

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
        url_to_pil(self.V.paint.mask).convert('RGB').save(save_im_path)

        save_mk_path = os.path.join(self.save_dir, 'masks', f'{t}.npz')
        mask_np = np.asarray(url_to_pil(self.V.paint.mask))
        mask_data = preprocess_mask(mask_np)
        np.savez(save_mk_path, **mask_data)

        status.set(f'stroke image saved at: {save_im_path}\n'
                   f'stroke data saved at: {save_mk_path}')

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
