from typing import Tuple, List
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.metrics import structural_similarity as calculate_ssim

def parameter_number(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_psnr(gt, pred):
    mse = torch.mean((gt - pred)**2)
    device = gt.device
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(device))
    psnr = psnr.cpu().item()
    return psnr 

def compute_ssim(gt, pred):
    '''image size: (h, w, 3)'''
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    ssim = calculate_ssim(pred, gt, data_range=gt.max() - gt.min(), multichannel=True)
    return ssim

def tensor2Image(
    tensor, 
    image_size: Tuple[int, int] = None, 
    resample: str = 'nearest'
):
    '''
    Args:
        tensor: shape (h, w, 3) in range (0, 1)
        image_size: (h, w)
        resample: 'nearest' or 'bilinear'
    Return:
        PIL image
    '''
    img = tensor.squeeze().detach().cpu()#.clamp(0.0, 1.0)
    img = (img * 255).numpy().astype(np.uint8)
    img = Image.fromarray(img)
    if image_size != None: 
        image_size = (int(image_size[0]), int(image_size[1]))
        resample_mode = Image.NEAREST if resample == 'nearest' else Image.BILINEAR
        img = img.resize(image_size, resample=resample_mode)
    return img

def output_images(
    output_dir: str,
    images_tensor: dict,
    image_size: Tuple[int, int] = None,
    prefix: str = '',
    postfix: str = ''
):
    for name, image_tensor in images_tensor.items():
        image = tensor2Image(image_tensor, image_size)
        out_name = '{}{}{}.png'.format(prefix, name, postfix)
        out_path = os.path.join(output_dir, out_name)
        image.save(out_path)

def generate_gif(
    gif_name: str,
    images: List[torch.tensor],
    size: Tuple[int, int],
    duration=50
):
    images = [tensor2Image(img, size) for img in images]
    images[0].save(
        gif_name,
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=duration,
        loop=0
    )

def list2txt(list, path):
    '''
    output list values into text file (rows) 
    '''
    with open(path, 'w') as f:
        for i in range(len(list)):
            val = list[i]
            f.write('{}\n'.format(val))

def is_image_file(file_name):
    if file_name.endswith('.png') or file_name.endswith('.jpg'):
        return True
    else:
        return False

def get_image_tensors(folder, channels:int=3):
    names = sorted(os.path.join(folder, name) for name in os.listdir(folder) if is_image_file(name))
    images = []
    for i in range(len(names)):
        img = Image.open(names[i])
        img_array = np.array(img)[...,:channels]
        img_tensor = torch.FloatTensor(img_array)
        img_tensor /= 255.0
        images.append(img_tensor)
    images = torch.stack(images, dim=0)
    return images

def random_sample_points(points, rate:float):
    points_n = points.size(0)
    sample_n = int(points_n * rate)
    sample_idx = torch.randperm(points_n)[:sample_n]
    points = points[sample_idx]
    return points

def to_numpy(tensor):
    array = tensor.detach().to('cpu').numpy()
    return array

class Timer:
    def __init__(self, cuda_sync:bool=False):
        self.cuda_sync = cuda_sync 
        self.reset()
    
    def reset(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.start = time.time()
    
    def get_time(self, reset=True):
        if self.cuda_sync:
            torch.cuda.synchronize()
        now = time.time()
        interval = now - self.start
        if reset:
            self.reset()
        return interval

    def print_time(self, info, reset=True):
        interval = self.get_time(reset)
        print('{:.5f} | {}'.format(interval, info))