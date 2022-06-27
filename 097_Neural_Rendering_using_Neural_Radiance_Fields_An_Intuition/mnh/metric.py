import os 
import numpy as np 
import torch
from .utils import compute_psnr, compute_ssim, list2txt, get_image_tensors
import argparse 
import lpips

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('-rewrite', dest='rewrite', action='store_true')
    parser.set_defaults(rewrite=False)
    args = parser.parse_args()

    folder = args.folder 
    device = 'cuda:0'
    psnr_path = os.path.join(folder, 'psnr.txt')
    ssim_path = os.path.join(folder, 'ssim.txt')
    lpips_path = os.path.join(folder, 'lpips.txt')

    if not os.path.isfile(psnr_path) or args.rewrite:
        psnr_list = folder_metric(folder, metric_func=compute_psnr, device=device)
        list2txt(psnr_list, psnr_path)
    psnr_list = np.genfromtxt(psnr_path)
    
    if not os.path.isfile(ssim_path) or args.rewrite:
        ssim_list = folder_metric(folder, metric_func=compute_ssim, device=device)
        list2txt(ssim_list, ssim_path)
    ssim_list = np.genfromtxt(ssim_path)

    if not os.path.isfile(lpips_path) or args.rewrite:
        lpips_fn = ComputeLPIPS(device)
        lpips_list = folder_metric(folder, metric_func=lpips_fn, device=device)
        list2txt(lpips_list, lpips_path)
    lpips_list = np.genfromtxt(lpips_path)

    print('Metrics evaluated in: {} \n------------'.format(folder))
    print('PSNR:  {:.2f}'.format(np.mean(psnr_list)))
    print('SSIM:  {:.3f}'.format(np.mean(ssim_list)))
    print('LPIPS: {:.3f}'.format(np.mean(lpips_list)))
    

def folder_metric(
    folder, 
    metric_func,
    device='cuda:0'
):
    images = get_image_tensors(folder) #(n, h, w, 3)

    view_num  = images.size(0) // 2
    metric_list = []
    for i in range(view_num):
        id_gt = 2*i
        id_pred = 2*i+1
        img_gt = images[id_gt].to(device)
        img_pred = images[id_pred].to(device)
        value = metric_func(img_gt, img_pred)
        metric_list.append(value)
    return metric_list

def pair_metric(gt, pred, metric_func, device='cuda:0'):
    '''
    gt, pred: (n, h, w, 3)
    '''
    gt = gt.to(device)
    pred = pred.to(device)
    
    metric_list = []
    num = gt.size(0)
    for i in range(num):
        img_gt = gt[i]
        img_pred = pred[i]
        val = metric_func(img_gt, img_pred)
        metric_list.append(val)
    return metric_list

class ComputeLPIPS():
    def __init__(self, device):
        self.device = device 
        self.model = lpips.LPIPS(net='alex').to(device)
    
    def __call__(self, image_0, image_1):
        image_0 = image_0.permute(2, 0, 1)
        image_1 = image_1.permute(2, 0, 1)
        out = self.model(image_0, image_1)
        val = out.item()
        return val 

if __name__ == '__main__':
    main()