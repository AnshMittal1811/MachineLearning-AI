import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import pad as tensor_pad
from torchvision import transforms
import numpy as np
import cv2
import pdb

import time
import argparse
import sys
sys.path.append('../')
sys.path.append('./modules')
sys.path.append('./modules/PerceptualSimilarity')
sys.path.append('./modules/pytorch_ssim')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from TorchTools.DataTools.DataSets import SRDataList, RealPairDataset
from TorchTools.Functions.Metrics import PSNR, psnr, YCbCr_psnr
from TorchTools.TorchNet.tools import calculate_parameters, load_weights
from TorchTools.Functions.functional import tensor_block_cat, tensor_block_crop, tensor_merge, tensor_divide
from TorchTools.DataTools.Loaders import to_pil_image, to_tensor
from modules.architecture import RRDBNet
import pytorch_ssim
from PerceptualSimilarity import perceptual_similarity

#sys.path.append('./modules/SpatialPropagation')
#from SpatialPropagationNet import SPNModel

import warnings
warnings.filterwarnings("ignore")
import pdb

local_test = False
debug_mode = False
use_cuda = True

parser = argparse.ArgumentParser()
## Test Dataset
parser.add_argument('--test_dataroot', type=str, default='/media/data1/xzw/Datasets/Photo-4/Realvalid93')
# parser.add_argument('--test_dataroot', type=str, default='/media/data1/xiezw/RealSR/Dataset/Photo-4/select_patch')
# parser.add_argument('--test_dataroot', type=str, default='/media/data1/xiezw/RealSR/Dataset/Photo-4/Vali100_rand36/test_HR')

# Test Options
parser.add_argument('--overlap', type=int, default=64, help='Overlap pixel when Divide input image, for edge effect')
parser.add_argument('--psize', type=int, default=256, help='Overlap pixel when Divide input image, for edge effect')
parser.add_argument('--real', type=bool, default=True, help='Whether to downsample input image')
parser.add_argument('--cat_result', type=bool, default=False, help='Concat result to one image')
parser.add_argument('--has_GT', type=bool, default=True, help='the low resolution image size')
parser.add_argument('--rgb_range', type=float, default=1., help='255 EDSR and RCAN, 1 for the rest')
parser.add_argument('--save_results', type=bool, default=True, help='Concat result to one image')
parser.add_argument('--bic', type=bool, default=False, help='Concat result to one image')

# Model Options
parser.add_argument('--sr_norm_type', default='IN', help='[srresnet | RRDB_net]')
parser.add_argument('--rrdb_nb', type=int, default=23, help='For RRDB, Blocks Number for RRD-block')
parser.add_argument('--model', type=str, default='RRDB_net', help='[srresnet | RRDB_net | EDSR]')
parser.add_argument('--inc', type=int, default=3, help='the low resolution image size')
parser.add_argument('--scala', type=int, default=4, help='the low resolution image size')
parser.add_argument('--n_HG', type=int, default=6, help='the low resolution image size')
parser.add_argument('--inter_supervis', type=bool, default=True, help='the low resolution image size')

# SPN Options
parser.add_argument('--use_spn', type=bool, default=False, help='the low resolution image size')
parser.add_argument('--spn_model', type=str, default='CSPN', help='the low resolution image size')
parser.add_argument('--spn_nHG', type=int, default=6, help='the low resolution image size')

# EDSR Settings
parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
# parser.add_argument('--n_resblocks', type=int, default=32, help='number of residual blocks')
# parser.add_argument('--n_feats', type=int, default=256, help='number of feature maps')
# parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
parser.add_argument('--n_colors', type=int, default=3, help='number of feature maps')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

## Barely change
# parser.add_argument('--sr_pretrain', type=str, default='./models/HGSR-MHR_X4_340.pth')
parser.add_argument('--pretrain', type=str, default='./experiments/HGSRn6-Test1/checkpoint/HGSR-MHR_X4_96.pth')
# parser.add_argument('--pretrain', type=str, default='')
parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')
# Logger
parser.add_argument('--result_dir', type=str, default='test_IN', help='folder to sr results')
parser.add_argument('--gpus', type=int, default=1, help='folder to sr results')
opt = parser.parse_args()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

# Make Result Folder
result_root_dir = 'results/'
exp_name = opt.result_dir
opt.result_dir = os.path.join(result_root_dir, opt.result_dir)
if not os.path.exists(opt.result_dir):
    os.makedirs(opt.result_dir)

# Save Test Info
log_f = open(os.path.join(opt.result_dir, 'test_log.txt'), 'a')
log_f.write('test_dataroot: ' + opt.test_dataroot + '\n')
log_f.write('real: ' + str(opt.real) + '\n')
log_f.write('test_patch_size: ' + str(opt.psize) + '\n')
log_f.write('test_overlap: ' + str(opt.overlap) + '\n')
log_f.write('test_model: ' + str(opt.pretrain) + '\n')

# Model Type Decide
if 'srresnet' in opt.pretrain:
    opt.model = 'srresnet'
if 'RRDB' in opt.pretrain:
    opt.model = 'RRDB_net'
if 'EDSR' in opt.pretrain:
    opt.model = 'EDSR'
    opt.rgb_range = 255.
if 'RCAN' in opt.pretrain:
    opt.model = 'RCAN'
    opt.rgb_range = 255.
if 'HGSR' in opt.pretrain:
    opt.model = 'HGSR'
if 'HGSR-MHR' in opt.pretrain:
    opt.model = 'HGSR-MHR'
if 'RRDB-Fea' in opt.pretrain:
    opt.model = 'RRDB-Fea'

# Init Dataset
if opt.has_GT and (not opt.bic):
    test_dataset = RealPairDataset(opt.test_dataroot, lr_patch_size=0, scala=opt.scala,
                                   mode='Y' if opt.inc == 1 else 'RGB', train=False, need_hr_down=False,
                                   need_name=True, rgb_range=opt.rgb_range)
else:
    test_dataset = SRDataList(opt.test_dataroot, lr_patch_size=0, scala=opt.scala, mode='Y' if opt.inc == 1 else 'RGB',
                          train=False, need_name=True, rgb_range=opt.rgb_range)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers, drop_last=True)

# Init Net
print('Build Generator Net...')
if opt.model == 'srresnet':
    # generator = SRResNet(scala=opt.scala, inc=opt.inc, norm_type=opt.sr_norm_type)
    sys.path.append('./modules/SRResNet')
    import srresnet
    generator = srresnet.SRResNet()
elif opt.model == 'RRDB_net':
    generator = RRDBNet(in_nc=opt.inc, out_nc=opt.inc, nf=64,
                        nb=opt.rrdb_nb, gc=32, upscale=opt.scala, norm_type=None,
                        act_type='leakyrelu', mode='CNA', upsample_mode='upconv')
elif opt.model == 'EDSR':
    sys.path.append('./modules/EDSR')
    sys.path.append('./modules/EDSR/model')
    import edsr
    generator = edsr.make_model(opt)
elif opt.model == 'RCAN':
    sys.path.append('./modules/RCAN')
    sys.path.append('./modules/RCAN/model')
    import rcan
    generator = rcan.make_model(opt)
elif opt.model == 'HGSR':
    sys.path.append('./modules/HourGlassSR')
    import model
    generator = model.HourGlassNet(in_nc=opt.inc, out_nc=opt.inc, upscale=opt.scala,
                                   nf=64, res_type='res', n_mid=2,
                                   n_HG=opt.n_HG, inter_supervis=opt.inter_supervis)
elif opt.model == 'HGSR-MHR':
    sys.path.append('./modules/HourGlassSR')
    import model
    generator = model.HourGlassNetMultiScaleInt(in_nc=opt.inc, out_nc=opt.inc, upscale=opt.scala,
                                   nf=64, res_type='res', n_mid=2,
                                   n_HG=opt.n_HG, inter_supervis=opt.inter_supervis)
elif opt.model == 'RRDB-Fea':
    from modules.architecture import RRDBNetFeature
    generator = RRDBNetFeature(in_nc=opt.inc, out_nc=opt.inc, nf=64,
                        nb=opt.rrdb_nb, gc=32, upscale=opt.scala, norm_type=None,
                        act_type='leakyrelu', mode='CNA', upsample_mode='upconv')
else:
    generator = RRDBNet(in_nc=opt.inc, out_nc=opt.inc, nf=64,
                        nb=opt.rrdb_nb, gc=32, upscale=opt.scala, norm_type=None,
                        act_type='leakyrelu', mode='CNA', upsample_mode='upconv')

generator = load_weights(generator, opt.pretrain, opt.gpus, just_weight=False, strict=True)
generator = generator.to(device)
generator.eval()
ps_loss = perceptual_similarity.PerceptualSimilarityLoss(use_cuda=use_cuda)

if opt.use_spn:
    spn_net = SPNModel(in_ch=opt.inc, model=opt.spn_model, n_HG=opt.spn_nHG)
    spn_net = load_weights(spn_net, opt.pretrain, opt.gpus, strict=False)
    spn_net = spn_net.to(device)

cnt = 1
psnr_sum = 0
ssim_sum = 0
LPIPS_sum = 0
result = []
result_imgs = []

for batch, data in enumerate(test_loader):
    lr = data['LR']
    hr = data['HR']
    im_path = data['HR_PATH'][0]

    with torch.no_grad():
        if opt.has_GT:
            tensor = lr
            hr_size = hr
        else:
            tensor = hr
            B, C, H, W = hr.shape
            hr_size = torch.zeros(B, C, H * opt.scala, W * opt.scala)
        blocks = tensor_divide(tensor, opt.psize, opt.overlap)
        blocks = torch.cat(blocks, dim=0)
        results = []

        iters = blocks.shape[0] // opt.gpus if blocks.shape[0] % opt.gpus == 0 else blocks.shape[0] // opt.gpus + 1
        for idx in range(iters):
            if idx + 1 == iters:
                input = blocks[idx * opt.gpus:]
            else:
                input = blocks[idx * opt.gpus : (idx + 1) * opt.gpus]
            hr_var = input.to(device)
            sr_var, SR_map = generator(hr_var)

            if isinstance(sr_var, list) or isinstance(sr_var, tuple):
                sr_var = sr_var[-1]

            if opt.use_spn:
                sr_var = spn_net(sr_var)

            results.append(sr_var.to('cpu'))
            print('Processing Image: %d Part: %d / %d'
                  % (batch + 1, idx + 1, iters), end='\r')
            sys.stdout.flush()

        results = torch.cat(results, dim=0)
        sr_img = tensor_merge(results, hr_size, opt.psize * opt.scala, opt.overlap * opt.scala)

    im_name = '%s_%s_x%d.png' % (os.path.basename(im_path).split('.')[0], exp_name, opt.scala)
    if opt.save_results:
        im = to_pil_image(torch.clamp(sr_img[0].cpu() / opt.rgb_range, min=0.0, max=1.0))
        im.save(os.path.join(opt.result_dir, im_name))
        print('[%d/%d] saving to: %s' % (batch + 1, len(test_loader), os.path.join(opt.result_dir, im_name)))
    # result_imgs.append([os.path.join(opt.result_dir, im_name), im])

    if opt.has_GT:
        # psnr_single = psnr(sr_img, hr, peak=opt.rgb_range)
        psnr_single = YCbCr_psnr(sr_img, hr, scale=opt.scala, peak=opt.rgb_range)
        # psnr_single = 0
        with torch.no_grad():
            ssim_single = pytorch_ssim.ssim(sr_img.to(device) / opt.rgb_range, hr.to(device) / opt.rgb_range).item()
            # ssim_single = 0
            LPIPS_single = ps_loss((sr_img / opt.rgb_range * 2 - 1).to(device),
                                    (hr / opt.rgb_range * 2 - 1).to(device)).item()
            # LPIPS_single = 0
        print('%s PSNR: %.4f SSIM: %.4f LPIPS: %.4f' % (im_name, psnr_single, ssim_single, LPIPS_single))
        psnr_sum += psnr_single
        ssim_sum += ssim_single
        LPIPS_sum += LPIPS_single
    sys.stdout.flush()

if opt.has_GT:
    psnr_sum /= len(test_loader)
    ssim_sum /= len(test_loader)
    LPIPS_sum /= len(test_loader)
    print('-----------\nAve: PSNR: %.4f SSIM: %.4f LPIPS: %.4f\n-----------' % (psnr_sum, ssim_sum, LPIPS_sum))
    sys.stdout.flush()



