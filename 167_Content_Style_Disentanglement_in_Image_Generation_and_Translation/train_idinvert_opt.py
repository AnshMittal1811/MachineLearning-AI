import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from dataset import MultiLabelResolutionDataset
from metrics.lpips import LPIPS

from model_mult import StyledGenerator, Discriminator, StyleEncoder,ContentEncoder

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=torch.float32).cuda()
        
    return zeros.scatter(scatter_dim, y_tensor, 1)
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(batch_size, path,image_size=4):
    dataset = MultiLabelResolutionDataset(path,resolution=image_size,is_val=True)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args,  generator,encoderS,encoderP):
    dim = args.code_size // args.groups
    step = args.step
    resolution = 4 * 2 ** step

    is_train =False
    loader = sample_data( args.batch_default, args.paths, resolution
    )
    data_loader = iter(loader)
    
    pbar = tqdm(range(3_000_000))

    requires_grad(generator, False)
    requires_grad(encoderS, False)
    requires_grad(encoderP, False)
    
    
    path_pix = args.pixpath
    path_sty = args.stypath
    
    lpips_loss_val=0

    alpha = 1.0
    step = args.step

    calc_lpips = LPIPS().cuda()
    requires_grad(calc_lpips, False)

    for i in pbar:
        try:
            real_image,y_org,y_trg = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image,y_org,y_trg= next(data_loader)
        
        b_size = real_image.size(0)

        real_image = real_image.cuda()
        y_org = y_org.cuda()
        y_trg = y_trg.cuda()

        z_rnd = torch.randn(real_image.size(0),args.code_size).cuda()
        zp_rnd = torch.randn(real_image.size(0),args.code_size).cuda()

        init_sty = encoderS(real_image,y_org,step=step,alpha=alpha)
        init_pix = encoderP(real_image,step=step,alpha=alpha)

        sty = init_sty
        pix = init_pix
        sty.requires_grad = True
        pix.requires_grad=True
        optimizer = optim.Adam([sty,pix], lr=args.baselr)
        for ii in range(100):
            optimizer.zero_grad()

            fake_image = generator.generator(sty,pix,step=step,alpha=alpha)
            sty_rec = encoderS(fake_image,y_org,step=step,alpha=alpha)
            pix_rec = encoderP(fake_image,step=step,alpha=alpha)
                  
            l2_loss = torch.mean((sty-sty_rec)**2)
            l2_lossp = torch.mean((pix-pix_rec)**2)

            rec_loss = torch.mean((fake_image-real_image)**2)
            lpips_loss = calc_lpips(fake_image,real_image)
            loss =  2*l2_loss + 2*l2_lossp +rec_loss + lpips_loss

            loss.backward()
            if ii%10 == 0:
                l2_loss_val = l2_loss.item()
                l2_lossp_val = l2_lossp.item()
                rec_loss_val = rec_loss.item()
                lpips_loss_val = lpips_loss.item()
            
            optimizer.step()
            
        for c in range(len(sty)):
            spath  = os.path.join(path_sty[y_org[c]],str(real_image.size(0)*i + c + 1).zfill(6)+'.pt')
            torch.save(sty[c],spath)
            ppath  = os.path.join(path_pix[y_org[c]],str(real_image.size(0)*i + c + 1).zfill(6)+'.pt')
            torch.save(pix[c],ppath)
        

        images = []

        images.append(real_image.data.cpu())
        images.append(fake_image.data.cpu())

        utils.save_image(
            torch.cat(images, 0),
            f'sample/sampleREC/{str(i + 1).zfill(6)}'+'IDINVERT.jpg',
            nrow=real_image.size(0),
            normalize=True,
            range=(-1, 1))        


if __name__ == '__main__':
    code_size = 512
    n_critic = 1
    
    parser = argparse.ArgumentParser(description='IDInvert code optimization')

    parser.add_argument('--code_size',type=int,default=512)
    parser.add_argument('--datapath',default = './data/Celeb/val',type=str,help='path of specified dataset')
    parser.add_argument('--dir_sty',default = './codes/sty',type=str,help='directory to save style code')
    parser.add_argument('--dir_pix',default = './codes/pix',type=str,help='directory to save content code')
    parser.add_argument('--num_domains', type=int, default=2)
    parser.add_argument('--baselr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--step', default=6, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument('--batch_default', default=8, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--enc_ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    
    args = parser.parse_args()
    
    domains = ['females','males']
    args.paths = [os.path.join(args.datapath,dom) for dom in domains]
    args.stypath = [os.path.join(args.dir_sty,dom) for dom in domains]
    args.pixpath = [os.path.join(args.dir_pix,dom) for dom in domains]
    
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    encoderS = StyleEncoder(num_domains=args.num_domains)
    encoderP = ContentEncoder()
    g_running = StyledGenerator(code_dim=args.code_size,num_domains=args.num_domains)
    g_running.train(False)

    
    diff_model = False
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt,map_location=lambda storage, loc: storage)
        g_running.load_state_dict(ckpt['g_running'])

    if args.enc_ckpt is not None:
        enc_ckpt = torch.load(args.enc_ckpt,map_location=lambda storage, loc: storage)
        encoderS.load_state_dict(enc_ckpt['encoderS'])
        encoderP.load_state_dict(enc_ckpt['encoderP'])

    encoderS = encoderS.cuda()
    encoderP = encoderP.cuda()
    g_running = g_running.cuda()
    

    train(args,g_running,encoderS,encoderP)
