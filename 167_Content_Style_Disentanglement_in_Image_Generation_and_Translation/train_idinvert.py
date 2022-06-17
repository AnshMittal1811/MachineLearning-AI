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
    dataset = MultiLabelResolutionDataset(path,resolution=image_size)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args,  generator,discriminator,encoderS,encoderP):

    step = args.step
    resolution = 4 * 2 ** step

    is_train =True
    loader = sample_data( args.batch_size, args.datapaths, resolution
    )
    data_loader = iter(loader)
    
    pbar = tqdm(range(200_000))

    requires_grad(generator, False)
    requires_grad(encoderS, False)
    requires_grad(encoderP, False)
    requires_grad(discriminator, False)
    
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
        
#***************************************************************************
#********************************* Discriminator Update ********************
#*************************************************************************** 
        
        requires_grad(encoderS, False)
        requires_grad(encoderP, False)
        requires_grad(discriminator, True)
        
        discriminator.zero_grad()

        real_image.requires_grad = True

        real_scores = discriminator(real_image,y_org,step=step,alpha=alpha)
        real_predict = F.softplus(-real_scores).mean()
        real_predict.backward(retain_graph=True)
        grad_real = grad(
            outputs=real_scores.sum(), inputs=real_image, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty
        grad_penalty.backward()
        if i%10 == 0:
            grad_loss_val = grad_penalty.item()
                
        sty_enc = encoderS(real_image,y_org,step=step,alpha=alpha)
        pix_enc = encoderP(real_image,step=step,alpha=alpha)

        fake_image = generator.generator(sty_enc,pix_enc,step=step,alpha=alpha)
        fake_predict = discriminator(fake_image,y_org,step=step,alpha=alpha)
        
        fake_predict = F.softplus(fake_predict).mean()
        fake_predict.backward()
        if i%10 == 0:
            disc_loss_val = (fake_predict+real_predict).item()
            
        d_optimizer.step()
        
#***************************************************************************
#**************** Style Encoder / Content Encoder Update********************
#*************************************************************************** 
        requires_grad(encoderS, True)
        requires_grad(encoderP, True)
        requires_grad(discriminator, False)
        
        encoderP.zero_grad()
        encoderS.zero_grad()
        
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
        
        sty_rnd, pix_rnd = generator.mapping(z_rnd,zp_rnd,y_org)
        
        fake_image = generator.generator(sty_rnd,pix_rnd,step=step,alpha=alpha)
        
        sty_enc = encoderS(fake_image,y_org,step=step,alpha=alpha)
        pix_enc = encoderP(fake_image,step=step,alpha=alpha)
        
        sty_real = encoderS(real_image,y_org,step=step,alpha=alpha)
        pix_real = encoderP(real_image,step=step,alpha=alpha)
        
        fake_enc = generator.generator(sty_real,pix_real,step=step,alpha=alpha)

        l2_loss = torch.mean((sty_rnd-sty_enc)**2)
        l2_lossp = torch.mean((pix_rnd-pix_enc)**2)

        rec_loss = torch.mean((fake_enc-real_image)**2)
        lpips_loss = calc_lpips(fake_enc,real_image)
        predict = F.softplus(-discriminator(fake_enc,y_org,step=step,alpha=alpha)).mean()

        loss =  rec_loss + lpips_loss + l2_loss + l2_lossp + 0.1*predict

        loss.backward()
        if i%10 == 0:
            l2_loss_val = l2_loss.item()
            l2_lossp_val = l2_lossp.item()
            rec_loss_val = rec_loss.item()
            lpips_loss_val = lpips_loss.item()
            adv_loss_val = predict.item()

        es_optimizer.step()
        ep_optimizer.step()

        if (i + 1) % 200 == 0:
            images = []
            images2 = []
            recs = []
            gen_j = 4

            if gen_j > real_image.size(0):
                gen_j = real_image.size(0)

            with torch.no_grad():
                rnd = torch.randn(gen_j,512).cuda()
                rnd2 = torch.randn(gen_j,512).cuda()

                sty_r,styp_r = generator.mapping(rnd,rnd2,y_trg)
                sty_enc = encoderS(real_image,y_org,step=step,alpha=alpha)
                pix_enc = encoderP(real_image,step=step,alpha=alpha)
                img = g_running.generator(sty_enc,pix_enc,step=step,alpha=alpha).data.cpu()
                imgP = g_running.generator(sty_enc,styp_r,step=step,alpha=alpha).data.cpu()
                imgS = g_running.generator(sty_r,pix_enc,step=step,alpha=alpha).data.cpu()
                
                images.append(real_image.data.cpu())
                images.append(img)
                images.append(imgP)
                images.append(imgS)

            utils.save_image(
                torch.cat(images, 0),
                f'sampleENC/{str(i + 1).zfill(6)}'+'.jpg',
                nrow=gen_j,
                normalize=True,
                range=(-1, 1),)


        if (i + 1) % 10000 == 0:
            torch.save(
                {
                    
                    'encoderS': encoderS.state_dict(),
                    'encoderP': encoderP.state_dict(),
                    'discriminator': discriminator.state_dict()
                },
                f'checkpoint/train_idinvert/{str(i + 1).zfill(6)}.model',
            ) 
            

        state_msg = (
            f'Size: {4 * 2 ** step};disc: {disc_loss_val:.3f};rec: {rec_loss_val:.3f};lpips: {lpips_loss_val:.3f};l2: {l2_loss_val:.3f};l2p: {l2_lossp_val:.3f};adv: {adv_loss_val:.3f};'
            
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1
    
    parser = argparse.ArgumentParser(description='IDInvert on Multidomain Diagonal GAN')
    parser.add_argument('--num_domains', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--code_size',type=int,default=512)
    parser.add_argument('--datapath',default = './data/Celeb/mult',type=str,help='path of specified dataset')

    parser.add_argument('--baselr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--step', default=6, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default='./checkpoint/train_mult/CelebAHQ_mult.model', type=str, help='load backbone model from previous checkpoints'
    )
    parser.add_argument(
        '--enc_ckpt', default=None, type=str, help='load encoder models from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )

    parser.add_argument('--resume_step' , type=int)
    parser.add_argument('--resume_full',action='store_true')
    args = parser.parse_args()


    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    encoderS = StyleEncoder(num_domains=args.num_domains)
    encoderP = ContentEncoder()

    g_running = StyledGenerator(code_dim=512,num_domains=args.num_domains)
    g_running.train(False)
    
    domains = ['females','males']
    args.datapaths = [os.path.join(args.datapath,dom) for dom in domains]
    
    es_optimizer = optim.Adam(
        encoderS.parameters(), lr=args.baselr, betas=(0.0, 0.99)
    )
    ep_optimizer = optim.Adam(
        encoderP.parameters(), lr=args.baselr, betas=(0.0, 0.99)
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),lr=args.baselr,betas=(0.0,0.99))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt,map_location=lambda storage, loc: storage)
        g_running.load_state_dict(ckpt['g_running'])
    
    if args.enc_ckpt is not None:
        enc_ckpt = torch.load(args.enc_ckpt,map_location=lambda storage, loc: storage)
        encoderS.load_state_dict(enc_ckpt['encoderS'])
        encoderP.load_state_dict(enc_ckpt['encoderP'])
        discriminator.load_state_dict(enc_ckpt['discriminator'])

    
    encoderS = encoderS.cuda()
    discriminator = discriminator.cuda()
    encoderP = encoderP.cuda()

    g_running = g_running.cuda()


    train(args,g_running,discriminator,encoderS,encoderP)
