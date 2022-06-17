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
from model_mult import StyledGenerator, Discriminator



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
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    if args.ckpt is not None:
        step = args.resume_step
        
    resolution = 4 * 2 ** step

    loader = sample_data( args.batch.get(resolution,args.batch_default),args.paths, resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(2_000_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))
        
        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step
            
            loader = sample_data(args.batch.get(resolution, args.batch_default), path,resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_mult/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
        if args.resume_full:
            alpha = 1.0
        try:
            real_image,y_org,y_trg = next(data_loader)
            
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image,y_org,y_trg = next(data_loader)

        used_sample += real_image.shape[0]
        b_size = real_image.size(0)
        real_image = real_image.cuda()


        real_image.requires_grad = True
        real_scores = discriminator(real_image,y_org, step=step, alpha=alpha)
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


        gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
            2, 0
        )
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)
        pix_in1= torch.randn(b_size, code_size, device='cuda')
        pix_in2= torch.randn(b_size, code_size, device='cuda')
        pix_in3= torch.randn(b_size, code_size, device='cuda')
        
        fake_image = generator(gen_in1, pix_in1,y_trg,step=step, alpha=alpha)
        fake_predict = discriminator(fake_image,y_trg, step=step, alpha=alpha)

        fake_predict = F.softplus(fake_predict).mean()
        fake_predict.backward()
        if i%10 == 0:
            disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, pix_in2,y_trg,step=step, alpha=alpha)
            fake_image2 = generator(gen_in2,pix_in3,y_trg,step=step,alpha=alpha)
            predict = discriminator(fake_image, y_trg,step=step, alpha=alpha)
            ds_loss = torch.mean(torch.abs(fake_image-fake_image2))
            ds_loss = torch.clamp(args.ds_lambda-ds_loss, min=0.0)
        
            loss = F.softplus(-predict).mean() + ds_loss

            if i%10 == 0:
                gen_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 200 == 0:
            images = []
            images2 = []

            gen_i = 5
            gen_j = 5
            p_fix = torch.randn(1, code_size).cuda()
            p_fix =  p_fix.repeat(5,1)
            s_fix = torch.randn(1, code_size).cuda()
            s_fix = s_fix.repeat(5,1)
            
            for s in range(args.num_domains):
                y_trg = torch.ones([5],dtype=torch.int64).cuda()
                y_trg *= s
                with torch.no_grad():
                    for _ in range(gen_i):
                        images.append(
                            g_running(
                                torch.randn(gen_j, code_size).cuda(), p_fix,y_trg,step=step, alpha=alpha
                            ).data.cpu()
                        )
                        images2.append(
                            g_running(
                                s_fix,torch.randn(gen_j, code_size).cuda() ,y_trg,step=step, alpha=alpha
                            ).data.cpu()
                        )

            utils.save_image(
                torch.cat(images, 0),
                f'sampleS/{str(i + 1).zfill(6)}.jpg',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )
            utils.save_image(
                torch.cat(images2, 0),
                f'sampleP/{str(i + 1).zfill(6)}.jpg',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_mult/train_step-{str(i + 1).zfill(6)}.model',
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--gpu_num',type=int)
    parser.add_argument('--num_domains',default=2,type=int)
    
    parser.add_argument('--datapath',default = './Celeb/mult',type=str,help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=400_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )

    parser.add_argument('--resume_step' , type=int)
    parser.add_argument('--resume_full' , action='store_true')
    args = parser.parse_args()

    generator = StyledGenerator(code_size,args.num_domains)
    discriminator = Discriminator(args.num_domains,from_rgb_activate=not args.no_from_rgb_activate)
    g_running = StyledGenerator(code_size,args.num_domains)
    g_running.train(False)

    accumulate(g_running, generator, 0)
    
    domains = ['females','males']
    args.paths = [os.path.join(args.datapath,dom) for dom in domains]
    
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt,map_location=lambda storage, loc: storage)

        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    g_running = g_running.cuda()
    g_optimizer = optim.Adam(
        generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.mapping.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )

    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.001}
        args.batch = {4: 512, 8: 512, 16: 256, 32: 64, 64: 16, 128: 8, 256: 2}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 8
    
    train(args,generator, discriminator)
