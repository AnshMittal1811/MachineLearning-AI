"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
# import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x, filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    p_ref = nets.style_encoderP(x_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, p_ref)
    s_src = nets.style_encoder(x_src, y_src)
    p_src = nets.style_encoderP(x_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src,p_src)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, z_trgp_list,psi, filename,filename2,filename3,fs,fp,y_org,fintp):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
#     pix_dim = z_trgp_list[0].size(1)
    
    x_concat = [x_src]
    x_concat2 = [x_src]
    x_concat3 = [x_src]
    x_porg = [x_src]
    x_sorg = [x_src]
    x_intp= [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    
    zp_many = torch.randn(10000, latent_dim).to(x_src.device)
    sp_many = nets.mapping_networkP(zp_many)
    sp_avg = torch.mean(sp_many, dim=0, keepdim=True)
    sp_avg = sp_avg.repeat(N, 1)
    p_org = nets.style_encoderP(x_src)
    s_org = nets.style_encoder(x_src,y_org)
    
#     pinit = z_trgp_list[0]
    pinit = p_org
    pfin = z_trgp_list[0]
#     pinit = nets.mapping_networkP(pinit)
    pfin = nets.mapping_networkP(pfin)
    pinit = torch.lerp(sp_avg, pinit, psi)
    pfin = torch.lerp(sp_avg,pfin,psi)
#         x_fake = nets.generator(x_src,s_org,p_init)
#         x_intp += [x_fake]
    for intp in range(10):
        pintp = torch.lerp(pinit,pfin,intp*0.1)
        x_fake = nets.generator(x_src, s_org,pintp)
        x_intp += [x_fake]
    x_fake = nets.generator(x_src,s_org,pfin)
    x_intp += [x_fake]
    
    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)
        
        
        
        
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, 0.5)
#             p_trg = nets.mapping_network(z_trg, y_trg)
#             p_trg = torch.lerp(sp_avg, sp_trg, psi)
            x_fake = nets.generator(x_src, s_trg, sp_avg)
#             x_same = nets.generator(x_src,s_trg,sp_avg,same=True)
            x_concat += [x_fake]
            x_fakep = nets.generator(x_src,s_trg,p_org)
            x_porg += [x_fakep]
            
        for z_trgp in z_trgp_list:
            p_trg = nets.mapping_networkP(z_trgp)
            p_trg = torch.lerp(sp_avg, p_trg, 0.5)
            x_fake = nets.generator(x_src, s_avg,p_trg)
            x_fakes = nets.generator(x_src,s_org,p_trg)
            x_concat2 += [x_fake]
            x_sorg += [x_fakes]
        
        
        
        for s in range(len(z_trg_list)):
            s_trg = nets.mapping_network(z_trg_list[s], y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            
            p_trg = nets.mapping_networkP(z_trgp_list[s])
            p_trg = torch.lerp(sp_avg, p_trg, psi)
            
            x_fake = nets.generator(x_src,s_trg,p_trg)
            
            x_concat3 += [x_fake]
            
    x_concat = torch.cat(x_concat, dim=0)
    x_concat2 = torch.cat(x_concat2,dim=0)
    x_concat3 = torch.cat(x_concat3,dim=0)
    x_porg = torch.cat(x_porg,dim=0)
    x_sorg = torch.cat(x_sorg,dim=0)
    
    x_intp = torch.cat(x_intp,dim=0)
    save_image(x_concat, N, filename)
    save_image(x_concat2, N, filename2)
    save_image(x_concat3, N, filename3)
    save_image(x_porg, N, fs)
    save_image(x_sorg, N, fp)
    save_image(x_intp, N, fintp)
def translate_latent(nets, args, x_src, y_trg_list, z_trg_list, z_trgp_list, filename,filename2):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    x_concat2 = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
#         z_many = torch.randn(10000, latent_dim).to(x_src.device)
#         y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
#         s_many = nets.mapping_network(z_many, y_many)
#         s_avg = torch.mean(s_many, dim=0, keepdim=True)
#         s_avg = s_avg.repeat(N, 1)
        
#         zp_many = torch.randn(10000, latent_dim).to(x_src.device)
#         sp_many = nets.mapping_networkP(zp_many, y_many)
#         sp_avg = torch.mean(sp_many, dim=0, keepdim=True)
#         sp_avg = sp_avg.repeat(N, 1)
        p_trg = nets.mapping_networkP(z_trgp_list[0])
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
#             s_trg = 
#             p_trg = nets.mapping_network(z_trg, y_trg)
#             p_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg,p_trg)
            x_concat += [x_fake]
        s_trg = nets.mapping_network(z_trg_list[0],y_trg)
        for z_trgp in z_trgp_list:
            p_trg = nets.mapping_networkP(z_trgp)
#             s_trg = 
#             p_trg = nets.mapping_network(z_trg, y_trg)
#             p_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg,p_trg)
            x_concat2 += [x_fake]
        
    x_concat = torch.cat(x_concat, dim=0)
    x_concat2 = torch.cat(x_concat2, dim=0)
    save_image(x_concat, N, filename)
    save_image(x_concat2, N, filename2)

@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, y_org,filename,filename2):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_src,y_org)
    p_org = nets.style_encoderP(x_src)
    s_refo = nets.style_encoder(x_ref, y_ref)
#     p_trg = nets.mapping_networkP(z_trgp)
    p_refo = nets.style_encoderP(x_ref)
    s_ref_list = s_refo.unsqueeze(1).repeat(1, N, 1)
    p_ref_list = p_refo.unsqueeze(1).repeat(1,N,1)
    x_concat = [x_src_with_wb]
    x_concat2 = [x_src_with_wb]
    
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, p_org)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
    for i, p_ref in enumerate(p_ref_list):
        x_fake = nets.generator(x_src, s_org, p_ref)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat2 += [x_fake_with_ref]
        
        
    x_concat = torch.cat(x_concat, dim=0)
    x_concat2 = torch.cat(x_concat2, dim=0)
    save_image(x_concat, N+1, filename)
    save_image(x_concat2, N+1, filename2)
    del x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src,x_ref,y_ref = inputs

    device = x_src.device
    N = x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)
    z_trg = torch.randn(N,args.latent_dim).to(device)
    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    z_trgp_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
#     for psi in [0.5, 0.7, 1.0]:
    filename = ospj(args.sample_dir, '%06d_latents.jpg' % (step))
    filename2 = ospj(args.sample_dir, '%06d_latentp.jpg' % (step))
    translate_latent(nets, args, x_src, y_trg_list, z_trg_list, z_trgp_list, filename,filename2)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_references.jpg' % (step))
    filename2 = ospj(args.sample_dir, '%06d_referencep.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, y_src,filename,filename2)

@torch.no_grad()
def debug_image2(nets, args, inputs,nums):
    
    
    # translate and reconstruct (reference-guided)
#     filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
#     translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)
    x_src, x_ref,x_ref2,y_org,y_trg = inputs
    x_src = x_src.cuda()
    x_ref = x_ref.cuda()
    filename = ospj(args.sample_dir, '%06d_references.jpg' % (0))
    filename2 = ospj(args.sample_dir, '%06d_referencep.jpg' % (0))
    translate_using_reference(nets, args, x_src, x_ref, y_trg, y_org,filename,filename2)
    
    for i in range(nums):
        
        
        device = x_src.device
        N = x_src.size(0)
#         z_trg = torch.randn(N,args.latent_dim).to(device)
        # latent-guided image synthesis
        y_trg_list = [torch.tensor(y).repeat(N).to(device)
                      for y in range(min(args.num_domains, 5))]
        z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
        z_trgp_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
#     for psi in [0.5, 0.7, 1.0]:
    
        filename = ospj(args.result_dir, '%06d_latents.jpg' % (i))
        filename2 = ospj(args.result_dir, '%06d_latentp.jpg' % (i))
        filename3 = ospj(args.result_dir, '%06d_latent_comp.jpg' % (i))
        
        file_recon = ospj(args.result_dir, '%06d_reconstruct.jpg' % (i))
        fs = ospj(args.result_dir, '%06d_changingS.jpg' % (i))
        fp = ospj(args.result_dir, '%06d_changingP.jpg' % (i))
        fintp = ospj(args.result_dir, '%06d_intp.jpg' % (i))
        psi = 0.7
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, z_trgp_list, psi,filename,filename2,filename3,fs,fp,y_org,fintp)
        translate_and_reconstruct(nets,args,x_src,y_org,x_ref,y_trg,file_recon)
        
        frefs = ospj(args.result_dir, '%06d_refs.jpg' % (i))
        frefp = ospj(args.result_dir, '%06d_refp.jpg' % (i))
        
        translate_using_reference(nets, args, x_src, x_ref, y_trg, y_org,frefs,frefp)
        
    # reference-guided image synthesis
#     filename = ospj(args.sample_dir, '%06d_references.jpg' % (step))
#     filename2 = ospj(args.sample_dir, '%06d_referencep.jpg' % (step))
#     translate_using_reference(nets, args, x_src, x_ref, y_ref, z_trg,filename,filename2)
# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


# @torch.no_grad()
# def video_ref(nets, args, x_src, x_ref, y_ref, fname):
#     video = []
#     s_ref = nets.style_encoder(x_ref, y_ref)
#     s_prev = None
#     for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
#         x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
#         if s_prev is None:
#             x_prev, y_prev, s_prev = x_next, y_next, s_next
#             continue
#         if y_prev != y_next:
#             x_prev, y_prev, s_prev = x_next, y_next, s_next
#             continue

#         interpolated = interpolate(nets, args, x_src, s_prev, s_next)
#         entries = [x_prev, x_next]
#         slided = slide(entries)  # (T, C, 256*2, 256)
#         frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
#         video.append(frames)
#         x_prev, y_prev, s_prev = x_next, y_next, s_next

#     # append last frame 10 time
#     for _ in range(10):
#         video.append(frames[-1:])
#     video = tensor2ndarray255(torch.cat(video))
#     save_video(fname, video)


# @torch.no_grad()
# def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
#     latent_dim = z_list[0].size(1)
#     s_list = []
#     for i, y_trg in enumerate(y_list):
#         z_many = torch.randn(10000, latent_dim).to(x_src.device)
#         y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
#         s_many = nets.mapping_network(z_many, y_many)
#         s_avg = torch.mean(s_many, dim=0, keepdim=True)
#         s_avg = s_avg.repeat(x_src.size(0), 1)

#         for z_trg in z_list:
#             s_trg = nets.mapping_network(z_trg, y_trg)
#             s_trg = torch.lerp(s_avg, s_trg, psi)
#             s_list.append(s_trg)

#     s_prev = None
#     video = []
#     # fetch reference images
#     for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
#         if s_prev is None:
#             s_prev = s_next
#             continue
#         if idx_ref % len(z_list) == 0:
#             s_prev = s_next
#             continue
#         frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
#         video.append(frames)
#         s_prev = s_next
#     for _ in range(10):
#         video.append(frames[-1:])
#     video = tensor2ndarray255(torch.cat(video))
#     save_video(fname, video)


# def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
#     assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
#     num_frames, height, width, channels = images.shape
#     stream = ffmpeg.input('pipe:', format='rawvideo', 
#                           pix_fmt='rgb24', s='{}x{}'.format(width, height))
#     stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
#     stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
#     stream = ffmpeg.overwrite_output(stream)
#     process = ffmpeg.run_async(stream, pipe_stdin=True)
#     for frame in tqdm(images, desc='writing video to %s' % fname):
#         process.stdin.write(frame.astype(np.uint8).tobytes())
#     process.stdin.close()
#     process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255