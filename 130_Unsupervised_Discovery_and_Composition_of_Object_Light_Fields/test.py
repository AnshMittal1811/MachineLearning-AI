import configargparse
import os, time, datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import piq, lpips, sklearn.metrics

import matplotlib.pyplot as plt

import models
import summaries

import dataio
from torch.utils.data import DataLoader
import util

import loss_functions

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Training options
p.add_argument('--data_root', required=True, help='Path to directory with test data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='path to directory where results will be saved.')

p.add_argument('--img_sidelength',type=int, default=64, help='image sidelength to train with.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--max_num_instances', type=int, default=None,
               help='If \'data_root\' has more instances, only the first max_num_instances are used.')
p.add_argument('--gtseg_path',      type=str, default=None, help='Path to gt segmentation masks.')

# Model options
p.add_argument('--phi_latent',      type=int, default=128, help='Dimensionality of the regressed object latent codes.')
p.add_argument('--phi_out_latent',  type=int, default=64,  help='Dimensionality of the features emitted by the phi networks.')
p.add_argument('--hyper_hidden',    type=int, default=1,   help='Number of layers of the hypernetwork.')
p.add_argument('--phi_hidden',      type=int, default=2,   help='Number of layers of the phi hyponetwork.')
p.add_argument('--zero_bg',         type=bool,default=False, help='Whether to zero-out the regressed background phi code.')
p.add_argument('--num_phi',         type=int, default=2, help='Number of objects to regress per scene.')

opt = p.parse_args()

def test():

    dataset = dataio.SceneClassDataset(root_dir=opt.data_root,
                                       max_num_instances=opt.max_num_instances,
                                       num_context=1,
                                       num_trgt=4,
                                       img_sidelength=opt.img_sidelength,)
    dataset.test=True

    model = models.COLF(phi_latent=opt.phi_latent, phi_out_latent=opt.phi_out_latent,
                hyper_hidden=opt.hyper_hidden,phi_hidden=opt.phi_hidden,
                num_phi=opt.num_phi).cuda().eval()

    if opt.checkpoint_path is not None:
        print("Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path)

    models.zero_bg = opt.zero_bg

    util.cond_mkdir(opt.logging_root)

    # Save command-line parameters log directory.
    with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.logging_root, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    iter = 0

    print('Beginning evaluation...')

    torch.set_grad_enabled(False)

    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            drop_last=True, num_workers=4,)

    run_seg = opt.gtseg_path is not None

    lpips_loss_all,ssim_loss_all,psnr_loss_all=0,0,0
    ari_loss=0 
    num_pred=0
    num_pred_ari=0
    max_i=-1
    for iter,(model_input, gt) in enumerate(dataloader):

        print(iter,"/",len(dataloader))
        if iter==max_i:break

        model_input,gt = [util.dict_to_gpu(x) for x in (model_input,gt)]

        model_out = model(model_input)

        pred_rgb,gt_rgb=[src["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(128,128))
                    for src in (model_out,gt)]

        for rgb,name in zip(pred_rgb,gt["imgname"]):
            plt.imsave(os.path.join(opt.logging_root,name[0].split("/")[-1]),rgb.permute(1,2,0).cpu().numpy()*.5+.5)

        if run_seg:
            model_segs = model_out["seg"].squeeze().max(0)[1]
            gtsegs,segnames=[],[]
            for i,imgname in enumerate(gt["imgname"]):
                name=gt["imgname"][0][0].split("/")[-1][:5]+imgname[0].split("/")[-1][5:-5]+"0_gt_mask%d.png"%i
                segnames.append(name)
                gtsegs.append(torch.from_numpy(plt.imread(os.path.join(opt.gtseg_path,name))))
            gt_segs = torch.stack(gtsegs,0)

            for model_seg,gt_seg_rgb in zip(model_segs,gt_segs):

                bg = (((gt_seg_rgb.flatten(0,1).unique(dim=1)/2+.5)*1000).round().int()==251).all(1)

                gt_seg=torch.zeros_like(model_seg)
                for unique_i,unique_rgb in enumerate(gt_seg_rgb.flatten(0,1).unique(dim=0)):
                    gt_seg[(gt_seg_rgb.flatten(0,1)==unique_rgb).all(1)]=unique_i

                plt.imsave(os.path.join(opt.logging_root,name.replace("_gt","")),model_seg.view(128,128).cpu())

                ari_loss += sklearn.metrics.cluster.adjusted_rand_score(model_seg.cpu(),gt_seg.cpu())
                num_pred_ari+=1

        ssim_loss=piq.ssim(pred_rgb*.5+.5,gt_rgb*.5+.5)
        psnr_loss=piq.psnr(pred_rgb*.5+.5,gt_rgb*.5+.5,1)
        lpips_loss= loss_fn_alex(pred_rgb,gt_rgb).mean()
        ssim_loss_all+=ssim_loss
        lpips_loss_all+=lpips_loss
        psnr_loss_all+=psnr_loss
        num_pred+=1
        
        if iter%10==0:
            if run_seg:
                print("num ari pred",num_pred_ari)
                print("num pred",num_pred)
                print("ari %02f"%(ari_loss/num_pred_ari))
            print("ssim",ssim_loss_all/num_pred)
            print("lpips",lpips_loss_all/num_pred)
            print("psnr",psnr_loss_all/num_pred)

if __name__ == '__main__':
    test()
