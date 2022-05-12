import configargparse
import os, time, datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import models
import summaries

import dataio
from torch.utils.data import DataLoader
import util

import loss_functions

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Training options
p.add_argument('--data_root', required=True, help='Path to directory with training data.')
p.add_argument('--val_root', required=False, help='Path to directory with validation data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='path to directory where checkpoints & tensorboard events will be saved.')

p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')

p.add_argument('--lpips_weight', type=float, default=0., help='lpips loss weight. default=0')
p.add_argument('--rgb_weight',   type=float, default=2.5*1e2, help='l2 rgb loss weight. default=250')
p.add_argument('--l1_rgb_weight',type=float, default=0., help='l1 rgb loss weight. default=0')
p.add_argument('--latent_weight',type=float, default=0., help='latent penalty weight. default=0')

p.add_argument('--img_sidelength',type=int, default=64, help='image sidelength to train with.')
p.add_argument('--num_query',type=int, default=1, help='Number of query images per scene batch.')
p.add_argument('--batch_size',type=int, default=1, help='Batch size.')

p.add_argument('--steps_til_ckpt', type=int, default=10000,
               help='Number of iterations until checkpoint is saved.')
p.add_argument('--steps_til_val', type=int, default=1000,
               help='Number of iterations until validation set is run.')
p.add_argument('--no_validation', action='store_true', default=False,
               help='If no validation set should be used.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--max_num_instances_train', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances_train are used.')
p.add_argument('--max_num_instances_val', type=int, default=5,
               help='If \'val_root\' has more instances, only the first max_num_instances_val are used.')

# Model options
p.add_argument('--phi_latent',      type=int, default=128, help='Dimensionality of the regressed object latent codes.')
p.add_argument('--phi_out_latent',  type=int, default=64,  help='Dimensionality of the features emitted by the phi networks.')
p.add_argument('--hyper_hidden',    type=int, default=1,   help='Number of layers of the hypernetwork.')
p.add_argument('--phi_hidden',      type=int, default=2,   help='Number of layers of the phi hyponetwork.')
p.add_argument('--zero_bg',         type=bool,default=False, help='Whether to zero-out the regressed background phi code.')
p.add_argument('--num_phi',         type=int, default=2, help='Number of objects to regress per scene.')

opt = p.parse_args()

def train():

    train_dataset = dataio.SceneClassDataset(root_dir=opt.data_root,
                                             max_num_instances=opt.max_num_instances_train,
                                             num_context=1,
                                             num_trgt=opt.num_query,
                                             img_sidelength=opt.img_sidelength,)

    if not opt.no_validation:
        assert (opt.val_root is not None), "No validation directory passed."

        val_dataset = dataio.SceneClassDataset(root_dir=opt.val_root,
                                               max_num_instances=opt.max_num_instances_val,
                                               num_context=1,
                                               num_trgt=opt.num_query,
                                               img_sidelength=opt.img_sidelength,)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=True)
    train_dataloader = DataLoader(train_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      drop_last=True,)

    model = models.COLF(phi_latent=opt.phi_latent, phi_out_latent=opt.phi_out_latent,
                hyper_hidden=opt.hyper_hidden,phi_hidden=opt.phi_hidden,
                num_phi=opt.num_phi,zero_bg=opt.zero_bg).cuda()

    if opt.checkpoint_path is not None:
        print("Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path)
    models.zero_bg = opt.zero_bg

    ckpt_dir = os.path.join(opt.logging_root, 'checkpoints')
    events_dir = os.path.join(opt.logging_root, 'events')

    util.cond_mkdir(opt.logging_root)
    util.cond_mkdir(ckpt_dir)
    util.cond_mkdir(events_dir)

    # Save command-line parameters log directory.
    with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.logging_root, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    writer = SummaryWriter(events_dir)
    iter = 0
    epoch = iter // len(train_dataset)
    step = 0

    print('Beginning training...')

    # Loops over epochs.
    while True:
        for model_input, ground_truth in train_dataloader:
            model_input,ground_truth = [util.dict_to_gpu(x) for x in (model_input,ground_truth)]

            model_outputs = model(model_input)

            optimizer.zero_grad()
    
            latent_loss = loss_functions.latent_penalty(model_outputs,ground_truth) * opt.latent_weight
            rgb_loss    = loss_functions.rgb(model_outputs,ground_truth) * opt.rgb_weight
            lpips_loss  = loss_functions.lpips_loss(model_outputs,ground_truth) * opt.lpips_weight
            l1_rgb      = loss_functions.l1_rgb(model_outputs,ground_truth) * opt.l1_rgb_weight

            total_loss = (latent_loss + rgb_loss + l1_rgb + lpips_loss)

            total_loss.backward()

            optimizer.step()

            print("Iter %07d   Epoch %03d   L_img %0.4f" % (iter, epoch, rgb_loss))

            if iter % 20 == 0:
                summaries.rgb(model_outputs,model_input,writer,iter)
                summaries.slot_attn_vid(model_outputs,model_input,writer,iter)
                summaries.seg_vid(model_outputs,model_input,writer,iter)

            writer.add_scalar("latent loss", latent_loss, iter)
            writer.add_scalar("rgb loss", rgb_loss, iter)
            writer.add_scalar("l1 rgb loss", l1_rgb, iter)
            writer.add_scalar("lpips loss", lpips_loss, iter)

            if iter % opt.steps_til_val == 0 and not opt.no_validation:
                print("Running validation set...")

                model.eval()
                with torch.no_grad():

                    rgb_loss    = loss_functions.rgb(model_outputs,ground_truth) * opt.rgb_weight
                    lpips_loss  = loss_functions.lpips_loss(model_outputs,ground_truth) * opt.lpips_weight

                    rgb = []
                    lpips = []
                    for i,(model_input, ground_truth) in enumerate(val_dataloader):
                        print(i,"/",len(val_dataloader))
                        model_input,ground_truth = [util.dict_to_gpu(x) for x in (model_input,ground_truth)]
                        model_outputs = model(model_input)
                        rgb.append(loss_functions.rgb(model_outputs,ground_truth) * opt.rgb_weight)
                        lpips.append(loss_functions.lpips_loss(model_outputs,ground_truth) * opt.lpips_weight)

                        if i%10==0:
                            summaries.rgb(model_outputs,model_input,writer,iter)
                            summaries.slot_attn_vid(model_outputs,model_input,writer,iter)
                            summaries.seg_vid(model_outputs,model_input,writer,iter)

                    writer.add_scalar("val_rgb_loss", torch.tensor(rgb).mean(), iter)
                    writer.add_scalar("val_lpips", torch.tensor(lpips).mean(), iter)
                model.train()

            iter += 1
            step += 1

            if iter % opt.steps_til_ckpt == 0:
                util.custom_save(model, os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
                                 discriminator=None, optimizer=optimizer)

        epoch += 1

    util.custom_save(model,
                     os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
                     discriminator=None, optimizer=optimizer)

if __name__ == '__main__':
    train()
