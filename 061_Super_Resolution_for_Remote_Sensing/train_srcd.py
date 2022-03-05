# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from math import log10
from configures import parser
from loss.BCL import BCL
from loss.Gloss import GeneratorLoss
from data_utils import LoadDatasetFromFolder, calMetric_iou
from model.CDNet import CDNet
from model.SRNet import Generator, Discriminator

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seeds
def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2021)

if __name__ == '__main__':
    mloss = 0

    # load data
    train_set = LoadDatasetFromFolder(args, args.hr1_train, args.lr2_train, args.hr2_train, args.lab_train)
    val_set = LoadDatasetFromFolder(args, args.hr1_val, args.lr2_val, args.hr2_val, args.lab_val)
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)

    # define model
    CDNet = CDNet(args).to(device, dtype=torch.float)
    netG = Generator(args.scale).to(device, dtype=torch.float)
    netD = Discriminator().to(device, dtype=torch.float)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))
        netG = torch.nn.DataParallel(netG, device_ids=range(torch.cuda.device_count()))
        netD = torch.nn.DataParallel(netD, device_ids=range(torch.cuda.device_count()))

    # set optimization
    optimizer = optim.Adam(CDNet.parameters(), lr= args.lr, betas=(0.9, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(0.9, 0.999))

    criterionCD = BCL().to(device, dtype=torch.float)
    criterionG = GeneratorLoss(args.w_cd).to(device, dtype=torch.float)


    results = {'train_loss':[], 'train_CD':[], 'train_SR':[],'val_IoU':[]}

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'SR_loss':0, 'd_score':0,'g_score':0,'CD_loss':0, 'g_loss': 0 ,'d_loss': 0 }

        CDNet.train()
        netG.train()
        netD.train()
        for hr_img1, lr_img2, hr_img2, label in train_bar:
            running_results['batch_sizes'] += args.batchsize

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            lr_img2 = lr_img2.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()

            fake_img = netG(lr_img2)
            dist = CDNet((hr_img1/0.5-1), (fake_img/0.5-1)) # (img/0.5-1) aims to normalized the value to [-1, 1]

            ############################
            # Update CD network
            ###########################
            CD_loss = criterionCD(dist, label)
            CDNet.zero_grad()
            CD_loss.backward(retain_graph=True)
            optimizer.step()

            ############################
            # Update D network
            ###########################
            netD.zero_grad()
            real_out = netD(hr_img2).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # Update G network
            ###########################
            netG.zero_grad()
            g_loss = criterionG(fake_out, fake_img, hr_img2, CD_loss)
            g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization
            running_results['CD_loss'] += CD_loss.item() * args.batchsize
            running_results['g_loss'] += g_loss.item() * args.batchsize
            running_results['d_loss'] += d_loss.item() * args.batchsize

            train_bar.set_description(desc='[%d/%d] D: %.3f G: %.3f  CD_loss: %.3f' % (
                epoch, args.num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['CD_loss'] / running_results['batch_sizes'],))

        # eval
        CDNet.eval()
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            valing_results = {'loss':0,'SR_loss': 0, 'CD_loss':0, 'batch_sizes': 0, 'IoU': 0, 'mse':0, 'psnr':0}

            for hr_img1, lr_img2, hr_img2, label in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                hr_img1 = hr_img1.to(device, dtype=torch.float)
                lr_img2 = lr_img2.to(device, dtype=torch.float)
                hr_img2 = hr_img2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()

                fake_img = netG(lr_img2)
                dist = CDNet((hr_img1/0.5-1), (fake_img/0.5-1))

                # calculate IoU
                gt_value = (label > 0).float()
                prob = (dist > 1).float()
                prob = prob.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                intr, unn = calMetric_iou(gt_value, result)
                inter = inter + intr
                unin = unin + unn

                batch_mse = ((fake_img - hr_img2) ** 2).data.mean()
                valing_results['mse'] += batch_mse * args.val_batchsize
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))

                valing_results['IoU'] = (inter * 1.0 / unin)
                val_bar.set_description(
                    desc='IoU: %.4f   PSNR: %.4f' % (valing_results['IoU'],valing_results['psnr']))

        # save model parameters
        val_loss = valing_results['IoU']

        if val_loss > mloss or epoch==1:
            mloss = val_loss
            torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_%d.pth' % epoch)
            torch.save(netG.state_dict(), args.sr_dir + 'netG_epoch_%d.pth' % epoch)

        results['train_SR'].append(running_results['SR_loss'] / running_results['batch_sizes'])
        results['train_CD'].append(running_results['CD_loss'] / running_results['batch_sizes'])
        results['val_IoU'].append(valing_results['IoU'])

        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'train_CD': results['train_CD'],
                      'val_IoU': results['val_IoU']},
                index=range(1, epoch + 1))
            data_frame.to_csv(args.sta_dir, index_label='Epoch')
