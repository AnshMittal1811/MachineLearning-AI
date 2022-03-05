# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import LoadDatasetFromFolder_CD, calMetric_iou
import numpy as np
import random
from model.CDNet import CDNet
from configures import parser
from loss.BCL import BCL
import pandas as pd
import itertools

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
    train_set = LoadDatasetFromFolder_CD(args, args.hr1_train, args.hr2_train, args.lab_train)
    val_set = LoadDatasetFromFolder_CD(args, args.hr1_val, args.hr2_val, args.lab_val)
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)

    # define model
    CDNet = CDNet(args).to(device, dtype=torch.float)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))

    # set optimization
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr= args.lr, betas=(0.9, 0.999))
    CDcriterionCD = BCL().to(device, dtype=torch.float)

    results = {'train_loss': [], 'train_CD':[], 'train_SR':[],'val_IoU': []}

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'SR_loss':0, 'CD_loss':0, 'loss': 0 }

        CDNet.train()
        for hr_img1, hr_img2, label in train_bar:
            running_results['batch_sizes'] += args.batchsize

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()

            dist = CDNet(hr_img1, hr_img2)
            CD_loss = CDcriterionCD(dist, label)

            CDNet.zero_grad()
            CD_loss.backward()
            optimizer.step()

            # loss for current batch before optimization
            running_results['CD_loss'] += CD_loss.item() * args.batchsize

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, args.num_epochs,
                    running_results['CD_loss'] / running_results['batch_sizes']))

        # eval
        CDNet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            valing_results = {'loss':0,'SR_loss': 0, 'CD_loss':0, 'batch_sizes': 0, 'IoU': 0}

            for hr_img1, hr_img2, label in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                hr_img1 = hr_img1.to(device, dtype=torch.float)
                hr_img2 = hr_img2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()

                dist = CDNet(hr_img1, hr_img2)

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

                # loss for current batch before optimization
                valing_results['IoU'] = (inter * 1.0 / unin)

                val_bar.set_description(
                    desc='IoU: %.4f' % ( valing_results['IoU'],
                    ))

        # save model parameters
        val_loss = valing_results['IoU']

        if val_loss > mloss or epoch==1:
            mloss = val_loss
            torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_%d.pth' % (epoch ))

        results['train_SR'].append(running_results['SR_loss'] / running_results['batch_sizes'])
        results['train_CD'].append(running_results['CD_loss'] / running_results['batch_sizes'])
        results['train_loss'].append(running_results['loss'] / running_results['batch_sizes'])
        results['val_IoU'].append(valing_results['IoU'])

        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'train_loss': results['train_loss'], 'train_CD': results['train_CD'],
                      'val_IoU': results['val_IoU']},
                index=range(1, epoch + 1))
            data_frame.to_csv(args.sta_dir, index_label='Epoch')
