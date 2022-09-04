#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import argparse
import os
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from PGNet import PGNet
from apex import amp
import torch.distributed as dist
from utils.lr_scheduler import LR_Scheduler

def flat(mask):
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask,size=(int(h),int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
    # print(x.shape)  b 28*28 1
    g = x @ x.transpose(-2,-1) # b 28*28 28*28
    g = g.unsqueeze(1) # b 1 28*28 28*28
    return g

def att_loss(pred,mask,p4,p5):
    g = flat(mask)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4)
    p5 = flat(np5)
    w1  = torch.abs(g-p4)
    w2  = torch.abs(g-p5)
    w = (w1+w2)*0.5+1
    attbce=F.binary_cross_entropy_with_logits(pred, g,weight =w*1.0,reduction='mean')
    return attbce
    
def bce_iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask,size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int) 
    parser.add_argument('--batchsize', default=-1, type=int)
    parser.add_argument('--savepath', default="../model/baseline", type=str)  
    parser.add_argument('--datapath', default="../data/DUTS-TR", type=str) 
    parser.parse_args()
    return parser.parse_args()

def train(Dataset, Network):
    # dataset
    args = parser()
    print(torch.cuda.device_count())
    ############################################################
    torch.distributed.init_process_group(backend="nccl")
    print('world_size', torch.distributed.get_world_size())
    torch.cuda.set_device(args.local_rank)
    cfg = Dataset.Config(datapath=args.datapath, savepath=args.savepath,mode='train', batch=args.batchsize, lr=0.03, momen=0.9,
                         decay=5e-4, epoch=32, snapshot=args.checkpoint)
    data = Dataset.Data(cfg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=args.batchsize,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=data.collate,
                                         sampler=train_sampler)


    net = Network(cfg)
    net.train(True)

    base, head = [], []
    for name, param in net.named_parameters():
        if 'swin' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    scheduler = LR_Scheduler('cos',cfg.lr,cfg.epoch,len(loader),warmup_epochs=cfg.epoch//2)
    net = net.cuda(args.local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank],output_device=args.local_rank, 
                                                     find_unused_parameters=True, 
                                                     broadcast_buffers=False)
    global_step = 0

    
    for epoch in range(cfg.epoch):
        train_sampler.set_epoch(epoch)
        net.train()
        for step, (image, mask) in enumerate(loader):
            image, mask = image.float().cuda(), mask.float().cuda()   
            image, mask = image.float().cuda(), mask.float().cuda()   
            optimizer.zero_grad()      
            scheduler(optimizer,step,epoch)
            p1,wr,ws,attmap= net(image)
            
            att_loss_ = att_loss(attmap,mask,wr,ws) # attention guided loss
            loss1 = bce_iou_loss(p1,mask) # loss_b+i
            loss2 = bce_iou_loss(wr,mask)*0.125+bce_iou_loss(ws,mask)*0.125 # loss_aux
            loss = loss1+loss2+att_loss_
            loss = reduce_mean(loss, dist.get_world_size())

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()      
            global_step += 1
            if step % 60 == 0 and args.local_rank==0:
                print('%s | step:%d/%d/%d | lr=%.6f  loss=%.6f attloss=%.6f' % (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[1]['lr'],loss.item(),att_loss_.item()))
        if epoch >= 27 and args.local_rank==0:
                if not os.path.exists(cfg.savepath):
                    os.makedirs(cfg.savepath)
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
    dist.barrier()


if __name__ == '__main__':
    train(dataset, PGNet)
