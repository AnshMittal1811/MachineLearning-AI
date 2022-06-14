# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)


from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchreid.data_manager import DatasetManager
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optim


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='veri',
                    help="name of the dataset")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256,
                    help="width of an image (default: 256)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=120, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="start epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[30, 60], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--lambda-xent', type=float, default=1,
                    help="weight to balance cross entropy loss")
parser.add_argument('--lambda-htri', type=float, default=1,
                    help="weight to balance hard triplet loss")
parser.add_argument('--lambda-vcolor', type=float, default=0.125,
                    help="weight to balance vehicle color classification loss")
parser.add_argument('--lambda-vtype', type=float, default=0.125,
                    help="weight to balance vehicle type classification loss")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--keyptaware', action='store_true', 
                    help="embed keypoints to deep features")
parser.add_argument('--heatmapaware', action='store_true', 
                    help="embed heatmaps to images")
parser.add_argument('--segmentaware', action='store_true', 
                    help="embed segments to images")
parser.add_argument('--multitask', action='store_true', 
                    help="use multi-task learning")
# Miscs
parser.add_argument('--print-freq', type=int, default=1,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = DatasetManager(dataset_dir=args.dataset, root=args.root)

    transform_train = T.Compose_Keypt([
        T.Random2DTranslation_Keypt((args.width, args.height)),
        T.RandomHorizontalFlip_Keypt(),
        T.ToTensor_Keypt(),
        T.Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose_Keypt([
        T.Resize_Keypt((args.width, args.height)),
        T.ToTensor_Keypt(),
        T.Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, keyptaware=args.keyptaware, heatmapaware=args.heatmapaware, segmentaware=args.segmentaware, 
                     transform=transform_train, imagesize=(args.width, args.height)), 
        sampler=RandomIdentitySampler(dataset.train, args.train_batch, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, keyptaware=args.keyptaware, heatmapaware=args.heatmapaware, segmentaware=args.segmentaware, 
                     transform=transform_test, imagesize=(args.width, args.height)), 
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, keyptaware=args.keyptaware, heatmapaware=args.heatmapaware, segmentaware=args.segmentaware, 
                     transform=transform_test, imagesize=(args.width, args.height)), 
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_vids=dataset.num_train_vids, num_vcolors=dataset.num_train_vcolors, 
                              num_vtypes=dataset.num_train_vtypes, keyptaware = args.keyptaware, 
                              heatmapaware = args.heatmapaware, segmentaware=args.segmentaware, 
                              multitask = args.multitask)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.label_smooth:
        criterion_xent_vid = CrossEntropyLabelSmooth(num_classes=dataset.num_train_vids, use_gpu=use_gpu)
        criterion_xent_vcolor = CrossEntropyLabelSmooth(num_classes=dataset.num_train_vcolors, use_gpu=use_gpu)
        criterion_xent_vtype = CrossEntropyLabelSmooth(num_classes=dataset.num_train_vtypes, use_gpu=use_gpu)
    else:
        criterion_xent_vid = nn.CrossEntropyLoss()
        criterion_xent_vcolor = nn.CrossEntropyLoss()
        criterion_xent_vtype = nn.CrossEntropyLoss()
    criterion_htri = TripletLoss(margin=args.margin)
    
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        if check_isfile(args.load_weights):
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = test(model, args.keyptaware, args.multitask, queryloader, galleryloader, use_gpu, 
                       dataset.vcolor2label, dataset.vtype2label, return_distmat=True)
        if args.vis_ranked_res:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(args.save_dir, 'ranked_results'),
                topk=100,
            )
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, args.keyptaware, args.multitask, criterion_xent_vid, criterion_xent_vcolor, criterion_xent_vtype, criterion_htri, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()

        if ((epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0) or (epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, args.keyptaware, args.multitask, queryloader, galleryloader, use_gpu, 
                         dataset.vcolor2label, dataset.vtype2label)
            is_best = rank1 > best_rank1
            
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # rank1 = 1
            # is_best = True

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.2%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, keyptaware, multitask, criterion_xent_vid, criterion_xent_vcolor, criterion_xent_vtype, criterion_htri, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            if keyptaware and multitask:
                imgs, vids, vcolors, vtypes, vkeypts = imgs.cuda(), vids.cuda(), vcolors.cuda(), vtypes.cuda(), vkeypts.cuda()
            elif keyptaware:
                imgs, vids, vkeypts = imgs.cuda(), vids.cuda(), vkeypts.cuda()
            elif multitask:
                imgs, vids, vcolors, vtypes = imgs.cuda(), vids.cuda(), vcolors.cuda(), vtypes.cuda()
            else:
                imgs, vids = imgs.cuda(), vids.cuda()

        if keyptaware and multitask:
            output_vids, output_vcolors, output_vtypes, features = model(imgs, vkeypts)
        elif keyptaware:
            output_vids, features = model(imgs, vkeypts)
        elif multitask:
            output_vids, output_vcolors, output_vtypes, features = model(imgs)
        else:
            output_vids, features = model(imgs)

        if args.htri_only:
            if isinstance(features, tuple):
                loss = DeepSupervision(criterion_htri, features, vids)
            else:
                loss = criterion_htri(features, vids)
        else:
            if isinstance(output_vids, tuple):
                xent_loss = DeepSupervision(criterion_xent_vid, output_vids, vids)
            else:
                xent_loss = criterion_xent_vid(output_vids, vids)
            
            if isinstance(features, tuple):
                htri_loss = DeepSupervision(criterion_htri, features, vids)
            else:
                htri_loss = criterion_htri(features, vids)
            
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss

        if multitask:
            if isinstance(output_vcolors, tuple):
                xent_loss_vcolor = DeepSupervision(criterion_xent_vcolor, output_vcolors, vcolors)
            else:
                xent_loss_vcolor = criterion_xent_vcolor(output_vcolors, vcolors)

            if isinstance(output_vtypes, tuple):
                xent_loss_vtype = DeepSupervision(criterion_xent_vtype, output_vtypes, vtypes)
            else:
                xent_loss_vtype = criterion_xent_vtype(output_vtypes, vtypes)

            loss += args.lambda_vcolor * xent_loss_vcolor + args.lambda_vtype* xent_loss_vtype

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), vids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        
        end = time.time()

def test(model, keyptaware, multitask, queryloader, galleryloader, use_gpu, 
         vcolor2label, vtype2label, ranks=range(1, 51), return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf = []
        q_vids = []
        q_camids = []
        q_vcolors = []
        q_vtypes = []
        pred_q_vcolors = []
        pred_q_vtypes = []
        for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(queryloader):
            if use_gpu:
                if keyptaware:
                    imgs, vkeypts = imgs.cuda(), vkeypts.cuda()
                else:
                    imgs = imgs.cuda()

            end = time.time()

            if keyptaware and multitask:
                output_vids, output_vcolors, output_vtypes, features = model(imgs, vkeypts)
            elif keyptaware:
                output_vids, features = model(imgs, vkeypts)
            elif multitask:
                output_vids, output_vcolors, output_vtypes, features = model(imgs)
            else:
                output_vids, features = model(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_vids.extend(vids)
            q_camids.extend(camids)
            if multitask:
                q_vcolors.extend(vcolors)
                q_vtypes.extend(vtypes)
                pred_q_vcolors.extend(output_vcolors.cpu().numpy())
                pred_q_vtypes.extend(output_vtypes.cpu().numpy())
        qf = torch.cat(qf, 0)
        q_vids = np.asarray(q_vids)
        q_camids = np.asarray(q_camids)
        if multitask:
            q_vcolors = np.asarray(q_vcolors)
            q_vtypes = np.asarray(q_vtypes)
            pred_q_vcolors = np.asarray(pred_q_vcolors)
            pred_q_vtypes = np.asarray(pred_q_vtypes)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf = []
        g_vids = [] 
        g_camids = []
        g_vcolors = []
        g_vtypes = []
        pred_g_vcolors = []
        pred_g_vtypes = []
        for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(galleryloader):
            if use_gpu:
                if keyptaware:
                    imgs, vkeypts = imgs.cuda(), vkeypts.cuda()
                else:
                    imgs = imgs.cuda()

            end = time.time()

            if keyptaware and multitask:
                output_vids, output_vcolors, output_vtypes, features = model(imgs, vkeypts)
            elif keyptaware:
                output_vids, features = model(imgs, vkeypts)
            elif multitask:
                output_vids, output_vcolors, output_vtypes, features = model(imgs)
            else:
                output_vids, features = model(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_vids.extend(vids)
            g_camids.extend(camids)
            if multitask:
                g_vcolors.extend(vcolors)
                g_vtypes.extend(vtypes)
                pred_g_vcolors.extend(output_vcolors.cpu().numpy())
                pred_g_vtypes.extend(output_vtypes.cpu().numpy())
        gf = torch.cat(gf, 0)
        g_vids = np.asarray(g_vids)
        g_camids = np.asarray(g_camids)
        if multitask:
            g_vcolors = np.asarray(g_vcolors)
            g_vtypes = np.asarray(g_vtypes)
            pred_g_vcolors = np.asarray(pred_g_vcolors)
            pred_g_vtypes = np.asarray(pred_g_vtypes)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_vids, g_vids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    if multitask:
        print("Compute attribute classification accuracy")

        for q in range(q_vcolors.size):
            q_vcolors[q] = vcolor2label[q_vcolors[q]]
        for g in range(g_vcolors.size):
            g_vcolors[g] = vcolor2label[g_vcolors[g]]
        q_vcolor_errors = np.argmax(pred_q_vcolors, axis=1) - q_vcolors
        g_vcolor_errors = np.argmax(pred_g_vcolors, axis=1) - g_vcolors
        vcolor_error_num = np.count_nonzero(q_vcolor_errors) + np.count_nonzero(g_vcolor_errors)
        vcolor_accuracy = 1.0 - (float(vcolor_error_num) / float(distmat.shape[0] + distmat.shape[1]))
        print("Color classification accuracy: {:.2%}".format(vcolor_accuracy))
        
        for q in range(q_vtypes.size):
            q_vtypes[q] = vcolor2label[q_vtypes[q]]
        for g in range(g_vtypes.size):
            g_vtypes[g] = vcolor2label[g_vtypes[g]]
        q_vtype_errors = np.argmax(pred_q_vtypes, axis=1) - q_vtypes
        g_vtype_errors = np.argmax(pred_g_vtypes, axis=1) - g_vtypes
        vtype_error_num = np.count_nonzero(q_vtype_errors) + np.count_nonzero(g_vtype_errors)
        vtype_accuracy = 1.0 - (float(vtype_error_num) / float(distmat.shape[0] + distmat.shape[1]))
        print("Type classification accuracy: {:.2%}".format(vtype_accuracy))

        print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
