import argparse

#training options
parser = argparse.ArgumentParser(description='Train SRCDNet')

# training parameters
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=32, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=24, type=int, help='num_workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg'], type=list, help='the suffix of the image files.')
parser.add_argument('--img_size', default=256, type=int, help='batchsize for validation')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for CDNet')
parser.add_argument('--w_cd', type=float, default=0.001, help='factor to balance the weight of CD loss in Generator loss')
parser.add_argument('--scale', default=8, type=int, help='resolution difference between images. [ 2| 4| 8]')

# path for loading data from folder
parser.add_argument('--hr1_train', default='../data/CDD/train/time1', type=str, help='hr image at t1 in training set')
parser.add_argument('--lr2_train', default='../data/CDD/train/time2_lr/X4.00', type=str, help='lr image at t2 in training set')
parser.add_argument('--hr2_train', default='../data/CDD/train/time2', type=str, help='hr image at t2 in training set')
parser.add_argument('--lab_train', default='../data/CDD/train/label', type=str, help='label image in training set')

parser.add_argument('--hr1_val', default='../data/CDD/val/time1', type=str, help='hr image at t1 in validation set')
parser.add_argument('--lr2_val', default='../data/CDD/val/time2_lr/X4.00', type=str, help='lr image at t2 in validation set')
parser.add_argument('--hr2_val', default='../data/CDD/val/time2', type=str, help='hr image at t2 in validation set')
parser.add_argument('--lab_val', default='../data/CDD/val/label', type=str, help='label image in validation set')

# network saving and loading parameters
parser.add_argument('--model_dir', default='epochs/X4.00/CD/', type=str, help='save path for CD model ')
parser.add_argument('--sr_dir', default='epochs/X4.00/SR/', type=str, help='save path for Generator')
parser.add_argument('--sta_dir', default='statistics/CDD_4x.csv', type=str, help='statistics save path')
