import os.path as osp
import numpy as np

from PIL import Image
from data.dataloaders import RobotCar
from torch.utils.data import DataLoader
from torchvision import transforms
from tools.options import Options

opt = Options().parse()

if opt.val:
    print('processing VAL data using {:d} cores'.format(opt.nThreads))
else:
    print('processing TRAIN data using {:d} cores'.format(opt.nThreads))

# create data loader
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(opt.cropsize),
                                transforms.Lambda(lambda x: np.asarray(x))])
dset = RobotCar(scene=opt.scene, data_path=opt.data_dir, train=not opt.val, transform=transform, undistort=True)
loader = DataLoader(dset, batch_size=opt.batchsize, num_workers=opt.nThreads)

# gather information about output filenames
base_dir = osp.join(opt.data_dir, opt.dataset, opt.scene)
if opt.val:
    split_filename = osp.join(base_dir, 'test_split.txt')
else:
    split_filename = osp.join(base_dir, 'train_split.txt')
with open(split_filename, 'r') as f:
    seqs = [l.rstrip() for l in f if not l.startswith('#')]

im_filenames = []
for seq in seqs:
    seq_dir = osp.join(base_dir, seq)
    ts_filename = osp.join(seq_dir, 'stereo.timestamps')
    with open(ts_filename, 'r') as f:
        ts = [l.rstrip().split(' ')[0] for l in f]
    im_filenames.extend([osp.join(seq_dir, 'stereo', 'centre_processed', '{:s}.png'.
                                  format(t)) for t in ts])
assert len(dset) == len(im_filenames)

# loop
for batch_idx, (imgs, _) in enumerate(loader):
    for idx, im in enumerate(imgs):
        im_filename = im_filenames[batch_idx * opt.batchsize + idx]
        im = Image.fromarray(im.numpy())
        try:
            im.save(im_filename)
        except IOError:
            print('IOError while saving {:s}'.format(im_filename))

    if batch_idx % 50 == 0:
        print('Processed {:d} / {:d}'.format(batch_idx * opt.batchsize, len(dset)))
