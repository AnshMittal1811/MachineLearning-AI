import os.path as osp
import numpy as np

from data.dataloaders import RobotCar, SevenScenes
from torchvision import transforms
from torch.utils.data import DataLoader
from tools.options import Options

opt = Options().parse()

data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.RandomCrop(opt.cropsize),
    transforms.ToTensor()])

# dataset loader
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=True, real=False, transform=data_transform)
if opt.dataset == '7Scenes':
    dset = SevenScenes(**kwargs)
elif opt.dataset == 'RobotCar':
    dset = RobotCar(**kwargs)
else:
    raise NotImplementedError

# accumulate
loader = DataLoader(dset, batch_size=opt.batch_size, num_workers=opt.nThreads)
acc = np.zeros((3, opt.cropsize, opt.cropsize))
sq_acc = np.zeros((3, opt.cropsize, opt.cropsize))
for batch_idx, (imgs, _) in enumerate(loader):
    imgs = imgs.numpy()
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs ** 2, axis=0)

    if batch_idx % 50 == 0:
        print('Accumulated {:d} / {:d}'.format(batch_idx * opt.batch_size, len(dset)))

N = len(dset) * acc.shape[1] * acc.shape[2]

mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
mean_p /= N
print('Mean pixel = ', mean_p)

# std = E[x^2] - E[x]^2
std_p = np.asarray([np.sum(sq_acc[c]) for c in range(3)])
std_p /= N
std_p -= (mean_p ** 2)
print('Std. pixel = ', std_p)

output_filename = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
np.savetxt(output_filename, np.vstack((mean_p, std_p)), fmt='%8.7f')
print('{:s} written'.format(output_filename))
