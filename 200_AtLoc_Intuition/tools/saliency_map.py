import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys
import cv2
from tools.options import Options

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network.atloc import AtLoc, AtLocPlus
from data.dataloaders import SevenScenes, RobotCar, MF
from tools.utils import load_state_dict
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, models

# config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"

# Model
feature_extractor = models.resnet34(pretrained=False)
atloc = AtLoc(feature_extractor, droprate=opt.test_dropout, pretrained=False, lstm=opt.lstm)
if opt.model == 'AtLoc':
    model = atloc
elif opt.model == 'AtLocPlus':
    model = AtLocPlus(atlocplus=atloc)
else:
    raise NotImplementedError
model.eval()

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
# transformer
data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.CenterCrop(opt.cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, transform=data_transform, target_transform=target_transform, seed=opt.seed)
if opt.model == 'AtLoc':
    if opt.dataset == '7Scenes':
        data_set = SevenScenes(**kwargs)
    elif opt.dataset == 'RobotCar':
        data_set = RobotCar(**kwargs)
    else:
        raise NotImplementedError
elif opt.model == 'AtLocPlus':
    kwargs = dict(kwargs, dataset=opt.dataset, skip=opt.skip, steps=opt.steps, variable_skip=opt.variable_skip)
    data_set = MF(real=opt.real, **kwargs)
else:
    raise NotImplementedError
L = len(data_set)
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

# load weights
model.to(device)
weights_filename = osp.expanduser(opt.weights)
if osp.isfile(weights_filename):
    checkpoint = torch.load(weights_filename, map_location=device)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)

# opencv init
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_filename = osp.join(opt.results_dir, '{:s}_{:s}_attention_{:s}.avi'.format(opt.dataset, opt.scene, opt.model))
# get frame size
img, _ = data_set[0]
vwrite = cv2.VideoWriter(out_filename, fourcc=fourcc, fps=20.0,
                         frameSize=(img.size(2), img.size(1)))
print('Initialized VideoWriter to {:s} with frames size {:d} x {:d}'.format(out_filename, img.size(2), img.size(1)))

# inference
cm_jet = plt.cm.get_cmap('jet')
for batch_idx, (data, target) in enumerate(loader):
    data = data.to(device)
    data_var = Variable(data, requires_grad=True)

    model.zero_grad()
    pose = model(data_var)
    pose.mean().backward()

    act = data_var.grad.data.cpu().numpy()
    act = act.squeeze().transpose((1, 2, 0))
    img = data[0].cpu().numpy()
    img = img.transpose((1, 2, 0))

    act *= img
    act = np.amax(np.abs(act), axis=2)
    act -= act.min()
    act /= act.max()
    act = cm_jet(act)[:, :, :3]
    act *= 255

    img *= stats[1]
    img += stats[0]
    img *= 255
    img = img[:, :, ::-1]

    img = 0.5 * img + 0.5 * act
    img = np.clip(img, 0, 255)

    vwrite.write(img.astype(np.uint8))

    if batch_idx % 200 == 0:
        print('{:d} / {:d}'.format(batch_idx, len(loader)))

vwrite.release()
print('{:s} written'.format(out_filename))
