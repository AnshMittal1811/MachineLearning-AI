import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.options import Options
from network.atloc import AtLoc, AtLocPlus
from torchvision import transforms, models
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from data.dataloaders import SevenScenes, RobotCar, MF
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Config
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

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
# transformer
data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.CenterCrop(opt.cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

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

pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses

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

# inference loop
for idx, (data, target) in enumerate(loader):
    if idx % 200 == 0:
        print('Image {:d} / {:d}'.format(idx, len(loader)))

    # output : 1 x 6
    data_var = Variable(data, requires_grad=False)
    data_var = data_var.to(device)

    with torch.set_grad_enabled(False):
        output = model(data_var)
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    target = target.numpy().reshape((-1, s[-1]))

    # normalize the predicted quaternions
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    # un-normalize the predicted and target translations
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m

    # take the middle prediction
    pred_poses[idx, :] = output[len(output) / 2]
    targ_poses[idx, :] = target[len(target) / 2]

# calculate losses
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
errors = np.zeros((L, 2))
print('Error in translation: median {:3.2f} m,  mean {:3.2f} m \nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'\
      .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))

fig = plt.figure()
real_pose = (pred_poses[:, :3] - pose_m) / pose_s
gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
plt.show(block=True)
image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
fig.savefig(image_filename)