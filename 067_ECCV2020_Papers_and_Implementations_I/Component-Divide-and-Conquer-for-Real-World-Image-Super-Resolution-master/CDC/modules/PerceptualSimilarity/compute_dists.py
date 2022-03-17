import argparse
from models import dist_model as dm
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path0', type=str, default='./imgs/test1/DSC_1454_x4.png')
parser.add_argument('--path1', type=str, default='./imgs/test1/DSC_1454_x1.png')
parser.add_argument('--use_gpu', type=bool, default=True, help='turn on flag to use GPU')
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# Load images
img0 = util.im2tensor(util.load_image(opt.path0)) # RGB image from [-1,1]
img1 = util.im2tensor(util.load_image(opt.path1))

# Compute distance
dist01 = model.forward(img0, img1)
# dist01 = model.forward(img1, img0)
print('Distance: %.3f'%dist01)
