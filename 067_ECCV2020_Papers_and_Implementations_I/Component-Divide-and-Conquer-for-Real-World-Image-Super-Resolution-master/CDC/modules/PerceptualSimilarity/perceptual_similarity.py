import argparse
from models import dist_model as dm
from util import util
from torch import nn
from PIL import Image
from TorchTools.DataTools.Loaders import to_tensor
import cv2

class PerceptualSimilarityLoss(nn.Module):
    def __init__(self, model='net-lin', net='alex', use_cuda=True):
        super(PerceptualSimilarityLoss, self).__init__()
        self.model = dm.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_cuda)

    def forward(self, img0, img1):
        return self.model.forward(img0, img1)

if __name__ == '__main__':
    loss = PerceptualSimilarityLoss()
    hr_down = to_tensor(Image.open('./imgs/test1/DSC_1454_x4.png')) * 2 - 1.
    lr = to_tensor(Image.open('./imgs/test1/DSC_1454_x1.png')) * 2 - 1
    print(loss(hr_down.unsqueeze(0), lr.unsqueeze(0)))
