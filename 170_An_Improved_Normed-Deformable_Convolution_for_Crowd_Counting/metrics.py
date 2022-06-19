import torch
import torch.nn as nn
import sys
from functools import reduce

class JointLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(JointLoss, self).__init__()
        self.MSELoss = nn.MSELoss(size_average=False)
        self.BCELoss = nn.BCELoss(size_average=True)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, gt_map, target_map):
        mse = self.MSELoss(x, gt_map) * self.alpha
        bce = self.BCELoss(x, target_map) * self.beta
#         sys.stdout.write("mse loss = {}, bce loss = {}\r".format(mse, bce))
        sys.stdout.flush()
        return  mse + bce
    
class MSEScalarLoss(nn.Module):
    def __init__(self):
        super(MSEScalarLoss, self).__init__()
    
    def forward(self, x, gt_map):
        return torch.pow(x.sum() - gt_map.sum(), 2) / (reduce(lambda a,b:a * b, x.shape))
        
class AEBatch(nn.Module):
    def __init__(self):
        super(AEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_num):
        return torch.abs(torch.sum(estimated_density_map, dim=(1, 2, 3)) - gt_num)


class SEBatch(nn.Module):
    def __init__(self):
        super(SEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_num):
        return torch.pow(torch.sum(estimated_density_map, dim=(1, 2, 3)) - gt_num, 2)
