from thop import profile
import torch


def count(model):
    img = torch.rand(1, 3, 256, 256).float().cuda()
    inputs = {'img': img}
    flops, params = profile(model, inputs=(inputs, {}, 'test'))
    print(flops)
    print(params)


from src.model.modules import *
class Testmodel(nn.Module):
    def __init__(self):
        super(Testmodel, self).__init__()
        feature_size = 16
        self.layer1 = ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4,
                                            stride=2)
        self.layer2 = ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4,
                                            stride=1)
        self.layer3 = ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4,
                                            stride=2)
        self.layer4 = ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4,
                                            stride=1)
        self.layer5 = ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1)
        self.layer6 = ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1)
        self.layer7 = ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1,
                                            activation=nn.Sequential())

    def forward(self, a,b):
        a = self.layer1(a)
        a = self.layer2(a)
        a = self.layer3(a)
        a = self.layer4(a)
        a = self.layer5(a)
        a = self.layer6(a)
        a = self.layer7(a)
        return a


tm = Testmodel()
img = torch.rand(1, 64, 64, 64).float()

flops, params = profile(tm, inputs=(img,None))
# 1706767392.0
# 62706


# from src.model.SADRNv2 import SADRNv2
# model=SADRNv2().cuda()
# count(model)
# 10323913856.0
# 15487044.0