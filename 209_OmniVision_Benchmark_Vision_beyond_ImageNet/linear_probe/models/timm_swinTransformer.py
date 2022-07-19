import timm

import torch
import copy
import torch.nn as nn
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

import torchvision
class MyswimTransformer(nn.Module):
    def __init__(self, num_classes=1000, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing swinTransformer model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('swin_tiny_patch4_window7_224',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)

    def forward_features(self, x):
        x = self.model.patch_embed(x)
        if self.model.absolute_pos_embed is not None:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        x = self.model.norm(x)  # B L C
        x = self.model.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.model.forward_features(x)
        # x = self.head(x)
        return x


def timm_swinTransformer(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return MyswimTransformer(**default_kwargs)

def test_build():
    model = MyswimTransformer(pretrain_path='/mnt/lustre/zhangyuanhan/architech/swin_tiny_patch4_window7_224.pth')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)
    # print(model.model.head)
    print(output.shape) #1024

if __name__ == '__main__':
    test_build()