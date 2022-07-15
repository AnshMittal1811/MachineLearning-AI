import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', coco='', input_extra=0, input_height=512):
        super(Resnet, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=True)
        del self.encoder.fc, self.encoder.avgpool
        if coco:
            coco_pretrain = getattr(models.segmentation, coco)(pretrained=True).backbone
            self.encoder.load_state_dict(coco_pretrain.state_dict())
        self.out_channels = [256, 512, 1024, 2048]
        self.feat_heights = [input_height//4//(2**i) for i in range(4)]
        if int(backbone[6:]) < 50:
            self.out_channels = [_//4 for _ in self.out_channels]

        # Patch for extra input channel
        if input_extra > 0:
            ori_conv1 = self.encoder.conv1
            new_conv1 = nn.Conv2d(
                3+input_extra, ori_conv1.out_channels,
                kernel_size=ori_conv1.kernel_size,
                stride=ori_conv1.stride,
                padding=ori_conv1.padding,
                bias=ori_conv1.bias)
            with torch.no_grad():
                for i in range(0, 3+input_extra, 3):
                    n = new_conv1.weight[:, i:i+3].shape[1]
                    new_conv1.weight[:, i:i+n] = ori_conv1.weight[:, :n]
            self.encoder.conv1 = new_conv1

        # Prepare for pre/pose down height filtering
        self.pre_down = None
        self.post_down = None

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        if self.pre_down is not None:
            x = self.pre_down(x)
        x = self.encoder.layer1(x);
        if self.post_down is not None:
            x = self.post_down(x)
        features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features
