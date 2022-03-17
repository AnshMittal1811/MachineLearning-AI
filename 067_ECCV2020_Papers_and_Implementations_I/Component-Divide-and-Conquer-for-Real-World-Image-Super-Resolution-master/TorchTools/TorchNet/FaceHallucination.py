import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

from .modules import Features4Layer, Features3Layer, residualBlock, upsampleBlock, LateUpsamplingBlock, LateUpsamplingBlockNoBN
from .activation import swish


class MaxActivationFusion(nn.Module):
    """
    model implementation of the Maximum-activation Detail Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, feature_extractor=Features4Layer, activation=relu):
        """
        :param features: the number of final feature maps
        """
        super(MaxActivationFusion, self).__init__()
        self.features = feature_extractor(features, activation=activation)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        frame_1_feature = self.features(frame_1)
        frame_2_feature = self.features(frame_2)
        frame_3_feature = self.features(frame_3)
        frame_4_feature = self.features(frame_4)
        frame_5_feature = self.features(frame_5)

        frame_1_feature = frame_1_feature.view((1, ) + frame_1_feature.size())
        frame_2_feature = frame_2_feature.view((1, ) + frame_2_feature.size())
        frame_3_feature = frame_3_feature.view((1, ) + frame_3_feature.size())
        frame_4_feature = frame_4_feature.view((1, ) + frame_4_feature.size())
        frame_5_feature = frame_5_feature.view((1, ) + frame_5_feature.size())

        cat = torch.cat((frame_1_feature, frame_2_feature, frame_3_feature, frame_4_feature, frame_5_feature), dim=0)
        return torch.max(cat, 0)[0]


class MeanActivationFusion(nn.Module):
    """
    model implementation of the Mean-activation Detail Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, feature_extractor=Features4Layer, activation=relu):
        """
        :param features: the number of final feature maps
        """
        super(MeanActivationFusion, self).__init__()
        self.features = feature_extractor(features, activation=activation)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        frame_1_feature = self.features(frame_1)
        frame_2_feature = self.features(frame_2)
        frame_3_feature = self.features(frame_3)
        frame_4_feature = self.features(frame_4)
        frame_5_feature = self.features(frame_5)

        frame_1_feature = frame_1_feature.view((1, ) + frame_1_feature.size())
        frame_2_feature = frame_2_feature.view((1, ) + frame_2_feature.size())
        frame_3_feature = frame_3_feature.view((1, ) + frame_3_feature.size())
        frame_4_feature = frame_4_feature.view((1, ) + frame_4_feature.size())
        frame_5_feature = frame_5_feature.view((1, ) + frame_5_feature.size())

        cat = torch.cat((frame_1_feature, frame_2_feature, frame_3_feature, frame_4_feature, frame_5_feature), dim=0)
        return torch.mean(cat, 0)


class EarlyMean(nn.Module):
    """
    model implementation of the Early Mean Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, feature_extractor=Features4Layer, activation=relu):
        """
        :param features: the number of final feature maps
        """
        super(EarlyMean, self).__init__()
        self.features = feature_extractor(features, activation=activation)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        frame_1 = frame_1.view((1, ) + frame_1.size())
        frame_2 = frame_2.view((1, ) + frame_2.size())
        frame_3 = frame_3.view((1, ) + frame_3.size())
        frame_4 = frame_4.view((1, ) + frame_4.size())
        frame_5 = frame_5.view((1, ) + frame_5.size())
        frames_mean = torch.mean(torch.cat((frame_1, frame_2, frame_3, frame_4, frame_5), dim=0), 0)
        return self.features(frames_mean)


class EaryFusion(nn.Module):
    """
    model implementation of the Early Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, feature_extractor=Features3Layer, activation=relu):
        """
        :param features: the number of final feature maps
        """
        super(EaryFusion, self).__init__()
        self.act = activation
        self.features = feature_extractor(features, activation=activation)
        self.conv = nn.Conv2d(features * 5, features, 3, stride=1, padding=1)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        return self.act(self.conv(
            torch.cat(
                (self.features(frame_1),
                 self.features(frame_2),
                 self.features(frame_3),
                 self.features(frame_4),
                 self.features(frame_5)),
                dim=1)
        ))


class EarlyEarly(nn.Module):
    """
    model implementation of the Early Fusion
    This is not a complete SR model, just **Fusion Part**
    """
    def __init__(self, features=64, activation=relu):
        """
        :param features: the number of final feature maps
        """
        super(EarlyEarly, self).__init__()
        self.act = activation
        self.conv = nn.Conv2d(5, features, 3, stride=1, padding=1)
        self.c1 = nn.Conv2d(features, features, 3, padding=1)
        self.c2 = nn.Conv2d(features, features, 3, padding=1)
        self.c3 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        """
        :param frame_1: frame t-2
        :param frame_2: frame t-1
        :param frame_3: frame t
        :param frame_4: frame t+1
        :param frame_5: frame t+2
        :return: features
        """
        return self.act(self.c3(self.act(self.c2(self.act(self.c1(self.act(self.conv(torch.cat(
                (frame_1,
                 frame_2,
                 frame_3,
                 frame_4,
                 frame_5),
                dim=1)))))))))


class _3DConv(nn.Module):
    def __init__(self,features=64, activation=relu):
        super(_3DConv, self).__init__()
        self.act = activation
        self.n_features = features
        self.conv3d = nn.Conv3d(1, features, (5, 3, 3), padding=(0, 1, 1))
        self.c1 = nn.Conv2d(features, features, 3, padding=1)
        self.c2 = nn.Conv2d(features, features, 3, padding=1)
        self.c3 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        batch, C, H, W = frame_1.size()
        frame_1 = frame_1.view((batch, C, 1, H, W))
        frame_2 = frame_2.view((batch, C, 1, H, W))
        frame_3 = frame_3.view((batch, C, 1, H, W))
        frame_4 = frame_4.view((batch, C, 1, H, W))
        frame_5 = frame_5.view((batch, C, 1, H, W))
        frame_block = torch.cat(
            (frame_1,
             frame_2,
             frame_3,
             frame_4,
             frame_5), dim=2)
        _3dfeatures = self.act(self.conv3d(frame_block))
        _3dfeatures = _3dfeatures.view((batch, self.n_features, H, W))
        return self.act(self.c3(
            self.act(self.c2(
                self.act(self.c1(_3dfeatures))
            ))
        ))


class HallucinationOrigin(nn.Module):
    """
    Original Video Face Hallucination Net
    |---------------------------------|
    |         Input features          |
    |---------------------------------|
    |   n   |    Residual blocks      |
    |---------------------------------|
    | Big short connect from features |
    |---------------------------------|
    |       Convolution and BN        |
    |---------------------------------|
    |    Pixel Shuffle Up-sampling    |
    |---------------------------------|
    |         Final Convolution       |
    |---------------------------------|
    |              Tanh               |
    |---------------------------------|
    """
    def __init__(self, scala=8, features=64, n_residual_blocks=9, big_short_connect=False, output_channel=1):
        """
        :param scala: scala factor
        :param n_residual_blocks: The number of residual blocks
        :param Big_short_connect: Weather the short connect between the input features and the Conv&BN
        """
        super(HallucinationOrigin, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.scala = scala
        self.connect = big_short_connect

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock(features))

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(features)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(features, features * 4))

        self.pad2 = nn.ReflectionPad2d(3)
        self.conv2 = nn.Conv2d(features, output_channel, 7, stride=1, padding=0)

    def forward(self, features):
        y = features.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        if self.connect:
            x = self.bn(self.conv(self.pad(y))) + features
        else:
            x = self.bn(self.conv(self.pad(y)))

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return  F.tanh(self.conv2(self.pad2(x)))


class StepHallucinationNet(nn.Module):
    """
    |-----------------------------------|
    |             features              |
    |-----------------------------------|
    | log2(scala) | LateUpsamplingBlock |
    |-----------------------------------|
    |       Convolution and Tanh        |
    |-----------------------------------|
    """
    def __init__(self, scala=8, features=64, little_res_blocks=3, output_channel=1):
        """
        :param scala: scala factor
        :param features:
        :param little_res_blocks: The number of residual blocks in every late upsample blocks
        :param output_channel: default to be 1 for Y channel
        """
        super(StepHallucinationNet, self).__init__()
        self.scala = scala
        self.features = features
        self.n_res_blocks = little_res_blocks

        for i in range(int(log2(self.scala))):
            self.add_module('lateUpsampling' + str(i + 1), LateUpsamplingBlock(features, n_res_block=little_res_blocks))

        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(features, output_channel, 7, stride=1, padding=0)

    def forward(self, features):
        for i in range(int(log2(self.scala))):
            features = self.__getattr__('lateUpsampling' + str(i + 1))(features)
        return F.tanh(self.conv(self.pad(features)))


class StepHallucinationNoBN(nn.Module):
    """
    |-----------------------------------|
    |             features              |
    |-----------------------------------|
    | log2(scala) | LateUpsamplingBlock |
    |-----------------------------------|
    |       Convolution and Tanh        |
    |-----------------------------------|
    """
    def __init__(self, scala=8, features=64, little_res_blocks=3, output_channel=1):
        """
        :param scala: scala factor
        :param features:
        :param little_res_blocks: The number of residual blocks in every late upsample blocks
        :param output_channel: default to be 1 for Y channel
        """
        super(StepHallucinationNoBN, self).__init__()
        self.scala = scala
        self.features = features
        self.n_res_blocks = little_res_blocks

        for i in range(int(log2(self.scala))):
            self.add_module('lateUpsampling' + str(i + 1), LateUpsamplingBlockNoBN(features, n_res_block=little_res_blocks))

        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(features, output_channel, 7, stride=1, padding=0)

    def forward(self, features):
        for i in range(int(log2(self.scala))):
            features = self.__getattr__('lateUpsampling' + str(i + 1))(features)
        return F.tanh(self.conv(self.pad(features)))


class FusionModel(nn.Module):
    """
    The Multi-frame face hallucination model
    """
    def __init__(self, scala=8, fusion='mdf', upsample='org'):
        """
        :param fusion: 'mdf'=MaxActivationFusion,
                       'early'=EaryFusion,
                       'mef'=MeanActivationFusion,
                       'earlyMean'=EarlyMean,
                       'ee'=EarlyEarly,
                       '3d'=_3DConv
        :param upsample: 'org'=HallucinationOrigin,
                         'step'=StepHallucinationNet,
                         'no'=StepHallucinationNoBN
        """
        super(FusionModel, self).__init__()
        if fusion == 'mdf':
            self.features = MaxActivationFusion()
        elif fusion == 'early':
            self.features = EaryFusion()
        elif fusion == 'mef':
            self.features = MeanActivationFusion()
        elif fusion == 'earlyMean':
            self.features = EarlyMean()
        elif fusion == 'ee':
            self.features = EarlyEarly()
        elif fusion == '3d':
            self.features = _3DConv()
        else:
            raise Exception('Wrong Parameter: fusion')

        if upsample == 'org':
            self.upsample = HallucinationOrigin(scala=scala)
        elif upsample == 'step':
            self.upsample = StepHallucinationNet(scala=scala)
        elif upsample == 'no':
            self.upsample = StepHallucinationNoBN(scala=scala)
        else:
            raise Exception('Wrong Parameter: upsample')

    def forward(self, frame_1, frame_2, frame_3, frame_4, frame_5):
        return self.upsample(self.features(frame_1, frame_2, frame_3, frame_4, frame_5))


class SingleImageBaseline(nn.Module):
    """
    Single Image Baseline
    """
    def __init__(self, scala=8, feature='f3', upsample='org'):
        """
        :param feature: 'f3'=Features3Layer
                        'f4'=Features4Layer
        :param upsample: 'org'=HallucinationOrigin,
                         'step'=StepHallucinationNet,
                         'no'=StepHallucinationNoBN
        """
        super(SingleImageBaseline, self).__init__()
        if feature == 'f3':
            self.features = Features3Layer()
        elif feature == 'f4':
            self.features = Features4Layer()
        else:
            raise Exception('Wrong Parameter: feature')

        if upsample == 'org':
            self.upsample = HallucinationOrigin(scala=scala)
        elif upsample == 'step':
            self.upsample = StepHallucinationNet(scala=scala)
        elif upsample == 'no':
            self.upsample = StepHallucinationNoBN(scala=scala)
        else:
            raise Exception('Wrong Parameter: upsample')

    def forward(self, frame):
        return self.upsample(self.features(frame))


def multi_to_single_step_upsample(Multi_state_dict):
    for key in dict(Multi_state_dict).keys():
        if key.startswith('features.features.'):
            Multi_state_dict[key[9:]] = Multi_state_dict.pop(key)
    return Multi_state_dict