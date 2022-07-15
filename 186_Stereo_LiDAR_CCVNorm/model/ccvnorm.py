"""
Network components: CBN and CCVNorm.
"""
import numpy as np
import torch
import torch.nn as nn


class CategoricalHierConditionalCostVolumeNorm(torch.nn.Module):
    """
    Categorical HierCCVNorm.
    """
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True, cbn_in_channels=1):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.cbn_in_channels = cbn_in_channels # NOTE: unused
        if self.affine:
            self.catcbn_weight_weight = torch.nn.Parameter(torch.ones(1, num_features, num_cats)) # NOTE +1 for background class (invalid point, represented by -1)
            self.catcbn_weight_bias = torch.nn.Parameter(torch.zeros(1, num_features, num_cats))
            self.catcbn_bias_weight = torch.nn.Parameter(torch.ones(1, num_features, num_cats)) # NOTE +1 for background class (invalid point, represented by -1)
            self.catcbn_bias_bias = torch.nn.Parameter(torch.zeros(1, num_features, num_cats))
            self.catcbn_weight = torch.nn.Parameter(torch.ones(1, num_features, num_cats))
            self.catcbn_bias = torch.nn.Parameter(torch.zeros(1, num_features, num_cats))
            self.invalid_weight = torch.nn.Parameter(torch.ones(1, num_features, num_cats, 1)) # NOTE 1 for background class
            self.invalid_bias = torch.nn.Parameter(torch.zeros(1, num_features, num_cats, 1))
        else:
            self.register_parameter('catcbn_weight_weight', None)
            self.register_parameter('catcbn_weight_bias', None)
            self.register_parameter('catcbn_bias_weight', None)
            self.register_parameter('catcbn_bias_bias', None)
            self.register_parameter('catcbn_weight', None)
            self.register_parameter('catcbn_bias', None)
            self.register_parameter('invalid_weight', None)
            self.register_parameter('invalid_bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.invalid_weight.data.fill_(1.0)
            self.invalid_bias.data.zero_()
            self.catcbn_weight_weight.data.fill_(1.0)
            self.catcbn_weight_bias.data.zero_()
            self.catcbn_bias_weight.data.fill_(1.0)
            self.catcbn_bias_bias.data.zero_()
            self.catcbn_weight.data.fill_(1.0)
            self.catcbn_bias.data.zero_()

    def forward(self, input, cats, feats=None):
        if feats is None:
            feats = cats.clone()
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        b, c, d, h, w = out.shape
        if self.affine:
            invalid_mask = (cats[0,0] == -1)[None,None,None,...].float()
            # Second CBN (CatCBN)
            catcbn_weight = (self.catcbn_weight[:,:,:,None,None]*cats.unsqueeze(1)).sum(2, keepdim=True)
            catcbn_bias = (self.catcbn_bias[:,:,:,None,None]*cats.unsqueeze(1)).sum(2, keepdim=True)
            # Apply 1st CBN to the output of 2nd CBN
            weight = catcbn_weight * self.catcbn_weight_weight[..., None, None] + self.catcbn_weight_bias[..., None, None]
            weight = weight * (1 - invalid_mask) + self.invalid_weight[...,None].repeat(1,1,1,h,w) * invalid_mask
            bias = catcbn_bias * self.catcbn_bias_weight[..., None, None] + self.catcbn_bias_bias[..., None, None]
            bias = bias * (1 - invalid_mask) + self.invalid_bias[...,None].repeat(1,1,1,h,w) * invalid_mask
            # Apply 2nd CBN to the original output
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class CategoricalConditionalCostVolumeNorm(torch.nn.Module):
    """
    Categorical version of CCVNorm.
    """
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(1, num_features*num_cats, num_cats+1)) # NOTE +1 for background class (invalid point, represented by -1)
            self.bias = torch.nn.Parameter(torch.zeros(1, num_features*num_cats, num_cats+1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        b, c, d, h, w = out.shape
        if self.affine:
            invalid_mask = (cats[0,0] == -1)[None,None,None,...].float()
            weight = (self.weight[:,:,:-1,None,None]*cats.unsqueeze(1)).sum(2, keepdim=True)
            weight = weight * (1 - invalid_mask) + \
                     self.weight[:,:,-2:-1,None,None].repeat(1,1,1,h,w) * invalid_mask
            weight = weight.view(b, self.num_features, self.num_cats, h, w)
            bias = (self.bias[:,:,:-1,None,None]*cats.unsqueeze(1)).sum(2, keepdim=True)
            bias = bias * (1 - invalid_mask) + \
                   self.bias[:,:,-2:-1,None,None].repeat(1,1,1,h,w) * invalid_mask
            bias = bias.view(b, self.num_features, self.num_cats, h, w)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class ContinuousConditionalCostVolumeNorm(torch.nn.Module):
    """
    Continuous version of CCVNorm (with one 1x1 conv as the continous mapping from sparse disparity to 
    feature modulation parameters).
    """
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True, cbn_in_channels=1, grad_masking=None):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.cbn_in_channels = cbn_in_channels
        self.grad_masking = grad_masking
        if self.affine:
            self.gamma_conv = nn.Conv2d(cbn_in_channels, num_features*num_cats, kernel_size=1, stride=1, padding=0)
            self.beta_conv = nn.Conv2d(cbn_in_channels, num_features*num_cats, kernel_size=1, stride=1, padding=0)
            self.weight = torch.nn.Parameter(torch.ones(1, num_features, num_cats, 1)) # NOTE 1 for background class
            self.bias = torch.nn.Parameter(torch.zeros(1, num_features, num_cats, 1))
        else:
            self.register_parameter('gamma_conv', None)
            self.register_parameter('beta_conv', None)
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def grad_masking_hook(self, grad_output):
        prob = self.grad_masking / self.valid_mask.float().sum()
        mask_keep = torch.empty(self.valid_mask[self.valid_mask].shape).uniform_().to(grad_output) < prob
        grad_mask = self.valid_mask.float()
        grad_mask[self.valid_mask] = mask_keep.float()
        grad_output.mul_(grad_mask) # NOTE: grad_output is pass-by-reference

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            net_init(self.gamma_conv)
            net_init(self.beta_conv)
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        b, c, d, h, w = out.shape
        if self.affine:
            invalid_mask = (cats[0,0] == -1)[None,None,None,...].float()
            self.valid_mask = (cats[0,0] > 0)[None,None,None,...] # NOTE: used for gradient masking
            weight = self.gamma_conv(cats).view(b, self.num_features, self.num_cats, h, w)
            bias = self.beta_conv(cats).view(b, self.num_features, self.num_cats, h, w)
            if self.training: # perform gradient masking of the raw output from gamma_conv and beta_conv
                if self.grad_masking is not None:
                    weight.register_hook(self.grad_masking_hook)
                    bias.register_hook(self.grad_masking_hook)
            weight = weight * (1 - invalid_mask) + self.weight[...,None].repeat(1,1,1,h,w) * invalid_mask
            bias = bias * (1 - invalid_mask) + self.bias[...,None].repeat(1,1,1,h,w) * invalid_mask
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class NaiveCategoricalConditionalBatchNorm(torch.nn.Module):
    """
    Naive version of CatCBN.
    """
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = torch.nn.Parameter(torch.zeros(1, num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        b, c, d, h, w = out.shape
        if self.affine:
            invalid_mask = (cats[0,0] == -1)[None,None,None,...].float()
            weight = (self.weight[:,:,:,None,None]*cats.unsqueeze(1)).sum(2, keepdim=True)
            weight = weight.repeat(1, 1, self.num_cats, 1, 1)
            bias = (self.bias[:,:,:,None,None]*cats.unsqueeze(1)).sum(2, keepdim=True)
            bias = bias.repeat(1, 1, self.num_cats, 1, 1)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class NaiveContinuousConditionalBatchNorm(torch.nn.Module):
    """
    Naive version of ContCBN.
    """
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True, cbn_in_channels=1, grad_masking=None):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.cbn_in_channels = cbn_in_channels
        self.grad_masking = grad_masking
        if self.affine:
            self.gamma_conv = torch.nn.Conv2d(cbn_in_channels, num_features, kernel_size=1, stride=1, padding=0)
            self.beta_conv = torch.nn.Conv2d(cbn_in_channels, num_features, kernel_size=1, stride=1, padding=0)
        else:
            self.register_parameter('gamma_conv', None)
            self.register_parameter('beta_conv', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def grad_masking_hook(self, grad_output):
        prob = self.grad_masking / self.valid_mask.float().sum()
        mask_keep = torch.empty(self.valid_mask[self.valid_mask].shape).uniform_().to(grad_output) < prob
        grad_mask = self.valid_mask.float()
        grad_mask[self.valid_mask] = mask_keep.float()
        grad_output.mul_(grad_mask) # NOTE: grad_output is pass-by-reference

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            net_init(self.gamma_conv)
            net_init(self.beta_conv)

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        b, c, d, h, w = out.shape
        if self.affine:
            invalid_mask = (cats[0,0] == -1)[None,None,None,...].float()
            self.valid_mask = (cats[0,0] > 0)[None,None,None,...] # NOTE: used for gradient masking
            weight = self.gamma_conv(cats).view(b, self.num_features, 1, h, w)
            bias = self.beta_conv(cats).view(b, self.num_features, 1, h, w)
            if self.training: # perform gradient masking of the raw output from gamma_conv and beta_conv
                if self.grad_masking is not None:
                    weight.register_hook(self.grad_masking_hook)
                    bias.register_hook(self.grad_masking_hook)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


def net_init(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, torch.nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, torch.nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, torch.nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
