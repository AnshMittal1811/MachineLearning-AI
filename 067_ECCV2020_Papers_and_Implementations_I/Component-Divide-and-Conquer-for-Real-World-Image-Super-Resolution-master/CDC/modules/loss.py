import torch
import torch.nn as nn


# Define GAN loss: [vanilla | lsgan | wgan-gp]

def make_ragan_input(pred_a, pred_b):
    """
    Make Input for GANLoss, pred_a - Mean(Pred_b)
    :param pred_a: returned by Multi-Scale Discriminator, [[B, 1, N, N]...]
    :param pred_b:
    :return: List
    """
    if isinstance(pred_a, list):
        assert len(pred_a) == len(pred_b)
        pred_result = []
        for i in range(len(pred_a)):
            pred_result.append([pred_a[i][-1] - torch.mean(pred_b[i][-1]).detach()])
        return pred_result
    else:
        return pred_a - torch.mean(pred_b).detach()


class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        # Modify for Multi-Scale Discriminator, whose result is list
        if isinstance(input, list):
            if isinstance(input[0], list):
                loss = 0
                for input_i in input:
                    pred = input_i[-1]
                    target_tensor = self.get_target_label(pred, target_is_real)
                    loss += self.loss(pred, target_tensor)
                return loss
            else:
                target_tensor = self.get_target_label(input[-1], target_is_real)
                return self.loss(input[-1], target_tensor)
        else:
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)
            return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


def ImageEntropy(img, bins = 256, min_value=0.0, max_value=255.0, is_gpu = False):
    # img：单通道灰度图
    # bins：划分的区间个数
    # min_value：图像类型的最小灰度值
    # max_value: 图像类型的最大灰度值
    interval = (max_value - min_value)/(bins-1)
    eps = torch.FloatTensor(1).fill_(0.5*interval)
    im_size = torch.numel(img)
    value = torch.range(min_value, max_value, interval)
    num = torch.numel(value)
    H = torch.zeros(num)
    zero = torch.zeros(1)
    if is_gpu:
        eps = eps.cuda()
        value = value.cuda()
        H = H.cuda()
        zero = zero.cuda()
    for i in range(0, num):
        if value[i] == 0:
            offset = 1
        else:
            offset = 0
        mask = (img > value[i]-eps).float() * (img < value[i]+eps).float()
        cur_img = img * mask
        pixel_num = torch.sum(mask)
        if pixel_num >= 1:
            cur_mean = torch.sum(cur_img + offset) / pixel_num
            pixel = cur_mean.detach()
            H[i] = torch.sum(cur_img + offset)/(pixel*im_size)
    H = H[torch.nonzero(H)[:, 0]]
    return torch.sum(-H*torch.log2(H))
    
    
class ImageEntropyLoss(nn.Module):
    def __init__(self):
        super(ImageEntropyLoss, self).__init__()
        
    def forward(self, x, sign = 1, multi_channel = False):
        b, c, h, w = x.size()
        if multi_channel:
            loss = torch.zeros(b, c, 1).type(x.type())
            for i in range(0, b):
                for j in range(0, c):
                    loss[i, j, :] = ImageEntropy(x[i, j, :, :], min_value=0.0, max_value=1.0, is_gpu = True)
        else:
            Y = 0.257*x[:, :1, :, :] + 0.564*x[:, 1:2, :, :] + 0.098*x[:, 2:, :, :] + 16/256  # rgb2ycbcr
#             Y =  (x[:, :1, :, :] + x[:, 1:2, :, :] + x[:, 2:, :, :])/3  # rgb2gray
            loss = torch.zeros(b, 1, 1).type(x.type())
            for i in range(0, b):
                loss[i, :, :] = ImageEntropy(Y[i, 0, :, :], min_value=0.0, max_value=1.0, is_gpu = True)
        return sign * torch.mean(loss)
        
        
        