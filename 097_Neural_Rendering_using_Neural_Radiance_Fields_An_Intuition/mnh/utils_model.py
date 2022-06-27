import torch
import torch.nn as nn
import torch.nn.functional as F

def grid_sample_planes(
    sample_points,
    planes_wh,
    planes_content,
    mode='bilinear',
    padding_mode='zeros'
):
    '''
    Args:
        sample_points: (plane_n, sample_n, 2)
        planes_wh: (plane_n, 2)
        planes_content: (plane_n, dim, h, w)
    Retrun:
        sampled_content: (plane_n, sample_n, dim)
        in_planes: (plane_n, sample_n) True if sample inside plane
    '''
    norm_scale = (-2/planes_wh).unsqueeze(1) #(plane_n, 1, 2)
    grid_points = sample_points * norm_scale 
    grid_points = grid_points.unsqueeze(1) #(plane_n, 1, sample_n, 2)
    sampled_content = F.grid_sample(
        planes_content, 
        grid_points, 
        mode=mode, 
        padding_mode=padding_mode, 
        align_corners=False
    ) #(plane_n, dim, 1, sample_n)
    sampled_content = sampled_content.squeeze(2).transpose(1, 2)
    return sampled_content

def check_inside_planes(
    planes_points,
    planes_wh,
):  
    '''
    Check if points are inside plane
    Args
        planes_points: (plane_n, point_n, 2)
        planes_wh: (plane_n, 2)
    Return
        inside_planes: (plane_n, point_n)
    '''
    norm_scale = (2/planes_wh).unsqueeze(1)
    planes_points = planes_points * norm_scale
    points_x, points_y = planes_points[:,:,0], planes_points[:,:,1]
    bound = 1.0
    in_width  = torch.logical_and(points_x >= -bound, points_x <= bound)
    in_height = torch.logical_and(points_y >= -bound, points_y <= bound)
    in_planes = torch.logical_and(in_width, in_height)
    return in_planes

def compute_alpha_weight(alpha_sorted, normalize=False):
    '''
    compute alpha weight for composite from raw alpha values (sorted)
    Args
        alpha_sored: (plane_n, sample_n)
        plane[0]: nearest, plane[-1]: farthest
    Return 
        alpha_weight: (plane_n, sample_n)
    '''
    plane_n, sample_n = alpha_sorted.size()
    alpha_comp = torch.cumprod(1-alpha_sorted, dim=0)
    alpha_comp = torch.cat([alpha_comp.new_ones(1, sample_n), alpha_comp[:-1,:]], dim=0)
    alpha_weight = alpha_sorted * alpha_comp #(plane_n, sample_n)
    # premultiplied alpha 
    if normalize == True:
        weight_sum = torch.sum(alpha_weight, dim=0, keepdim=True)
        weight_sum[weight_sum == 0] = 1e-5
        alpha_weight /= weight_sum
    return alpha_weight

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False 

def detect_invalid_values(name:str, tensor):
    numel = tensor.numel()
    inf_num = torch.sum(torch.isinf(tensor)) 
    inf_rate = inf_num / numel
    nan_num = torch.sum(torch.isnan(tensor))
    nan_rate = nan_num / numel

    if inf_num > 0 or nan_num > 0:
        print('invalid | inf num: {:}, nan num: {:} | {}'.format(
            inf_num.item(), nan_num.item(), name))

def print_tensor_type(name:str, tensor):
    print('dtype | {} <- {}'.format(tensor.dtype, name))

def check_valid_model(name:str, model):
    for p_name, param in model.named_parameters():
        detect_invalid_values('[{}][{}], param.'.format(name, p_name), param.data)
        if param.grad is not None:
            detect_invalid_values('[{}][{}], grad.'.format(name, p_name), param.grad)
