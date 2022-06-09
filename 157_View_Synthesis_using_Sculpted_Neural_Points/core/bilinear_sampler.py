import torch
import torch.nn.functional as F



def bilinear_sampler1(img, coords, mode='bilinear', mask=False, cpu=False):
    """ Wrapper for grid_sample, uses pixel coordinates 
        Because it is 1-D sampler, one dimension is of size 1
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    # ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    if cpu:
        img = F.grid_sample(img.cpu(), grid.cpu(), align_corners=True).cuda()
    else:
        img = F.grid_sample(img, grid, align_corners=True)
    
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img



