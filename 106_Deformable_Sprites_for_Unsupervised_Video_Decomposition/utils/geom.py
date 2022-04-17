import torch
import torch.nn.functional as F


def resample_batch(src, coords, **kwargs):
    """
    does 2D grid sample but for tensors of dimensions > 4
    """
    *sdims, C, H, W = src.shape
    *cdims, h, w, d = coords.shape
    assert sdims == cdims
    src = src.view(-1, C, H, W)
    coords = coords.view(-1, h, w, d)
    out = F.grid_sample(src, coords, **kwargs)
    return out.view(*sdims, C, h, w)


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def get_flow_coords(flow, align_corners=False):
    """
    :param flow (*, H, W, 2) normalized flow vectors
    :returns (*, H, W, 2)
    """
    device = flow.device
    *dims, H, W, _ = flow.shape
    uv = get_uv_grid(H, W, homo=False, align_corners=align_corners, device=device)
    uv = uv.view(*(1,) * len(dims), H, W, 2)
    return uv + flow


def inverse_flow_warp(I2, F_12, O_12=None):
    """
    Given image I2 and the flow field from I1 to I2, sample I1 from I2,
    except at points that are disoccluded
    :param I2 (B, C, H, W)
    :param F_12 flow field from I1 to I2 in uv coords (B, H, W, 2)
    :param O_12 (optional) mask of disocclusions (B, 1, H, W)
    """
    C_12 = get_flow_coords(F_12, align_corners=False)
    I1 = F.grid_sample(I2, C_12, align_corners=False)
    if O_12 is not None:
        mask = ~(O_12 == 1)
        I1 = mask * I1
    return I1


def compute_occlusion_locs(fwd, bck, gap, method="brox", thresh=1.5, ret_locs=False):
    """
    compute the locations of the occluding pixels using round-trip flow
    :param fwd (N, 2, H, W) flow from 1->2, 2->3, etc
    :param bck (N, 2, H, W) flow from 2->1, 3->2, etc
    :param method (str) brox implementation taken from
        https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py#L312
        otherwise use a threshold on the forward backward distance
    :param thresh (float) if not using the brox method, the fb distance threshold to use
    :return occ_map (N, 1, H, W) bool binary mask of pixels that get occluded
            occ_locs (N, H, W, 2) O[i,j] location of the pixel that occludes the pixel at i, j
    """
    N, _, H, W = fwd.shape

    ## get the backward flow at the points the forward flow maps points to
    fwd_vec = fwd.permute(0, 2, 3, 1)
    inv_flo = inverse_flow_warp(bck, fwd_vec)  # (N, 2, H, W)
    fb_sq_diff = torch.square(fwd + inv_flo).sum(dim=1, keepdim=True)

    sq_thresh = (thresh / H) ** 2
    if method == "brox":
        fb_sum_sq = (fwd ** 2 + inv_flo ** 2).sum(dim=1, keepdim=True)
        occ_map = fb_sq_diff > (0.01 * fb_sum_sq + sq_thresh)
    else:  # use a simple threshold
        occ_map = fb_sq_diff > sq_thresh

    # get the mask of points that don't go out of frame
    uv_fwd = get_flow_coords(fwd_vec, align_corners=False)  # (N, H, W, 2)
    valid = ((uv_fwd < 0.99) & (uv_fwd > -0.99)).all(dim=-1)  # (N, H, W)
    occ_map = valid[:, None] & occ_map

    out = [occ_map]

    if ret_locs:
        # the inverse warped locs in the original image
        occ_locs = uv_fwd + inv_flo.permute(0, 2, 3, 1)
        out.append(occ_locs)
    return out


def compute_rect_area(vertices):
    """
    :param vertices (*, 4, 3) rectangle coordinates counterclockwise
    :returns areas of the four triangles
    """
    ## triangle indices in counter-clockwise order
    tri_idcs = torch.tensor(
        [[0, 1, 2], [2, 3, 0], [3, 0, 1], [1, 2, 3]], device=vertices.device
    )  # (4, 3)
    view_tris = vertices[..., tri_idcs, :]  # (*, 4, 3, 3)
    tri_areas = 0.5 * torch.linalg.det(view_tris).abs()  # (*, 4)
    return tri_areas


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z ** 2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err
