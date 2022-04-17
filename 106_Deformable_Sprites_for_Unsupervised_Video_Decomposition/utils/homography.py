import torch
import torch.nn.functional as F

from .geom import get_uv_grid, get_flow_coords


def identity_homography(dims, scale=1.0, device=None):
    if device is None:
        device = torch.device("cpu")
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device).view(-1)
    if len(scale) == 1:
        scale = scale.repeat(2)
    scale = torch.cat([scale[:2], torch.ones_like(scale[:1])])
    identity = torch.diag(scale)
    identity = identity.flatten()[:8]
    return identity.view(*(1,) * len(dims), 8).repeat(*dims, 1)  # (*dims, 8)


def apply_homography_xy1(mat, xy1):
    """
    :param mat (*, 3, 3) (# * dims must match uv dims)
    :param xy1 (*, H, W, 3)
    :returns warped coordinates (*, H, W, 2)
    """
    out_h = torch.matmul(mat[..., None, None, :, :], xy1[..., None])  # (*, H, W, 3, 1)
    return out_h[..., :2, 0] / out_h[..., 2:, 0]


def apply_homography(mat, uv):
    """
    :param mat (*, 3, 3) (# * dims must match uv dims)
    :param uv (*, H, W, 2)
    :returns warped coordinates (*, H, W, 2)
    """
    uv_h = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)  # (*, H, W, 3)
    return apply_homography_xy1(mat, uv_h)


def apply_homography_grid(mat, H, W):
    """
    :param mat (*, 3, 3)
    :returns warped coordinates (*, H, W, 2)
    """
    *dims, _, _ = mat.shape
    xy1 = get_uv_grid(H, W, homo=True, device=mat.device)
    xy1 = xy1.view(*(1,) * len(dims), H, W, 3)  # (*, H, W, 3)
    return apply_homography_xy1(mat, xy1)


def homography_params_to_mat(thetas):
    """
    :param thetas (*, 8)
    :returns homography matrices (*, 3, 3)
    """
    *dims, _ = thetas.shape
    homo_flat = torch.cat([thetas, torch.ones_like(thetas[..., :1])], dim=-1)
    return homo_flat.view(*dims, 3, 3)


def invert_homography(mat):
    return torch.linalg.inv(mat)


def flow_warp_homography(T_1c, T_2c, F_12, ord=2):
    """
    T_1c(A) -> A', FLOW_12(A) -> B, T_2c(B) -> B' = A'
    :param T_1c (B, *, 8) transform from 1 to cano
    :param T_2c (B, *, 8) transform from 2 to cano
    :param F_12 (B, 2, H, W) dense flow field from 1 to 2
    :returns the distance between A' and B'
    """
    B, *dims, _ = T_1c.shape
    _, _, H, W = F_12.shape
    F_12 = F_12.permute(0, 2, 3, 1)
    F_12 = F_12.view(B, *(1,) * len(dims), H, W, 2)  # (B, *, H, W, 2)
    uv = get_uv_grid(H, W, align_corners=False, device=F_12.device)  # (H, W, 2)
    uv = uv.view(1, *(1,) * len(dims), H, W, 2)  # (1, *, H, W, 2)
    uv_fwd = uv + F_12
    warp_a = apply_homography(T_1c, uv)  # (B, *, H, W, 2)
    warp_b = apply_homography(T_2c, uv_fwd)
    dist = torch.linalg.norm(warp_a - warp_b, dim=-1, ord=ord)[
        ..., None, :, :
    ]  # (B, *, 1, H, W)
    return dist
