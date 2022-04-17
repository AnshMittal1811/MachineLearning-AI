"""
Pytorch adaptation of
https://github.com/google-research/google-research/tree/master/factorize_a_city
"""

import torch
import torch.nn.functional as F


def _constant(position):
    """B-Spline basis function of degree 0 for positions in the range [0, 1]."""
    # A piecewise constant spline is discontinuous at the knots.
    return torch.clip(1.0 + position, 1.0, 1.0).unsqueeze(-1)


def _linear(position):
    """B-Spline basis functions of degree 1 for positions in the range [0, 1]."""
    # Piecewise linear splines are C0 smooth.
    return torch.stack((1.0 - position, position), dim=-1)


def _quadratic(position):
    """B-Spline basis functions of degree 2 for positions in the range [0, 1]."""
    # We pre-calculate the terms that are used multiple times.
    pos_sq = torch.pow(position, 2.0)

    # Piecewise quadratic splines are C1 smooth.
    return torch.stack(
        (torch.pow(1.0 - position, 2.0) / 2.0, -pos_sq + position + 0.5, pos_sq / 2.0),
        dim=-1,
    )


def _cubic(position):
    """B-Spline basis functions of degree 3 for positions in the range [0, 1]."""
    # We pre-calculate the terms that are used multiple times.
    neg_pos = 1.0 - position
    pos_sq = torch.pow(position, 2.0)
    pos_cb = torch.pow(position, 3.0)

    # Piecewise cubic splines are C2 smooth.
    return torch.stack(
        (
            torch.pow(neg_pos, 3.0) / 6.0,
            (3.0 * pos_cb - 6.0 * pos_sq + 4.0) / 6.0,
            (-3.0 * pos_cb + 3.0 * pos_sq + 3.0 * position + 1.0) / 6.0,
            pos_cb / 6.0,
        ),
        dim=-1,
    )


def _quartic(position):
    """B-Spline basis functions of degree 4 for positions in the range [0, 1]."""
    # We pre-calculate the terms that are used multiple times.
    neg_pos = 1.0 - position
    pos_sq = torch.pow(position, 2.0)
    pos_cb = torch.pow(position, 3.0)
    pos_qt = torch.pow(position, 4.0)

    # Piecewise quartic splines are C3 smooth.
    return torch.stack(
        (
            torch.pow(neg_pos, 4.0) / 24.0,
            (
                -4.0 * torch.pow(neg_pos, 4.0)
                + 4.0 * torch.pow(neg_pos, 3.0)
                + 6.0 * torch.pow(neg_pos, 2.0)
                + 4.0 * neg_pos
                + 1.0
            )
            / 24.0,
            (pos_qt - 2.0 * pos_cb - pos_sq + 2.0 * position) / 4.0 + 11.0 / 24.0,
            (-4.0 * pos_qt + 4.0 * pos_cb + 6.0 * pos_sq + 4.0 * position + 1.0) / 24.0,
            pos_qt / 24.0,
        ),
        dim=-1,
    )


def knot_weights(positions, num_knots, degree, sparse_mode=False):
    """Function that converts cardinal B-spline positions to knot weights.

    Note:
    In the following, A1 to An are optional batch dimensions.

    Args:
    positions: A tensor with shape `[A1, .. An]`. Positions must be between `[0,
      C - D)`, where `C` is the number of knots and `D` is the spline degree.
    num_knots: A strictly positive `int` describing the number of knots in the
      spline.
    degree: An `int` describing the degree of the spline, which must be smaller
      than `num_knots`.
    sparse_mode: A `bool` describing whether to return a result only for the
      knots with nonzero weights. If set to True, the function returns the
      weights of only the `degree` + 1 knots that are non-zero, as well as the
      indices of the knots.

    Returns:
    A tensor with dense weights for each control point, with the shape
    `[A1, ... An, C]` if `sparse_mode` is False.
    Otherwise, returns a tensor of shape `[A1, ... An, D + 1]` that contains the
    non-zero weights, and a tensor with the indices of the knots, with the type
    tf.int32.

    Raises:
    ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
    InvalidArgumentError: If positions are not in the right range.
    """
    if degree > 4 or degree < 0:
        raise ValueError("Degree should be between 0 and 4.")
    if degree > num_knots - 1:
        raise ValueError("Degree cannot be >= number of knots.")
    device = positions.device

    # Maps valid degrees to functions.
    all_basis_functions = [
        _constant,
        _linear,
        _quadratic,
        _cubic,
        _quartic,
    ]
    basis_functions = all_basis_functions[degree]

    if num_knots - degree == 1:
        # In this case all weights are non-zero and we can just return them.
        if sparse_mode:
            shift = torch.zeros_like(positions, dtype=torch.int32)
            return basis_functions(positions), shift
        return basis_functions(positions)

    shape_batch = positions.shape
    positions = positions.reshape(-1)

    # Calculate the nonzero weights from the decimal parts of positions.
    shift = torch.floor(positions)
    sparse_weights = basis_functions(positions - shift)
    shift = shift.long()

    if sparse_mode:
        # Returns just the weights and the shift amounts, so that tf.gather_nd on
        # the knots can be used to sparsely activate knots if needed.
        sparse_weights = sparse_weights.reshape(*shape_batch, degree + 1)
        shift = shift.reshape(*shape_batch)
        return sparse_weights, shift

    N = positions.numel()
    ind_row, ind_col = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(degree + 1, device=device),
        indexing="ij",
    )

    tiled_shifts = torch.tile(shift[..., None], (1, degree + 1))
    ind_col = ind_col + tiled_shifts
    weights = torch.zeros(N, num_knots, dtype=torch.float32, device=device)
    weights[ind_row, ind_col] = sparse_weights
    return weights.reshape(*shape_batch, num_knots)


def bspline_warp(cps, image, degree, w_stiff=0):
    """Differentiable 2D alignment of images
    Args:
      cps: Control points [bsz, H_CP, W_CP, d] defining the deformations.
      image: An image tensor [bsz, H, W, 3] from which we sample deformed
        coordinates.
      degree: Defines the degree of the b-spline interpolation.
      w_stiff: A float ranging from [0, 1] that smooths the extremes of the
        control points. The effect is that the network has some leeway in fitting
        the original control points exactly.
    Returns:
      A warped image based on deformations specified by control points at various
      positions. Has shape [bsz, H, W, d]
    Raises:
      ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
      InvalidArgumentError: If positions are not in the right range.
    """
    if w_stiff > 0.0:
        cps = regularize_knots(cps, w_stiff)

    coords = bspline_warp_coords(cps, degree, image.shape[-2:])
    return F.grid_sample(image, coords, mode="bilinear", align_corners=True)


def regularize_knots(cps, w_stiff):
    """
    :param cps (..., H_cp, W_cp, d)
    :param w_stiff (...)
    """
    *dims, nky, nkx, d = cps.shape
    if isinstance(w_stiff, torch.Tensor):
        w_stiff = w_stiff[..., None, None, None]

    # Regularizing constraint on the local structure of control points.
    #   New control points is:
    #     w_stiff * avg_neighbor + (1-w_stiff) * cp
    cps_down = torch.cat([cps[..., 1:, :, :], cps[..., -1:, :, :]], dim=1)
    cps_up = torch.cat([cps[..., :1, :, :], cps[..., :-1, :, :]], dim=1)
    cps_left = torch.cat([cps[..., :1, :], cps[..., :-1, :]], dim=2)
    cps_right = torch.cat([cps[..., 1:, :], cps[..., -1:, :]], dim=2)
    cps_reg = (cps_left + cps_right + cps_up + cps_down) / 4.0
    cps = cps * (1 - w_stiff) + cps_reg * (w_stiff)
    return cps


def bspline_warp_coords(cps, degree, target_shape, clip_coords=False):
    """
    Get the query coordinates for resampling a source image warped with bspline
    :param cps (B, h, w, 2) control points defining the deformations
    :param degree (int) degree of b-spline interpolation
    :param target_shape (int, int) tuple with H, W dims of image to warp
    :returns coords (B, H, W, 2) in range [-1, 1] for resampling image
    """
    B, small_h, small_w, _ = cps.shape
    big_h, big_w = target_shape
    device = cps.device

    # Control points are "normalized" in the sense that they're agnostic to the
    # resolution of the image being warped.
    cps = cps * torch.tensor([big_h, big_w], dtype=torch.float32, device=device)

    y_coord = torch.linspace(
        0.0, small_h - 3 - 1e-4, big_h - 4, dtype=torch.float32, device=device
    )
    y_coord = F.pad(y_coord[None, None], (2, 2), mode="replicate")[0, 0]

    x_coord = torch.linspace(0.0, small_w - 3 - 1e-4, big_w - 4, device=device)
    x_coord = F.pad(x_coord[None, None], (2, 2), mode="replicate")[0, 0]

    yy, xx = torch.meshgrid(y_coord, x_coord, indexing="ij")
    stacked_coords = torch.stack([yy, xx], dim=-1)[None].repeat(B, 1, 1, 1)
    offsets = interpolate_2d(cps, stacked_coords, degree, [False, False])

    yy, xx = torch.meshgrid(
        torch.arange(big_h, device=device),
        torch.arange(big_w, device=device),
        indexing="ij",
    )

    xx = xx + offsets[..., 1]
    yy = yy + offsets[..., 0]
    if clip_coords:
        xx = torch.clip(xx, 0, big_w - 1)
        yy = torch.clip(yy, 0, big_h - 1)

    return torch.stack([2 * xx / (big_w - 1) - 1, 2 * yy / (big_h - 1) - 1], dim=-1)


def interpolate_2d(knots, positions, degree):
    """Interpolates the knot values at positions of a bspline surface warp.
    Args:
      knots: (B, KH, KW, Kc) the control_points
      positions: (B, H, W, 2) the desired query points.
        Values must be between [0, num_knots - degree).
        The last dimension of positions record [y, x] coordinates.
      degree: (int) the degree of the spline. Must be > num_knots - 1
    Returns:
      interpolated points (B, H, W, Kc)
    Raises:
      ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
      InvalidArgumentError: If positions are not in the right range.
    """
    B, KH, KW, _ = knots.shape
    device = positions.device
    y_weights, y_ind = knot_weights(positions[..., 0], KH, degree, sparse_mode=True)

    x_weights, x_ind = knot_weights(positions[..., 1], KW, degree, sparse_mode=True)
    base_y_ind = torch.arange(0, degree + 1, device=device)
    base_y_ind = base_y_ind.view(*(1,) * y_ind.dim(), len(base_y_ind))
    stacked_y = torch.clip(base_y_ind + y_ind[..., None], 0, KH - 1)

    base_x_ind = torch.arange(0, degree + 1, device=device)
    base_x_ind = base_x_ind.view(*(1,) * x_ind.dim(), len(base_x_ind))
    stacked_x = torch.clip(base_x_ind + x_ind[..., None], 0, KW - 1)

    ## stacked_x (B, H, W, D+1, D+1), stacked_y (B, H, W, D+1, D+1)
    stacked_y = stacked_y[:, :, :, :, None].long()
    stacked_x = stacked_x[:, :, :, None, :].long()
    batch_ind = torch.arange(0, B, device=device).view(B, 1, 1, 1, 1)
    sel_knots = knots[batch_ind, stacked_y, stacked_x]  # (B, H, W, D+1, D+1, C)

    mixed = y_weights[:, :, :, :, None] * x_weights[:, :, :, None, :]
    return (sel_knots * mixed[..., None]).sum(dim=(-2, -3))  # (B, H, W, C)


def interpolate_1d(knots, positions, degree):
    """Applies B-spline interpolation to input control points (knots).
    Args:
    :param knots (B, K, C) control points, K is number of knots
    :param positions (B, N) query points, must be between [0, C-D)
    :param degree (0 < int < 4) degree of spline
    :returns interpolated points (B, N, C)
    """
    device = knots.device
    B, K, _ = knots.shape
    weights, ind = knot_weights(positions, K, degree, sparse_mode=True)
    base_ind = torch.arange(degree + 1, device=device)
    base_ind = base_ind.view(*(1,) * ind.dim(), len(base_ind))
    stacked = (torch.clip(base_ind + ind[..., None], 0, K - 1)).long()  # (B, K, D+1)
    batch_ind = torch.arange(0, B, device=device).view(B, 1, 1)
    sel_knots = knots[batch_ind, stacked]  # (B, N, D+1, C)
    return (sel_knots * weights[..., None]).sum(dim=-2)  # (B, N, C)
