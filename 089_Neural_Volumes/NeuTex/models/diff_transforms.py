import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.
    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    Example:
        >>> quaternion = torch.tensor([1., 0., 1., 0.])
        >>> kornia.normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
        torch.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example:
        >>> quaternion = torch.tensor([0., 0., 1., 0.])
        >>> kornia.quaternion_to_rotation_matrix(quaternion)
        tensor([[[-1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0.,  1.]]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )
    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.0)

    matrix: torch.Tensor = torch.stack(
        [
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix.view(quaternion.shape[:-1] + (3, 3))


def qinv(quaternion: torch.Tensor):
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )

    return torch.cat((-quaternion[..., :3], quaternion[..., [-1]]), dim=-1)


def qmult(q1: torch.Tensor, q2: torch.Tensor):
    if not isinstance(q1, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(q1)))
    if not isinstance(q2, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(q2)))
    if q1.shape != q2.shape:
        raise ValueError(
            "Input has different shapes {} and {}".format(q1.shape, q2.shape)
        )
    if not q1.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(q1.shape)
        )

    w1 = q1[..., [-1]]
    w2 = q2[..., [-1]]
    v1 = q1[..., :3]
    v2 = q2[..., :3]

    w = w1 * w2 - (v1 * v2).sum(-1, keepdim=True)
    v = w1 * v2 + w2 * v1 + torch.cross(v1, v2)
    return torch.cat((v, w), dim=-1)


def quaternion_to_angle(q: torch.Tensor):
    if not isinstance(q, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(q)))
    if not q.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(q.shape)
        )

    return 2 * torch.acos(q[..., -1]).abs()


def quaternion_similarity(q1: torch.Tensor, q2: torch.Tensor):
    """
    cosine similarity `[0,1]` for the half rotation angle between quaternions.
    """
    if not isinstance(q1, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(q1)))
    if not isinstance(q2, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(q2)))
    if q1.shape != q2.shape:
        raise ValueError(
            "Input has different shapes {} and {}".format(q1.shape, q2.shape)
        )
    if not q1.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(q1.shape)
        )

    q = qmult(q1, qinv(q2))
    return q[..., -1]


if __name__ == "__main__":
    q1 = torch.FloatTensor([[0.67774235, -0.26440982, 0.25804417, -0.63574048]])
    q2 = torch.FloatTensor([[0.50144721, -0.46370333, 0.38666788, 0.61969175]])
    import transforms3d.quaternions as tq
    print(qinv(q1))
    print(tq.qinverse([-0.63574048, 0.67774235, -0.26440982, 0.25804417]))

    print(qmult(q1, q2))
    print(
        tq.qmult(
            [-0.63574048, 0.67774235, -0.26440982, 0.25804417],
            [0.61969175, 0.50144721, -0.46370333, 0.38666788],
        )
    )

    print(quaternion_similarity(q1, q1))
    print(quaternion_similarity(q2, q2))
    print(quaternion_similarity(q1, q2))
