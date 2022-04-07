import numpy as np
import torch
from torch.autograd import gradcheck

from completion.ops.ball_query import ball_query


def bpdist2(feature1, feature2, data_format='NWC'):
    """This version has a high memory usage but more compatible(accurate) with optimized Chamfer Distance."""
    if data_format == 'NCW':
        diff = feature1.unsqueeze(3) - feature2.unsqueeze(2)
        distance = torch.sum(diff ** 2, dim=1)
    elif data_format == 'NWC':
        diff = feature1.unsqueeze(2) - feature2.unsqueeze(1)
        distance = torch.sum(diff ** 2, dim=3)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))
    return distance


def ball_query_torch(xyz1, xyz2, radius, data_format='NWC'):
    assert torch.is_tensor(xyz1) and xyz1.dim() == 3
    assert torch.is_tensor(xyz2) and xyz2.dim() == 3
    if data_format == 'NCW':
        assert xyz1.size(1) == 3 and xyz2.size(1) == 3
    elif data_format == 'NWC':
        assert xyz1.size(2) == 3 and xyz2.size(2) == 3
    distance = bpdist2(xyz1, xyz2, data_format)
    distance = torch.sqrt(distance)
    mask = (distance < radius) * (distance > 1e-6)
    count = mask.sum(2)
    mean_distance = (distance * mask.to(dtype=distance.dtype)).sum(2) / count.clamp(min=1).to(dtype=distance.dtype)
    return mean_distance, count


def test_ball_query():
    # ---------------------------------------------------------------------------- #
    # NWC format
    # ---------------------------------------------------------------------------- #
    batch_size = 32
    num_points = 2048
    radius = 0.05
    xyz1 = torch.rand(batch_size, num_points, 3).float()
    xyz2 = torch.rand(batch_size, num_points, 3).float()
    # xyz1 = torch.zeros(batch_size, num_points, 3).float()
    # xyz2 = torch.zeros(batch_size, num_points, 3).float()
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()

    # check forward
    dist_actual, count_actual = ball_query(xyz1, xyz2, radius=radius, transpose=False)
    dist_desired, count_desired = ball_query_torch(xyz1, xyz2, radius, 'NWC')

    # np.testing
    np.testing.assert_allclose(dist_actual.cpu().numpy(), dist_desired.cpu().numpy(), atol=1e-6)
    np.testing.assert_equal(count_actual.cpu().numpy(), count_desired.cpu().numpy())

    # # timing
    # import time
    # torch.cuda.synchronize()
    # tic = time.time()
    # for _ in range(5):
    #     ball_query(xyz1, xyz2, radius=radius, transpose=False)
    #     # ball_query_torch(xyz1, xyz2, radius=radius, data_format='NWC')
    # torch.cuda.synchronize()
    # print('forward', (time.time() - tic) / 5)

    # # profile
    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     ball_query(xyz1, xyz2, radius=radius, transpose=False)
    # print(prof)

    # check backward. float32 is not enough for numerical accuracy.
    batch_size = 2
    num_points = 64
    radius = 0.05
    xyz1 = torch.rand(batch_size, num_points, 3).double()
    # xyz2 = torch.rand(batch_size, num_points, 3).double()
    xyz2 = xyz1 + 0.03
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()
    xyz1.requires_grad = True
    xyz2.requires_grad = True
    # gradcheck(ball_query_torch, (xyz1, xyz2, radius, 'NWC'), atol=1e-6)
    func = lambda xyz1, xyz2, radius, transpose: ball_query(xyz1, xyz2, radius, transpose)[0]
    gradcheck(func, (xyz1, xyz2, radius, False), atol=1e-6)

    # ---------------------------------------------------------------------------- #
    # NCW format
    # ---------------------------------------------------------------------------- #
    batch_size = 32
    num_points = 2048
    radius = 0.05
    xyz1 = torch.rand(batch_size, 3, num_points).float()
    xyz2 = torch.rand(batch_size, 3, num_points).float()
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()

    # check forward
    dist_actual, count_actual = ball_query(xyz1, xyz2, radius=radius, transpose=True)
    dist_desired, count_desired = ball_query_torch(xyz1, xyz2, radius, 'NCW')

    # np.testing
    np.testing.assert_allclose(dist_actual.cpu().numpy(), dist_desired.cpu().numpy(), atol=1e-6)
    np.testing.assert_equal(count_actual.cpu().numpy(), count_desired.cpu().numpy())

    # check backward. float32 is not enough for numerical accuracy.
    batch_size = 2
    num_points = 64
    radius = 0.05
    xyz1 = torch.rand(batch_size, 3, num_points).double()
    # xyz2 = torch.rand(batch_size, num_points, 3).double()
    xyz2 = xyz1 + 0.03
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()
    xyz1.requires_grad = True
    xyz2.requires_grad = True
    func = lambda xyz1, xyz2, radius, transpose: ball_query(xyz1, xyz2, radius, transpose)[0]
    gradcheck(func, (xyz1, xyz2, radius, True), atol=1e-6)

