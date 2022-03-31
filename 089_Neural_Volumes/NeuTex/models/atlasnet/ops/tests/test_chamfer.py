import numpy as np
import torch
from torch.autograd import gradcheck

from completion.ops.chamfer import chamfer_distance, nn_distance


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


def nn_distance_torch(xyz1, xyz2, data_format='NWC'):
    assert torch.is_tensor(xyz1) and xyz1.dim() == 3
    assert torch.is_tensor(xyz2) and xyz2.dim() == 3
    if data_format == 'NCW':
        assert xyz1.size(1) == 3 and xyz2.size(1) == 3
    elif data_format == 'NWC':
        assert xyz1.size(2) == 3 and xyz2.size(2) == 3
    distance = bpdist2(xyz1, xyz2, data_format)
    dist1, idx1 = distance.min(2)
    dist2, idx2 = distance.min(1)
    return dist1, idx1, dist2, idx2


def test_chamfer():
    # ---------------------------------------------------------------------------- #
    # NWC format
    # ---------------------------------------------------------------------------- #
    batch_size = 32
    num_points = 2048
    xyz1 = torch.rand(batch_size, num_points, 3).float()
    xyz2 = torch.rand(batch_size, num_points, 3).float()
    # xyz1 = torch.tensor([[[0, 0, 1], [1, 0, 0]]]).float()
    # xyz2 = torch.tensor([[[0, 0, 1.1], [1.2, 0, 0]]]).float()
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()

    # check forward
    dist1_actual, idx1_actual, dist2_actual, idx2_actual = nn_distance(xyz1, xyz2, transpose=False)
    dist1_desired, idx1_desired, dist2_desired, idx2_desired = nn_distance_torch(xyz1, xyz2, 'NWC')

    # np.testing
    np.testing.assert_allclose(dist1_actual.cpu().numpy(), dist1_desired.cpu().numpy(), atol=1e-6)
    np.testing.assert_equal(idx1_actual.cpu().numpy(), idx1_desired.cpu().numpy())
    # torch built-in
    assert dist2_desired.allclose(dist2_actual)
    assert idx2_desired.equal(idx2_actual)

    # # timing
    # import time
    # torch.cuda.synchronize()
    # tic = time.time()
    # for _ in range(5):
    #     nn_distance(xyz1, xyz2, False)
    # torch.cuda.synchronize()
    # print('chamfer_forward', (time.time() - tic) / 5)
    #
    # # profile
    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     nn_distance(xyz1, xyz2, False)
    # print(prof)

    # check backward. float32 is not enough for numerical accuracy.
    batch_size = 2
    num_points = 64
    xyz1 = torch.rand(batch_size, num_points, 3).double()
    xyz2 = torch.rand(batch_size, num_points, 3).double()
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()
    xyz1.requires_grad = True
    xyz2.requires_grad = True
    gradcheck(chamfer_distance, (xyz1, xyz2, False))

    # ---------------------------------------------------------------------------- #
    # NCW format
    # ---------------------------------------------------------------------------- #
    batch_size = 32
    num_points = 2043
    xyz1 = torch.rand(batch_size, 3, num_points).float()
    xyz2 = torch.rand(batch_size, 3, num_points).float()
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()

    # check forward
    dist1_actual, idx1_actual, dist2_actual, idx2_actual = nn_distance(xyz1, xyz2, transpose=True)
    dist1_desired, idx1_desired, dist2_desired, idx2_desired = nn_distance_torch(xyz1, xyz2, 'NCW')

    # np.testing
    np.testing.assert_allclose(dist1_actual.cpu().numpy(), dist1_desired.cpu().numpy(), atol=1e-6)
    np.testing.assert_equal(idx1_actual.cpu().numpy(), idx1_desired.cpu().numpy())
    # torch built-in
    assert dist2_desired.allclose(dist2_actual)
    assert idx2_desired.equal(idx2_actual)

    # check backward. float32 is not enough for numerical accuracy.
    batch_size = 2
    num_points = 64
    xyz1 = torch.rand(batch_size, 3, num_points).double()
    xyz2 = torch.rand(batch_size, 3, num_points).double()
    xyz1 = xyz1.cuda()
    xyz2 = xyz2.cuda()
    xyz1.requires_grad = True
    xyz2.requires_grad = True
    gradcheck(chamfer_distance, (xyz1, xyz2, True))
