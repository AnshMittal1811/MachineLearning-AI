import pytest
import numpy as np
import torch
from completion.ops.knn_distance import knn_distance


def bpdist2(feature1, feature2, data_format='NCW'):
    """This version has a high memory usage but more compatible(accurate)."""
    if data_format == 'NCW':
        diff = feature1.unsqueeze(3) - feature2.unsqueeze(2)
        distance = torch.sum(diff ** 2, dim=1)
    elif data_format == 'NWC':
        diff = feature1.unsqueeze(2) - feature2.unsqueeze(1)
        distance = torch.sum(diff ** 2, dim=3)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))
    return distance


def knn_distance_torch(query_xyz, key_xyz, num_neighbors, transpose=True):
    distance = bpdist2(query_xyz, key_xyz, data_format='NCW' if transpose else 'NWC')
    distance, index = torch.topk(distance, num_neighbors, dim=2, largest=False, sorted=True)
    return index, distance


test_data = [
    (2, 512, 1024, 5, True, False),
    (3, 513, 1025, 5, True, False),
    (3, 513, 1025, 5, False, False),
    (3, 31, 63, 5, True, False),
    (32, 4096, 4096, 3, True, True),
    (32, 4096, 4096, 5, True, True),
    (32, 4096, 4096, 10, True, True),
]


@pytest.mark.parametrize('b, n1, n2, k, transpose, profile', test_data)
def test(b, n1, n2, k, transpose, profile):
    np.random.seed(0)

    if transpose:
        query_np = np.random.randn(b, 3, n1).astype(np.float32)
        key_np = np.random.randn(b, 3, n2).astype(np.float32)
    else:
        query_np = np.random.randn(b, n1, 3).astype(np.float32)
        key_np = np.random.randn(b, n2, 3).astype(np.float32)

    query_tensor = torch.tensor(query_np).cuda()
    key_tensor = torch.tensor(key_np).cuda()

    if not profile:
        index_actual, distance_actual = knn_distance(query_tensor, key_tensor, k, transpose=transpose)
        index_desired, distance_desired = knn_distance_torch(query_tensor, key_tensor, k, transpose=transpose)
        np.testing.assert_equal(index_actual.cpu().numpy(), index_desired.cpu().numpy())
        np.testing.assert_allclose(distance_actual.cpu().numpy(), distance_desired.cpu().numpy(), atol=1e-6)
    else:
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            knn_distance(query_tensor, key_tensor, k, transpose=transpose)
        print(prof)
