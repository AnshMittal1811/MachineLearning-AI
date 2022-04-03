import numpy as np
import torch

from completion.ops.fps import farthest_point_sample


def farthest_point_sample_np(points, num_centroids):
    """Farthest point sample

    Args:
        points: (batch_size, 3, num_points)
        num_centroids (int): the number of centroids

    Returns:
        index (np.ndarray): index of centroids. (batch_size, num_centroids)

    """
    index = []
    for points_per_batch in points:
        index_per_batch = [0]
        cur_ind = 0
        dist2set = None
        for ind in range(1, num_centroids):
            cur_xyz = points_per_batch[:, cur_ind]
            dist2cur = points_per_batch - cur_xyz[:, None]
            dist2cur = np.square(dist2cur).sum(0)
            if dist2set is None:
                dist2set = dist2cur
            else:
                dist2set = np.minimum(dist2cur, dist2set)
            cur_ind = np.argmax(dist2set)
            index_per_batch.append(cur_ind)
        index.append(index_per_batch)
    return np.asarray(index)


def test_farthest_point_sample():
    batch_size = 16
    channels = 3
    num_points = 1024
    num_centroids = 128

    np.random.seed(0)
    points = np.random.rand(batch_size, channels, num_points)

    index = farthest_point_sample_np(points, num_centroids)
    point_tensor = torch.from_numpy(points).cuda()
    index_tensor = farthest_point_sample(point_tensor, num_centroids)
    index_tensor = index_tensor.cpu().numpy()
    np.testing.assert_equal(index, index_tensor)

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     # warmup = point_tensor * 2
    #     farthest_point_sample(point_tensor, num_centroids)
    # print(prof)
