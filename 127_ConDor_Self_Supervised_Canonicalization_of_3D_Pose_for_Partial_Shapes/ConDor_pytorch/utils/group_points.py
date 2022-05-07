import torch
import numpy as np



def patches_radius(radius, sq_norm):
    batch_size = sq_norm.shape[0]
    rad = radius
    if isinstance(radius, float):
        rad = radius * torch.ones((batch_size, 1, 1))
    if isinstance(radius, str):
        rad = torch.sqrt(torch.maximum(torch.max(sq_norm, dim=2, keepdims=False), torch.tensor(0.0000001).type_as(sq_norm)))
        if radius == "avg":
            rad = torch.mean(rad, dim=-1, keepdims=False)
        elif radius == 'min':
            rad = torch.min(rad, dim=-1, keepdims=False)
        elif radius.isnumeric():
            rad = torch.sort(rad, dim=-1)
            i = int((float(int(radius)) / 100.) * sq_norm.shape[1])
            i = max(i, 1)
            rad = torch.mean(rad[:, :i], dim=-1, keepdims=False)
        rad = torch.reshape(rad, (batch_size, 1, 1))
    return rad


def gather_idx(x, idx):


    """
    x - B, N, 3
    idx - B, N, K, 2/3

    out - B, N, K, 3
    """
    num_idx = idx.shape[-1]
    
    if idx.shape[-1] == 3:
        if len(x.shape) == 3:
            out = x[idx[..., 0], idx[..., 1], idx[..., 2]]
            out[(idx[..., 2] < 0) * (idx[..., 1] < 0)] = 0
            return out

    if len(x.shape) == 2:
        out = x[idx[..., 0], idx[..., 1]]
        out[idx[..., 1] < 0] = 0
    else:
        out = x[idx[..., 0], idx[..., 1], :]
        out[idx[..., 1] < 0, :] = 0

    # print(idx[..., 1].shape, out.shape)

    return out


def compute_patches_(source, target, sq_distance_mat, num_samples, spacing, radius, source_mask=None):
    batch_size = source.shape[0]
    num_points_source = source.shape[1]
    num_points_target = target.shape[1]
    assert (num_samples * (spacing + 1) <= num_points_source)

    sq_patches_dist, patches_idx = torch.topk(-sq_distance_mat, k=num_samples * (spacing + 1))
    sq_patches_dist = -sq_patches_dist
    if spacing > 0:
        sq_patches_dist = sq_patches_dist[:, :, 0::(spacing + 1), ...]
        patches_idx = patches_idx[:, :, 0::(spacing + 1), ...]

    rad = patches_radius(radius, sq_patches_dist).type_as(sq_distance_mat)
    patches_size = patches_idx.shape[-1]

    # mask = sq_patches_dist < radius ** 2
    mask = torch.greater_equal(rad.type_as(sq_distance_mat) ** 2, sq_patches_dist)
    patches_idx = (torch.where(mask, patches_idx, torch.tensor(-1).type_as(patches_idx))).to(torch.int64)
    if source_mask is not None:
        source_mask = source_mask < 1
        source_mask = source_mask.unsqueeze(-1).repeat(1, 1, patches_idx.shape[-1])
        patches_idx = torch.where(source_mask, patches_idx, torch.tensor(-1).type_as(patches_idx))

    batch_idx = torch.arange(batch_size).type_as(patches_idx)
    batch_idx = torch.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = batch_idx.repeat(1, num_points_target, num_samples)
    patches_idx = torch.stack([batch_idx, patches_idx], dim = -1).to(torch.long)

    source = (source / (rad + 1e-6))
    target = (target / (rad + 1e-6))
    
    # patches = source[batch_idx.to(torch.long), patches_idx.to(torch.long)]
    patches = gather_idx(source, patches_idx)
    # patches = source[patches_idx[..., 0], patches_idx[..., 1], :]
    # print(patches.shape, "patch")
    patches = patches - target.unsqueeze(-2)


    if source_mask is not None:
        mask = source_mask
    else:
        mask = torch.ones((batch_size, num_points_source)).type_as(patches)
    
    patch_size = gather_idx(mask, patches_idx.to(torch.long))
    # patch_size = mask[patches_idx[..., 0], patches_idx[..., 1]]
    patches_size = torch.sum(patch_size, dim=-1, keepdims=False)
    patches_dist = torch.sqrt(torch.maximum(sq_patches_dist, torch.tensor(0.000000001).type_as(sq_patches_dist)))
    patches_dist = patches_dist / (rad + 1e-6)

    return {"patches": patches, "patches idx": patches_idx, "patches size": patches_size, "patches radius": rad,
            "patches dist": patches_dist}

class GroupPoints(torch.nn.Module):
    def __init__(self, radius, patch_size_source, radius_target=None, patch_size_target=None,
                 spacing_source=0, spacing_target=0):
        super(GroupPoints, self).__init__()

        """
        Group points and different scales for pooling
        """
        self.radius = radius
        self.radius_target = radius_target
        self.patch_size_source = patch_size_source
        self.patch_size_target = patch_size_target
        self.spacing_source = spacing_source
        self.spacing_target = spacing_target

    def forward(self, x):
        """
        :param x: [source, target]
        :return: [patches_idx_source, num_incident_points_target]

        Returns:
            source patches - B, N, K, 3
            patches idx source - B, N, K, 2
            patches size source - B, N
            patches radius source - B, 1, 1
            patches dist source - B, N, K
        """
        assert isinstance(x, dict)
        source = x["source points"]
        target = x["target points"]

        source_mask = None
        if "source mask" in x:
            source_mask = x["source mask"]

        target_mask = None
        if "target mask" in x:
            target_mask = x["target mask"]

        num_points_source = source.shape[1]

        # assert (num_points_source >= self.patch_size_source)
        if self.patch_size_target is not None:
            num_points_target = target.shape[1]
            # assert (num_points_target >= self.patch_size_source)

        # compute distance mat
        r0 = target * target
        r0 = torch.sum(r0, dim=2, keepdims=True)
        r1 = (source * source)
        r1 = torch.sum(r1, dim=2, keepdims=True)
        r1 = r1.permute(0, 2, 1)
        sq_distance_mat = r0 - 2. * (target @ source.permute(0, 2, 1)) + r1

        # Returns 
        patches = compute_patches_(source, target, sq_distance_mat,
                                   min(self.patch_size_source, num_points_source),
                                   self.spacing_source, self.radius,
                                   source_mask=source_mask)
        # print(patches["patches"].shape)
        y = dict()
        y["patches source"] = patches["patches"] # B, N, K, 3
        y["patches idx source"] = patches["patches idx"]
        y["patches size source"] = patches["patches size"]
        y["patches radius source"] = patches["patches radius"]
        y["patches dist source"] = patches["patches dist"]

        # y = [patches_source, patches_idx_source, patches_size_source]
        if self.patch_size_target is not None:
            sq_distance_mat_t = sq_distance_mat.permute(0, 2, 1)
            patches = compute_patches_(target, source, sq_distance_mat_t,
                                       min(self.patch_size_target, num_points_target),
                                       self.spacing_target, self.radius_target,
                                       source_mask=target_mask)
            # y += [patches_target, patches_idx_target, patches_size_target]

            y["patches target"] = patches["patches"]
            y["patches idx target"] = patches["patches idx"]
            y["patches size target"] = patches["patches size"]
            y["patches radius target"] = patches["patches radius"]
            y["patches dist target"] = patches["patches dist"]
        # y.append(radius)

        return y


if __name__ == "__main__":

    N_pts = 100
    start = 10
    x = torch.ones((2, N_pts, 3)) * torch.arange(N_pts).unsqueeze(-1).unsqueeze(0)
    y = torch.ones((2, N_pts, 3)) * torch.arange(start, N_pts + start).unsqueeze(-1).unsqueeze(0)
    # print(x, y)
    gi = GroupPoints(0.2, 32)
    out = gi({"source points": x, "target points": y})

    for k in out:
        print(k, " ", out[k].shape)#, " ", k)