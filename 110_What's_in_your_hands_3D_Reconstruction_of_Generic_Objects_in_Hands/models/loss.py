# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import torch
from nnutils import mesh_utils

def repulsive_loss(xVerts, sdf):
    sdf_value = sdf(xVerts)
    # if it's <= -eps (inside of the object), penalize 
    eps = 1e-3
    mask = sdf_value  < -eps
    masked_value = sdf_value[mask]
    if len(masked_value) == 0:
        return torch.tensor(0.0).cuda().float()

    loss = (-masked_value).mean()
    return loss


def contact_loss(hand_wrapper, sdf, yHand, nTy, reduction='mean', ep=.25):
    loss = 0
    for ind in range(6):
        index = getattr(hand_wrapper, 'contact_index_%d' % ind)
        yVerts_hand = torch.index_select(yHand.verts_padded(), 1, index)  # (N, P, 3)

        nVerts_hand = mesh_utils.apply_transform(yVerts_hand, nTy)
        # nSdf_hand = self.model.dec(nVerts_hand, z, batch['hA'])
        nSdf_hand = sdf(nVerts_hand, )  # (N, P, 1)

        # contact
        # for those > 0 (area not in contact ) and < ep (not too far away), penalize for the distance
        min_sdf = torch.min(nSdf_hand.clamp(min=0), dim=1)[0]  # (N, 1)
        m = (min_sdf < ep)
        loss += m * min_sdf

    if reduction == 'mean':
        loss = loss.mean()
    elif isinstance(reduction, int):
        loss = loss.reshape(reduction, -1).mean(-1)
    return loss