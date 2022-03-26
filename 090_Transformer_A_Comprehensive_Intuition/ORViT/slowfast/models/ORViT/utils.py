import torch
from slowfast.utils.box_ops import box_cxcywh_to_xyxy
from .layout import boxes_to_layout
from torchvision.ops import roi_align
from slowfast.utils import box_ops
from torch import nn

def box2spatial_layout(box_tensors, action_map, H, W):
    """
    box_tensor: [BS, T, O, 4] cxcywh
    action_map: [BS, T, O, d]
    """
    # Interpolate boxes
    BS, T, O, _ = box_tensors.shape
    segs_batches = []
    for b in range(BS):
        segs_frames = []
        for t in range(T):
            boxes = box_tensors[b][t]
            boxes = box_cxcywh_to_xyxy(boxes)
            objs_vecs = action_map[b][t]
            seg = boxes_to_layout(objs_vecs, boxes, H, W)
            segs_frames.append(seg)
        segs_batches.append(torch.cat(segs_frames, dim=0))
    # padding layouts
    segs = torch.stack(segs_batches, dim=0)
    segs = segs.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    return segs

class ObjectsCrops(nn.Module):
    def __init__(self, cfg):
        """
        video_size: original boxes scale
        spatial_slots: output spatial size
        """
        super().__init__()
        self.cfg = cfg
        self.aligned = True
        self.sampling_ratio = -1
        self.video_hw = (self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.TRAIN_CROP_SIZE)

    def prepare_outdim(self, outdim):
        if isinstance(outdim, (int, float)):
            outdim = [outdim, outdim]
        assert len(outdim) == 2
        return tuple(outdim)

    def forward(self, features, boxes):
        """
        boxes: List[Tensor[T, N_OBJ, 4]]
        features: [BS, d, T, H=14, W=14]
        """
        BS, d, T, H, W = features.shape
        Horig, Worig = self.video_hw
        O = boxes.size(2)
        output_size = (H, W)
        spatial_scale = [H/Horig, W/Worig][0]
        features = features.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [BS * T, d, H, W]
        boxes = boxes.flatten(0, 1)  # [BS * T, O, 4]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # unnormalize boxes
        boxes[...,[1,3]] = boxes[...,[1,3]] * Horig
        boxes[...,[0,2]] = boxes[...,[0,2]] * Worig
        ret = roi_align(
            features,
            list(boxes.float()),
            output_size,
            spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )  # [BS * T * O, d, H, W]
        # T = T // 2
        ret = ret.reshape(BS * T, O, d, *output_size)  # [BS * T, O, d, H, W]
        ret = ret.reshape(BS, T, O, d, *output_size)  # [BS , T, O, d, H, W]
        ret = ret.permute(0, 2, 1, 3, 4, 5).contiguous()  # [BS, O, T, d, H, W]
        return ret


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

