import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

from tqdm import tqdm
sys.path.append(os.getcwd())



def init(smpl_layer, target, device, cfg):
    params = {}
    params["pose_params"] = torch.zeros(target.shape[0], 72)
    params["shape_params"] = torch.zeros(target.shape[0], 10)
    params["scale"] = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)

    optimizer = optim.Adam([params["pose_params"], params["shape_params"], params["scale"]],
                           lr=cfg.TRAIN.LEARNING_RATE)

    index = {}
    smpl_index = []
    dataset_index = []
    for tp in cfg.DATASET.DATA_MAP:
        if not torch.any(torch.isnan(target[:, tp[1], :])): 
            smpl_index.append(tp[0])
            dataset_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)

    return smpl_layer, params, target, optimizer, index


def train(smpl_layer, target, device, args, cfg, meters):
    res = []
    smpl_layer, params, target, optimizer, index = \
        init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]

    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
    # for epoch in range(cfg.TRAIN.MAX_EPOCH):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss = F.smooth_l1_loss(Jtr.index_select(1, index["smpl_index"]) * 100,
                                target.index_select(1, index["dataset_index"]) * 100 * scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts, Jtr]
        if meters.early_stop:
            print("Early stop at epoch {} !".format(epoch))
            break

    print('Train ended, min_loss = {:.4f}'.format(
        float(meters.min_loss)))
    return res
