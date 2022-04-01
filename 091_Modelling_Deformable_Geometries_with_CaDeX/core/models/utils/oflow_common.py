# some utils for oflow

import numpy as np


def eval_iou(occ_gt, occ_pred, threshold):
    assert occ_gt.shape == occ_pred.shape
    batch_size, n_steps, n_pts = occ_pred.shape
    occ_pred = occ_pred > threshold
    occ_gt = occ_gt >= 0.5
    iou = compute_iou(occ_pred.reshape(-1, n_pts), occ_gt.reshape(-1, n_pts))
    iou = iou.reshape(batch_size, n_steps)
    return iou  # eval_dict is for running tensorboard


def compute_iou(occ1, occ2):
    """Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = area_intersect / area_union

    return iou


def eval_oflow_all(
    pcl_tgt, points_tgt, occ_tgt, mesh_t_list, evaluator, corr_project_to_final_mesh, eval_corr=True
):
    # pcl_tgt/points_tgt T, 100000, 3, occ_tgt T, 1000000
    eval_dict_mean, eval_dict_t = {}, {}
    eval_dict_mesh = {}
    T = pcl_tgt.shape[0]
    # eval IOU and CD
    for t, mesh in enumerate(mesh_t_list):
        # print(t)
        _eval_dict_mesh = evaluator.eval_mesh(mesh, pcl_tgt[t], None, points_tgt[t], occ_tgt[t])
        for k, v in _eval_dict_mesh.items():
            # ! Modify here 2021.10.5, skip the normal metrics, to avoid ignoring nan in other metrics
            if k.startswith("normal"):
                continue
            if not np.isnan(v):
                if k not in eval_dict_mesh.keys():
                    eval_dict_mesh[k] = [v]
                else:
                    eval_dict_mesh[k].append(v)
            else:
                raise ValueError("Evaluator meets nan")
    for k, v in eval_dict_mesh.items():
        mean_v = np.array(v).mean()
        eval_dict_mean["{}".format(k)] = mean_v
        for t in range(T):
            eval_dict_t["{}_t{}".format(k, t)] = v[t]
    if eval_corr:
        # eval correspondence
        eval_dict_corr = evaluator.eval_correspondences_mesh(
            mesh_t_list,
            pcl_tgt,
            project_to_final_mesh=corr_project_to_final_mesh,
        )
        corr_list = []
        for k, v in eval_dict_corr.items():
            t = int(k.split(" ")[1])
            eval_dict_t["corr_l2_t%d" % t] = v
            corr_list.append(v)
        eval_dict_mean["corr_l2"] = np.array(corr_list).mean()
    return eval_dict_mean, eval_dict_t


def eval_atc_all(
    pcl_corr,
    pcl_chamfer,
    points_tgt,
    occ_tgt,
    mesh_t_list,
    evaluator,
    corr_project_to_final_mesh,
    eval_corr=True,
):
    # pcl_tgt/points_tgt T, 100000, 3, occ_tgt T, 1000000
    eval_dict_mean, eval_dict_t = {}, {}
    eval_dict_mesh = {}
    T = pcl_corr.shape[0]
    # eval IOU and CD
    for t, mesh in enumerate(mesh_t_list):
        # print(t)
        _eval_dict_mesh = evaluator.eval_mesh(mesh, pcl_chamfer[t], None, points_tgt[t], occ_tgt[t])
        for k, v in _eval_dict_mesh.items():
            # ! Modify here 2021.10.5, skip the normal metrics, to avoid ignoring nan in other metrics
            if k.startswith("normal"):
                continue
            if not np.isnan(v):
                if k not in eval_dict_mesh.keys():
                    eval_dict_mesh[k] = [v]
                else:
                    eval_dict_mesh[k].append(v)
            else:
                raise ValueError("Evaluator meets nan")
    for k, v in eval_dict_mesh.items():
        mean_v = np.array(v).mean()
        eval_dict_mean["{}".format(k)] = mean_v
        for t in range(T):
            eval_dict_t["{}_t{}".format(k, t)] = v[t]
    if eval_corr:
        # eval correspondence
        eval_dict_corr = evaluator.eval_correspondences_mesh(
            mesh_t_list,
            pcl_corr,
            project_to_final_mesh=corr_project_to_final_mesh,
        )
        corr_list = []
        for k, v in eval_dict_corr.items():
            t = int(k.split(" ")[1])
            eval_dict_t["corr_l2_t%d" % t] = v
            corr_list.append(v)
        eval_dict_mean["corr_l2"] = np.array(corr_list).mean()
    return eval_dict_mean, eval_dict_t
