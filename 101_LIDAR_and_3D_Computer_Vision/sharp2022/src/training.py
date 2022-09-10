import torch
import torch.optim as optim
from torch.nn import functional as F
from pytorch_lightning import LightningModule
import numpy as np
from src.models import get_models
from pytorch_lightning.utilities.distributed import rank_zero_info
import data_processing.utils as utils
import os
import trimesh
from scipy.spatial import cKDTree as KDTree
import mcubes


def get_trainers():
    return {'TextureTrainer': TextureTrainer,
            'GeometryTrainer': GeometryTrainer,
            'GeometryTrainer_Pose': GeometryTrainer_Pose,
            'GeometryTrainer_SDF': GeometryTrainer_SDF}


def focal_loss(pred, gt, balance=False):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    # neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    if balance:
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

    else:
        num_pos = pos_inds.float().sum()
        num_neg = neg_inds.float().sum()

        pos_loss = pos_loss.sum() * (num_neg)/(num_pos+num_neg)
        neg_loss = neg_loss.sum() * (num_pos)/(num_pos+num_neg)

    loss = loss - (pos_loss + neg_loss)

    return loss

    # if num_pos == 0:
    #     loss = loss - neg_loss
    # else:
    #     loss = loss - (pos_loss + neg_loss) / num_pos


class IFNetTrainer(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = get_models()[cfg['model']]()
        # sync_batchnorm = true
        rank_zero_info(self.model)
        self.cfg = cfg
        self.loss_weight_near = self.cfg["training"]["loss"]["weight_near"]
        self.loss_weight_far = self.cfg["training"]["loss"]["weight_far"]
        self.path = cfg['data_path']
        self.n_points = cfg['n_points_score']
        self.threshold = cfg['generation']['retrieval_threshold']
        self.resolution = cfg['generation']['retrieval_resolution']
        self.bbox = cfg['data_bounding_box']
        self.min = self.bbox[::2]
        self.max = self.bbox[1::2]
        self.balanced_loss = cfg['balanced_loss']
        self.reweighted_loss = cfg['reweighted_loss']
        self.focal_loss = cfg['focal_loss']

    def configure_optimizers(self):
        optimizer = self.cfg["training"]["optimizer"]
        lr = self.cfg["training"]["lr"]
        if optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(
            ), lr=lr, momentum=0.98, weight_decay=0.000001)
        if optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(
                self.model.parameters(), momentum=0.9, lr=lr)
        if self.cfg["training"]["scheduler"]:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.98)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        loss_dict, metric_dict = self.compute_loss(batch)
        loss_dict["loss"] = sum(list(loss_dict.values()))

        loss_dict_detached = {k: v.detach() for k, v in loss_dict.items()}
        # tensorboard logging with prefix
        self.log_dict(
            {"train/" + k: v for k, v in loss_dict_detached.items()},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        self.log_dict(
            {"train/" + k: v for k, v in metric_dict.items()},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # progress bar logging without prefix
        self.log_dict(
            loss_dict_detached,
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )
        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss_dict, metric_dict = self.compute_score(batch)
        loss_dict["loss"] = sum(list(loss_dict.values()))

        loss_dict_detached = {k: v for k, v in loss_dict.items()}
        # tensorboard logging with prefix
        self.log_dict(
            {"val/" + k: v for k, v in loss_dict_detached.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        self.log_dict(
            {"val/" + k: v for k, v in metric_dict.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        loss_dict, metric_dict = self.compute_score(batch)
        loss_dict["loss"] = sum(list(loss_dict.values()))

        loss_dict_detached = {k: v for k, v in loss_dict.items()}
        # tensorboard logging with prefix
        self.log_dict(
            {"test/" + k: v for k, v in loss_dict_detached.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        self.log_dict(
            {"test/" + k: v for k, v in metric_dict.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError


class TextureTrainer(IFNetTrainer):
    def compute_loss(self, batch):
        # p = batch.get('grid_coords')
        gt_rgb = batch.get('rgb')
        # inputs = batch.get('inputs')

        # print(p[:,:3])
        pred_rgb = self.model(batch)
        pred_rgb = pred_rgb.transpose(-1, -2)
        # print(gt_rgb.shape)
        loss_i = torch.nn.L1Loss(reduction='none')(
            pred_rgb, gt_rgb)  # out = (B,num_points,3)

        # loss_i summed 3 rgb channels for all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        loss = loss_i.sum(-1).mean()

        return {"rgb_loss": loss}, {"occ_accuracy": 1}

    def compute_score(self, batch):
        # p = batch.get('grid_coords')
        gt_rgb = batch.get('rgb')
        # inputs = batch.get('inputs')

        # print(p[:,:3])
        pred_rgb = self.model(batch)
        pred_rgb = pred_rgb.transpose(-1, -2)
        # print(gt_rgb.shape)
        loss_i = torch.nn.L1Loss(reduction='none')(
            pred_rgb, gt_rgb)  # out = (B,num_points,3)

        # loss_i summed 3 rgb channels for all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        loss = loss_i.sum(-1).mean()

        return {"rgb_loss": loss.detach()}, {"occ_accuracy": 1}

    def predict_step(self, batch, batch_idx):
        inputs = batch['inputs']
        pred_mesh_path = batch['mesh_path'][0]

        pred_mesh = trimesh.load(pred_mesh_path)
        colors_pred_surface = self.generate_colors(batch)
        # attach predicted colors to the mesh
        pred_mesh.visual.vertex_colors = colors_pred_surface
        return pred_mesh

    def generate_colors(self, batch):

        grid_coords = batch['grid_coords']
        inputs = batch['inputs']
        # grid_points_split = torch.split(grid_coords, 200000, dim=1)
        grid_points_split = torch.split(
            grid_coords, self.cfg["generation"]["batch_points"], dim=1)
        full_pred = []

        for points in grid_points_split:
            with torch.no_grad():
                # hard fix to keep arguments consistent
                mini_batch = {'grid_coords': points, 'inputs': inputs,
                              }
                pred_rgb = self.model(mini_batch)
            full_pred.append(pred_rgb.squeeze(
                0).detach().cpu().transpose(0, 1))

        pred_rgb = torch.cat(full_pred, dim=0).numpy()
        pred_rgb.astype(np.int)[0]
        pred_rgb = np.clip(pred_rgb, 0, 255)

        return pred_rgb


class GeometryTrainer(IFNetTrainer):
    def compute_loss(self, batch):
        occ = batch.get('occupancies')
        dt_mask = batch.get('dt_mask')
        dt_mask = dt_mask.to(dtype=torch.bool)

        # General points
        logits = self.model(batch)

        if not self.focal_loss:
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')  # out = (B,num_points) by componentwise comparing vecots of size num_samples:
            # l(logits[n],occ[n]) for each n in B. i.e. l(logits[n],occ[n]) is vector of size num_points again.
        else:
            pred = F.sigmoid(logits)
            pred = torch.stack([1-pred, pred], dim=-1)
            labels = F.one_hot(occ.to(torch.int64), num_classes=2)
            # assume gamma = 2
            # loss_i = torch.sum(-labels * torch.log(pred) * (1-pred)**2, dim=-1)
            loss_i = focal_loss(pred, labels, self.balanced_loss)

        # weights = torch.ones_like(occ)
        # w_negative = occ.mean()
        # w_positive = 1 - w_negative

        # weights[occ < 0.5] = w_negative
        # weights[occ >= 0.5] = w_positive
        # loss_i = loss_i * weights
        # # w_class_loss = torch.mean(weights * class_loss)

        if not self.focal_loss and self.balanced_loss:
            weights = torch.ones_like(occ)
            w_negative = occ.mean()
            w_positive = 1 - w_negative
            weights[occ < 0.5] = w_negative
            weights[occ >= 0.5] = w_positive
            loss_i = loss_i * weights

        if self.reweighted_loss:
            loss_i[dt_mask] *= self.loss_weight_far
            loss_i[~dt_mask] *= self.loss_weight_near

        occ_accuracy_raw = logits >= 0
        occ_accuracy = occ_accuracy_raw.int() == occ.int()
        occ_accuracy = occ_accuracy.float().mean()

        occ_accuracy_far = occ_accuracy_raw[dt_mask].int(
        ) == occ[dt_mask].int()
        occ_accuracy_far = occ_accuracy_far.float().mean()

        occ_accuracy_near = occ_accuracy_raw[~dt_mask].int(
        ) == occ[~dt_mask].int()
        occ_accuracy_near = occ_accuracy_near.float().mean()
        # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        loss = loss_i.sum(-1).mean()
        return {"geometry_loss": loss}, {"occ_accuracy": occ_accuracy, "occ_accuracy_near": occ_accuracy_near, "occ_accuracy_far": occ_accuracy_far}

    def compute_score(self, batch):
        grid_coords = batch['grid_coords']
        # grid_points_split = torch.split(grid_coords, 200000, dim=1)
        grid_points_split = torch.split(
            grid_coords, self.cfg["generation"]["batch_points"], dim=1)
        inputs = batch['inputs']
        logits_list = []
        path = batch['path'][0]
        path = os.path.normpath(path)
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]

        normalized_path = os.path.join(
            self.path, split[:-8], gt_file_name, gt_file_name + '_normalized.obj')

        mesh_tgt = utils.as_mesh(trimesh.load(normalized_path))
        pointcloud_tgt = mesh_tgt.sample(self.n_points)

        for points in grid_points_split:
            with torch.no_grad():
                # hard fix to keep arguments consistent
                mini_batch = {'grid_coords': points, 'inputs': inputs,
                              #   'landmarks3d': batch['landmarks3d'],
                              'smpl_inputs': batch['smpl_inputs']
                              }
                logits = self.model(mini_batch)
            logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0)
        mesh = self.mesh_from_logits(logits.numpy())

        try:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
        except IndexError:
            print("mesh empty!")
            pointcloud = np.zeros((self.n_points, 3))

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, None, None)

        return {"chamfer_loss": out_dict['chamfer-L1']}, out_dict

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)),
        #                 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        # remove translation due to padding
        # vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += self.min

        return trimesh.Trimesh(vertices, triangles)

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1./1000, 1, 1000)):
        ''' Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        # # Return maximum losses if pointcloud is empty
        # if pointcloud.shape[0] == 0:
        #     logger.warn('Empty pointcloud / mesh detected!')
        #     out_dict = EMPTY_PCL_DICT.copy()
        #     if normals is not None and normals_tgt is not None:
        #         out_dict.update(EMPTY_PCL_DICT_NORMALS)
        #     return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        # completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        # precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        # accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        # normals_correctness = (
        # 0.5 * completeness_normals + 0.5 * accuracy_normals
        # )
        chamferL1 = 0.5 * (completeness + accuracy)

        # # F-Score
        # F = [
        #     2 * precision[i] * recall[i] / (precision[i] + recall[i])
        #     for i in range(len(precision))
        # ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            # 'normals completeness': completeness_normals,
            # 'normals accuracy': accuracy_normals,
            # 'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            # 'f-score': F[9],  # threshold = 1.0%
            # 'f-score-15': F[14],  # threshold = 1.5%
            # 'f-score-20': F[19],  # threshold = 2.0%
        }

        return out_dict

    def predict_step(self, batch, batch_idx):
        # for mesh generation
        grid_coords = batch['grid_coords'].clone()
        grid_points_split = torch.split(
            grid_coords, self.cfg["generation"]["batch_points"], dim=1)
        inputs = batch['inputs']
        logits_list = []

        ##
        # path = batch['path'][0]
        # path = os.path.normpath(path)
        # gt_file_name = path.split(os.sep)[-2]
        # filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]

        # file_out_path = '/itet-stor/leilil/net_scratch/if-net/3dv/experiments/IFNetGeometrySMPLGT_EarlyFusion/estimated_smpl_128_balanced_loss/geometry_reconstruction' + \
        #     '/{}/'.format(gt_file_name)

        # if os.path.exists(file_out_path + f'{filename_partial}_reconstruction.obj'):
        #     print('Path exists - skip! {}'.format(file_out_path))
        #     return

        for points in grid_points_split:
            with torch.no_grad():
                batch['grid_coords'] = points
                logits = self.model(batch)
            logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0)

        return logits.numpy()


class GeometryTrainer_SDF(IFNetTrainer):
    def compute_loss(self, batch):
        sdf = batch.get('sdf')
        dt_mask = batch.get('dt_mask')
        dt_mask = dt_mask.to(dtype=torch.bool)
        # General points
        pred = self.model(batch)

        loss_i = F.l1_loss(
            pred, sdf, reduction='none')  # out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # l(logits[n],occ[n]) for each n in B. i.e. l(logits[n],occ[n]) is vector of size num_points again.

        pred_accuracy_raw = pred >= 0
        sdf_accuracy_raw = sdf >= 0
        occ_accuracy = pred_accuracy_raw.int() == sdf_accuracy_raw.int()
        occ_accuracy = occ_accuracy.float().mean()

        occ_accuracy_far = pred_accuracy_raw[dt_mask].int(
        ) == sdf_accuracy_raw[dt_mask].int()
        occ_accuracy_far = occ_accuracy_far.float().mean()

        occ_accuracy_near = pred_accuracy_raw[~dt_mask].int(
        ) == sdf_accuracy_raw[~dt_mask].int()
        occ_accuracy_near = occ_accuracy_near.float().mean()
        # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        loss = loss_i.sum(-1).mean()
        return {"geometry_loss": loss}, {"occ_accuracy": occ_accuracy, "occ_accuracy_near": occ_accuracy_near, "occ_accuracy_far": occ_accuracy_far}

    def predict_step(self, batch, batch_idx):
        # for mesh generation
        grid_coords = batch['grid_coords'].clone()
        grid_points_split = torch.split(grid_coords, 200000, dim=1)
        inputs = batch['inputs']
        pred_list = []
        for points in grid_points_split:
            with torch.no_grad():
                batch['grid_coords'] = points
                pred = self.model(batch)
            pred_list.append(pred.squeeze(0).detach().cpu())

        pred = torch.cat(pred_list, dim=0)

        return pred.numpy()


class GeometryTrainer_Pose(IFNetTrainer):
    def compute_loss(self, batch):
        p = batch.get('grid_coords')
        occ = batch.get('occupancies')
        inputs = batch.get('inputs')
        pose_gt = batch.get('landmarks3d')

        # General points
        pose_pred = self.model(p, inputs)
        # loss_p = mse(pose_pred, pose_gt)  # MSE loss
        loss_p = torch.norm(pose_gt-pose_pred, p=1,
                            dim=[-1, -2]).mean()  # L1 loss
        # loss_i = F.binary_cross_entropy_with_logits(
        # logits, occ, reduction='none')# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # l(logits[n],occ[n]) for each n in B. i.e. l(logits[n],occ[n]) is vector of size num_points again.

        # loss_i = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        # occ_accuracy = logits >=0
        # occ_accuracy = occ_accuracy.int() == occ.int()
        # occ_accuracy = occ_accuracy.float().mean()

        # TODO: loss weight

        return {"pose_loss": loss_p}, {"pose_loss": loss_p}

    def predict_step(self, batch, batch_idx):
        # for mesh generation
        p = batch['grid_coords']
        # grid_points_split = torch.split(grid_coords, 200000, dim=1)
        inputs = batch['inputs']
        # logits_list = []
        # for points in grid_points_split:
        with torch.no_grad():
            pose = self.model(p, inputs)
        pose = pose.squeeze(0).detach().cpu()

        return pose.numpy()


class GeometryTrainer_PoseGT_Full(IFNetTrainer):
    def compute_loss(self, batch):
        p = batch.get('grid_coords')
        occ = batch.get('occupancies')
        inputs = batch.get('inputs')
        inputs_full = batch.get('inputs_full')
        pose_gt = batch.get('landmarks3d')

        # General points
        logits = self.model(p, inputs, inputs_full, pose_gt)

        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')  # out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # l(logits[n],occ[n]) for each n in B. i.e. l(logits[n],occ[n]) is vector of size num_points again.

        occ_accuracy = logits >= 0
        occ_accuracy = occ_accuracy.int() == occ.int()
        occ_accuracy = occ_accuracy.float().mean()

        # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        loss = loss_i.sum(-1).mean()
        return {"geometry_loss": loss, "occ_accuracy": occ_accuracy}

    def predict_step(self, batch, batch_idx):
        # for mesh generation
        grid_coords = batch['grid_coords']
        grid_points_split = torch.split(grid_coords, 200000, dim=1)
        inputs = batch['inputs']
        inputs_full = batch.get('inputs_full')
        pose_gt = batch.get('landmarks3d')
        logits_list = []
        for points in grid_points_split:
            with torch.no_grad():
                logits = self.model(points, inputs, inputs_full, pose_gt)
            logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0)

        return logits.numpy()


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.
    Args:
        points (numpy array): points array
        mesh (trimesh): mesh
    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold
