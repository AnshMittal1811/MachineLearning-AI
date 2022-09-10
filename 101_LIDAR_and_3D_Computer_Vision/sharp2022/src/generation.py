import mcubes
import trimesh
import os
import numpy as np
from pytorch_lightning.callbacks import Callback
from data_processing.utils import form_grid_to_original_coords


class TextureGenerationCallback(Callback):
    def __init__(self, cfg, output_dir):
        self.out_path = os.path.join(output_dir, "texture_reconstruction")
        os.makedirs(self.out_path, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        path = batch['path'][0]
        path = os.path.normpath(path)
        gt_file_name = path.split(os.sep)[-2]
        filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]
        file_out_path = self.out_path + '/{}/'.format(gt_file_name)

        os.makedirs(file_out_path, exist_ok=True)
        if os.path.exists(file_out_path + f'{gt_file_name}-completed.obj'):
            return
        pred_mesh = outputs
        pred_mesh.export(
            file_out_path + f'{gt_file_name}-completed.obj')


class GeometryGenerationCallback(Callback):
    def __init__(self, cfg, output_dir):
        self.threshold = cfg['generation']['retrieval_threshold']
        self.resolution = cfg['generation']['retrieval_resolution']
        self.bbox = cfg['data_bounding_box']
        self.min = self.bbox[::2]
        self.max = self.bbox[1::2]
        self.out_path = os.path.join(output_dir, "geometry_reconstruction")
        os.makedirs(self.out_path, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Hook for on_predict_batch_end."""
        logits = outputs
        path = batch['path'][0]
        path = os.path.normpath(path)
        gt_file_name = path.split(os.sep)[-2]
        filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]

        file_out_path = self.out_path + '/{}/'.format(gt_file_name)

        if os.path.exists(file_out_path + f'{filename_partial}_reconstruction.obj'):
            print('Path exists - skip! {}'.format(file_out_path))
            return

        mesh = self.mesh_from_logits(logits)

        mesh = self.mesh_from_logits(logits)
        if not os.path.exists(file_out_path):
            os.makedirs(file_out_path)

        mesh.export(file_out_path + f'{filename_partial}_reconstruction.obj')

        # if os.path.exists(file_out_path + f'{gt_file_name}-completed.obj'):
        #     print('Path exists - skip! {}'.format(gt_file_name))
        #     return

        # mesh = self.mesh_from_logits(logits)

        # if not os.path.exists(file_out_path):
        #     os.makedirs(file_out_path)

        # mesh.export(file_out_path + f'{gt_file_name}-completed.obj')

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)),
        #                 'constant', constant_values=1)
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


class PoseGenerationCallback(Callback):
    def __init__(self, cfg, output_dir):
        self.threshold = cfg['generation']['retrieval_threshold']
        self.resolution = cfg['generation']['retrieval_resolution']
        self.bbox = cfg['data_bounding_box']
        self.min = self.bbox[::2]
        self.max = self.bbox[1::2]
        self.out_path = os.path.join(output_dir, "pose_estimation")
        os.makedirs(self.out_path, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Hook for on_predict_batch_end."""
        pose_predict = form_grid_to_original_coords(outputs, self.bbox)
        path = batch['path'][0]
        path = os.path.normpath(path)
        gt_file_name = path.split(os.sep)[-2]
        filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]

        file_out_path = self.out_path + '/{}/'.format(gt_file_name)

        if os.path.exists(file_out_path + f'{filename_partial}_pose.npy'):
            print('Path exists - skip! {}'.format(file_out_path))
            return

        if not os.path.exists(file_out_path):
            os.makedirs(file_out_path)

        np.save(file_out_path + f'{filename_partial}_pose', pose_predict)
