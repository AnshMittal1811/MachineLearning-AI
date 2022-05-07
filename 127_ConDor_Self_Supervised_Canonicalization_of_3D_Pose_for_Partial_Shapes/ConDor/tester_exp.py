import tensorflow as tf
from data_providers.provider import Provider, load_dataset
import os
import time
from datetime import datetime
from utils.data_prep_utils import save_h5_data, save_h5_upsampling, compute_centroids, tf_random_rotate, tf_center, tf_dist, var_normalize, registration, tf_random_dir, partial_shapes
from data_providers.classification_datasets import get_dataset_files
from utils.losses import hausdorff_distance_l1, hausdorff_distance_l2, chamfer_distance_l1, chamfer_distance_l2, orthogonality_loss, tf_directional_loss
from utils.losses import sq_distance_mat
from auto_encoder.tfn_capsules import TFN
from auto_encoder.tfn_capsules_multi_frame import TFN_multi
from auto_encoder.tfn_capsules_multi_frame_scale_translation import TFN_all
#from auto_encoder.condor_partial import TFN_all
from network_utils.pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import logging, hydra
from utils.vis_utils import save_pointcloud
from trainers.trainer_siamese import Trainer
from trainers.trainer_multi_frame import TrainerMulti
import trainers, auto_encoder

@hydra.main(config_path="cfgs", config_name="config_capsules_exp.yaml")
def run(cfg):
    
    if cfg.model.weights != False:
        weights_path = os.path.join(hydra.utils.get_original_cwd(), cfg.model.weights)
    else:
        weights_path = False
        pass
        #weights_path = "/gpfs/scratch/rsajnani/rsajnani/research/clean/EquiNet/TFN-capsules/outputs/2021-10-06/17-46-07/checkpoints/weights_model.h5"
    cfg.dataset.batch_size = 1
    print("using weights:", weights_path)
    print(os.path.realpath(__file__), os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path))
    dataset_dict = get_dataset_files(os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path))
    train_provider, val_provider, test_provider = load_dataset(dataset_dict, cfg.dataset.batch_size, cfg.dataset.num_points, cfg.dataset.shuffle, cfg.dataset.train_list, cfg.dataset.val_list, cfg.dataset.test_list)
    inputs = tf.keras.layers.Input(shape=(None, 3), batch_size = cfg.dataset.batch_size, dtype=tf.float64)
    
    #inputs = tf.keras.layers.Input(batch_shape=(None,None, 3))
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN_all(num_frames=5, num_capsules=10, num_classes=1)(inputs))
    autoencoder.load_weights(weights_path) 
    #autoencoder = tf.keras.models.Model(inputs=inputs,
    #    outputs=getattr(getattr(auto_encoder, cfg.model.file), cfg.model.type)(**cfg.model.args)(inputs))
    #autoencoder.load_weights(weights_path)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    trainer = getattr(getattr(trainers, cfg.trainer.file), cfg.trainer.type)(cfg, autoencoder, optimizer, train_set = train_provider, val_set = val_provider, test_set = test_provider)
    trainer.test(test_provider, "./pointclouds", max_iters = cfg.test.max_iter)

if __name__ == '__main__':

    run()
