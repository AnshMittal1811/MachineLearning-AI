import tensorflow as tf
from data_providers.provider import Provider, load_dataset
import os
import time
from datetime import datetime
from utils.data_prep_utils import save_h5_data, save_h5_upsampling, compute_centroids, tf_random_rotate, tf_center, tf_dist, var_normalize, registration, tf_random_dir, partial_shapes
from data_providers.classification_datasets import get_dataset_files
from utils.losses import hausdorff_distance_l1, hausdorff_distance_l2, chamfer_distance_l1, chamfer_distance_l2, orthogonality_loss, tf_directional_loss
from utils.losses import sq_distance_mat
from auto_encoder.tfn_capsules_multi_frame_scale_translation import TFN_all as TFN
from network_utils.pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import logging, hydra
from utils.vis_utils import save_pointcloud
# from trainers.trainer_siamese import Trainer
from data_providers.NOCS_provider import NOCSProvider
from trainers.trainer_nocs import TrainerNOCS as Trainer

@hydra.main(config_path="cfgs", config_name="config_nocs.yaml")
def run(cfg):
    
    
    weights_path = cfg.model.weights
    max_iter = cfg.test.max_iter
    
    print(weights_path, max_iter)
    cfg.dataset.batch_size = 1
    print(os.path.realpath(__file__), os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path))
    
    dataset_path = os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path)
    
    train_provider = NOCSProvider(os.path.join(dataset_path, cfg.dataset.train_list[0]), batch_size = 1)
    val_provider =   NOCSProvider(os.path.join(dataset_path, cfg.dataset.val_list[0]), batch_size = 1, shuffle=False)
    test_provider =  NOCSProvider(os.path.join(dataset_path, cfg.dataset.test_list[0]), batch_size = 1, shuffle=False)

    print("#"* 1000)
    inputs = tf.keras.layers.Input(shape=(None, 3), batch_size = cfg.dataset.batch_size, dtype=tf.float64)
    print("#"* 1000, "model")
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(**cfg.model.args)(inputs))
    print("#"* 1000, "model")
    autoencoder.load_weights(weights_path)
    print("#"* 1000, "weights")
    optimizer = tf.keras.optimizers.Adam(cfg.trainer.optimizer.args.lr)
    print("#"* 1000)
    trainer = Trainer(cfg, autoencoder, optimizer, train_set = train_provider, val_set = val_provider, test_set = test_provider) 
    trainer.test(test_provider, "./pointclouds", max_iters = max_iter)

if __name__ == '__main__':

    run()
