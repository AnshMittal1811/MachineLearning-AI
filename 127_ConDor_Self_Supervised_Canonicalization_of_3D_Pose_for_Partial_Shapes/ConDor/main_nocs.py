import tensorflow as tf
from data_providers.provider import Provider, load_dataset
from data_providers.NOCS_provider import NOCSProvider
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
from network_utils.pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import logging, hydra
import trainers
from utils.helper_functions import apply_regularizer, step_scheduler
from functools import partial
import auto_encoder


@hydra.main(config_path="cfgs", config_name="config_nocs.yaml")
def run(cfg):
    seed_value = cfg.trainer.seed
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    print(os.path.realpath(__file__), os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path))
    
    dataset_path = os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path)
    
    
    train_provider = NOCSProvider(os.path.join(dataset_path, cfg.dataset.train_list[0]), batch_size = cfg.dataset.batch_size)
    
    val_provider =   NOCSProvider(os.path.join(dataset_path, cfg.dataset.val_list[0]), batch_size = cfg.dataset.batch_size, shuffle=False)
    
    test_provider =  NOCSProvider(os.path.join(dataset_path, cfg.dataset.test_list[0]), batch_size = 1, shuffle=False)

    
    inputs = tf.keras.layers.Input(shape=(None, 3), batch_size = cfg.dataset.batch_size, dtype=tf.float64)
    
    model = getattr(getattr(auto_encoder, cfg.model["file"]), cfg.model["type"])(**cfg.model.args)
    if cfg.trainer.regularizer.weight >= 0:
        print(model.layers)
        print("#"*100)
        apply_regularizer(model, cfg.trainer.regularizer.layer, cfg.trainer.regularizer.type, cfg.trainer.regularizer.weight)
    
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=model(inputs))
    optimizer = getattr(tf.keras.optimizers, cfg.trainer.optimizer.type)(**cfg.trainer.optimizer.args)
    scheduler_callback = partial(step_scheduler, cfg.trainer.scheduler.steps, cfg.trainer.scheduler.decay_rate)
    trainer = getattr(getattr(trainers, cfg.trainer.file), cfg.trainer.type)(cfg, autoencoder, optimizer, train_set = train_provider, val_set = val_provider, test_set = test_provider, scheduler = scheduler_callback)
    trainer.train()

if __name__ == '__main__':

    run()
