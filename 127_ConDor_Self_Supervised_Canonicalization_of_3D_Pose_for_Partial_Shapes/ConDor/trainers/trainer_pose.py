import tensorflow as tf
from data_providers.provider import Provider, load_dataset
import os
import time
from datetime import datetime
from utils.data_prep_utils import save_h5_data, save_h5_upsampling, compute_centroids, tf_random_rotate, tf_center, tf_dist, var_normalize, registration, tf_random_dir, partial_shapes
from data_providers.classification_datasets import datsets_list
from utils.losses import hausdorff_distance_l1, hausdorff_distance_l2, chamfer_distance_l1, chamfer_distance_l2, orthogonality_loss, tf_directional_loss, l2_loss_, equilibrium_loss, localization_loss, localization_loss_new
from utils.losses import sq_distance_mat
#from auto_encoder.tfn_capsules import TFN
from network_utils.pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import logging
import open3d as o3d
from utils.vis_utils import save_pointcloud, distribute_capsules






class TrainerPose:
    def __init__(self, config_params, model, optimizer, train_set = None, val_set = None, test_set = None):

        self.config_params = config_params
        self.model = model
        self.optimizer = optimizer
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.min_loss = None
        self.configure_logger()
        logging.info(self.config_params)
        
    def train(self):
        '''
        Begin training
        '''
        
        total_iters = 0
        total_iters_val = 0
        for epoch in range(self.config_params.trainer.max_epochs):
            
            self.train_set.on_epoch_end()
            start = time.time()
            print('epoch: ', epoch)
            iters_per_epoch = 0
            loss_dict_epoch = {}

            for x in self.train_set:
                
                # Perform forward pass and obtain loss dictionary
                loss_dict = self.training_step(x)
                
                # Aggregate losses
                for loss_key in loss_dict:
                    if loss_dict_epoch.get(loss_key) is None:
                        loss_dict_epoch[loss_key] = 0.0   
                    # print(loss_key, loss_dict[loss_key])                 
                    loss_dict_epoch[loss_key] += float(loss_dict[loss_key])
                    self.log(loss_key, float(loss_dict[loss_key]), step = total_iters)
                
                # Incrementing iterations
                iters_per_epoch += 1
                total_iters += 1
            logging.info("\nTrain metrics")

            # Averaging loss values
            for loss_key in loss_dict_epoch:
                loss_dict_epoch[loss_key] /= iters_per_epoch
                # print(loss_key, " ", loss_dict_epoch[loss_key], " ")
                self.log("epoch_" + loss_key, loss_dict_epoch[loss_key], step = epoch)
                logging.info("epoch_" + loss_key + " " + str(loss_dict_epoch[loss_key]) + " epoch " + str(epoch))

            if self.val_set is not None:
                
                self.val_set.on_epoch_end()
                start = time.time()
                iters_per_epoch_val = 0
                loss_val_dict_epoch =  {}
                for x in self.val_set:
                    
                    # Perform forward pass and obtain loss dictionary
                    loss_dict = self.validation_step(x)
                    
                    # Aggregate losses
                    for loss_key in loss_dict:
                        if loss_val_dict_epoch.get("val_" + loss_key) is None:
                            loss_val_dict_epoch["val_" + loss_key] = 0.0                    
                        loss_val_dict_epoch["val_" + loss_key] += float(loss_dict[loss_key])
                        self.log("val_" + loss_key, float(loss_val_dict_epoch["val_" + loss_key]), step = total_iters_val)

                    # Incrementing iterations
                    iters_per_epoch_val += 1
                    total_iters_val +=1
            
                # Averaging loss values
                logging.info("\nValidation metrics")
                for loss_key in loss_val_dict_epoch:
                    loss_val_dict_epoch[loss_key] /= iters_per_epoch_val
                    # print(loss_key, " ", loss_val_dict_epoch[loss_key], " ")
                    self.log("epoch_" + loss_key, loss_val_dict_epoch[loss_key], step = epoch)
                    logging.info("epoch_" + loss_key + " " + str(loss_val_dict_epoch[loss_key]) + " epoch " + str(epoch))

                
                print("\n")
                print(' time: ', time.time()-start)
                if self.min_loss is None:
                    self.min_loss = loss_val_dict_epoch["val_loss"]
                    self.save_model()
                
                elif loss_val_dict_epoch["val_loss"] < self.min_loss:
                    # print(loss_val_dict_epoch["val_loss"])
                    self.min_loss = loss_val_dict_epoch["val_loss"]
                    self.save_model()
    @tf.function
    def forward_pass(self, x, val_step = False):
        '''
        Perform a forward pass
        '''

        x = tf_center(x)
        x = tf_random_rotate(x)
        x = kdtree_indexing(x)
        # yzx = tf_center(tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1))
        eps = self.config_params.utils.eps
        x_partial, x_partial_idx, kd_idx_inv = partial_shapes(x)
        x_restriction = tf.gather_nd(x, x_partial_idx)
        x_restriction, center_x = tf_center(x_restriction, return_center = True)
        x_partial_input_centered, center_x_partial = tf_center(x_partial, return_center = True)
        
        with tf.GradientTape() as tape:
            
            if val_step:
                caps, inv, basis = self.model(x, training=False)
            else:
                caps, inv, basis = self.model(x, training=True)

            x_partial_input = x_partial

            if val_step:
                p_caps, p_inv, p_basis = self.model(x_partial_input_centered, training=False)
            else:
                p_caps, p_inv, p_basis = self.model(x_partial_input_centered, training=True)


            basis_last = tf.linalg.cross(basis[:, 0], basis[:, 1])
            p_basis_last = tf.linalg.cross(p_basis[:, 0], p_basis[:, 1])


            basis = tf.stack([basis[:, 0], basis[:, 1], basis_last], axis = -1)
            p_basis = tf.stack([p_basis[:, 0], p_basis[:, 1], p_basis_last], axis = -1)

            basis = tf.linalg.l2_normalize(basis, epsilon=1e-4, axis=-2)
            p_basis = tf.linalg.l2_normalize(p_basis, epsilon=1e-4, axis=-2)
            
            
            p_caps = tf.gather_nd(p_caps, kd_idx_inv)
            p_inv = tf.gather_nd(p_inv, kd_idx_inv)
            x_partial_input_centered_inv = tf.gather_nd(x_partial_input_centered, kd_idx_inv)


            pp_caps = tf.gather_nd(caps, x_partial_idx) # B, N, K
            pp_inv = tf_center(tf.gather_nd(inv, x_partial_idx))

            inv_partiality_loss = l2_loss_(p_inv, pp_inv)
            caps_partiality_loss = tf.reduce_mean(tf.keras.losses.cosine_similarity(pp_caps, p_caps, axis = -1))
            
            caps_partial_sum = tf.reduce_sum(p_caps, axis = 1, keepdims = True)
            caps_sum = tf.reduce_sum(caps, axis=1, keepdims=True)
            caps_sum_restriction = tf.reduce_sum(pp_caps, axis=1, keepdims=True)


            # Encouraging more spaced capsules
            caps_max = tf.reduce_max(caps, axis=-1, keepdims = True)
            categorical_predictions = tf.cast((caps == caps_max), tf.float32)
            caps_spatial_loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(tf.stop_gradient( categorical_predictions), caps))

            # Normalizing capsules
            normalized_caps_partial = tf.divide(p_caps, caps_partial_sum + eps)
            normalized_caps = tf.divide(caps, caps_sum + eps)
            normalized_caps_partial_restriction = tf.divide(pp_caps, caps_sum_restriction + eps)

            centroids_partial = compute_centroids(x_partial, normalized_caps_partial)
            centroids = compute_centroids(x, normalized_caps) # B, K, 3
            
            centroid_partial_inv = compute_centroids(x_partial_input_centered_inv, normalized_caps_partial)
            centroid_full_inv = compute_centroids(x_restriction, normalized_caps_partial_restriction)

            capsule_position_loss = l2_loss_(centroid_partial_inv, centroid_full_inv)

           
            directional_loss_partial = tf_directional_loss(centroid_full_inv, centroid_partial_inv)


            s_p, u_p, v_p = tf.linalg.svd(p_basis)
            orth_p_basis = tf.matmul(u_p, v_p, transpose_b=True)
            orth_p_loss = tf.reduce_mean(tf.abs(p_basis - tf.stop_gradient(orth_p_basis)))

            s, u, v = tf.linalg.svd(basis)
            orth_basis = tf.matmul(u, v, transpose_b=True)
            orth_loss = tf.reduce_mean(tf.abs(basis - tf.stop_gradient(orth_basis))) + orth_p_loss


            y_p = tf.einsum('bvj,bmj->bvm', p_inv, orth_p_basis)
            y_p = tf.stack([y_p[..., 2], y_p[..., 0], y_p[..., 1]], axis=-1)

            y = tf.einsum('bvj,bmj->bvm', inv, orth_basis)
            y = tf.stack([y[..., 2], y[..., 0], y[..., 1]], axis=-1)


            # directional_loss_partial = tf_directional_loss(centroids, tf.add(centroids_partial, center_x), tf.stop_gradient(caps_sum), tf.stop_gradient(caps_partial_sum))

            # print(directional_loss_partial)
            eq_loss  = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x, centroids) #+ 10* chamfer_distance_l2(x_restriction, centroids_partial)



            l2_loss = x - y
            l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
            mean_root_square = l2_loss
            l2_loss = tf.sqrt(l2_loss + 0.000000001)
            l2_loss = tf.reduce_mean(l2_loss) + tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.multiply(x_restriction - y_p, x_restriction - y_p), axis = -1, keepdims=False) + 1e-8))
            l2_loss = l2_loss
            mean_root_square = tf.reduce_sum(mean_root_square, axis=1, keepdims=False)
            mean_root_square = tf.sqrt(mean_root_square + 0.000000001) / x.shape[1]
            mean_root_square = tf.reduce_mean(mean_root_square)

            chamfer_loss = chamfer_distance_l2(y, x) + chamfer_distance_l2(y_p, x_restriction)
            hausdorff_loss = hausdorff_distance_l2(y, x)

            s_p_shape_actual = tf.linalg.svd(x_restriction, compute_uv = False)
            s_p_shape_prediction = tf.linalg.svd(y_p, compute_uv = False)

            s_shape_actual = tf.linalg.svd(x, compute_uv=False) # B, 3
            s_shape_prediction = tf.linalg.svd(y, compute_uv = False) #B, 3


            loss_shape_svd = tf.reduce_mean(tf.abs(s_shape_actual - s_shape_prediction)) + tf.reduce_mean(tf.abs(s_p_shape_actual - s_p_shape_prediction))
            


            loss = self.config_params.loss.l2_loss * l2_loss + \
            self.config_params.loss.eq_loss * eq_loss + \
            self.config_params.loss.loc_loss * loc_loss + \
            self.config_params.loss.caps_chamf_loss * caps_chamf_loss + \
            self.config_params.loss.orth_loss + orth_loss + \
            self.config_params.loss.caps_partiality_loss * caps_partiality_loss + \
            self.config_params.loss.directional_loss_partial * directional_loss_partial + \
            self.config_params.loss.capsule_position_loss * capsule_position_loss + \
            self.config_params.loss.caps_spatial_loss * caps_spatial_loss + \
            self.config_params.loss.inv_partiality_loss * inv_partiality_loss + \
            self.config_params.loss.chamfer_loss * chamfer_loss + \
            self.config_params.loss.hausdorff_loss * hausdorff_loss + \
            self.config_params.loss.loss_shape_svd * loss_shape_svd
            if not val_step:
                # print ("backward pass")
                grad = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))    


        loss_dict = {"loss": loss, "l2_loss": l2_loss, "eq_loss": eq_loss, "caps_chamf_loss": caps_chamf_loss, 
                    "orth_loss": orth_loss, "chamfer_loss": chamfer_loss, "mean_root_square": mean_root_square, 
                    "hausdorff_loss": hausdorff_loss, "loc_loss": loc_loss, "caps_partiality_loss": caps_partiality_loss,
                    "inv_partiality_loss": inv_partiality_loss,
                    "directional_loss": directional_loss_partial,
                    "caps_position_loss": capsule_position_loss,
                    "caps_spatial_category_loss": caps_spatial_loss
                    }
        return loss_dict


    def configure_logger(self):
        '''
        Configure logger
        '''
        log_dir = './logs/graphs/'
        logging.basicConfig(filename='./logs/train.log', encoding='utf-8',level=logging.DEBUG)
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        
    def log(self, loss_key, loss_value, step):
        '''
        Log the losses
        '''
        with self.summary_writer.as_default():
            tf.summary.scalar(loss_key, loss_value, step=step)


    def training_step(self, x):
        '''
        Perform the training step
        '''
        
        loss_dict = self.forward_pass(x)

        return loss_dict

    def validation_step(self, x):
        '''
        Perform validation step
        '''

        loss_dict = self.forward_pass(x, val_step=True)
        return loss_dict

    def test(self, test_set, output_dir):
        '''
        Perform test pass
        '''

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_iters = 0
        test_set.on_epoch_end()
        start = time.time()
        iters_per_epoch = 0
        loss_dict_epoch = {}
        
        for x in test_set:
            
            # Perform forward pass and obtain loss dictionary
            loss_dict = self.validation_step(x)
            self.test_pass(x, os.path.join(output_dir, str(iters_per_epoch)))
            # Aggregate losses
            for loss_key in loss_dict:
                if loss_dict_epoch.get(loss_key) is None:
                    loss_dict_epoch[loss_key] = 0.0   
                loss_dict_epoch[loss_key] += float(loss_dict[loss_key])
            
            # Incrementing iterations
            iters_per_epoch += 1
            total_iters += 1
            if total_iters > 50:
                break
        print("\nTest metrics")

        # Averaging loss values
        for loss_key in loss_dict_epoch:
            loss_dict_epoch[loss_key] /= iters_per_epoch
            # print(loss_key, " ", loss_dict_epoch[loss_key], " ")
            logging.info("epoch_test_" + loss_key + " " + str(loss_dict_epoch[loss_key]))

    def test_pass(self, x, save_file_name):
        '''
        Perform test pass
        x - [1, N, 3]
        '''
        x = tf_center(x)
        x = tf_random_rotate(x)
        x = kdtree_indexing(x)

        eps = self.config_params.utils.eps
        x_partial, x_partial_idx, kd_idx_inv = partial_shapes(x)
        x_partial_input = tf_center(x_partial)
        x_restriction = tf.gather_nd(x, x_partial_idx)
        x_restriction, center_x = tf_center(x_restriction, return_center = True)
        
        caps, inv, basis = self.model(x, training=False)
        p_caps, p_inv, p_basis = self.model(x_partial_input, training = False)

        s, u, v = tf.linalg.svd(p_basis)
        orth_p_basis = tf.matmul(u, v, transpose_b=True)
        center_x_can = tf.einsum('bvj,bmj->bvm', center_x, orth_p_basis)
        center_x_can = tf.stack([center_x_can[..., 2], center_x_can[..., 0], center_x_can[..., 1]], axis=-1)

        # Summing capsules
        caps_partial_sum = tf.reduce_sum(p_caps, axis = 1, keepdims = True)
        caps_sum = tf.reduce_sum(caps, axis=1, keepdims=True)

        # Normalizing capsules
        normalized_caps_partial = tf.divide(p_caps, caps_partial_sum + eps)
        normalized_caps = tf.divide(caps, caps_sum + eps)

        centroids_partial = compute_centroids(tf.add(x_partial_input, center_x), normalized_caps_partial)
        centroids = compute_centroids(x, normalized_caps)


        pcd_caps = distribute_capsules(inv, caps)
        pcd_caps_partial = distribute_capsules(tf.add(p_inv, center_x_can), p_caps)

        # Saving pointclouds
        save_pointcloud(centroids_partial, save_file_name + "_partial_centroids_inv.ply")
        save_pointcloud(centroids, save_file_name + "_full_centroids_inv.ply")

        save_pointcloud(inv, save_file_name + "_full_inv.ply")
        save_pointcloud(tf.add(p_inv, center_x_can), save_file_name + "_partial_inv.ply")

        save_pointcloud(pcd_caps, save_file_name + "_full_inv_splits.ply")
        save_pointcloud(pcd_caps_partial, save_file_name + "_partial_inv_splits.ply")

        save_pointcloud(x, save_file_name + "_input_full.ply")
        save_pointcloud(x_partial, save_file_name + "_input_partial.ply")


    def save_model(self):
        '''
        Saves model
        '''

        weights_path = os.path.join(os.getcwd(), self.config_params.save.path)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        save_path = os.path.join(weights_path, 'weights_model.h5')
        self.model.save_weights(save_path)
        logging.info("Saved model to path: " + save_path)

# inputs = tf.keras.layers.Input(shape=(None, 3), batch_size = BATCH_SIZE, dtype=tf.float64)
# autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs))
# optimizer = tf.keras.optimizers.Adam(1e-4)


if __name__=="__main__":
    pass
