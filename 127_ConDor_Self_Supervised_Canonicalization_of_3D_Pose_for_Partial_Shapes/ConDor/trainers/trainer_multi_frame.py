import tensorflow as tf
import os, sys
sys.path.append("../")
from utils.losses import hausdorff_distance_l2, chamfer_distance_l2, tf_directional_loss, l2_loss_, equilibrium_loss, localization_loss_new
from utils.losses import sq_distance_mat

from utils.data_prep_utils import kdtree_indexing, kdtree_indexing_, compute_centroids, tf_random_rotate, tf_center, partial_shapes
from trainers.trainer_siamese import Trainer
from utils.helper_functions import slice_idx_data, orthonormalize_basis, compute_l2_loss, normalize_caps
import open3d as o3d
from utils.vis_utils import save_pointcloud, distribute_capsules
import numpy as np
import logging



class TrainerMulti(Trainer):

    def __init__(self, config_params, model, optimizer, train_set = None, val_set = None, test_set = None, scheduler = None):
    
        super().__init__(config_params, model, optimizer, train_set, val_set, test_set, scheduler)

    @tf.function
    def forward_pass(self, x, val_step = False):
        '''
        Perform a forward pass
        '''
        
        x = tf_center(x)
        x = tf_random_rotate(x)
        x = kdtree_indexing(x)

        B, N, dim = x.shape
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

            if val_step:
                p_caps, p_inv, p_basis = self.model(x_partial_input_centered, training=False)
            else:
                p_caps, p_inv, p_basis = self.model(x_partial_input_centered, training=True)


            p_caps = tf.gather_nd(p_caps, kd_idx_inv)
            p_inv = tf.gather_nd(p_inv, kd_idx_inv)
            x_partial_input_centered_inv = tf.gather_nd(x_partial_input_centered, kd_idx_inv)


            pp_caps = tf.gather_nd(caps, x_partial_idx) # B, N, K
            pp_inv = tf_center(tf.gather_nd(inv, x_partial_idx))

            inv_partiality_loss = l2_loss_(p_inv, pp_inv)
            caps_partiality_loss = tf.reduce_mean(tf.keras.losses.cosine_similarity(pp_caps, p_caps, axis = -1))


            # Normalize capsules
            normalized_caps_partial = normalize_caps(p_caps, eps = eps)
            normalized_caps = normalize_caps(caps, eps = eps)            
            normalized_caps_partial_restriction = normalize_caps(pp_caps, eps = eps)


            # Encouraging more spaced capsules
            caps_max = tf.reduce_max(caps, axis=-1, keepdims = True)
            categorical_predictions = tf.cast((caps == caps_max), tf.float32)
            caps_spatial_loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(tf.stop_gradient( categorical_predictions), caps))


            centroids_partial = compute_centroids(x_partial, normalized_caps_partial)
            centroids = compute_centroids(x, normalized_caps) # B, K, 3
            
            centroid_partial_inv = compute_centroids(x_partial_input_centered_inv, normalized_caps_partial)
            centroid_full_inv = compute_centroids(x_restriction, normalized_caps_partial_restriction)

            capsule_position_loss = l2_loss_(centroid_partial_inv, centroid_full_inv)
            directional_loss_partial = tf_directional_loss(centroid_full_inv, centroid_partial_inv)
            eq_loss  = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x, centroids) #+ 10* chamfer_distance_l2(x_restriction, centroids_partial)

            basis = tf.stack(basis, axis = 1) # B, 5, 3, 3
            p_basis = tf.stack(p_basis, axis = 1)

            orth_basis = orthonormalize_basis(basis)
            orth_p_basis = orthonormalize_basis(p_basis)


            inv_combined = tf.einsum('bvij,bkj->bvki', orth_basis, inv)# B, V, 1024, 3
            p_inv_combined = tf.einsum('bvij,bkj->bvki', orth_p_basis, p_inv)

            inv_combined = tf.stack([inv_combined[..., 2], inv_combined[..., 0], inv_combined[..., 1]], axis=-1)
            p_inv_combined = tf.stack([p_inv_combined[..., 2], p_inv_combined[..., 0], p_inv_combined[..., 1]], axis=-1)

            separation_loss_basis = -tf.reduce_mean(tf.abs(basis[:, None] - basis[:, :, None]))


            error_full = tf.reduce_mean(tf.reduce_mean(tf.abs(x[:, None] - inv_combined), axis = -1), axis = -1)
            error_partial = tf.reduce_mean(tf.reduce_mean(tf.abs(x_restriction[:, None] - p_inv_combined), axis = -1), axis = -1)

            values, indices = tf.math.top_k(-error_full, k = 1)
            values_p, indices_p = tf.math.top_k(-error_partial, k = 1)
            
            orth_loss_f = tf.reduce_mean(tf.abs(basis - tf.stop_gradient( orth_basis))) + tf.reduce_mean(tf.abs(p_basis - tf.stop_gradient(orth_p_basis)))

            orth_basis = tf.squeeze(tf.gather(orth_basis, indices, batch_dims = 1), axis = 1) # B, 3, 3
            orth_p_basis = tf.squeeze(tf.gather(orth_p_basis, indices_p, batch_dims = 1), axis = 1)

            y_p = tf.einsum('bvj,bmj->bvm', p_inv, orth_p_basis)
            y_p = tf.stack([y_p[..., 2], y_p[..., 0], y_p[..., 1]], axis=-1)

            y = tf.einsum('bvj,bmj->bvm', inv, orth_basis)
            y = tf.stack([y[..., 2], y[..., 0], y[..., 1]], axis=-1)


            # directional_loss_partial = tf_directional_loss(centroids, tf.add(centroids_partial, center_x), tf.stop_gradient(caps_sum), tf.stop_gradient(caps_partial_sum))

            # print(directional_loss_partial)
            eq_loss  = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x, centroids) #+ 10* chamfer_distance_l2(x_restriction, centroids_partial)


            l2_loss_f, mean_root_square_loss_f = compute_l2_loss(x, y)
            l2_loss_p, mean_root_square_p = compute_l2_loss(x_restriction, y_p)
            l2_loss_f += l2_loss_p
            mean_root_square_loss_f += mean_root_square_p
            
            chamfer_loss_f = chamfer_distance_l2(y, x) + chamfer_distance_l2(y_p, x_restriction)
            hausdorff_loss_f = hausdorff_distance_l2(y, x)
            

            loss = self.config_params.loss.l2_loss * l2_loss_f + \
            self.config_params.loss.eq_loss * eq_loss + \
            self.config_params.loss.loc_loss * loc_loss + \
            self.config_params.loss.caps_chamf_loss * caps_chamf_loss + \
            self.config_params.loss.orth_loss + orth_loss_f + \
            self.config_params.loss.caps_partiality_loss * caps_partiality_loss + \
            self.config_params.loss.directional_loss_partial * directional_loss_partial + \
            self.config_params.loss.capsule_position_loss * capsule_position_loss + \
            self.config_params.loss.caps_spatial_loss * caps_spatial_loss + \
            self.config_params.loss.inv_partiality_loss * inv_partiality_loss + \
            self.config_params.loss.chamfer_loss * chamfer_loss_f + \
            self.config_params.loss.hausdorff_loss * hausdorff_loss_f + \
            self.config_params.loss.separation_loss_basis * separation_loss_basis

            if not val_step:
                # print ("backward pass")
                grad = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))    


        loss_dict = {"loss": loss, "l2_loss": l2_loss_f, "eq_loss": eq_loss, "caps_chamf_loss": caps_chamf_loss, 
                    "orth_loss": orth_loss_f, "chamfer_loss": chamfer_loss_f, "mean_root_square": mean_root_square_loss_f, 
                    "hausdorff_loss": hausdorff_loss_f, "loc_loss": loc_loss, "caps_partiality_loss": caps_partiality_loss,
                    "inv_partiality_loss": inv_partiality_loss,
                    "directional_loss": directional_loss_partial,
                    "caps_position_loss": capsule_position_loss,
                    "caps_spatial_category_loss": caps_spatial_loss,
                    "separation_loss_basis": separation_loss_basis,
                    }
        return loss_dict
    
    def test_pass(self, x, save_file_name):
        '''
        Perform test pass
        x - [1, N, 3]
        '''



        x = tf_center(x)
        x = tf_random_rotate(x)
        x = kdtree_indexing(x)

        B, N, dim = x.shape
        eps = self.config_params.utils.eps
        x_partial, x_partial_idx, kd_idx_inv = partial_shapes(x)
        x_restriction = tf.gather_nd(x, x_partial_idx)
        x_restriction, center_x = tf_center(x_restriction, return_center = True)
        x_partial_input_centered, center_x_partial = tf_center(x_partial, return_center = True)
        
            
        caps, inv, basis = self.model(x, training=False)
        p_caps, p_inv, p_basis = self.model(x_partial_input_centered, training=False)

        p_caps = tf.gather_nd(p_caps, kd_idx_inv)
        p_inv = tf.gather_nd(p_inv, kd_idx_inv)
        x_partial_input_centered_inv = tf.gather_nd(x_partial_input_centered, kd_idx_inv)

        pp_caps = tf.gather_nd(caps, x_partial_idx) # B, N, K
        pp_inv = tf_center(tf.gather_nd(inv, x_partial_idx))


        basis = tf.stack(basis, axis = 1) # B, 5, 3, 3
        p_basis = tf.stack(p_basis, axis = 1)

        orth_basis = orthonormalize_basis(basis)
        orth_p_basis = orthonormalize_basis(p_basis)


        inv_combined = tf.einsum('bvij,bkj->bvki', orth_basis, inv) # B, V, 1024, 3
        p_inv_combined = tf.einsum('bvij,bkj->bvki', orth_p_basis, p_inv)
        
        inv_combined = tf.stack([inv_combined[..., 2], inv_combined[..., 0], inv_combined[..., 1]], axis=-1)
        p_inv_combined = tf.stack([p_inv_combined[..., 2], p_inv_combined[..., 0], p_inv_combined[..., 1]], axis=-1)


        error_full = tf.reduce_mean(tf.reduce_mean(tf.abs(x[:, None] - inv_combined), axis = -1), axis = -1)
        error_partial = tf.reduce_mean(tf.reduce_mean(tf.abs(x_restriction[:, None] - p_inv_combined), axis = -1), axis = -1)

        values, indices = tf.math.top_k(-error_full, k = 1)
        values_p, indices_p = tf.math.top_k(-error_partial, k = 1)
        

        orth_basis = tf.squeeze(tf.gather(orth_basis, indices, batch_dims = 1), axis = 1) # B, 1, 3, 3
        orth_p_basis = tf.squeeze(tf.gather(orth_p_basis, indices_p, batch_dims = 1), axis = 1)

        # Normalizing capsules
        normalized_caps = normalize_caps(caps, eps)
        normalized_caps_partial = normalize_caps(p_caps, eps)

        centroids_partial = compute_centroids(x_partial_input_centered, normalized_caps_partial)
        centroids = compute_centroids(x, normalized_caps)   
        centroids_partial_pred = compute_centroids(p_inv, normalized_caps_partial)
        centroids_pred = compute_centroids(inv, normalized_caps)
        
        pcd_caps = distribute_capsules(inv, caps)
        pcd_caps_partial = distribute_capsules(p_inv, p_caps)
        pcd_caps_in = distribute_capsules(x, caps)
        pcd_caps_in_partial = distribute_capsules(x_partial_input_centered, p_caps)
        



        # Saving pointclouds
        save_pointcloud(centroids_partial, save_file_name + "_partial_centroids_inv.ply")
        save_pointcloud(centroids, save_file_name + "_full_centroids_inv.ply")

        save_pointcloud(inv, save_file_name + "_full_inv.ply")
        save_pointcloud(p_inv, save_file_name + "_partial_inv.ply")

        save_pointcloud(pcd_caps, save_file_name + "_full_inv_splits.ply")
        save_pointcloud(pcd_caps_partial, save_file_name + "_partial_inv_splits.ply")

        save_pointcloud(x, save_file_name + "_input_full.ply")
        save_pointcloud(x_partial_input_centered, save_file_name + "_input_partial.ply")

        save_pointcloud(pcd_caps_in, save_file_name + "_full_input_splits.ply")
        save_pointcloud(pcd_caps_in_partial, save_file_name + "_partial_input_splits.ply")

        save_pointcloud(centroids_partial_pred, save_file_name + "_partial_centroids_inv_pred.ply")
        save_pointcloud(centroids_pred, save_file_name + "_full_centroids_inv_pred.ply")


        for view in range(inv_combined.shape[1]):

            pointcloud_view = distribute_capsules(inv_combined[:, view], normalized_caps)
            pointcloud_view_partial = distribute_capsules(p_inv_combined[:, view], p_caps)
            
            save_pointcloud(pointcloud_view, save_file_name + "_full_input_pred_frame_%d.ply" % view)
            save_pointcloud(pointcloud_view_partial, save_file_name + "_partial_input_pred_frame_%d.ply" % view)

