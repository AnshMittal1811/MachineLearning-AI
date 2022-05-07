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



class TrainerFeaturesCanonical(Trainer):

    def __init__(self, config_params, model, optimizer, train_set = None, val_set = None, test_set = None, scheduler = None):
    
        super().__init__(config_params, model, optimizer, train_set, val_set, test_set, scheduler)
        self.features = config_params.features
        if self.features.translate.use == True:
            logging.info("Predicting translation\n")

        if self.features.scale.use == True:
            logging.info("Predicting scale\n")


    @tf.function
    def forward_pass(self, x, val_step = False):
        '''
        Perform a forward pass
        '''
        
        x = tf_center(x)
        if self.features.rotation.use:
            x = tf_random_rotate(x)
       
        x = kdtree_indexing(x)

        B, N, dim = x.shape
        eps = self.config_params.utils.eps
        x_partial, x_partial_idx, kd_idx_inv = partial_shapes(x)
        x_restriction = tf.gather_nd(x, x_partial_idx)        
        x_restriction_not_centered = tf.identity(x_restriction)
        x_restriction, center_x = tf_center(x_restriction, return_center = True)
        
        x_partial_input_centered, center_x_partial = tf_center(x_partial, return_center = True)
        
        if self.features.scale.use:
            scale_factor = np.random.uniform(self.features.scale.args.factor_min, self.features.scale.args.factor_max)
            x_scaled = scale_factor * x
        
        training_bool = True
        with tf.GradientTape() as tape:
            
            if val_step:
                training_bool = False
            else:
                training_bool = True

            # Full branch
            out_dict = self.model(x, training=training_bool)
            caps, inv, basis = out_dict['caps'], out_dict['points_inv'], out_dict["basis_list"]
            if self.features.scale.use == True:
                scale = out_dict["scale"]
            if self.features.translate.use == True:
                translation = out_dict["translation"]
            
            # Partial branch
            if self.features.partiality.use:
                out_dict_partial = self.model(x_partial_input_centered, training=training_bool)
                p_caps, p_inv, p_basis = out_dict_partial['caps'], out_dict_partial['points_inv'], out_dict_partial["basis_list"]
                if self.features.scale.use == True:
                    p_scale = out_dict_partial["scale"]
                if self.features.translate.use == True:
                    p_translation = out_dict_partial["translation"]
                
            # Scale branch
            if self.features.scale.use:
                scale_factor = np.random.uniform(self.features.scale.args.factor_min, self.features.scale.args.factor_max)
                x_scaled = scale_factor * x

                out_dict_partial = self.model(x_scaled, training=training_bool)
                s_caps, s_inv, s_basis = out_dict_partial['caps'], out_dict_partial['points_inv'], out_dict_partial["basis_list"]
                if self.features.scale.use == True:
                    s_scale = out_dict_partial["scale"]
                if self.features.translate.use == True:
                    s_translation = out_dict_partial["translation"]
            


            p_caps = tf.gather_nd(p_caps, kd_idx_inv)
            p_inv = tf.gather_nd(p_inv, kd_idx_inv)
            x_partial_input_centered_inv = tf.gather_nd(x_partial_input_centered, kd_idx_inv)
                
            pp_caps = tf.gather_nd(caps, x_partial_idx) # B, N, K
            pp_inv_not_centered = tf.gather_nd(inv, x_partial_idx)        
            pp_inv = tf_center(pp_inv_not_centered)
            
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

            centroids_partial = compute_centroids(x_restriction, normalized_caps_partial)
            centroids = compute_centroids(x, normalized_caps) # B, K, 3
            
            centroid_partial_inv = compute_centroids(x_restriction, normalized_caps_partial)
            centroid_full_inv = compute_centroids(x_restriction, normalized_caps_partial_restriction)

            capsule_position_loss = l2_loss_(centroid_partial_inv, centroid_full_inv)
            directional_loss_partial = tf_directional_loss(centroid_full_inv, centroid_partial_inv)
            eq_loss  = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x, centroids) + chamfer_distance_l2(x_restriction, centroids_partial)

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

            eq_loss  = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x, centroids) + chamfer_distance_l2(x_restriction, centroids_partial)


            l2_loss_f, mean_root_square_loss_f = compute_l2_loss(x, y)
            l2_loss_p, mean_root_square_p = compute_l2_loss(x_restriction, y_p)
            l2_loss_f += l2_loss_p
            mean_root_square_loss_f += mean_root_square_p
            
            chamfer_loss_f = chamfer_distance_l2(y, x) + chamfer_distance_l2(y_p, x_restriction)
            hausdorff_loss_f = hausdorff_distance_l2(y, x) + hausdorff_distance_l2(y_p, x_restriction)
            

            if self.features.scale.use:

                s_p_caps = tf.gather_nd(s_caps, kd_idx_inv)
                s_p_inv = tf.gather_nd(s_inv, kd_idx_inv)
                s_basis = tf.stack(s_basis, axis = 1)
                orth_s_basis = orthonormalize_basis(s_basis)
                
                # Selecting the best frame
                s_inv_combined = tf.einsum('bvij,bkj->bvki', orth_s_basis, s_inv)
                s_inv_combined = tf.stack([s_inv_combined[..., 2], s_inv_combined[..., 0], s_inv_combined[..., 1]], axis=-1)
                error_full_scaled = tf.reduce_mean(tf.reduce_mean(tf.abs(x_scaled[:, None] - s_inv_combined), axis = -1), axis = -1)
                values_s, indices_s = tf.math.top_k(-error_full_scaled, k = 1)
                orth_s_basis = tf.squeeze(tf.gather(orth_s_basis, indices_s, batch_dims = 1), axis = 1)

                # Centroids scaled 
                normalized_caps_s = normalize_caps(s_caps, eps = eps)            
                centroids_s = compute_centroids(x_scaled, normalized_caps_s)

                # Scaled prediction in the input frame
                y_s = tf.einsum('bvj,bmj->bvm', s_inv, orth_s_basis)
                y_s = tf.stack([y_s[..., 2], y_s[..., 0], y_s[..., 1]], axis=-1)
                
                # Loss computations with respect to the scaled cloud
                l2_loss_s, mean_root_square_s = compute_l2_loss(x_scaled, y_s)
                l2_loss_f += l2_loss_s
                mean_root_square_loss_f += mean_root_square_s
                chamfer_loss_f += chamfer_distance_l2(y_s, x_scaled) 
                hausdorff_loss_f += hausdorff_distance_l2(y_s, x_scaled)
                eq_loss += equilibrium_loss(s_caps)
                loc_loss += localization_loss_new(x_scaled, s_caps, centroids_s)
                caps_chamf_loss += chamfer_distance_l2(x_scaled, centroids_s)
                ## Ensuring the capsules are same irrespective of scale
                caps_scaled_loss = tf.reduce_mean(tf.keras.losses.cosine_similarity(s_caps, caps, axis = -1))
                caps_scaled_l2_loss, _ = compute_l2_loss(scale_factor * tf.stop_gradient(inv), s_inv)     
                scale_loss = tf.reduce_mean(tf.abs((scale_factor - s_scale))) + tf.reduce_mean(tf.abs((1.0 - scale))) 

            if self.features.translate.use == True:
                translation_loss_full = tf.reduce_mean((tf.abs(translation - tf.zeros(translation.shape))))
                if self.features.scale.use == True:
                    translation_loss_full += tf.reduce_mean((tf.abs(s_translation - tf.zeros(s_translation.shape))))

                translation_loss_partial = tf.reduce_mean((tf.abs(p_translation + tf.stop_gradient(p_inv) -  tf.stop_gradient(pp_inv_not_centered))))

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

            if self.features.scale.use:
                loss += self.config_params.loss.caps_scaled_loss * caps_scaled_loss
                loss += self.config_params.loss.caps_scaled_l2_loss * caps_scaled_l2_loss
                loss += self.config_params.loss.scale_loss * scale_loss

            if self.features.translate.use:
                loss += self.config_params.loss.translation_loss_full * translation_loss_full
                loss += self.config_params.loss.translation_loss_partial * translation_loss_partial
                
            
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

        if self.features.scale.use:
            loss_dict["caps_scaled_loss"] = caps_scaled_loss
            loss_dict["caps_scaled_l2_loss"] = caps_scaled_l2_loss
            loss_dict["scale_loss"] = scale_loss
            
        if self.features.translate.use:
            loss_dict["translation_loss_full"] = translation_loss_full
            loss_dict["translation_loss_partial"] = translation_loss_partial
            
        return loss_dict
    
    def test_pass(self, x, save_file_name):
        '''
        Perform test pass
        x - [1, N, 3]
        '''



        x = tf_center(x)
        
        if self.features.scale.use:
            scale_factor = np.random.uniform(self.features.scale.args.factor_min, self.features.scale.args.factor_max)
            x = scale_factor * x
        
        if self.features.rotation.use:
            x = tf_random_rotate(x)
        
        x = kdtree_indexing(x)

        B, N, dim = x.shape
        eps = self.config_params.utils.eps
        x_partial, x_partial_idx, kd_idx_inv = partial_shapes(x)
        x_restriction = tf.gather_nd(x, x_partial_idx)
        x_restriction, center_x = tf_center(x_restriction, return_center = True)
        x_partial_input_centered, center_x_partial = tf_center(x_partial, return_center = True)
        
        out_dict = self.model(x, training=False)
        caps, inv, basis = out_dict['caps'], out_dict['points_inv'], out_dict["basis_list"]

        if self.features.translate.use:
            translation = out_dict['translation']    
        
        if self.features.scale.use:
            scale = out_dict['scale']

        out_dict_partial = self.model(x_partial_input_centered, training=False)
        p_caps, p_inv, p_basis = out_dict_partial['caps'], out_dict_partial['points_inv'], out_dict_partial["basis_list"]

        if self.features.translate.use:
            p_translation = out_dict_partial['translation']    
        
        if self.features.scale.use:
            p_scale = out_dict_partial['scale']
            

        p_caps = tf.gather_nd(p_caps, kd_idx_inv)
        p_inv = tf.gather_nd(p_inv, kd_idx_inv)
        x_partial_input_centered_inv = tf.gather_nd(x_partial_input_centered, kd_idx_inv)

        pp_caps = tf.gather_nd(caps, x_partial_idx) # B, N, K
        pp_inv_not_centered = tf.gather_nd(inv, x_partial_idx) 
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
        
        if self.features.translate.use:

            inv = inv + translation
            p_inv = p_inv + p_translation

        if self.features.scale.use:
            inv_s = inv / (scale + 1e-8)
            p_inv_s = p_inv / (p_scale + 1e-8)

            save_pointcloud(inv_s, save_file_name + "_full_inv_scale.ply")
            save_pointcloud(p_inv_s, save_file_name + "_partial_inv_scaled.ply")
        
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
            

            if self.features.translate.use:

                pointcloud_view = distribute_capsules(inv_combined[:, view] + translation, normalized_caps)
                pointcloud_view_partial = distribute_capsules(p_inv_combined[:, view] + p_translation, p_caps)
            else:
                pointcloud_view = distribute_capsules(inv_combined[:, view], normalized_caps)
                pointcloud_view_partial = distribute_capsules(p_inv_combined[:, view], p_caps)


            save_pointcloud(pointcloud_view, save_file_name + "_full_input_pred_frame_%d.ply" % view)
            save_pointcloud(pointcloud_view_partial, save_file_name + "_partial_input_pred_frame_%d.ply" % view)

