import tensorflow as tf
import os, sys
sys.path.append("../")
from utils.losses import hausdorff_distance_l2, chamfer_distance_l2, tf_directional_loss, l2_loss_, equilibrium_loss, localization_loss_new
from utils.losses import sq_distance_mat

from utils.data_prep_utils import kdtree_indexing_, partial_shapes, compute_centroids, tf_random_rotate, tf_center
from trainers.trainer_siamese import Trainer
from utils.helper_functions import slice_idx_data
import open3d as o3d
from utils.vis_utils import save_pointcloud, distribute_capsules
from network_utils.pooling import kdtree_indexing


class TrainerFrame(Trainer):

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

            x_full_permuted = tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1)
            x_partial_permuted = tf.stack([x_restriction[..., 1], x_restriction[..., 2], x_restriction[..., 0]], axis=-1)

            x_full_inv = tf.einsum('bvj,bmj->bvm', x_full_permuted, tf.transpose(orth_basis, perm=[0, 2, 1]))
            x_partial_inv = tf.einsum('bvj,bmj->bvm', x_partial_permuted, tf.transpose(orth_p_basis, perm=[0, 2, 1]))


            # directional_loss_partial = tf_directional_loss(centroids, tf.add(centroids_partial, center_x), tf.stop_gradient(caps_sum), tf.stop_gradient(caps_partial_sum))

            # print(directional_loss_partial)
            eq_loss  = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x, centroids) #+ 10* chamfer_distance_l2(x_restriction, centroids_partial)



            l2_loss = x_full_inv - inv
            l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
            mean_root_square = l2_loss
            l2_loss = tf.sqrt(l2_loss + 0.000000001)
            l2_loss = tf.reduce_mean(l2_loss) + tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.multiply(p_inv - x_partial_inv, p_inv - x_partial_inv), axis = -1, keepdims=False) + 1e-8))
            l2_loss = l2_loss
            mean_root_square = tf.reduce_sum(mean_root_square, axis=1, keepdims=False)
            mean_root_square = tf.sqrt(mean_root_square + 0.000000001) / x.shape[1]
            mean_root_square = tf.reduce_mean(mean_root_square)

            chamfer_loss = chamfer_distance_l2(y, x) + chamfer_distance_l2(y_p, x_restriction)
            hausdorff_loss = hausdorff_distance_l2(y, x)
    
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
            self.config_params.loss.hausdorff_loss * hausdorff_loss

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
