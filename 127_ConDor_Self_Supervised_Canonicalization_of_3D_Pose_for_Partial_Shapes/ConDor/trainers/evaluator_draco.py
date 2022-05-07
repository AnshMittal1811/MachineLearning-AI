import logging
import numpy as np
from utils.vis_utils import save_pointcloud, distribute_capsules
import open3d as o3d
from utils.helper_functions import slice_idx_data, orthonormalize_basis, compute_l2_loss, normalize_caps, convert_yzx_to_xyz_basis
from trainers.trainer_siamese import Trainer
from utils.data_prep_utils import kdtree_indexing, kdtree_indexing_, compute_centroids, tf_random_rotate, tf_center, partial_shapes
from utils.losses import sq_distance_mat
from utils.losses import hausdorff_distance_l2, chamfer_distance_l2, tf_directional_loss, l2_loss_, equilibrium_loss, localization_loss_new
import tensorflow as tf
import os
import sys
sys.path.append("../")


class EvaluatorDRACO(Trainer):

    def __init__(self, config_params, model, optimizer, train_set=None, val_set=None, test_set=None, scheduler=None):

        super().__init__(config_params, model, optimizer,
                         train_set, val_set, test_set, scheduler)
        self.features = config_params.features

    @tf.function
    def forward_pass(self, data, val_step=False):
        '''
        Perform a forward pass
        '''

        x_input = data["full"]
        x_partial_input = data["depth"]
        idx_partial_to_full = data["idx_partial_to_full"]

        # Center
        x = tf_center(x_input)

        # Center
        x_partial, center = tf_center(x_partial_input, return_center = True)

        x_kd, idx_kd, idx_inv_kd = kdtree_indexing_(x)
        x_partial_kd, idx_partial_kd, idx_partial_inv_kd = kdtree_indexing_(
            x_partial)
        eps = self.config_params.utils.eps

        with tf.GradientTape() as tape:

            if val_step:
                training_bool = False
            else:
                training_bool = True

            ##################### Full branch #############################################
            out_dict = self.model(x_kd, training=training_bool)
            caps, inv, basis = out_dict['caps'], out_dict['points_inv'], out_dict["basis_list"]

            # Computing basis
            basis = tf.stack(basis, axis=1)  # B, 5, 3, 3
            orth_basis = orthonormalize_basis(basis)

            # Encouraging more spaced capsules
            caps_max = tf.reduce_max(caps, axis=-1, keepdims=True)
            categorical_predictions = tf.cast((caps == caps_max), tf.float32)
            caps_spatial_loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(
                tf.stop_gradient(categorical_predictions), caps))

            # Normalize capsules and compute centroids
            normalized_caps = normalize_caps(caps, eps=eps)
            centroids = compute_centroids(x, normalized_caps)  # B, K, 3

            # Rotate canonical frame to reproduce the input for all the V basis
            inv_combined = tf.einsum(
                'bvij,bkj->bvki', orth_basis, inv)  # B, V, 1024, 3
            inv_combined = tf.stack(
                [inv_combined[..., 2], inv_combined[..., 0], inv_combined[..., 1]], axis=-1)

            separation_loss_basis = - \
                tf.reduce_mean(tf.abs(basis[:, None] - basis[:, :, None]))

            # Compute the frame that gives the least error
            error_full = tf.reduce_mean(tf.reduce_mean(
                tf.abs(x_kd[:, None] - inv_combined), axis=-1), axis=-1)
            values, indices = tf.math.top_k(-error_full, k=1)

            orth_loss_f = tf.reduce_mean(
                tf.abs(basis - tf.stop_gradient(orth_basis)))
            orth_basis = tf.squeeze(
                tf.gather(orth_basis, indices, batch_dims=1), axis=1)  # B, 3, 3

            # Rotate the canonical frame to reproduce
            y = tf.einsum('bvj,bmj->bvm', inv, orth_basis)
            y = tf.stack([y[..., 2], y[..., 0], y[..., 1]], axis=-1)

            # Losses for full shape
            eq_loss = equilibrium_loss(caps)
            loc_loss = localization_loss_new(x_kd, caps, centroids)
            caps_chamf_loss = chamfer_distance_l2(x_kd, centroids)
            l2_loss_f, mean_root_square_loss_f = compute_l2_loss(x_kd, y)
            chamfer_loss_f = chamfer_distance_l2(y, x_kd)
            hausdorff_loss_f = hausdorff_distance_l2(y, x_kd)

            ##################### Partial branch #############################################
            out_dict_partial = self.model(x_partial_kd, training=training_bool)
            p_caps, p_inv, p_basis = out_dict_partial['caps'], out_dict_partial[
                'points_inv'], out_dict_partial["basis_list"]

            pp_caps = slice_idx_data(
                caps, idx_partial_to_full, idx_inv_kd, idx_partial_kd)
            pp_inv = tf_center(slice_idx_data(
                inv, idx_partial_to_full, idx_inv_kd, idx_partial_kd))
            x_restriction_not_centered = slice_idx_data(
                x_kd, idx_partial_to_full, idx_inv_kd, idx_partial_kd)
            x_restriction = tf_center(x_restriction_not_centered)

            inv_partiality_loss = l2_loss_(p_inv, pp_inv)
            caps_partiality_loss = tf.reduce_mean(
                tf.keras.losses.cosine_similarity(pp_caps, p_caps, axis=-1))

            # Normalize capsules
            normalized_caps_partial = normalize_caps(p_caps, eps=eps)
            normalized_caps_partial_restriction = normalize_caps(
                pp_caps, eps=eps)

            # Compute centroids
            centroids_partial = compute_centroids(
                x_partial, normalized_caps_partial)
            centroid_partial_inv = compute_centroids(
                x_partial_kd, normalized_caps_partial)
            centroid_full_inv = compute_centroids(
                x_restriction, normalized_caps_partial_restriction)

            # Penalize partiality
            capsule_position_loss = l2_loss_(
                centroid_partial_inv, centroid_full_inv)
            directional_loss_partial = tf_directional_loss(
                centroid_full_inv, centroid_partial_inv)

            # Orthonormalize the basis
            p_basis = tf.stack(p_basis, axis=1)
            orth_p_basis = orthonormalize_basis(p_basis)
            orth_loss_f += tf.reduce_mean(tf.abs(p_basis -
                                          tf.stop_gradient(orth_p_basis)))

            # Choose the best frame that minimizes the error
            p_inv_combined = tf.einsum('bvij,bkj->bvki', orth_p_basis, p_inv)
            p_inv_combined = tf.stack(
                [p_inv_combined[..., 2], p_inv_combined[..., 0], p_inv_combined[..., 1]], axis=-1)
            error_partial = tf.reduce_mean(tf.reduce_mean(
                tf.abs(x_partial_kd[:, None] - p_inv_combined), axis=-1), axis=-1)
            values_p, indices_p = tf.math.top_k(-error_partial, k=1)
            orth_p_basis = tf.squeeze(
                tf.gather(orth_p_basis, indices_p, batch_dims=1), axis=1)

            # Predict the input by transformaining the canonical frame to the input frame for the best frame
            y_p = tf.einsum('bvj,bmj->bvm', p_inv, orth_p_basis)
            y_p = tf.stack([y_p[..., 2], y_p[..., 0], y_p[..., 1]], axis=-1)

            if self.features.translate.use:
                translation = out_dict["translation"][0]
                p_translation = out_dict_partial["translation"][0]

                translation_loss_full = tf.reduce_mean(
                    (tf.abs(translation - tf.zeros(translation.shape))))
                translation_loss_partial = tf.reduce_mean(
                    (tf.abs(p_translation + x_restriction - x_restriction_not_centered)))

            # Losses
            l2_loss_p, mean_root_square_p = compute_l2_loss(x_partial_kd, y_p)
            l2_loss_f += l2_loss_p
            mean_root_square_loss_f += mean_root_square_p
            chamfer_loss_f += chamfer_distance_l2(y_p, x_partial_kd)

            # Logging and adding losses
            loss = self.config_params.loss.l2_loss * l2_loss_f + \
                self.config_params.loss.eq_loss * eq_loss + \
                self.config_params.loss.loc_loss * loc_loss + \
                self.config_params.loss.caps_chamf_loss * caps_chamf_loss + \
                self.config_params.loss.orth_loss + orth_loss_f + \
                self.config_params.loss.chamfer_loss * chamfer_loss_f + \
                self.config_params.loss.hausdorff_loss * hausdorff_loss_f + \
                self.config_params.loss.caps_spatial_loss * caps_spatial_loss + \
                self.config_params.loss.separation_loss_basis * separation_loss_basis

            if self.features.translate.use:
                loss += self.config_params.loss.translation_loss_full * translation_loss_full
                loss += self.config_params.loss.translation_loss_partial * translation_loss_partial

            loss += self.config_params.loss.caps_partiality_loss * caps_partiality_loss + \
                self.config_params.loss.directional_loss_partial * directional_loss_partial + \
                self.config_params.loss.capsule_position_loss * capsule_position_loss + \
                self.config_params.loss.inv_partiality_loss * inv_partiality_loss

            loss_dict = {"loss": loss, "l2_loss": l2_loss_f,
                         "eq_loss": eq_loss, "caps_chamf_loss": caps_chamf_loss,
                         "orth_loss": orth_loss_f, "chamfer_loss": chamfer_loss_f,
                         "mean_root_square": mean_root_square_loss_f,
                         "hausdorff_loss": hausdorff_loss_f, "loc_loss": loc_loss,
                         "caps_spatial_category_loss": caps_spatial_loss,
                         "separation_loss_basis": separation_loss_basis
                         }

            loss_dict.update(
                {"caps_partiality_loss": caps_partiality_loss,
                 "inv_partiality_loss": inv_partiality_loss,
                 "directional_loss": directional_loss_partial,
                 "caps_position_loss": capsule_position_loss})

            if self.features.translate.use:
                loss_dict["translation_loss_full"] = translation_loss_full
                loss_dict["translation_loss_partial"] = translation_loss_partial

            if not val_step:
                # print ("backward pass")
                gradients = tape.gradient(loss, self.model.trainable_variables)
                #self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
                self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(
                    gradients, self.model.trainable_variables) if grad is not None)

        return loss_dict

    @tf.function
    def test_pass(self, data):
        '''
        Perform test pass
        x - [1, N, 3]
        '''

        x = data["full"]
        x_partial_input = data["depth"]
        idx_partial_to_full = data["idx_partial_to_full"]

        x = tf_center(x)
        x, _, _ = kdtree_indexing_(x)

        x_partial_input, init_translation = tf_center(x_partial_input, return_center = True)
        x_partial_input, _, _ = kdtree_indexing_(x_partial_input)

        eps = self.config_params.utils.eps

        out_dict = self.model(x, training=False)
        caps, inv, basis = out_dict['caps'], out_dict['points_inv'], out_dict["basis_list"]

        if self.features.translate.use:
            translation = out_dict['translation'][0]

        out_dict_partial = self.model(x_partial_input, training=False)
        p_caps, p_inv, p_basis = out_dict_partial['caps'], out_dict_partial[
            'points_inv'], out_dict_partial["basis_list"]

        if self.features.translate.use:
            p_translation = out_dict_partial['translation'][0]

        basis = tf.stack(basis, axis=1)
        orth_basis = orthonormalize_basis(basis)

        p_basis = tf.stack(p_basis, axis=1)
        orth_p_basis = orthonormalize_basis(p_basis)

        inv_combined = tf.einsum(
            'bvij,bkj->bvki', orth_basis, inv)  # B, V, 1024, 3
        p_inv_combined = tf.einsum('bvij,bkj->bvki', orth_p_basis, p_inv)

        inv_combined = tf.stack(
            [inv_combined[..., 2], inv_combined[..., 0], inv_combined[..., 1]], axis=-1)
        p_inv_combined = tf.stack(
            [p_inv_combined[..., 2], p_inv_combined[..., 0], p_inv_combined[..., 1]], axis=-1)

        error_full = tf.reduce_mean(tf.reduce_mean(
            tf.abs(x[:, None] - inv_combined), axis=-1), axis=-1)
        error_partial = tf.reduce_mean(tf.reduce_mean(
            tf.abs(x_partial_input[:, None] - p_inv_combined), axis=-1), axis=-1)

        values, indices = tf.math.top_k(-error_full, k=1)
        values_p, indices_p = tf.math.top_k(-error_partial, k=1)

        orth_basis = tf.squeeze(
            tf.gather(orth_basis, indices, batch_dims=1), axis=1)  # B, 1, 3, 3
        orth_p_basis = tf.squeeze(
            tf.gather(orth_p_basis, indices_p, batch_dims=1), axis=1)

        # Normalizing capsules
        normalized_caps_partial = normalize_caps(p_caps, eps=eps)
        normalized_caps = normalize_caps(caps, eps=eps)

        # Compute centroids
        centroids_partial = compute_centroids(
            x_partial_input, normalized_caps_partial)
        centroids = compute_centroids(x, normalized_caps)

        pcd_caps = distribute_capsules(inv, caps)
        pcd_caps_partial = distribute_capsules(p_inv, p_caps)

        pcd_caps_in = distribute_capsules(x, caps)
        pcd_caps_in_partial = distribute_capsules(x_partial_input, p_caps)

        centroids_partial_pred = compute_centroids(
            p_inv, normalized_caps_partial)
        centroids_pred = compute_centroids(inv, normalized_caps)

        out_dict = {"centroids_partial": centroids_partial,
                    "centroids": centroids,
                    "inv": inv, "p_inv": p_inv,
                    "pcd_caps": pcd_caps,
                    "pcd_caps_partial": pcd_caps_partial,
                    "x": x, "x_partial_input_centered": x_partial_input,
                    "pcd_caps_in": pcd_caps_in, "pcd_caps_in_partial": pcd_caps_in_partial,
                    "centroids_partial_pred": centroids_partial_pred, "centroids_pred": centroids_pred,
                    "inv_combined": inv_combined, "p_inv_combined": p_inv_combined, "normalized_caps": normalized_caps,
                    "normalized_caps_partial": normalized_caps_partial,
                    "orth_basis": orth_basis, "orth_p_basis": orth_p_basis}

        if self.features.translate.use:
            out_dict["p_translation"] = p_translation
            out_dict["translation"] = translation
            out_dict["init_translation"] = init_translation

        return out_dict

    def save_files_all(self, output_dict, save_file_name):
        '''
        Save all the point clouds returned in the test pass
        '''

        # Saving pointclouds
        save_pointcloud(output_dict["centroids_partial"], save_file_name + "_partial_centroids_inv.ply")
        save_pointcloud(output_dict["centroids"], save_file_name + "_full_centroids_inv.ply")

        save_pointcloud(output_dict["inv"], save_file_name + "_full_inv.ply")
        save_pointcloud(output_dict["p_inv"], save_file_name + "_partial_inv.ply")

        save_pointcloud(output_dict["pcd_caps"], save_file_name + "_full_inv_splits.ply")
        save_pointcloud(output_dict["pcd_caps_partial"], save_file_name + "_partial_inv_splits.ply")

        save_pointcloud(output_dict["x"], save_file_name + "_input_full.ply")
        save_pointcloud(output_dict["x_partial_input_centered"], save_file_name + "_input_partial.ply")

        save_pointcloud(output_dict["pcd_caps_in"], save_file_name + "_full_input_splits.ply")
        save_pointcloud(output_dict["pcd_caps_in_partial"], save_file_name + "_partial_input_splits.ply")

        save_pointcloud(output_dict["centroids_partial_pred"], save_file_name + "_partial_centroids_inv_pred.ply")
        save_pointcloud(output_dict["centroids_pred"], save_file_name + "_full_centroids_inv_pred.ply")

        inv_combined = output_dict["inv_combined"]
        p_inv_combined = output_dict["p_inv_combined"]

        normalized_caps = output_dict["normalized_caps"]
        p_caps = output_dict["normalized_caps_partial"]

        p_translation = output_dict["p_translation"]

        for view in range(inv_combined.shape[1]):

            if self.features.translate.use:
                pointcloud_view = distribute_capsules(inv_combined[:, view], normalized_caps)
                pointcloud_view_partial = distribute_capsules(p_inv_combined[:, view] + p_translation, p_caps)
            else:
                pointcloud_view = distribute_capsules(inv_combined[:, view], normalized_caps)
                pointcloud_view_partial = distribute_capsules(p_inv_combined[:, view], p_caps)

            save_pointcloud(pointcloud_view, save_file_name + "_full_input_pred_frame_%d.ply" % view)
            save_pointcloud(pointcloud_view_partial, save_file_name + "_partial_input_pred_frame_%d.ply" % view)

        np.save(save_file_name + "_full_basis.npy", convert_yzx_to_xyz_basis(output_dict["orth_basis"])[0])
        np.save(save_file_name + "_partial_basis.npy", convert_yzx_to_xyz_basis(output_dict["orth_p_basis"])[0])

        np.save(save_file_name + "_full_translation.npy", output_dict["translation"])
        np.save(save_file_name + "_partial_translation.npy", output_dict["p_translation"])
        np.save(save_file_name + "_init_translation.npy", output_dict["init_translation"])

        np.save(save_file_name + "_partial_translation.npy", p_translation[0])
        np.save(save_file_name + "_full_translation.npy", translation[0])

        basis_instance_to_canonical = tf.linalg.pinv(convert_yzx_to_xyz_basis(output_dict["orth_basis"]))
        p_basis_instance_to_canonical = tf.linalg.pinv(convert_yzx_to_xyz_basis(output_dict["orth_p_basis"]))


        translated_p_input = output_dict["x_partial_input_centered"] + p_translation
        
        canonical_cloud = (tf.einsum('bij,bkj->bki', basis_instance_to_canonical, output_dict["x"]))
        p_canonical_cloud = (tf.einsum('bij,bkj->bki', p_basis_instance_to_canonical, translated_p_input))

        canonical_cloud_splits = distribute_capsules(canonical_cloud, normalized_caps)
        p_canonical_cloud_splits = distribute_capsules(p_canonical_cloud, p_caps)

        save_pointcloud(canonical_cloud, save_file_name + "_canonical_pointcloud_full.ply")
        save_pointcloud(p_canonical_cloud, save_file_name + "_canonical_pointcloud_partial.ply")


        save_pointcloud( canonical_cloud_splits, save_file_name + "_canonical_pointcloud_full_splits.ply")
        save_pointcloud( p_canonical_cloud_splits, save_file_name + "_canonical_pointcloud_partial_splits.ply")