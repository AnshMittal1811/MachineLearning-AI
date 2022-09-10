from typing import Dict, Any, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

import MinkowskiEngine as Me

from lib import modeling
from lib.config import config
from lib.modeling.backbone import UNetSparse, GeometryHeadSparse, ClassificationHeadSparse
from lib.structures.field_list import collect
from lib.modeling.utils import ModuleResult


class FrustumCompletion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNetSparse(config.MODEL.FRUSTUM3D.UNET_OUTPUT_CHANNELS,
                                num_features=config.MODEL.FRUSTUM3D.UNET_FEATURES)

        self.occupancy_256_head = None
        self.surface_head = None
        self.semantic_head = None
        self.instance_head = None

        self.init_heads("cuda")

        # Class / Instance weightings
        self.semantic_weights = torch.FloatTensor(config.MODEL.FRUSTUM3D.CLASS_WEIGHTS).to("cuda")
        self.instance_weights = torch.tensor([0.05] + [1.0 for _ in range(config.MODEL.INSTANCE2D.MAX)]).to("cuda")

        # losses
        self.criterion_occupancy = F.binary_cross_entropy_with_logits
        self.criterion_surface = F.l1_loss
        self.criterion_semantics = F.cross_entropy  #nn.CrossEntropyLoss(reduction="none")
        self.criterion_instances = F.cross_entropy  #nn.CrossEntropyLoss(reduction="none")

        self.truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        self.frustum_dimensions = [256, 256, 256]

    def init_heads(self, device="cpu"):
        # Proxy occupancy head
        self.occupancy_256_head = GeometryHeadSparse(config.MODEL.FRUSTUM3D.UNET_OUTPUT_CHANNELS, 1,
                                                     config.MODEL.FRUSTUM3D.TRUNCATION,
                                                     config.MODEL.FRUSTUM3D.GEOMETRY_HEAD.RESNET_BLOCKS)
        self.occupancy_256_head = self.occupancy_256_head.to(device)

        # Surface head
        self.surface_head = GeometryHeadSparse(config.MODEL.FRUSTUM3D.UNET_OUTPUT_CHANNELS, 1,
                                               config.MODEL.FRUSTUM3D.TRUNCATION,
                                               config.MODEL.FRUSTUM3D.GEOMETRY_HEAD.RESNET_BLOCKS)

        self.surface_head = self.surface_head.to(device)

        # Semantic head
        self.semantic_head = ClassificationHeadSparse(config.MODEL.FRUSTUM3D.UNET_OUTPUT_CHANNELS,
                                                      config.MODEL.FRUSTUM3D.NUM_CLASSES,
                                                      config.MODEL.FRUSTUM3D.SEMANTIC_HEAD.RESNET_BLOCKS)
        self.semantic_head = self.semantic_head.to(device)

        # Instance head
        self.instance_head = ClassificationHeadSparse(config.MODEL.FRUSTUM3D.UNET_OUTPUT_CHANNELS,
                                                      config.MODEL.INSTANCE2D.MAX + 1,
                                                      config.MODEL.FRUSTUM3D.SEMANTIC_HEAD.RESNET_BLOCKS)
        self.instance_head = self.instance_head.to(device)

    def forward(self, frustum_features: Me.SparseTensor, targets) -> ModuleResult:
        batch_size = len(targets)

        frustum_mask = collect(targets, "frustum_mask")
        frustum_mask_64 = F.max_pool3d(frustum_mask.float(), kernel_size=2, stride=4).bool()
        unet_output = self.model(frustum_features, batch_size, frustum_mask_64)

        losses = {}
        results = {}

        if unet_output is None:
            return {}, {}

        predictions = unet_output.data

        if predictions[2] is None:
            return {}, {}

        # Hierarchy level: 64 -> Occupancy, Instances, Semantics
        weighting_64 = collect(targets, "weighting3d_64")
        losses_64, results_64 = self.forward_64(predictions[2], targets, frustum_mask_64, weighting_64)
        losses.update(losses_64)
        results.update(results_64)

        # Hierarchy level: 128 -> Occupancy, Instances, Semantics
        if predictions[1] is None:
            return losses, results

        weighting_128 = collect(targets, "weighting3d_128")
        losses_128, results_128 = self.forward_128(predictions[1], targets, weighting_128)
        losses.update(losses_128)
        results.update(results_128)

        # Hierarchy level: 256 -> Occupancy
        if predictions[0] is None:
            return losses, results

        weighting_256 = collect(targets, "weighting3d")
        losses_256, results_256, features_256 = self.forward_256(predictions[0], targets, weighting_256)
        losses.update(losses_256)
        results.update(results_256)

        # Output level -> Surface, Instances, Semantics
        losses_output, results_output = self.forward_output(features_256, targets, weighting_256)
        losses.update(losses_output)
        results.update(results_output)

        return losses, results

    def forward_64(self, predictions, targets, frustum_mask, weighting_mask) -> Tuple[Dict, Dict]:
        hierarchy_losses = {}
        hierarchy_results = {}

        # Preprocess frustum and weighting masks
        weighting_mask = torch.masked_select(weighting_mask, frustum_mask)

        # Occupancy 64
        occupancy_prediction = predictions[0]
        occupancy_ground_truth = collect(targets, "occupancy_64")
        occupancy_loss, occupancy_result = self.compute_occupancy_64_loss(occupancy_prediction, occupancy_ground_truth,
                                                                          frustum_mask, weighting_mask)
        hierarchy_losses.update(occupancy_loss)
        hierarchy_results.update(occupancy_result)

        # Instances 64
        instance_prediction = predictions[1]
        instance_ground_truth = collect(targets, "instance3d_64")
        instance_loss, instance_result = self.compute_instance_64_loss(instance_prediction, instance_ground_truth,
                                                                       frustum_mask, weighting_mask)
        hierarchy_losses.update(instance_loss)
        hierarchy_results.update(instance_result)

        # Semantic 64
        semantic_prediction = predictions[2]
        semantic_ground_truth = collect(targets, "semantic3d_64")
        semantic_loss, semantic_result = self.compute_semantic_64_loss(semantic_prediction, semantic_ground_truth,
                                                                       frustum_mask, weighting_mask)
        hierarchy_losses.update(semantic_loss)
        hierarchy_results.update(semantic_result)

        return hierarchy_losses, hierarchy_results

    def compute_occupancy_64_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor,
                                  mask: torch.Tensor, weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        loss = self.criterion_occupancy(prediction, ground_truth, reduction="none")

        # Only consider loss within the camera frustum
        loss = torch.masked_select(loss, mask)
        loss = (loss * weighting_mask)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.COMPLETION_WEIGHT

        occupancy_probability = torch.sigmoid(prediction)
        occupancy = torch.masked_fill(occupancy_probability, mask == False, 0.0)  # mask out regions outside of frustum

        return {"occupancy_64": loss_weighted}, {"occupancy_64": occupancy}

    def compute_instance_64_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor,
                                 mask: torch.Tensor, weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        loss = self.criterion_instances(prediction, ground_truth, weight=self.instance_weights, reduction="none")

        # Only consider loss within the camera frustum
        loss = torch.masked_select(loss, mask)
        loss = (loss * weighting_mask)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.INSTANCE_WEIGHT

        prediction = torch.argmax(prediction, dim=1)
        prediction = torch.masked_fill(prediction, mask == False, 0.0)

        return {"instance3d_64": loss_weighted}, {"instance3d_64": prediction}

    def compute_semantic_64_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor,
                                  mask: torch.Tensor, weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        loss = self.criterion_semantics(prediction, ground_truth, weight=self.semantic_weights, reduction="none")

        # Only consider loss within the camera frustum
        loss = torch.masked_select(loss, mask)
        loss = (loss * weighting_mask)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.SEMANTIC_WEIGHT

        prediction = torch.argmax(prediction, dim=1)
        prediction = torch.masked_fill(prediction, mask == False, 0)

        return {"semantic3d_64": loss_weighted}, {"semantic3d_64": prediction}

    def forward_128(self, predictions, targets, weighting_mask) -> Tuple[Dict, Dict]:
        hierarchy_losses = {}
        hierarchy_results = {}

        # occupancy at 128
        occupancy_prediction: Me.SparseTensor = predictions[0]

        if occupancy_prediction is not None:
            occupancy_ground_truth = collect(targets, "occupancy_128")
            occupancy_loss, occupancy_result = self.compute_occupancy_128_loss(occupancy_prediction, occupancy_ground_truth, weighting_mask)

            hierarchy_losses.update(occupancy_loss)
            hierarchy_results.update(occupancy_result)

        # instances at 128
        instance_prediction: Me.SparseTensor = predictions[1]

        if instance_prediction is not None:
            instance_ground_truth = collect(targets, "instance3d_128").long().squeeze(1)
            instance_loss, instance_result = self.compute_instance_128_loss(instance_prediction, instance_ground_truth, weighting_mask)

            hierarchy_losses.update(instance_loss)
            hierarchy_results.update(instance_result)

        # semantics at 128
        semantic_prediction: Me.SparseTensor = predictions[2]

        if semantic_prediction is not None:
            semantic_ground_truth = collect(targets, "semantic3d_128").long().squeeze(1)
            semantic_loss, semantic_result = self.compute_semantic_128_loss(semantic_prediction, semantic_ground_truth, weighting_mask)

            hierarchy_losses.update(semantic_loss)
            hierarchy_results.update(semantic_result)

        return hierarchy_losses, hierarchy_results

    def compute_occupancy_128_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor,
                                   weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        loss = self.criterion_occupancy(prediction.F, ground_truth_values, reduction="none")

        # Get sparse weighting values from dense tensor
        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates)
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.COMPLETION_128_WEIGHT

        occupancy = Me.MinkowskiSigmoid()(prediction)

        return {"occupancy_128": loss_weighted}, {"occupancy_128": occupancy}

    def compute_instance_128_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor,
                                  weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        loss = self.criterion_instances(prediction.F, ground_truth_values.squeeze(1), weight=self.instance_weights, reduction="none")

        # Get sparse weighting values from dense tensor
        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates).squeeze(1)
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.INSTANCE_WEIGHT

        instance_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        instance_labels = Me.SparseTensor(torch.argmax(instance_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)

        return {"instance3d_128": loss_weighted}, {"instance3d_128": instance_labels}

    def compute_semantic_128_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor,
                                  weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        loss = self.criterion_semantics(prediction.F, ground_truth_values.squeeze(), weight=self.semantic_weights, reduction="none")

        # Get sparse weighting values from dense tensor
        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates).squeeze()
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.SEMANTIC_WEIGHT

        semantic_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)

        return {"semantic3d_128": loss_weighted}, {"semantic3d_128": semantic_labels}

    def forward_256(self, predictions, targets, weighting_mask) -> Tuple[Dict, Dict, Me.SparseTensor]:
        hierarchy_losses = {}
        hierarchy_results = {}

        feature_prediction: Me.SparseTensor = predictions

        if feature_prediction is not None:
            # occupancy at 256
            occupancy_prediction = self.occupancy_256_head(feature_prediction)
            occupancy_ground_truth = collect(targets, "occupancy_256")
            occupancy_loss, occupancy_result = self.compute_occupancy_256_loss(occupancy_prediction,
                                                                               occupancy_ground_truth,
                                                                               weighting_mask)

            hierarchy_losses.update(occupancy_loss)
            hierarchy_results.update(occupancy_result)

            # Use occupancy prediction to refine sparse voxels
            occupancy_masking_threshold = config.MODEL.FRUSTUM3D.SPARSE_THRESHOLD_256
            occupancy_mask = (occupancy_result["occupancy_256"].F > occupancy_masking_threshold).squeeze()
            feature_prediction = Me.MinkowskiPruning()(feature_prediction, occupancy_mask)

        return hierarchy_losses, hierarchy_results, feature_prediction

    def compute_occupancy_256_loss(self, prediction, ground_truth, weighting_mask) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        loss = self.criterion_occupancy(prediction.F, ground_truth_values)

        # Get sparse weighting values from dense tensor
        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates)
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.COMPLETION_256_WEIGHT

        occupancy = Me.MinkowskiSigmoid()(prediction)

        return {"occupancy_256": loss_weighted}, {"occupancy_256": occupancy}

    def forward_output(self, predictions: List[Me.SparseTensor], targets, weighting_mask) -> Tuple[Dict, Dict]:
        hierarchy_losses = {}
        hierarchy_results = {}

        # Surface
        surface_prediction = self.surface_head(predictions)
        if surface_prediction is not None:
            surface_ground_truth = collect(targets, "geometry")
            surface_loss, surface_result = self.compute_surface_loss(surface_prediction, surface_ground_truth, weighting_mask)
            hierarchy_losses.update(surface_loss)
            hierarchy_results.update(surface_result)

        # Instances
        instance_prediction = self.instance_head(predictions)
        if instance_prediction is not None:
            instance_ground_truth: torch.LongTensor = collect(targets, "instance3d").long().squeeze(1)
            instance_loss, instance_result = self.compute_instance_256_loss(instance_prediction, instance_ground_truth, weighting_mask)
            hierarchy_losses.update(instance_loss)
            hierarchy_results.update(instance_result)

        # Semantics
        semantic_prediction = self.semantic_head(predictions)
        if semantic_prediction is not None:
            semantic_ground_truth: torch.LongTensor = collect(targets, "semantic3d").long().squeeze(1)
            semantic_loss, semantic_result = self.compute_semantic_256_loss(semantic_prediction, semantic_ground_truth, weighting_mask)
            hierarchy_losses.update(semantic_loss)
            hierarchy_results.update(semantic_result)

        return hierarchy_losses, hierarchy_results

    def compute_surface_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor,
                             weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        predicted_values = prediction.F

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        if config.MODEL.FRUSTUM3D.IS_SDF:
            predicted_values = torch.clamp(predicted_values, -self.truncation, self.truncation)
        else:
            predicted_values = torch.clamp(predicted_values, 0.0, self.truncation)

        loss = self.criterion_surface(predicted_values, ground_truth_values, reduction="none")

        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates)
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = config.MODEL.FRUSTUM3D.L1_WEIGHT * loss_mean
        surface = Me.SparseTensor(predicted_values, prediction.C, coordinate_manager=prediction.coordinate_manager)

        return {"geometry": loss_weighted}, {"geometry": surface}

    def compute_instance_256_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor,
                                  weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        loss = self.criterion_instances(prediction.F, ground_truth_values.squeeze(1), weight=self.instance_weights, reduction="none")

        # Get sparse weighting values from dense tensor
        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates).squeeze(1)
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.INSTANCE_WEIGHT

        instance_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        instance_labels = Me.SparseTensor(torch.argmax(instance_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)

        return {"instance3d": loss_weighted}, {"instance3d": instance_labels, "instance3d_prediction": instance_softmax}

    def compute_semantic_256_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor,
                                  weighting_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = self.mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = modeling.get_sparse_values(ground_truth, predicted_coordinates)

        loss = self.criterion_semantics(prediction.F, ground_truth_values.squeeze(1), weight=self.semantic_weights, reduction="none")

        # Get sparse weighting values from dense tensor
        weighting_values = modeling.get_sparse_values(weighting_mask, predicted_coordinates).squeeze(1)
        loss = (loss * weighting_values)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        loss_weighted = loss_mean * config.MODEL.FRUSTUM3D.INSTANCE_WEIGHT

        semantic_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)

        return {"semantic3d": loss_weighted}, {"semantic3d": semantic_softmax, "semantic3d_label": semantic_labels}

    def mask_invalid_sparse_voxels(self, grid: Me.SparseTensor) -> Me.SparseTensor:
        # Mask out voxels which are outside of the grid
        valid_mask = (grid.C[:, 1] < self.frustum_dimensions[0] - 1) & (grid.C[:, 1] >= 0) & \
                     (grid.C[:, 2] < self.frustum_dimensions[1] - 1) & (grid.C[:, 2] >= 0) & \
                     (grid.C[:, 3] < self.frustum_dimensions[2] - 1) & (grid.C[:, 3] >= 0)
        num_valid_coordinates = valid_mask.sum()

        if num_valid_coordinates == 0:
            return {}, {}

        num_masked_voxels = grid.C.size(0) - num_valid_coordinates
        grids_needs_to_be_pruned = num_masked_voxels > 0

        # Fix: Only prune if there are invalid voxels
        if grids_needs_to_be_pruned:
            grid = Me.MinkowskiPruning()(grid, valid_mask)

        return grid

    def inference(self, frustum_results, frustum_mask):
        # downscale frustum_mask
        frustum_mask_64 = F.max_pool3d(frustum_mask.float(), kernel_size=2, stride=4).bool()
        unet_output = self.model(frustum_results, 1, frustum_mask_64)
        predictions = unet_output.data
        occupancy_prediction = self.occupancy_256_head(predictions[0])
        occupancy_prediction = self.mask_invalid_sparse_voxels(occupancy_prediction)
        occupancy_prediction = Me.MinkowskiSigmoid()(occupancy_prediction)
        predicted_coordinates = occupancy_prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // occupancy_prediction.tensor_stride[0]

        occupancy_masking_threshold = config.MODEL.FRUSTUM3D.SPARSE_THRESHOLD_256
        occupancy_mask = (occupancy_prediction.F > occupancy_masking_threshold).squeeze()
        prediction_pruned = Me.MinkowskiPruning()(predictions[0], occupancy_mask)

        surface_prediction = self.surface_head(prediction_pruned)
        surface_values = torch.clamp(surface_prediction.F, 0.0, self.truncation)
        surface_prediction = Me.SparseTensor(surface_values, surface_prediction.C,
                                             coordinate_manager=surface_prediction.coordinate_manager)

        instance_prediction = self.instance_head(prediction_pruned)

        instance_softmax = Me.MinkowskiSoftmax(dim=1)(instance_prediction)
        instance_labels = Me.SparseTensor(torch.argmax(instance_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=instance_prediction.coordinate_map_key,
                                          coordinate_manager=instance_prediction.coordinate_manager)

        semantic_prediction = self.semantic_head(prediction_pruned)

        semantic_softmax = Me.MinkowskiSoftmax(dim=1)(semantic_prediction)
        semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=semantic_prediction.coordinate_map_key,
                                          coordinate_manager=semantic_prediction.coordinate_manager)

        frustum_result = {
            "geometry": surface_prediction,
            "instance3d": instance_labels,
            "semantic3d_label": semantic_labels
        }

        return frustum_result
