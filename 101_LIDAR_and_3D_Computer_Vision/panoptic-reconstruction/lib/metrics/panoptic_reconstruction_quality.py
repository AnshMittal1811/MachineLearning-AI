from collections import OrderedDict

import torch
from lib.metrics.intersection_over_union import compute_iou
from lib.metrics.panoptic_quality import PQStatCategory
from typing import Tuple, Dict

from lib.config import config

from lib.metrics import Metric


class PanopticReconstructionQuality(Metric):
    def __init__(self, matching_threshold=0.25, category_information=None, ignore_labels=None, reduction="mean"):
        super().__init__()

        # Ignore freespace label and ceiling
        if ignore_labels is None:
            ignore_labels = [0, 12]
        self.ignore_labels = ignore_labels

        self.matching_threshold = matching_threshold
        # self.extract_mesh = extract_mesh

        if category_information is None:
            self.category_information = {
                -1: True,
                1: True,
                2: True,
                3: True,
                4: True,
                5: True,
                6: True,
                7: True,
                8: True,
                9: True,
                10: False,
                11: False
            }

            if config.DATASETS.NAME == "matterport":
                self.category_information[12] = False

        self.categories = {}

        for label, is_thing in self.category_information.items():
            self.categories[label] = PQStatCategory(is_thing)

        self.reduction = reduction

    def add(self, prediction: Dict[int, Tuple[torch.Tensor, int]], ground_truth: Dict[int, Tuple[torch.Tensor, int]]) -> Dict:
        matched_ids_ground_truth = set()
        matched_ids_prediction = set()

        per_sample_result = {}
        for label, is_thing in self.category_information.items():
            per_sample_result[label] = PQStatCategory(is_thing)

        # True Positives
        for ground_truth_instance_id, (ground_truth_instance_mask, ground_truth_semantic_label) in ground_truth.items():
            self.categories[ground_truth_semantic_label].n += 1
            per_sample_result[ground_truth_semantic_label].n += 1

            for prediction_instance_id, (prediction_instance_mask, prediction_semantic_label) in prediction.items():

                # 0: Check if prediction was already matched
                if prediction_instance_id in matched_ids_prediction:
                    continue

                # 1: Check if they have the same label
                are_same_category = ground_truth_semantic_label == prediction_semantic_label

                if not are_same_category:
                    # self.logger.info(f"{ground_truth_instance_id} vs {prediction_instance_id} --> Not same category {ground_truth_semantic_label} vs {prediction_semantic_label}")
                    continue

                # 2: Compute overlap and check if they are overlapping enough
                overlap = compute_iou(ground_truth_instance_mask, prediction_instance_mask)
                is_match = overlap > self.matching_threshold

                if is_match:
                    self.categories[ground_truth_semantic_label].iou += overlap
                    self.categories[ground_truth_semantic_label].tp += 1

                    per_sample_result[ground_truth_semantic_label].iou += overlap
                    per_sample_result[ground_truth_semantic_label].tp += 1

                    matched_ids_ground_truth.add(ground_truth_instance_id)
                    matched_ids_prediction.add(prediction_instance_id)
                    break

        # False Negatives
        for ground_truth_instance_id, (_, ground_truth_semantic_label) in ground_truth.items():
            # 0: Check if ground truth has not yet been matched
            if ground_truth_instance_id not in matched_ids_ground_truth:
                # self.logger.info(f"Not matched, counted as FN: {ground_truth_instance_id}, num voxels: {mask.sum()}")
                self.categories[ground_truth_semantic_label].fn += 1
                per_sample_result[ground_truth_semantic_label].fn += 1

        # False Positives
        for prediction_instance_id, (_, prediction_semantic_label) in prediction.items():
            # 0: Check if prediction has not yet been matched
            if prediction_instance_id not in matched_ids_prediction:
                # self.logger.info(f"Not matched, counted as FP: {prediction_instance_id}, num voxels: {mask.sum()}")
                self.categories[prediction_semantic_label].fp += 1
                per_sample_result[prediction_semantic_label].fp += 1

        return per_sample_result

    def add_sample(self, sample):
        for k in sample.keys():
            if k in self.categories:
                self.categories[k] += sample[k]

    def reduce(self) -> Dict:
        if self.reduction == "mean":
            return self.reduce_mean()

        return None

    def reduce_mean(self):
        pq, sq, rq, n = 0, 0, 0, 0

        per_class_results = {}

        for class_label, class_stats in self.categories.items():
            iou = class_stats.iou
            tp = class_stats.tp
            fp = class_stats.fp
            fn = class_stats.fn
            num_samples = class_stats.n

            if tp + fp + fn == 0:
                per_class_results[class_label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'n': 0}
                continue

            if num_samples > 0:
                n += 1
                pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
                sq_class = iou / tp if tp != 0 else 0
                rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
                per_class_results[class_label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'n': num_samples}
                pq += pq_class
                sq += sq_class
                rq += rq_class

        results = OrderedDict()
        results.update({'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n})

        for label, per_class_result in per_class_results.items():
            if per_class_result["n"] > 0:
                results[f"pq_{label}"] = per_class_result["pq"]
                results[f"sq_{label}"] = per_class_result["sq"]
                results[f"rq_{label}"] = per_class_result["rq"]
                results[f"n_{label}"] = per_class_result["n"]

        return results
