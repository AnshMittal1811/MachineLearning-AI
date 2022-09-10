from collections import defaultdict, namedtuple, OrderedDict

from typing import Tuple, List, Dict

import torch
import numpy as np

from lib.metrics import Metric

PredictionSample = namedtuple("PredictionSample", ["item", "label", "confidence"])
GroundTruthSample = namedtuple("GroundTruthSample", ["item", "label"])


class MeanAveragePrecision(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.predictions: List[List[PredictionSample]] = []
        self.ground_truths: List[List[GroundTruthSample]] = []
        self.use_07_metric = True
        self.threshold = 0.5

    def add(self, prediction: List[PredictionSample], ground_truth: List[GroundTruthSample]) -> None:
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)

    def reduce(self):
        assert len(self.predictions) == len(self.ground_truths)

        # Group by label
        predictions_by_label = self._group_by_label(self.predictions)
        ground_truths_by_label = self._group_by_label(self.ground_truths)

        average_precision_per_label = OrderedDict()
        sorted_labels = sorted(list(ground_truths_by_label.keys()))
        for label in sorted_labels:
            if label not in predictions_by_label.keys():
                average_precision_per_label[label] = 0.0
                continue

            # Compute AP per label
            average_precision_per_label[label] = self._compute_average_precision(predictions_by_label[label],
                                                                                 ground_truths_by_label[label])

        mean_average_precision = sum(v for v in average_precision_per_label.values())/len(ground_truths_by_label.keys())
        reduction = {
            "mean": mean_average_precision
        }
        reduction.update(average_precision_per_label)
        return reduction

    def _group_by_label(self, list_of_samples) -> Dict[int, Dict[int, List]]:
        grouped_by_label = defaultdict(lambda: defaultdict(list))
        for sample_index, sample in enumerate(list_of_samples):
            for idx in range(sample.label.shape[0]):
                # convert to PredictionSample
                if len(sample) == 3:
                    res = PredictionSample(item=sample.item[idx], label=sample.label[idx], confidence=sample.confidence[idx])
                elif len(sample) == 2:
                    res = GroundTruthSample(item=sample.item[idx], label=sample.label[idx])

                grouped_by_label[int(sample.label[idx])][sample_index].append(res)
        return grouped_by_label

    def _compute_average_precision(self, predictions: Dict[int, List[PredictionSample]],
                                   ground_truths: Dict[int, List[GroundTruthSample]]) -> float:

        class_recs: Dict[int, Dict[str, List]] = {}
        npos = 0

        for idx, samples in ground_truths.items():
            npos += len(samples)
            class_recs[idx] = {"sample": samples, "detected": [False] * len(samples)}

        for idx in predictions.keys():
            if idx not in ground_truths:
                class_recs[idx] = {"sample": [], "detected": []}

        indices = []
        confidences = []
        bounding_boxes = []

        for idx, samples in predictions.items():
            for sample in samples:
                indices.append(idx)
                confidences.append(sample.confidence)
                bounding_boxes.append(sample.item)

        indices = np.array(indices)
        confidences = np.array(confidences)
        bounding_boxes = np.array(bounding_boxes)

        # sort by confidence
        sorted_indices = np.argsort(-confidences)
        sorted_confidences = np.sort(-confidences)
        sorted_bounding_boxes = bounding_boxes[sorted_indices]
        sorted_indices = indices[sorted_indices]

        # go down detections
        num_detected = indices.shape[0]
        tp = np.zeros(indices.shape[0])
        fp = np.zeros(indices.shape[0])
        for d in range(num_detected):
            rec = class_recs[sorted_indices[d]]

            prediction = sorted_bounding_boxes[d]
            overlap_max = -np.inf
            index_max = 0

            ground_truths: GroundTruthSample = rec["sample"]

            if len(ground_truths) > 0:
                for idx, ground_truth in enumerate(ground_truths):
                    overlap = self.evaluation_function(prediction, ground_truth.item)

                    if overlap > overlap_max:
                        overlap_max = overlap
                        index_max = idx

            if overlap_max > self.threshold:
                if not rec["detected"][index_max]:
                    tp[d] = 1.0
                    rec["detected"][index_max] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0

        fps = np.cumsum(fp)
        tps = np.cumsum(tp)
        recall = tps / float(npos)
        precision = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)

        average_precision = self.voc_ap(recall, precision)

        return average_precision

    def voc_ap(self, recall, precision):
        if self.use_07_metric:
            return self.voc_average_precision_07(recall, precision)
        else:
            return self.voc_average_precision(recall, precision)

    def voc_average_precision_07(self, recall, precision):
        # 11 point metric
        average_precision = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            average_precision = average_precision + p / 11.

        return average_precision

    def voc_average_precision(self, recall, precision):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        average_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return average_precision

    def evaluation_function(self, prediction, ground_truth):
        return bbox2d(prediction, ground_truth)


""" Benchmark Evaluation for Object Detection.

    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/sunrgbd/sunrgbd_detection/eval_det.py

    Author: Ji Hou
    Date: March 10th 2018
"""

def mask2d(maskA, maskB):
    interArea = np.count_nonzero(np.logical_and(maskA, maskB))
    unionArea = np.count_nonzero(np.logical_or(maskA, maskB))

    if unionArea == 0:
        return 0

    return interArea / float(unionArea)


def mask3d(maskA, maskB):
    interArea = np.count_nonzero(np.logical_and(maskA, maskB))
    unionArea = np.count_nonzero(np.logical_or(maskA, maskB))

    if unionArea == 0:
        return 0

    return interArea / float(unionArea)


def bbox2d(bboxA, bboxB):
    """ Compute IoU of two bounding boxes.
    """
    # determine the (x, y, z)-coordinates of the intersection rectangle
    minx_overlap = max(bboxA[0], bboxB[0])
    miny_overlap = max(bboxA[1], bboxB[1])

    maxx_overlap = min(bboxA[2], bboxB[2])
    maxy_overlap = min(bboxA[3], bboxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, maxx_overlap - minx_overlap) * max(0, maxy_overlap - miny_overlap)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    boxBArea = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    div = float(boxAArea + boxBArea - interArea)
    iou = interArea / div if div != 0.0 else 0.0

    # return the intersection over union value
    return iou
