"""
Definition of evaluation metric. Please modify this code very carefully!!
"""
import math
import numpy as np
from easydict import EasyDict


class MovingAverageEstimator(object):
    """ Estimate moving average of the given results """
    def __init__(self, field_names):
        self.field_names = field_names
        self.results = EasyDict()
        self.reset()

    def update(self, results):
        for name in self.field_names:
            self.results[name] += results[name]
        self.count += 1

    def compute(self):
        avg_results = EasyDict()
        for name in self.field_names:
            avg_results[name] = self.results[name] / self.count

        return avg_results

    def reset(self):
        for name in self.field_names:
            self.results[name] = 0.
        self.count = 0

    def __repr__(self):
        return 'Moving Average Estimator: ' + ', '.join(self.field_names)


class Metrics(object):
    """ Benchmark """
    def __init__(self, field_names):
        """ Metrics to be evaluated are specified in `field_names`.
            Make sure you used metrics are defined in this file. """
        self.metric_fn = EasyDict()
        self.results = EasyDict()
        for name in field_names:
            self.metric_fn[name] = globals()[name]
            self.results[name] = 0.
        self.field_names = field_names

    def compute(self, pred, target):
        """ Compute results. Note that `pred` and `target` are numpy array
            and they should have the same shape. """
        valid_mask = (target > 0)
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        for name in self.field_names:
            self.results[name] = self.metric_fn[name](pred_valid, target_valid)

        return EasyDict(self.results.copy())

    def __repr__(self):
        return 'Metrics: ' + ', '.join(self.field_names)


def mae(pred, target):
    """ Mean Average Error (MAE) """
    return np.absolute(pred - target).mean()


def imae(pred, target):
    """ inverse Mean Average Error in 1/km (iMAE) """
    return np.absolute(1000*(1./pred-1./target)).mean()


def rmse(pred, target):
    """ Root Mean Square Error (RMSE) """
    return math.sqrt(np.power((pred - target), 2).mean())


def irmse(pred, target):
    """ inverse Root Mean Square Error in 1/km (iRMSE) """
    return math.sqrt(np.power(1000*(1./pred - 1./target), 2).mean())


def mre(pred, target):
    """ Mean Absolute Relative Error (MRE) """
    return (np.absolute(pred - target) / target).mean()


def log10(pred, target):
    """ Mean log10 Error (LOG10) """
    return np.absolute(np.log10(np.clip(pred, a_min=1E-8, a_max=None)) - np.log10(target)).mean()


def delta1(pred, target, delta=1.25):
    """ Threshold delta1 """
    thr = np.maximum(target/pred, pred/target)
    return (thr < delta).astype(np.float32).mean()


def delta2(pred, target, delta=1.25):
    """ Threshold delta2 """
    thr = np.maximum(target/pred, pred/target)
    return (thr < delta**2).astype(np.float32).mean()


def delta3(pred, target, delta=1.25):
    """ Threshold delta2 """
    thr = np.maximum(target/pred, pred/target)
    return (thr < delta**3).astype(np.float32).mean()


def log_rmse(pred, target):
    """ Log RMSE. """
    log_pred = np.log(np.clip(pred, a_min=1E-8, a_max=None))
    log_target = np.log(target)
    return math.sqrt(np.power((log_pred - log_target), 2).mean())


def scale_inv_log_rmse(pred, target):
    """ Scale invariant log RMSE.
        NOTE: can only be used with batch size = 1. """
    log_pred = np.log(np.clip(pred, a_min=1E-8, a_max=None))
    log_target = np.log(target)
    diff = log_pred - log_target
    n = diff.shape[0]
    return (diff ** 2).mean() - (diff.sum() ** 2) / (n ** 2)


def err_1px(pred, target):
    """ 1-pix error; used in stereo depth """
    abs_err = np.absolute(pred - target)
    correct = (abs_err < 1) | (abs_err < (target * 0.05))
    return 1 - (float(correct.sum()) / target.shape[0])


def err_2px(pred, target):
    """ 2-pix error; used in stereo depth """
    abs_err = np.absolute(pred - target)
    correct = (abs_err < 2) | (abs_err < (target * 0.05))
    return 1 - (float(correct.sum()) / target.shape[0])


def err_3px(pred, target):
    """ 3-pix error; used in stereo depth """
    abs_err = np.absolute(pred - target)
    correct = (abs_err < 3) | (abs_err < (target * 0.05))
    return 1 - (float(correct.sum()) / target.shape[0])
