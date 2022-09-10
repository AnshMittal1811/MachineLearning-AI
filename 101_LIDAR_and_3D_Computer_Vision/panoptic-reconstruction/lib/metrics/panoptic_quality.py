# following evaluation of
# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
class PQStatCategory:
    def __init__(self, is_thing=True):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n = 0
        self.is_thing = is_thing

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.n += pq_stat_cat.n
        return self

    @property
    def as_metric(self):
        return {"iou": self.iou, "tp": self.tp, "fp": self.fp, "fn": self.fn, "n": self.n}

    def __repr__(self):
        return self.as_metric
