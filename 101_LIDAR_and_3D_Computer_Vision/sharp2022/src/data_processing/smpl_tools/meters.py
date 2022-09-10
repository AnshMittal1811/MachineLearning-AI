class Meters:
    def __init__(self, eps=-1e-3, stop_threshold=10) -> None:
        self.eps = eps
        self.stop_threshold = stop_threshold
        self.avg = 0
        self.cnt = 0
        self.reset_early_stop()

    def reset_early_stop(self):
        self.min_loss = float('inf')
        self.satis_num = 0
        self.update_res = True
        self.early_stop = False

    def update_avg(self, val, k=1):
        self.avg = self.avg + (val - self.avg) * k / (self.cnt + k)
        self.cnt += k

    def update_early_stop(self, val):
        delta = (val - self.min_loss) / self.min_loss
        if float(val) < self.min_loss:
            self.min_loss = float(val)
            self.update_res = True
        else:
            self.update_res = False
        self.satis_num = self.satis_num + 1 if delta >= self.eps else 0
        self.early_stop = self.satis_num >= self.stop_threshold