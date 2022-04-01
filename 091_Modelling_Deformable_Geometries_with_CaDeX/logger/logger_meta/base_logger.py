class BaseLogger(object):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__()
        self.cfg = cfg
        self.NAME = "base"
        self.tb = tb_logger
        self.log_path = log_path
        self.total_epoch = cfg["training"]["total_epoch"]
        self.batch_size = cfg["training"]["batch_size"]
        if cfg["evaluation"]["batch_size"] < 0:
            self.eval_batch_size = cfg["training"]["batch_size"]
        else:
            self.eval_batch_size = cfg["evaluation"]["batch_size"]
        # make dir

    def log_phase(self):
        pass

    def log_batch(self, batch):
        pass
