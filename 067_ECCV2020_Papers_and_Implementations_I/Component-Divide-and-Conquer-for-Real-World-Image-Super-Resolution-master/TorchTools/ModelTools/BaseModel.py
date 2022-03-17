import os
import torch


class BaseModel(object):
    """
    The base class of one experiment
    """
    def __init__(self, logger):
        self.logger = logger

    def name(self):
        return 'BaseModel'

    def initialize(self):
        pass

    def set_input(self):
        pass

    def backward(self):
        pass

    def optimize_parameters(self):
        pass

    def train_step(self):
        pass

    def test(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def save_filename(self, network_label, epoch_label):
        return '%s_net_%s.pth' % (epoch_label, network_label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_path = os.path.join(self.logger.checkpoint_dir, self.save_filename(network_label, epoch_label))
        network.load_state_dict(torch.load(save_path))

    def get_current_errors(self):
        return {}


