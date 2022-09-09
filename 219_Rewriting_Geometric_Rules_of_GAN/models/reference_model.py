import torch

import models.networks as networks
from lib.dissect.nethook import Trace
from util.util import slice_ordered_dict


class ReferenceModel(torch.nn.Module):
    """
    Model used for reference.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--target_layer", default='output', help="which layer to provide supervision")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = 'cuda' if self.opt.num_gpus > 0 else 'cpu'
        self.netG = self.initialize_networks(opt).to(self.device)

        # setup layers of interest
        module_dict = networks.get_modules(opt.archG, self.netG, mode=opt.finetune_mode)
        in_sizes, out_sizes = networks.get_module_resolution(opt.archG, module_dict)

        # setup target layer for supervision
        if opt.target_layer == 'output':
            self.target_layer = 'output'
            self.target_res = self.netG.img_resolution
        else:
            ind = int(opt.target_layer)
            self.target_layer = slice_ordered_dict(module_dict, ind, ind + 1)
            self.target_res = out_sizes[ind]

    def forward(self, w_latents, mode):
        w_latents = w_latents.to(self.device)

        # output features at the target layer
        if mode == 'target' and self.target_layer != 'output':
            assert len(self.target_layer) == 1, f"self.target_layer should have length 1, but get {len(self.target_layer)}"
            tgt_name = list(self.target_layer.keys())[0]
            with Trace(self.netG, tgt_name, stop=True) as ret:
                self.netG.synthesis(w_latents)
            features = ret.output
            return features

        # return the model output
        elif mode == 'output' or self.target_layer == 'output':
            return self.netG.synthesis(w_latents)

        # return both model output and target features
        elif mode == 'all':
            assert len(self.target_layer) == 1, f"self.target_layer should have length 1, but get {len(self.target_layer)}"
            tgt_name = list(self.target_layer.keys())[0]
            with Trace(self.netG, tgt_name, stop=False) as ret:
                output = self.netG.synthesis(w_latents)
            features = ret.output
            return output, features

        else:
            raise KeyError(f"mode should be either 'target', 'output' or 'all', but get {mode}")

    def get_all_params(self):
        """Return all parameters for the model."""
        return list(self.netG.parameters())

    def get_trainable_params(self):
        """Return a subset of parameters that is trainable."""
        return []

    def get_output_res(self):
        """Return the resolution of the output."""
        return self.netG.img_resolution  # TODO: too specific too stylegan

    def get_target_res(self):
        """Return the resolution of the target."""
        return self.target_res

    def save(self, iters):
        """Saves the trainable parameters to the checkpoint directory."""
        pass

    def load(self, iters=None, load_path=None):
        """Loads the subset of parameters from the checkpoint directory."""
        pass

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        return netG
