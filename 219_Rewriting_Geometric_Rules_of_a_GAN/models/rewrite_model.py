import torch
from os.path import join

import models.networks as networks
from models import wrapper
from lib.dissect.nethook import Trace
from util.util import slice_ordered_dict


class RewriteModel(torch.nn.Module):
    """
    A general interface for finetuning generative models.
    The goal is to tune any specified layers with supervision at any specified target features.
    Each layer allows flexible tuning using the Wrapper module; for example, low-rank updates of weights.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--finetune_mode", default='conv', choices=['conv', 'affine', 'all'])
        parser.add_argument("--update_layers", default='5-5', help="i-j, update params from layer i to layer j (inclusive)")
        parser.add_argument("--target_layer", default='output', help="which layer to provide supervision")
        parser.add_argument("--only_weight", action='store_true')
        parser.add_argument("--rank", default=None, type=int)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.rank = opt.rank
        self.device = 'cuda' if self.opt.num_gpus > 0 else 'cpu'  # TODO Cleaner way to do this needed, currently need this member still
        self.netG = self.initialize_networks(opt).to(self.device)

        # setup layers of interest
        module_dict = networks.get_modules(opt.archG, self.netG, mode=opt.finetune_mode)
        in_sizes, out_sizes = networks.get_module_resolution(opt.archG, module_dict)

        # setup layers to update
        if opt.update_layers == 'all':
            start, end = 0, len(module_dict) - 1
        elif '-' in opt.update_layers:
            start, end = [int(s) for s in opt.update_layers.split('-')]
        else:
            ind = int(opt.update_layers)
            start, end = ind, ind
        self.update_layers = slice_ordered_dict(module_dict, start, end + 1)
        self.key_res = in_sizes[start:end + 1]

        # setup target layer for supervision
        if opt.target_layer == 'output':
            self.target_layer = 'output'
            self.target_res = self.netG.img_resolution
        else:
            ind = int(opt.target_layer)
            self.target_layer = slice_ordered_dict(module_dict, ind, ind + 1)
            self.target_res = out_sizes[ind]

        if self.opt.isTrain:
            # get the updatable parameters to train
            self.update_params = self.set_update_params()
            self.print_trainable_params()

    def forward(self, w_latents, mode):
        w_latents = w_latents.to(self.device)

        # output features at the target layer
        if mode == 'target' and self.target_layer != 'output':
            assert len(self.target_layer) == 1, f"self.target_layer should have length 1, but get {len(self.target_layer)}"
            tgt_name = list(self.target_layer.keys())[0]
            with Trace(self.netG, tgt_name, stop=True) as ret:
                self.netG.synthesis(w_latents, force_fp32=True)
            features = ret.output
            return features

        # return the model output
        elif mode == 'output' or self.target_layer == 'output':
            return self.netG.synthesis(w_latents, force_fp32=True)

        # return both model output and target features
        elif mode == 'all':
            assert len(self.target_layer) == 1, f"self.target_layer should have length 1, but get {len(self.target_layer)}"
            tgt_name = list(self.target_layer.keys())[0]
            with Trace(self.netG, tgt_name, stop=False) as ret:
                output = self.netG.synthesis(w_latents, force_fp32=True)
            features = ret.output
            return output, features

        else:
            raise KeyError(f"mode should be either 'target', 'output' or 'all', but get {mode}")

    def get_all_params(self):
        """Return all parameters for the model."""
        return list(self.netG.parameters())

    def get_trainable_params(self):
        """Return a subset of parameters that is trainable."""
        return list(self.update_params.values())

    def get_weight_reg_loss(self):
        """Return weight regularization loss, if defined."""
        if self.schatten_p is not None:
            return wrapper.weight_reg_loss_from_wrappers(self.module_wrappers)
        return None

    def get_output_res(self):
        """Return the resolution of the output."""
        return self.netG.img_resolution

    def get_target_res(self):
        """Return the resolution of the target."""
        return self.target_res

    def save(self, iters):
        """Saves the trainable parameters to the checkpoint directory."""
        save_path = join(self.opt.checkpoints_dir, self.opt.name, f"{iters}_checkpoint.pth")
        save_params = wrapper.save_params_from_wrappers(self.module_wrappers)
        torch.save(save_params, save_path)

    def load(self, iters=None, load_path=None):
        """Loads the subset of parameters from the checkpoint directory."""
        assert (iters is None) != (load_path is None), "need specify just one argument -- either iters or load_path"
        if load_path is None:
            load_path = join(self.opt.checkpoints_dir, self.opt.name, f"{iters}_checkpoint.pth")
        state_dict = torch.load(load_path, map_location=self.device)
        self.netG.load_state_dict(state_dict, strict=False)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        return netG

    def set_update_params(self):
        """Returns a state dict of traininable parameters."""
        # setup the module wrappers to enable different training paradigms (e.g. low-rank updates)
        only_weight = self.opt.only_weight
        if self.rank is None:
            wrapper_name = 'standard'
            self.module_wrappers = wrapper.create_wrappers(wrapper_name, self.update_layers, only_weight=only_weight)
        else:
            wrapper_name = 'lowrank'
            self.module_wrappers = []
            for layer, module in self.update_layers.items():
                w = wrapper.create_wrapper(wrapper_name, module, layer, context=None, rank=self.rank, only_weight=only_weight)
                self.module_wrappers.append(w)

        # get the updatable parameters from the list of wrappers
        update_params = wrapper.update_params_from_wrappers(self.module_wrappers)
        return update_params

    def print_trainable_params(self):
        param_names = list(self.update_params)
        self.logprint(
            '\n' +
            '-------------- Trainables ---------------\n' +
            '(Trainable parameters)\n' +
            '\n'.join(param_names) +
            '\n----------------- End -------------------'
        )

    def logprint(self, log):
        print(log)
