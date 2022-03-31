'''Modules for hypernetwork experiments
'''

import torch
from torch import nn
from collections import OrderedDict
import modules
import numpy as np

import utils.common_utils as common_utils
from torchmeta.modules.utils import get_subdict


class Reptile(nn.Module):
    def __init__(self, opt, hypo_module, loss):
        super().__init__()

        self.opt = opt
        self.device = hypo_module.device
        num_meta_steps = opt.num_meta_steps
        lr_sdf = opt.lr_sdf
        lr = opt.lr

        self.hypo_module = hypo_module  # The module who's weights we want to meta-learn.
        self.loss = loss
        self.log = []

        # self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())
        # self.register_buffer('lr_sdf', torch.Tensor([lr_sdf]))
        # self.register_buffer('lr', torch.Tensor([lr]))

        self.num_meta_steps = num_meta_steps
        self.lr_sdf = lr_sdf
        self.lr = lr

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(f'Meta-paramater count: {param_count}')

        slow_params = OrderedDict()
        for name, param in self.hypo_module.meta_named_parameters():
            slow_params[name] = param.detach().clone().requires_grad_(True)
        self.slow_params = slow_params

    def renormalize(self):
        """
        Normalizes parameters of hypo_module.
        """
        self.hypo_module.renormalize()

    def update_lr(self, lr_sdf_new=None, lr_new=None, meta_lr=None, meta_steps=None):
        if lr_sdf_new is not None:
            self.lr_sdf = lr_sdf_new
        if lr_new is not None:
            self.lr = lr_new
        if meta_lr is not None:
            self.opt.meta_lr = meta_lr
        if meta_steps is not None:
            self.num_meta_steps = meta_steps

    def _update_meta_params(self, fast_params, outer_lr):
        params_cur = OrderedDict()
        for name, param in self.slow_params.items():
            params_cur[name] = param

        for i, (name, param) in enumerate(fast_params.items()):
            if name[:len('decoder_sdf')] == 'decoder_sdf':
                params_cur[name] = params_cur[name] + (outer_lr*(param - params_cur[name]))
            else:
                # params_cur[name] = param
                params_cur[name] = params_cur[name] + (5*outer_lr * (param - params_cur[name]))

        for name, param in self.slow_params.items():
            param.data = params_cur[name].data

        # Update the parameters to store them inside the hypo-module.
        for name, param in self.hypo_module.named_parameters():
            if self.slow_params.get(name, False) is not False:
                param.data = self.slow_params[name].data

    def _update_step(self, loss, param_dict, step):
        # Compute gradients.
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False, allow_unused=False)

        # Clip gradients.
        if self.opt.clip_gradients:
            max_norm = 1.
            total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), 2) for grad in grads]), 2)
            clip_coef = max_norm / (total_norm + 1e-6)

            if clip_coef < 1:
                for grad in grads:
                    grad.mul_(clip_coef)

        # Ensure we are not differentiating through the update step. Update parameters.
        with torch.no_grad():
            params = OrderedDict()
            for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
                if grad is None:
                    grad = 0

                if name[:len('decoder_sdf')] == 'decoder_sdf':
                    lr = self.lr_sdf
                else:
                    lr = self.lr
                params[name] = param - lr * grad
                params[name] = params[name].detach().requires_grad_(True)

        return params

    def _update_step_Adam(self, loss, p, step, optimizer):
        # Create OrderedDict of parameters based on the underlying parameters.
        param_dict = OrderedDict()
        for pd in p:
            param_dict[pd['name']] = pd['params'][0]

        # Compute gradients.
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False, allow_unused=False)

        # Clip gradients.
        if self.opt.clip_gradients:
            max_norm = 1.
            total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), 2) for grad in grads]), 2)
            clip_coef = max_norm / (total_norm + 1e-6)

            if clip_coef < 1:
                for grad in grads:
                    grad.mul_(clip_coef)

        # Update gradients of the parameters using the computed gradients.
        with torch.no_grad():
            for i, (pd, grad) in enumerate(zip(p, grads)):
                pd['params'][0].grad = grad

        # Update the underlying parameters with the optimizer.
        optimizer.step()

        # Manually zero out grads
        with torch.no_grad():
            for pd in p:
                del pd['params'][0].grad

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, context_dict, dataset_num):
        """Specializes the model"""
        x = context_dict.get('inputs')
        y = context_dict.get('gt')

        # Create underlying parameters based on slow params, and create optimizer on these parameters.
        p = []
        for name, param in self.slow_params.items():
            if name[:len('decoder_sdf')] == 'decoder_sdf':
                lr = self.lr_sdf
            else:
                lr = self.lr
            p += [{'name': name, 'params': param.detach().clone().requires_grad_(True),
                   'lr': lr}]
        optimizer = torch.optim.Adam(params=p, lr=self.lr)

        self.hypo_module.precompute_3D_buffers(dataset_num=dataset_num)

        intermed_loss = []
        for j in range(self.num_meta_steps):
            self.renormalize()

            # Generate fast parameters based on underlying parameters
            fast_params = OrderedDict()
            for param_dict in p:
                fast_params[param_dict['name']] = param_dict['params'][0]

            # Using the current set of parameters, perform a forward pass with the context inputs.
            predictions = self.hypo_module(x[j], params=fast_params, dataset_num=dataset_num)

            # Add output of predictions
            # Add model parameters for regularization losses.
            predictions['weights_sdf'] = {k: v for k, v in get_subdict(fast_params, 'decoder_sdf').items()}

            # Compute the loss on the context labels.
            losses = self.loss(predictions, y[j])

            inner_loss = 0.
            for loss_name, (loss, loss_enabled) in losses.items():
                single_loss = loss.mean()
                if torch.isnan(single_loss).any().item():
                    print('We have NAN in loss!!!!')
                    import pdb
                    pdb.set_trace()
                    raise Exception('NaN in loss!')

                if loss_enabled:
                    # Sum only active losses.
                    inner_loss += single_loss

            inner_loss_numpy = inner_loss.detach().clone().cpu().numpy()
            intermed_loss.append(inner_loss_numpy)

            # Update the underlying parameters using the computed loss
            self._update_step_Adam(inner_loss, p, j, optimizer)

        fast_params_final = OrderedDict()
        for param_dict in p:
            fast_params_final[param_dict['name']] = param_dict['params'][0]

        return fast_params_final, {'loss': intermed_loss,
                                   'init_params': self.slow_params}

    def forward(self, meta_batch, **kwargs):
        # The meta_batch conists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)
        context = meta_batch['context']
        query_x = meta_batch['query']['inputs']
        dataset_num = int(meta_batch['dataset_number'].numpy())

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(context, dataset_num)

        # Compute the final outputs.
        with torch.no_grad():
            model_output = self.hypo_module(query_x, params=fast_params, dataset_num=dataset_num)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions,
                    'fast_params': fast_params}

        return out_dict

    def load_checkpoint(self, checkpoint_file, load_sdf=True, strict=False,
                        load_img_encoder=True, load_img_decoder=True, load_aggregation=True, load_poses=True):
        """
        Loads checkpoint.
        """
        if checkpoint_file is None:
            return
        print(
            f'Loading checkpoint from {checkpoint_file} (load_sdf={load_sdf},'
            f' load img_encoder={load_img_encoder}, load img_decoder={load_img_decoder}, load_aggregation={load_aggregation}).')
        state = torch.load(checkpoint_file, map_location=self.device)

        state_filtered = {}
        for k, v in state.items():
            if not load_sdf and k.startswith('hypo_module.decoder_sdf'):
                continue
            if not load_img_encoder and k.startswith('hypo_module.enc_net'):
                continue
            if not load_img_decoder and k.startswith('hypo_module.dec_net'):
                continue
            if not load_aggregation and k.startswith('hypo_module.agg_net'):
                continue
            state_filtered[k] = v
        self.load_state_dict(state_filtered, strict=strict)
