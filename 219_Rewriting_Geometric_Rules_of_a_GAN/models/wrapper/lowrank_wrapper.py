import torch
from .base_wrapper import BaseWrapper
from lib.dissect.param_tool import get_params_from_module


class LowRankWrapper(BaseWrapper):
    """This wrapper takes in the module and apply low-rank update on the `.weight` parameter."""

    def __init__(self, module, module_name, context=None, rank=None, only_weight=True):
        assert context is not None or rank is not None, "need to specify either rank or context."
        if context is not None:
            assert context.shape[0] == rank, f"the rank of the context should match the specified rank {rank}, but got {context.shape[0]}."
        super().__init__(module, module_name, context=context, rank=None, only_weight=only_weight)

        # setup bookkeeping
        self.module = module
        self.module_name = module_name
        self.original_weight = module.weight

        ws = self.original_weight.shape
        device = self.original_weight.device

        self.update_params = {}
        self.save_params = {}

        # options to include params other than .weight
        # note that these params will be updated fully (low-rank is only on `.weight`)
        param_dict = get_params_from_module(module, prefix=module_name, exclude_children=True)
        weight_name = f'{module_name}.weight'
        assert weight_name in param_dict.keys(), \
            f"If only_weight is True, parameters with name `{weight_name}` needs to exists in module."
        self.module._parameters.pop('weight')
        if not only_weight:
            other_params = {k: v for k, v in param_dict.items() if '.weight' not in k}
            self.update_params.update(other_params)
            self.save_params.update(other_params)

        # if make a random, updatable context direction if context is None
        if context is None:
            self.context_updatable = True
            self.context = torch.randn(rank, ws[1], device=device, requires_grad=True)
            self.update_params[f'{module_name}.weight_context'] = self.context
        else:
            self.context_updatable = False
            self.context = context

        # create the tunable lambda parameter
        self.lambda_param = torch.zeros(   # TODO: this shape is too specific too stylegan
            ws[0], self.context.shape[0], ws[2], ws[3], device=device, requires_grad=True)
        self.update_params[f'{module_name}.weight_lambda'] = self.lambda_param

        # update the temporary new forward with low rank updates
        self.cache_forward = self.module.forward

        def new_forward(*args, **kwargs):
            # weight_1 = weight_0 + Lambda D
            self.module.weight = self.original_weight + \
                torch.einsum('odyx, di -> oiyx', self.lambda_param, self.context)
            result = self.cache_forward(*args, **kwargs)
            return result
        self.module.forward = new_forward

    def get_update_params(self):
        """Get the updatable parameters. In other words, params that will be send to the optimizer."""
        return self.update_params

    def get_save_params(self):
        """Get the parameters to save. In other words, params that will be saved using torch.save()."""
        with torch.no_grad():
            # Fill in the learned weights
            new_weight = self.original_weight + \
                torch.einsum('odyx, di -> oiyx', self.lambda_param, self.context)
        self.save_params[f'{self.module_name}.weight'] = new_weight
        return self.save_params

    def resume_module_forward(self, use_new_weight=True):
        """Warning: the wrapper can no longer be used after this method is called."""
        if use_new_weight:
            with torch.no_grad():
                # Fill in the learned weights and undo the hook.
                new_weight = self.original_weight + \
                    torch.einsum('odyx, di -> oiyx', self.lambda_param, self.context)
            self.module.register_parameter('weight', new_weight)
        else:
            self.module.register_parameter('weight', self.original_weight)
        self.module.forward = self.cache_forward
