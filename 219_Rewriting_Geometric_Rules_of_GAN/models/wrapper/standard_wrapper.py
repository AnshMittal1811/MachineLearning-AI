from .base_wrapper import BaseWrapper
from lib.dissect.param_tool import get_params_from_module


class StandardWrapper(BaseWrapper):
    """This wrapper takes in the module and apply low-rank update on the `.weight` parameter."""

    def __init__(self, module, module_name, only_weight=False):
        super().__init__(module, module_name, only_weight=only_weight)

        # setup bookkeeping
        self.module = module
        self.module_name = module_name

        # gather update_params and save_params
        param_dict = get_params_from_module(module, prefix=module_name, exclude_children=True)
        if only_weight:
            weight_name = f'{module_name}.weight'
            assert weight_name in param_dict.keys(), \
                f"If only_weight is True, parameters with name `{weight_name}` needs to exists in module."
            self.update_params = {weight_name: param_dict[weight_name]}
            self.save_params = {weight_name: param_dict[weight_name]}
        else:
            self.update_params = param_dict.copy()
            self.save_params = param_dict.copy()

    def get_update_params(self):
        """Get the updatable parameters. In other words, params that will be send to the optimizer."""
        return self.update_params

    def get_save_params(self):
        """Get the parameters to save. In other words, params that will be saved using torch.save()."""
        return self.save_params
