from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    @abstractmethod
    def __init__(self, module, module_name, **kwargs):
        pass

    def get_weight_reg_loss(self):
        """Get the weight regularization loss. Raises NotImplementedError by default if not overridden."""
        raise NotImplementedError(f"Weight regularization loss not implemented in {self.__class__.__name__}")

    @abstractmethod
    def get_update_params(self):
        """Get the updatable parameters. In other words, params that will be send to the optimizer."""
        pass

    @abstractmethod
    def get_save_params(self):
        """Get the parameters to save. In other words, params that will be saved using torch.save()."""
        pass
