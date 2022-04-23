import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from .solvers import anderson, broyden


def make_pair(target, source):
    if len(target) == len(source):
        return source
    elif len(source) == 1:
        return [source[0] for _ in range(len(target))]
    else:
        raise ValueError('Unable to align the arg squence!')


def backward_factory(
        grad_type='ift',
        safe_ift=False, 
        b_solver=anderson, 
        b_solver_kwargs=dict(),
        sup_all=False,
        tau=1.0):
    """
    [2019-NeurIPS] Deep Equilibrium Models
    [2021-ICLR] Is Attention Better Than Matrix Decomposition? 
    [2021-NeurIPS] On Training Implicit Models
    [2022-AAAI] JFB: Jacobian-Free Backpropagation for Implicit Networks

    This function implements a factory for the backward pass of implicit deep learning,
    e.g., DEQ (implicit models), Hamburger (optimization layer), etc.
    It now supports IFT, 1-step Grad, and Phantom Grad.
    
    Kwargs:
        grad_type (string, int): 
            grad_type should be ``ift`` or an int. Default ``ift``.
            Set to ``ift`` to enable the implicit differentiation mode.
            When passing a number k to this function, it runs UPG with steps k and damping tau.
        safe_ift (bool): 
            Replace the O(1) hook implementeion with a safer one. Default ``False``.
            Set to ``True`` to avoid the (potential) segment fault (under previous versions of Pytorch).
        b_solver (type):
            Solver for the IFT backward pass. Default ``anderson``.
            Supported solvers: anderson, broyden.
        b_solver_kwargs (dict):
            Colllection of backward solver kwargs, e.g., 
                threshold (int), max steps for the backward solver, 
                stop_mode (string), criterion for convergence,
                etc.
            See solver.py to check all the kwargs.
        sup_all (bool):
            Indicate whether to supervise all the trajectories by Phantom Grad.
            Set ``True`` to return all trajectory in Phantom Grad.
        tau (float):
            Damping factor for Phantom Grad. Default ``1.0``.
            0.5 is recommended for CIFAR-10. 1.0 for DEQ flow.
            For DEQ flow, the gating function in GRU naturally produces adaptive tau values. 
    
    Return:
        A gradient functor for implicit deep learning.
        Args:
            trainer (nn.Module): the module that employs implicit deep learning.
            z_pred (torch.Tensor): latent state to run the backward pass.
            func (type): function that defines the ``f`` in ``z = f(z)``.
        
        Return:
            (list(torch.Tensor)): a list of tensors that tracks the gradient info.

    """
    
    if grad_type == 'ift':
        assert b_solver in [anderson, broyden]
        
        if safe_ift:
            def plain_ift_grad(trainer, z_pred, func):
                z_pred = z_pred.requires_grad_()
                new_z_pred = func(z_pred) # 1-step grad for df/dtheta

                z_pred_copy = new_z_pred.clone().detach().requires_grad_()
                new_z_pred_copy = func(z_pred_copy)
                def backward_hook(grad):
                    result = b_solver(lambda y: autograd.grad(new_z_pred_copy, z_pred_copy, y, retain_graph=True)[0] + grad, 
                            torch.zeros_like(grad), **b_solver_kwargs)
                    return result['result']
                new_z_pred.register_hook(backward_hook)
                
                return [new_z_pred]
            return plain_ift_grad
        else:
            def hook_ift_grad(trainer, z_pred, func):
                z_pred = z_pred.requires_grad_()
                new_z_pred = func(z_pred) # 1-step grad for df/dtheta

                def backward_hook(grad):
                    if trainer.hook is not None:
                        trainer.hook.remove()    # To avoid infinite loop
                    result = b_solver(lambda y: autograd.grad(new_z_pred, z_pred, y, retain_graph=True)[0] + grad, 
                            torch.zeros_like(grad), **b_solver_kwargs)
                    return result['result']
                trainer.hook = new_z_pred.register_hook(backward_hook)
                
                return [new_z_pred]
            return hook_ift_grad
    else:
        assert type(grad_type) is int and grad_type >= 1
        n_phantom_grad = grad_type
        
        if sup_all:
            def sup_all_phantom_grad(trainer, z_pred, func):
                z_out = []
                for _ in range(n_phantom_grad):
                    z_pred = (1 - tau) * z_pred + tau * func(z_pred)
                    z_out.append(z_pred)

                return z_out
            return sup_all_phantom_grad
        else:
            def phantom_grad(trainer, z_pred, func):
                for _ in range(n_phantom_grad):
                    z_pred = (1 - tau) * z_pred + tau * func(z_pred)

                return [z_pred]
            return phantom_grad
