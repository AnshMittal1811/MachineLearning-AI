# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import gc
from torchdiffeq import odeint
from models.utils import get_score_fn


def get_div_fn(fn):
    def div_fn(u, t, eps):
        '''
        Estimates the divergence of the function "fn".
        '''
        with torch.enable_grad():
            u.requires_grad_(True)
            fn_eps = torch.sum(fn(u, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, u)[0]
        u.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(u.shape))))

    return div_fn


def get_likelihood_fn(config, sde):
    return get_ode_likelihood_fn(config, sde)


def get_ode_likelihood_fn(config, sde):
    '''
    Evaluating the likelihood (bound) of the model using the ProbabilityFlow formulation.
    '''
    def probability_flow_ode(model, u, t):
        score_fn = get_score_fn(config, sde, model, train=False)
        rsde = sde.get_reverse_sde(score_fn, probability_flow=True)
        return -rsde(u, 1. - t)[0]

    def div_fn(model, u, t, noise):
        return get_div_fn(lambda uu, tt: probability_flow_ode(model, uu, tt))(u, t, noise)

    def likelihood_fn(model, data):
        gc.collect()

        with torch.no_grad():
            shape = data.shape
            if config.likelihood_hutchinson_type == 'gaussian':
                epsilon = torch.randn_like(data)
            elif config.likelihood_hutchinson_type == 'rademacher':
                epsilon = torch.randint_like(
                    data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(
                    'Hutchinson type %s is not implemented.' % config.likelihood_hutchinson_type)

            def ode_func(t, state):
                global nfe_counter
                nfe_counter += 1
                sample = state[0]
                vec_t = torch.ones(
                    sample.shape[0], device=sample.device, dtype=torch.float64) * t
                dudt = probability_flow_ode(model, sample, vec_t)
                dlogpdt = div_fn(model, sample, vec_t, epsilon)
                return (dudt, dlogpdt)

            global nfe_counter
            nfe_counter = 0
            solution = odeint(ode_func,
                              (data, torch.zeros(
                                  shape[0], device=config.device, dtype=torch.float64)),
                              torch.tensor(
                                  [config.likelihood_eps, 1.], device=config.device),
                              rtol=config.likelihood_rtol,
                              atol=config.likelihood_atol,
                              method=config.likelihood_solver,
                              options=config.likelihood_solver_options)
            u_T = solution[0][-1]
            delta_logp = solution[1][-1]

            if sde.is_augmented:
                prior_logpx, prior_logpz = sde.prior_logp(u_T)
                nll = -(prior_logpx + prior_logpz + delta_logp)
            else:
                prior_logpx, _ = sde.prior_logp(u_T)
                nll = -(prior_logpx + delta_logp)
            return nll, u_T, nfe_counter
    return likelihood_fn
