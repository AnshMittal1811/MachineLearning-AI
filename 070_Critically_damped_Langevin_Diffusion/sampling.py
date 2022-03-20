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


def get_sampling_fn(config, sde, sampling_shape, eps):
    sampler_name = config.sampling_method
    if sampler_name == 'ode':
        return get_ode_sampler(config, sde, sampling_shape, eps)
    elif sampler_name == 'em':
        return get_em_sampler(config, sde, sampling_shape, eps)
    elif sampler_name == 'sscs':
        return get_sscs_sampler(config, sde, sampling_shape, eps)
    else:
        raise NotImplementedError(
            'Sampler %s is not implemened.' % sampler_name)


def get_ode_sampler(config, sde, sampling_shape, eps):
    ''' 
    Sampling from ProbabilityFlow formulation. 
    '''
    gc.collect()

    def denoising_fn(model, u, t):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, eps)
        return u_mean

    def probability_flow_ode(model, u, t):
        ''' 
        The "Right-Hand Side" of the ODE. 
        '''
        score_fn = get_score_fn(config, sde, model, train=False)
        rsde = sde.get_reverse_sde(score_fn, probability_flow=True)
        return rsde(u, t)[0]

    def ode_sampler(model, u=None):
        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    u = x

            def ode_func(t, u):
                global nfe_counter
                nfe_counter += 1
                vec_t = torch.ones(
                    sampling_shape[0], device=u.device, dtype=torch.float64) * t
                dudt = probability_flow_ode(model, u, vec_t)
                return dudt

            global nfe_counter
            nfe_counter = 0
            time_tensor = torch.tensor(
                [0., 1. - eps], dtype=torch.float64, device=config.device)
            solution = odeint(ode_func,
                              u,
                              time_tensor,
                              rtol=config.sampling_rtol,
                              atol=config.sampling_atol,
                              method=config.sampling_solver,
                              options=config.sampling_solver_options)

            u = solution[-1]

            if config.denoising:
                u = denoising_fn(model, u, 1. - eps)
                nfe_counter += 1

            if sde.is_augmented:
                x, v = torch.chunk(u, 2, dim=1)
                return x, v, nfe_counter
            else:
                return u, None, nfe_counter

    return ode_sampler


def get_em_sampler(config, sde, sampling_shape, eps):
    ''' 
    Sampling from the ReverseSDE using Euler--Maruyama. 
    '''

    gc.collect()

    def step_fn(model, u, t, dt):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, dt)
        return u, u_mean

    def em_sampler(model, u=None):
        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    u = x

            n_discrete_steps = config.n_discrete_steps if not config.denoising else config.n_discrete_steps - 1
            t_final = 1. - eps
            t = torch.linspace(
                0., t_final, n_discrete_steps + 1, dtype=torch.float64)
            if config.striding == 'linear':
                pass
            elif config.striding == 'quadratic':
                t = t_final * torch.flip(1 - (t / t_final) ** 2., dims=[0])

            for i in range(n_discrete_steps):
                dt = t[i + 1] - t[i]
                u, _ = step_fn(model, u, t[i], dt)

            if config.denoising:
                _, u = step_fn(model, u, 1. - eps, eps)

            if sde.is_augmented:
                x, v = torch.chunk(u, 2, dim=1)
                return x, v, config.n_discrete_steps
            else:
                return u, None, config.n_discrete_steps

    return em_sampler


def get_sscs_sampler(config, sde, sampling_shape, eps):
    ''' 
    Sampling from the ReverseSDE using our SSCS. Only applicable to CLD-SGM.
    '''

    gc.collect()

    n_discrete_steps = config.n_discrete_steps if not config.denoising else config.n_discrete_steps - 1
    t_final = 1. - eps
    t = torch.linspace(0., t_final, n_discrete_steps + 1, dtype=torch.float64)
    if config.striding == 'linear':
        pass
    elif config.striding == 'quadratic':
        t = t_final * torch.flip(1 - (t / t_final) ** 2., dims=[0])

    beta_fn = sde.beta_fn
    beta_int_fn = sde.beta_int_fn
    num_stab = config.sscs_num_stab

    def denoising_fn(model, u, t):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, eps)
        return u_mean

    def compute_mean_of_analytical_dynamics(u, t, dt):
        B = (beta_int_fn(1. - (t + dt)) - beta_int_fn(1. - t))

        x, v = torch.chunk(u, 2, dim=1)
        coeff = torch.exp(2. * sde.g * B)

        mean_x = coeff * ((1. - 2. * sde.g * B) * x + 4. * sde.g ** 2. * B * v)
        mean_v = coeff * (-B * x + (1. + 2. * sde.g * B) * v)
        return torch.cat((mean_x, mean_v), dim=1)

    def compute_variance_of_analytical_dynamics(t, dt):
        B = beta_int_fn(1. - (t + dt)) - beta_int_fn(1. - t)
        coeff = torch.exp(4. * sde.g * B)
        var_xx = coeff * (1. / coeff - 1. + 4. * sde.g *
                          B - 8. * sde.g**2 * B ** 2.)
        var_xv = -coeff * (4. * sde.g * B ** 2.)
        var_vv = coeff * (-sde.f ** 2. * (-(1. / coeff) +
                          1.) / 4. - sde.f * B - 2. * B ** 2.)

        return [var_xx + num_stab, var_xv, var_vv + num_stab]

    def analytical_dynamics(u, t, dt, half_step):
        if half_step:
            dt_hd = dt / 2.
        else:
            dt_hd = dt

        mean = compute_mean_of_analytical_dynamics(u, t, dt_hd)
        var = compute_variance_of_analytical_dynamics(t, dt_hd)

        cholesky11 = (torch.sqrt(var[0]))
        cholesky21 = (var[1] / cholesky11)
        cholesky22 = (torch.sqrt(var[2] - cholesky21 ** 2.))

        if torch.sum(torch.isnan(cholesky11)) > 0 or torch.sum(torch.isnan(cholesky22)) > 0:
            raise ValueError('Numerical precision error.')

        batch_randn = torch.randn_like(u, device=u.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        noise = torch.cat((noise_x, noise_v), dim=1)

        mean = mean
        noise = noise
        perturbed_data = mean + noise
        return perturbed_data

    def euler_score_dynamics(model, u, t, dt, half_step):
        if half_step:
            raise ValueError('Avoid half steps in score dynamics.')

        score_fn = get_score_fn(config, sde, model, train=False)
        score = score_fn(u, torch.ones(
            u.shape[0], device=u.device, dtype=torch.float64) * (1. - t))

        x, v = torch.chunk(u, 2, dim=1)
        v_new = v + 2. * sde.f * (score + sde.m_inv * v) * beta_fn(1. - t) * dt

        return torch.cat((x, v_new), dim=1)

    def sscs_sampler(model, u=None):
        ''' 
        The SSCS sampler takes analytical "half-steps" for the Ornstein--Uhlenbeck
        and the Hamiltonian components, and evaluates the score model using "full-steps". 
        '''

        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    raise ValueError('SSCS sampler does only work for CLD.')
            else:
                if not sde.is_augmented:
                    raise ValueError('SSCS sampler does only work for CLD.')

            for i in range(n_discrete_steps):
                dt = t[i + 1] - t[i]
                u = analytical_dynamics(u, t[i], dt, True)
                u = euler_score_dynamics(model, u, t[i], dt, False)
                u = analytical_dynamics(u, t[i], dt, True)

            if config.denoising:
                u = denoising_fn(model, u, 1.0 - eps)

            x, v = torch.chunk(u, 2, dim=1)
            return x, v, config.n_discrete_steps

    return sscs_sampler
