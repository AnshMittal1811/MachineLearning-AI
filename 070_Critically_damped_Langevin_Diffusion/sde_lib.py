# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from util.utils import add_dimensions


class CLD(nn.Module):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn
        self.m_inv = config.m_inv
        self.f = 2. / np.sqrt(config.m_inv)
        self.g = 1. / self.f
        self.gamma = config.gamma
        self.numerical_eps = config.numerical_eps

    @property
    def type(self):
        return 'cld'

    @property
    def is_augmented(self):
        return True

    def sde(self, u, t):
        '''
        Evaluating drift and diffusion of the SDE.
        '''
        x, v = torch.chunk(u, 2, dim=1)

        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

        drift_x = self.m_inv * beta * v
        drift_v = -beta * x - self.f * self.m_inv * beta * v

        diffusion_x = torch.zeros_like(x)
        diffusion_v = torch.sqrt(2. * self.f * beta) * torch.ones_like(v)

        return torch.cat((drift_x, drift_v), dim=1), torch.cat((diffusion_x, diffusion_v), dim=1)

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            '''
            Evaluating drift and diffusion of the ReverseSDE.
            '''
            drift, diffusion = sde_fn(u, 1. - t)
            score = score if score is not None else score_fn(u, 1. - t)

            drift_x, drift_v = torch.chunk(drift, 2, dim=1)
            _, diffusion_v = torch.chunk(diffusion, 2, dim=1)

            reverse_drift_x = -drift_x
            reverse_drift_v = -drift_v + diffusion_v ** 2. * \
                score * (0.5 if probability_flow else 1.)

            reverse_diffusion_x = torch.zeros_like(diffusion_v)
            reverse_diffusion_v = torch.zeros_like(
                diffusion_v) if probability_flow else diffusion_v

            return torch.cat((reverse_drift_x, reverse_drift_v), dim=1), torch.cat((reverse_diffusion_x, reverse_diffusion_v), dim=1)

        return reverse_sde

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), torch.randn(*shape, device=self.config.device) / np.sqrt(self.m_inv)

    def prior_logp(self, u):
        x, v = torch.chunk(u, 2, dim=1)
        N = np.prod(x.shape[1:])

        logx = -N / 2. * np.log(2. * np.pi) - \
            torch.sum(x.view(x.shape[0], -1) ** 2., dim=1) / 2.
        logv = -N / 2. * np.log(2. * np.pi / self.m_inv) - torch.sum(
            v.view(v.shape[0], -1) ** 2., dim=1) * self.m_inv / 2.
        return logx, logv

    def mean(self, u, t):
        '''
        Evaluating the mean of the conditional perturbation kernel.
        '''
        x, v = torch.chunk(u, 2, dim=1)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        coeff_mean = torch.exp(-2. * beta_int * self.g)

        mean_x = coeff_mean * (2. * beta_int * self.g *
                               x + 4. * beta_int * self.g ** 2. * v + x)
        mean_v = coeff_mean * (-beta_int * x - 2. * beta_int * self.g * v + v)
        return torch.cat((mean_x, mean_v), dim=1)

    def var(self, t, var0x=None, var0v=None):
        '''
        Evaluating the variance of the conditional perturbation kernel.
        '''
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image)
        if var0v is None:
            if self.config.cld_objective == 'dsm':
                var0v = torch.zeros_like(
                    t, dtype=torch.float64, device=t.device)
            elif self.config.cld_objective == 'hsm':
                var0v = (self.gamma / self.m_inv) * torch.ones_like(t,
                                                                    dtype=torch.float64, device=t.device)

            var0v = add_dimensions(var0v, self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        multiplier = torch.exp(-4. * beta_int * self.g)

        var_xx = var0x + (1. / multiplier) - 1. + 4. * beta_int * self.g * (var0x - 1.) + 4. * \
            beta_int ** 2. * self.g ** 2. * \
            (var0x - 2.) + 16. * self.g ** 4. * beta_int ** 2. * var0v
        var_xv = -var0x * beta_int + 4. * self.g ** 2. * beta_int * var0v - 2. * self.g * \
            beta_int ** 2. * (var0x - 2.) - 8. * \
            self.g ** 3. * beta_int ** 2. * var0v
        var_vv = self.f ** 2. * ((1. / multiplier) - 1.) / 4. + self.f * beta_int - 4. * self.g * beta_int * \
            var0v + 4. * self.g ** 2. * beta_int ** 2. * \
            var0v + var0v + beta_int ** 2. * (var0x - 2.)
        return [var_xx * multiplier + self.numerical_eps, var_xv * multiplier, var_vv * multiplier + self.numerical_eps]

    def mean_and_var(self, u, t, var0x=None, var0v=None):
        return self.mean(u, t), self.var(t, var0x, var0v)

    def noise_multiplier(self, t, var0x=None, var0v=None):
        '''
        Evaluating the -\ell_t multiplier. Similar to -1/standard deviaton in VPSDE.
        '''
        var = self.var(t, var0x, var0v)
        coeff = torch.sqrt(var[0] / (var[0] * var[2] - var[1]**2))

        if torch.sum(torch.isnan(coeff)) > 0:
            raise ValueError('Numerical precision error.')

        return -coeff

    def loss_multiplier(self, t):
        '''
        Evaluating the "maximum likelihood" multiplier.
        '''
        return self.beta_fn(t) * self.f

    def perturb_data(self, batch, t, var0x=None, var0v=None):
        '''
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and 
        \gamma * M for HSM.
        '''
        mean, var = self.mean_and_var(batch, t, var0x, var0v)

        cholesky11 = (torch.sqrt(var[0]))
        cholesky21 = (var[1] / cholesky11)
        cholesky22 = (torch.sqrt(var[2] - cholesky21 ** 2.))

        if torch.sum(torch.isnan(cholesky11)) > 0 or torch.sum(torch.isnan(cholesky21)) > 0 or torch.sum(torch.isnan(cholesky22)) > 0:
            raise ValueError('Numerical precision error.')

        batch_randn = torch.randn_like(batch, device=batch.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        noise = torch.cat((noise_x, noise_v), dim=1)

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(
                u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn(*u.shape, device=u.device)

            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean
        return discrete_step_fn


class VPSDE(nn.Module):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn

    @property
    def type(self):
        return 'vpsde'

    @property
    def is_augmented(self):
        return False

    def sde(self, u, t):
        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

        drift = -0.5 * beta * u
        diffusion = torch.sqrt(beta) * torch.ones_like(u,
                                                       device=self.config.device)

        return drift, diffusion

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            drift, diffusion = sde_fn(u, 1. - t)
            score = score if score is not None else score_fn(u, 1. - t)

            reverse_drift = -drift + diffusion**2 * \
                score * (0.5 if probability_flow else 1.0)
            reverse_diffusion = torch.zeros_like(
                diffusion) if probability_flow else diffusion

            return reverse_drift, reverse_diffusion

        return reverse_sde

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), None

    def prior_logp(self, u):
        shape = u.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2. * np.pi) - torch.sum(u.view(u.shape[0], -1) ** 2., dim=1) / 2., None

    def var(self, t, var0x=None):
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

        coeff = torch.exp(-beta_int)
        return [1. - (1. - var0x) * coeff]

    def mean(self, x, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

        return x * torch.exp(-0.5 * beta_int)

    def mean_and_var(self, x, t, var0x=None):
        if var0x is None:
            var0x = torch.zeros_like(x, device=self.config.device)

        return self.mean(x, t), self.var(t, var0x)

    def noise_multiplier(self, t, var0x=None):
        _var = self.var(t, var0x)[0]
        return -1. / torch.sqrt(_var)

    def loss_multiplier(self, t):
        return 0.5 * self.beta_fn(t)

    def perturb_data(self, batch, t, var0x=None):
        mean, var = self.mean_and_var(batch, t, var0x)
        cholesky = torch.sqrt(var[0])

        batch_randn = torch.randn_like(batch, device=batch.device)
        noise = cholesky * batch_randn

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(
                u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn_like(u, device=u.device)
            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean
        return discrete_step_fn
