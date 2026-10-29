"""IR-SDE sampler: the parts of IRSDE that reverse_sde() uses.

From Algolzw/image-restoration-sde@2598d73 codes/utils/sde_utils.py. The maths is unchanged;
removed are training helpers, the ODE/posterior samplers, the tqdm bar, state saving and the
schedule print-outs.
"""
import math

import torch


class IRSDE:
    """Timestep t runs from 1 to T; state t=0 is never used."""

    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01):
        self.T = T
        self.dt = 1 / T
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            raise ValueError(f"unknown schedule: {schedule}")

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas
        self.sigmas = sigmas
        self.thetas_cumsum = thetas_cumsum
        self.sigma_bars = sigma_bars

        self.mu = 0.
        self.model = None

    def set_mu(self, mu):
        self.mu = mu

    def set_model(self, model):
        self.model = model

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt))

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_sde(self, xt, T=-1, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in reversed(range(1, T + 1)):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)
        return x

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma
