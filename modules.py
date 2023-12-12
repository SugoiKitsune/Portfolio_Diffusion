# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:37:35 2023

@author: Andrey
"""

import gc
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import tensorflow as tf
from attention import SelfAttention, CrossAttention

class MLP(nn.Module):
    def __init__(self, N=40, data_dim=5341, hidden_dim=64, d_cross=2100):
        super(MLP, self).__init__()

        # Add the first cross attention layer at the beginning
        self.first_cross_attention_layer = CrossAttention(n_heads=1, d_embed=data_dim, d_cross=d_cross)

        self.network_head = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Add the second cross attention layer in between
        self.cross_attention_layer = CrossAttention(n_heads=4, d_embed=hidden_dim, d_cross=d_cross)

        self.network_tail = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, data_dim * 2)
            ) for _ in range(N)
        ])

        # Add the third cross attention layer at the end
        self.last_cross_attention_layer = CrossAttention(n_heads=1, d_embed=data_dim*2, d_cross=d_cross)

    def forward(self, x, y, t: int):
        x = x.to(self.network_head[0].weight.device)
        y = y.to(self.network_head[0].weight.device)

        # Apply the first cross attention layer
        x = self.first_cross_attention_layer(x, y)

        h = self.network_head(x)

        # Apply the second cross attention layer
        k = self.cross_attention_layer(h, y)

        p = self.network_tail[t](k)

        # Apply the third cross attention layer
        output = self.last_cross_attention_layer(p, y)

        output = output.view(output.size(0), -1)
        return output

class SimpleDiffusion(nn.Module):
    def __init__(self, model: nn.Module, n_steps=40, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        betas = torch.linspace(-18, 10, n_steps)
        self.beta = torch.sigmoid(betas) * (3e-1 - 1e-5) + 1e-5

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def forward_process(self, x0, t):

        t = t - 1  # Start indexing at 0
        beta_forward = self.beta[t]
        alpha_forward = self.alpha[t]
        alpha_cum_forward = self.alpha_bar[t]
        xt = x0 * torch.sqrt(alpha_cum_forward) + torch.randn_like(x0) * torch.sqrt(1. - alpha_cum_forward)
        # Ensure that the dimensions of xt are correct for the subsequent operations
        #assert xt.size(-2) == self.model.network_head[0].in_features, "Dimension mismatch in xt"
                
        mu1_scl = torch.sqrt(alpha_cum_forward / alpha_forward)
        mu2_scl = 1. / torch.sqrt(alpha_forward)    
        cov1 = 1. - alpha_cum_forward / alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = 1. / cov1 + 1. / cov2
        mu = (x0 * mu1_scl / cov1 + xt * mu2_scl / cov2) / lam
        sigma = torch.sqrt(1. / lam)
        return mu, sigma, xt

    def reverse(self, xt, y, t):
        # Assuming self.model.parameters() returns a generator of model parameters
        model_params = list(self.model.parameters())

        # Convert the input tensors to the same data type as model parameters
        xt = xt.to(model_params[0].dtype)
        y = y.to(model_params[0].dtype)
        
        t = t - 1  # Start indexing at 0
        if t == 0: 
            return None, None, xt
        mu, h = self.model(xt, y, t).chunk(2, dim=1)
        #  output = self.model(xt, t)
        #mu, h = torch.split(output, output.size(1)//2, dim=1)
        # Print dimensions for debugging
        #print(f"mu: {mu.size()}, xt: {xt.size()}, h: {h.size()}")
        
        sigma = torch.sqrt(torch.exp(h))
        noise = torch.randn_like(xt[:, :mu.size(1)])
        samples = mu + noise * sigma
        return mu, sigma, samples

    def sample(self, size, y, device):
        noise = torch.randn((size, self.model.network_head[0].in_features)).to(device)
        samples = [noise]
        for t in range(self.n_steps):
            _, _, x = self.reverse(samples[-1], y, self.n_steps - t - 1 + 1)
            samples.append(x)
        return samples
    
    
    
    
class GaussianDiffusion(nn.Module):
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(     
        self,
        model: nn.Module,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
        ):
        
        super(GaussianDiffusion, self).__init__()  # Add this line
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model. 

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise    
    
    
    
    
    
    
    