import torch
from torch import nn

class NoiseScheduler(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initialize a NoiseScheduler. This class follows the original DDPM 2020 for the forward process. Variance scheduler is linear.

        Parameters:
        T (int): The total number of timesteps.
        beta_start (float): The starting value of the beta schedule.
        beta_end (float): The ending value of the beta schedule.

        Returns:
        None
        """
        super().__init__()
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        
        alphas = 1. - betas
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def forward(self, x_0, t):
        """
        Forward process of a diffusion model. Denoise data at a given timestep using the closed form solution from DDPM (2020)

        Parameters:
        x_0 (torch.Tensor): The original data.
        t (int): The current timestep.

        Returns:
        torch.Tensor: The noisy data at the current timestep.
        torch.Tensor: The noise added at the current timestep.
        """
        noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].reshape(x_0.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(x_0.shape[0], 1, 1, 1)
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, noise