import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPMScheduler:
    def __init__(self, timesteps=1000, schedule="linear", device=None):
        self.timesteps = timesteps
        self.device = device if device is not None else torch.device('cpu')
        
        # Define beta schedule
        if schedule == "linear":
            self.betas = linear_beta_schedule(timesteps).to(self.device)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps).to(self.device)
            
        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, context, noise=None):
        """Training loss computation"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, context)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, context):
        """Single DDPM sampling step"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, context) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, context, shape, device, noise=None):
        """Complete sampling loop for image generation"""
        if noise is None:
            noise = torch.randn(shape, device=device)
            
        img = noise

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, device=device, dtype=torch.long),
                i,
                context
            )
        return img

    @torch.no_grad()
    def ddim_sample(self, model, context, shape, device, noise=None, eta=0.0):
        """DDIM sampling - faster than regular DDPM sampling"""
        if noise is None:
            noise = torch.randn(shape, device=device)
            
        img = noise
        timesteps = self.timesteps
        step_size = timesteps // 50  # Use 50 steps instead of 1000

        for i in reversed(range(0, timesteps, step_size)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(img, t, context)
            
            # Get alpha values
            alpha = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod[i - step_size] if i > 0 else torch.tensor(1.)
            
            # Compute sigma
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            
            # Compute "direction" pointing to x_t
            pred_x0 = (img - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            
            # Direction pointing to x_{t-1}
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * pred_noise
            
            # Random noise for stochasticity
            noise = sigma * torch.randn_like(img) if eta > 0 else 0
            
            # Compute x_{t-1}
            img = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise

        return img 