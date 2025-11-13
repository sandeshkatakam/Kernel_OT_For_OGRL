"""
MAFNet utilities - Adapted from OTFlowRL
Masked Autoencoder Flow for policy modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class MaskedLinear(nn.Linear):
    """Linear layer with learnable mask (MADE-style)"""
    
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


def made_masks(d, hidden, n_hidden):
    """Generate MADE masks for autoregressive transformation"""
    rng = np.random.RandomState(0)

    # Degrees
    in_deg = np.arange(1, d + 1)
    degrees = [rng.randint(1, d, size=hidden) for _ in range(n_hidden)]
    out_deg = np.arange(1, d + 1)

    masks = []
    # input -> h1
    masks.append(torch.from_numpy((in_deg[:, None] <= degrees[0][None, :]).astype(np.float32)).T)
    # h -> h
    for i in range(1, n_hidden):
        masks.append(torch.from_numpy((degrees[i - 1][:, None] <= degrees[i][None, :]).astype(np.float32)).T)
    # hL -> out
    masks.append(torch.from_numpy((degrees[-1][:, None] < out_deg[None, :]).astype(np.float32)).T)
    
    return masks


class ContextMADE(nn.Module):
    """MADE with state context conditioning"""
    
    def __init__(self, action_dim, state_dim, hidden=512, n_hidden=3, out_dim=None):
        super().__init__()
        self.d = action_dim
        out_dim = out_dim or (2 * action_dim)
        masks = made_masks(action_dim, hidden, n_hidden)

        self.in_lin = MaskedLinear(action_dim, hidden, masks[0])
        self.hiddens = nn.ModuleList([
            MaskedLinear(hidden, hidden, masks[i]) for i in range(1, n_hidden)
        ])
        
        # Output mask
        out_mask = masks[-1]
        out_mask_2d = torch.cat([out_mask, out_mask], dim=0)
        self.out_lin = MaskedLinear(hidden, out_dim, out_mask_2d)

        # Unmasked state context
        self.ctx_in = nn.Linear(state_dim, hidden)
        self.ctx_h = nn.ModuleList([nn.Linear(state_dim, hidden) for _ in range(n_hidden - 1)])
        self.act = nn.SiLU()
        self.ln_in = nn.LayerNorm(hidden)
        self.ln_h = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_hidden - 1)])

        nn.init.zeros_(self.out_lin.bias)
        nn.init.normal_(self.out_lin.weight, mean=0.0, std=5e-3)
    
    def forward(self, x_ar, s):
        """Forward pass with state conditioning"""
        h = self.in_lin(x_ar) + self.ctx_in(s)
        h = self.ln_in(self.act(h))
        for lin, ctx, ln in zip(self.hiddens, self.ctx_h, self.ln_h):
            h = lin(h) + ctx(s)
            h = ln(self.act(h))
        return self.out_lin(h)


class ActNorm(nn.Module):
    """Activation normalization layer"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.log_scale = nn.Parameter(torch.zeros(1, dim))
    
    def initialize(self, x):
        with torch.no_grad():
            m = x.mean(0, keepdim=True)
            v = x.var(0, unbiased=False, keepdim=True).clamp_min(self.eps)
            self.bias.data.copy_(-m)
            self.log_scale.data.copy_(-0.5 * torch.log(v))
            self.initialized.fill_(1)
    
    def forward(self, x):
        if self.initialized.item() == 0:
            self.initialize(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
        logdet = self.log_scale.sum(dim=1).expand(x.size(0))
        return y, logdet
    
    def inverse(self, y):
        x = y * torch.exp(-self.log_scale) - self.bias
        logdet = (-self.log_scale.sum(dim=1)).expand(y.size(0))
        return x, logdet


class AffineAutoregressiveBlock(nn.Module):
    """One affine autoregressive flow block"""
    
    def __init__(self, action_dim, state_dim, hidden=512, n_hidden=2, 
                 logscale_clip=(-5.0, 2.0)):
        super().__init__()
        self.cond = ContextMADE(action_dim, state_dim, hidden=hidden, 
                                 n_hidden=n_hidden, out_dim=2 * action_dim)
        self.actnorm = ActNorm(action_dim)
        self.logscale_clip = logscale_clip
    
    def forward(self, x, s):
        """Forward transformation with log-det"""
        y, logdet = self.actnorm(x)
        out = self.cond(y, s)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, *self.logscale_clip)
        z = (y - mu) * torch.exp(-log_sigma)
        logdet = logdet - log_sigma.sum(-1)
        return z, logdet
    
    def inverse(self, z, s):
        """Inverse transformation (autoregressive)"""
        B, D = z.shape
        y = torch.zeros_like(z)
        logdet_sigma = torch.zeros(B, device=z.device)
        
        for i in range(D):
            out = self.cond(y, s)
            mu, log_sigma = out.chunk(2, dim=-1)
            log_sigma = torch.clamp(log_sigma, *self.logscale_clip)
            y[:, i] = z[:, i] * torch.exp(log_sigma[:, i]) + mu[:, i]
            logdet_sigma += log_sigma[:, i]
        
        y_n, logdet_an = self.actnorm.inverse(y)
        return y_n, logdet_an + logdet_sigma


class MAFNet(nn.Module):
    """
    Masked Autoencoder Flow for π(a|s)
    
    Provides action sampling conditioned on state
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 n_blocks: int = 4, hidden: int = 512, device=None):
        """
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            n_blocks: Number of flow blocks
            hidden: Hidden dimension
            device: Device to use
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")
        
        # Build flow blocks
        self.blocks = nn.ModuleList([
            AffineAutoregressiveBlock(action_dim, state_dim, hidden=hidden, n_hidden=3)
            for _ in range(n_blocks)
        ])
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: z ~ N(0,I), a = f^{-1}(z|s)
        
        Args:
            s: States [B, state_dim]
            a: Actions [B, action_dim]
        
        Returns:
            z: Latent [B, action_dim]
            log_det: Log determinant [B]
        """
        z = a
        log_det = torch.zeros(s.shape[0], device=s.device)
        
        for block in self.blocks:
            z, ld = block.forward(z, s)
            log_det = log_det + ld
        
        return z, log_det
    
    def sample(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse: a = f^{-1}(z|s)
        
        Args:
            s: States [B, state_dim]
            z: Latent noise [B, action_dim]
        
        Returns:
            a: Sampled actions [B, action_dim]
        """
        a = z
        
        for block in reversed(self.blocks):
            a, _ = block.inverse(a, s)
        
        return a
    
    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability log π(a|s)
        
        Args:
            s: States [B, state_dim]
            a: Actions [B, action_dim]
        
        Returns:
            log_prob: [B] log probabilities
        """
        z, log_det = self.forward(s, a)
        
        # Standard normal log prob
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.action_dim * math.log(2 * math.pi)
        
        # Change of variables
        log_pa = log_pz + log_det
        
        return log_pa


if __name__ == "__main__":
    print("MAFNet Test")
    print("=" * 60)
    
    state_dim = 29
    action_dim = 8
    batch_size = 4
    
    maf = MAFNet(state_dim, action_dim, n_blocks=4)
    
    # Test sampling
    s = torch.randn(batch_size, state_dim)
    z = torch.randn(batch_size, action_dim)
    a = maf.sample(s, z)
    
    print(f"✓ Sampling: {s.shape}, {z.shape} → {a.shape}")
    
    # Test log prob
    log_p = maf.log_prob(s, a)
    print(f"✓ Log prob: {log_p.shape}")
    
    print("\n✓ All tests passed!")
