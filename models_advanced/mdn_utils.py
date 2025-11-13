"""
MDN Model utilities - Adapted from OTFlowRL
Mixture Density Network for action-conditioned transition modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_tril(param: torch.Tensor,
                d: int,
                min_diag: float = 1e-3,
                max_diag: float = 50.0,
                offdiag_scale: float = 0.5,
                temp: float | None = None) -> torch.Tensor:
    """
    Build lower-triangular L with stable diagonals and controlled off-diagonals
    
    Args:
        param: Raw parameters to convert to triangular matrix
        d: Dimension of the triangular matrix
        min_diag: Minimum diagonal value
        max_diag: Maximum diagonal value (stability cap)
        offdiag_scale: Scale factor for off-diagonals relative to diagonals
        temp: Temperature scaling (optional)
    
    Returns:
        L: Stable lower-triangular matrix
    """
    expected = d * (d + 1) // 2
    if param.shape[-1] != expected:
        raise ValueError(f"stable_tril: last dim={param.shape[-1]} but expected {expected} for d={d}")

    L = param.new_zeros(*param.shape[:-1], d, d)
    i, j = torch.tril_indices(d, d, 0)
    L[..., i, j] = param

    # Split diag/offdiag views
    diag_view = torch.diagonal(L, dim1=-2, dim2=-1)
    off_mask = torch.tril(torch.ones(d, d, device=L.device, dtype=torch.bool), diagonal=-1)

    # Stabilize diagonals
    diag_pos = F.softplus(diag_view) + float(min_diag)
    if max_diag is not None:
        diag_pos = torch.clamp(diag_pos, max=float(max_diag))
    L = L + torch.diag_embed(diag_pos - diag_view)

    # Tame off-diagonals
    off_view = L.masked_select(off_mask).view(*L.shape[:-2], off_mask.sum())
    raw_off = off_view
    tamed_off = torch.tanh(raw_off)
    
    rows, cols = torch.tril_indices(d, d, -1)
    row_diag = diag_pos[..., rows]
    tamed_off = tamed_off * (offdiag_scale * row_diag)
    
    L = L.clone()
    L[..., rows, cols] = tamed_off

    if temp is not None:
        t = torch.clamp(torch.as_tensor(temp, dtype=L.dtype, device=L.device), min=1e-8)
        L = L * t

    return L


def log_prob_gaussian_full_safe(x, mu, L, diag_floor: float = 1e-6):
    """
    Numerically stable log probability of Gaussian with diagonal + low-rank structure
    
    Computes log N(x|mu, Σ) where Σ = L L^T using fp64 internally
    
    Args:
        x: Data points
        mu: Gaussian means
        L: Lower triangular Cholesky factor
        diag_floor: Minimum diagonal value for numerical stability
    
    Returns:
        log_prob: Log probability values
    """
    # Promote to float64 for numerical stability
    x64 = x.double()
    mu64 = mu.double()
    L64 = L.double()

    d = x64.shape[-1]

    # Re-floor the diagonal
    diag = torch.diagonal(L64, dim1=-2, dim2=-1)
    diag_safe = torch.clamp(diag, min=float(diag_floor))
    L64 = L64 + torch.diag_embed(diag_safe - diag)

    # Solve L y = (x - mu)
    xc = x64 - mu64
    y = torch.linalg.solve_triangular(L64, xc.unsqueeze(-1), upper=False).squeeze(-1)

    # Guard extreme magnitudes
    y = torch.clamp(y, min=-1e6, max=1e6)

    quad = (y * y).sum(dim=-1)
    log_det = torch.log(diag_safe).sum(dim=-1)

    out64 = -0.5 * (quad + d * math.log(2.0 * math.pi)) - log_det
    return out64.to(x.dtype)


class MDNFullCov(nn.Module):
    """
    Mixture Density Network with full covariance
    
    Models p(y|x) as mixture of K Gaussians with full covariance matrices
    Learned via Cholesky decomposition for numerical stability
    """
    
    def __init__(self, state_dim, action_dim, K=5, hidden=512):
        """
        Args:
            state_dim: Dimension of next-state output
            action_dim: Dimension of action input
            K: Number of mixture components
            hidden: Hidden layer dimension
        """
        super().__init__()
        self.d = state_dim
        self.K = K
        in_dim = state_dim + action_dim
        out_mu = K * state_dim
        out_L = K * (state_dim * (state_dim + 1) // 2)
        out_pi = K

        # Shared encoder
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        
        # Output heads for mixture parameters
        self.head_mu = nn.Linear(hidden, out_mu)
        self.head_L = nn.Linear(hidden, out_L)
        self.head_pi = nn.Linear(hidden, out_pi)

        # Stable initialization
        nn.init.zeros_(self.head_mu.bias)
        nn.init.constant_(self.head_L.bias, -2.0)
        nn.init.zeros_(self.head_pi.bias)
        nn.init.xavier_uniform_(self.head_mu.weight)
        nn.init.xavier_uniform_(self.head_L.weight, gain=0.1)
        nn.init.xavier_uniform_(self.head_pi.weight)
    
    def forward_params(self, s, a):
        """
        Compute MDN parameters
        
        Args:
            s: States [B, state_dim]
            a: Actions [B, action_dim]
        
        Returns:
            log_pi: [B, K] log mixture weights
            mu: [B, K, state_dim] component means
            L: [B, K, state_dim, state_dim] Cholesky factors
        """
        h = self.net(torch.cat([s, a], dim=-1))
        B = s.shape[0]
        K, d = self.K, self.d
        
        mu = self.head_mu(h).view(B, K, d)
        Lv = self.head_L(h).view(B, K, d * (d + 1) // 2)
        L = stable_tril(Lv, d)
        
        logit_pi = self.head_pi(h)
        log_pi = logit_pi - torch.logsumexp(logit_pi, dim=-1, keepdim=True)
        
        return log_pi, mu, L
    
    def log_prob(self, s, a, sp):
        """
        Compute log probability log p(s'|s,a)
        
        Args:
            s: Current states [B, state_dim]
            a: Actions [B, action_dim]
            sp: Next states [B, state_dim]
        
        Returns:
            log_prob: [B] log probabilities
        """
        log_pi, mu, L = self.forward_params(s, a)
        
        # Component log probs: [B, K]
        comp = log_prob_gaussian_full_safe(sp.unsqueeze(1), mu, L)
        
        # Mixture log prob: log sum exp
        return torch.logsumexp(log_pi + comp, dim=1)
    
    def sample(self, s, a, n_samples=1):
        """
        Sample from the mixture
        
        Args:
            s: States [B, state_dim]
            a: Actions [B, action_dim]
            n_samples: Number of samples per (s,a) pair
        
        Returns:
            samples: [B, n_samples, state_dim]
        """
        log_pi, mu, L = self.forward_params(s, a)
        B = s.shape[0]
        
        # Sample component indices
        pi = torch.exp(log_pi)
        k_indices = torch.multinomial(pi, n_samples, replacement=True)
        # [B, n_samples]
        
        # Gather parameters
        b_idx = torch.arange(B, device=s.device).unsqueeze(1).repeat(1, n_samples).reshape(B * n_samples)
        k_idx = k_indices.reshape(B * n_samples)
        
        mu_k = mu[b_idx, k_idx]  # [B*n_samples, d]
        L_k = L[b_idx, k_idx]    # [B*n_samples, d, d]
        
        # Sample epsilon
        eps = torch.randn(B * n_samples, self.d, device=s.device)
        
        # Transform: s' = mu + L @ eps
        samples = mu_k + torch.matmul(L_k, eps.unsqueeze(-1)).squeeze(-1)
        
        return samples.reshape(B, n_samples, self.d)
