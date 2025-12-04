"""
Kernel Module
Fixed (non-learnable) kernel implementations for COT loss
"""

import torch
import torch.nn as nn


class IMQKernel(nn.Module):
    """
    Inverse Multi-Quadratic (IMQ) Kernel
    
    K(x, y) = (c + ||x - y||^2)^(-beta)
    
    This kernel is FIXED (non-learnable) and serves as the embedding metric
    for matching kernel embeddings across transitions and goals.
    
    Reference: Gretton et al., "A Kernel Method for the Two-Sample Problem" (2012)
    """
    
    def __init__(self, alpha=1.0, c=1.0, beta=0.5):
        """
        Args:
            alpha: legacy parameter (deprecated, kept for backward compatibility)
            c: offset parameter, controls kernel offset
               larger c -> smoother kernel
               typically 1.0 works well
            beta: power parameter, controls kernel decay rate
                  typically 0.5 works well
        """
        super().__init__()
        self.c = c
        self.beta = beta
        self.alpha = alpha  # kept for backward compatibility
        self.register_buffer('_c', torch.tensor(c))
        self.register_buffer('_beta', torch.tensor(beta))
        
    def forward(self, x, y=None, c=None, beta=None):
        """
        Compute IMQ kernel between points
        
        K(x,y) = (c + ||x - y||^2)^(-beta)
        
        Args:
            x: tensor of shape [n, d]
            y: tensor of shape [m, d]
               if None, computes self-similarity
            c: optional override for c parameter
            beta: optional override for beta parameter
        
        Returns:
            K: kernel matrix of shape [n, m]
        """
        
        if y is None:
            y = x
        
        # Use provided parameters or defaults
        c_val = torch.tensor(c) if c is not None else self._c
        beta_val = torch.tensor(beta) if beta is not None else self._beta
        
        # Compute squared Euclidean distance: ||x - y||^2
        dist_sq = self._pairwise_dist_sq(x, y)
        
        # K(x,y) = (c + ||x - y||^2)^(-beta)
        kernel = torch.pow(c_val + dist_sq, -beta_val)
        
        return kernel
    
    def _pairwise_dist_sq(self, x, y):
        """
        Compute pairwise squared Euclidean distances
        ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i^T * y_j
        
        Args:
            x: [n, d]
            y: [m, d]
        
        Returns:
            dist_sq: [n, m]
        """
        
        # Norms: [n, 1], [m, 1]
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm = (y ** 2).sum(dim=-1, keepdim=True)
        
        # Cross term: [n, m]
        xy = torch.matmul(x, y.t())
        
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        dist_sq = x_norm + y_norm.t() - 2 * xy
        
        # Numerical stability
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        return dist_sq



class RBFKernel(nn.Module):
    """
    Radial Basis Function (RBF) Kernel (alternative, not used in main experiment)
    
    K(x, y) = exp(-gamma * ||x - y||^2)
    
    Included for reference/comparison
    """
    
    def __init__(self, gamma=1.0):
        """
        Args:
            gamma: bandwidth parameter
                   larger gamma -> narrower kernel
        """
        super().__init__()
        self.gamma = gamma
        self.register_buffer('_gamma', torch.tensor(gamma))
        
    def forward(self, x, y=None):
        """
        Compute RBF kernel: K(x,y) = exp(-gamma * ||x - y||^2)
        
        Args:
            x: tensor of shape [n, d]
            y: tensor of shape [m, d], if None uses x
        
        Returns:
            K: kernel matrix of shape [n, m]
        """
        
        if y is None:
            y = x
        
        dist_sq = self._pairwise_dist_sq(x, y)
        kernel = torch.exp(-self._gamma * dist_sq)
        
        return kernel
    
    def _pairwise_dist_sq(self, x, y):
        """Compute pairwise squared Euclidean distances"""
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm = (y ** 2).sum(dim=-1, keepdim=True)
        xy = torch.matmul(x, y.t())
        dist_sq = x_norm + y_norm.t() - 2 * xy
        return torch.clamp(dist_sq, min=0.0)


def get_kernel(kernel_type='IMQ', **kwargs):
    """
    Factory function to get kernel instances
    
    Args:
        kernel_type: 'IMQ' or 'RBF'
        **kwargs: kernel-specific parameters
    
    Returns:
        kernel: instance of kernel class
    """
    
    if kernel_type == 'IMQ':
        alpha = kwargs.get('alpha', 1.0)
        return IMQKernel(alpha=alpha)
    
    elif kernel_type == 'RBF':
        gamma = kwargs.get('gamma', 1.0)
        return RBFKernel(gamma=gamma)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


# Test usage
if __name__ == "__main__":
    
    # Create kernel
    kernel = IMQKernel(alpha=1.0)
    
    # Test data
    x = torch.randn(10, 5)
    y = torch.randn(8, 5)
    
    # Compute kernel
    K_xy = kernel(x, y)
    
    print("IMQ Kernel Test")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  K(x,y) shape: {K_xy.shape}")
    print(f"  K(x,y) range: [{K_xy.min():.4f}, {K_xy.max():.4f}]")
    print(f"  K values sum to ~0.5-1.0 for close points")
    
    # Self-similarity
    K_xx = kernel(x)
    print(f"\n  K(x,x) shape: {K_xx.shape}")
    print(f"  K(x,x) diagonal mean: {K_xx.diag().mean():.4f}")
