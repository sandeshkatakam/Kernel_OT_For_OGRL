"""
Policy-Augmented Action-Conditioned Transition Model

Architecture:
  q_θ(s'|s, π(s)) where:
  - π(a|s): Policy model (frozen, pretrained MAFNet)
  - a ~ π(a|s): Sample multiple actions from policy
  - q_θ(s'|s,a): Action-conditioned transition kernel (MDN, learnable)

This generates multiple candidate trajectories for a fixed state by:
  1. Sampling actions from frozen policy π(a|s)
  2. For each action, generating next-state distributions via q_θ(s'|s,a)
  3. Providing multi-sample transitions for kernel OT optimization
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Tuple, Optional, Dict
from pathlib import Path

# Handle both module and script imports
try:
    from .mdn_utils import MDNFullCov, log_prob_gaussian_full_safe
    from .maf_utils import MAFNet
except (ImportError, ValueError):
    # If relative import fails, try absolute import (for script execution)
    sys.path.insert(0, str(Path(__file__).parent))
    from mdn_utils import MDNFullCov, log_prob_gaussian_full_safe
    from maf_utils import MAFNet


class FrozenPolicyModel(nn.Module):
    """
    Frozen policy model π(a|s)
    
    Loads pretrained MAFNet checkpoint and prevents gradient updates
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 ckpt_path: Optional[str] = None, device: torch.device = None):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            ckpt_path: Path to pretrained MAFNet checkpoint (optional)
            device: Device to load model on
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load pretrained MAFNet
        self.maf_net = MAFNet(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)
        
        # Load checkpoint
        if ckpt_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # Try different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # Handle checkpoint with 'model' key (preferred)
                    self.maf_net.load_state_dict(checkpoint['model'], strict=False)
                    print(f"✓ Loaded pretrained policy from {ckpt_path} (model key)")
                elif 'model_state_dict' in checkpoint:
                    # Handle checkpoint with 'model_state_dict' key
                    self.maf_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"✓ Loaded pretrained policy from {ckpt_path} (model_state_dict key)")
                else:
                    # Try direct load
                    self.maf_net.load_state_dict(checkpoint, strict=False)
                    print(f"✓ Loaded pretrained policy from {ckpt_path} (direct)")
            else:
                self.maf_net.load_state_dict(checkpoint, strict=False)
                print(f"✓ Loaded pretrained policy from {ckpt_path}")
        elif ckpt_path is not None:
            print(f"⚠ Warning: Policy checkpoint not found at {ckpt_path}")
        else:
            print(f"⊘ Using random-initialized policy (no checkpoint provided)")
        
        # Freeze all parameters
        for param in self.maf_net.parameters():
            param.requires_grad = False
        
        self.maf_net.eval()
    
    @torch.no_grad()
    def sample_actions(self, s: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample multiple actions from π(a|s)
        
        Args:
            s: State tensor [batch_size, state_dim]
            n_samples: Number of action samples per state
        
        Returns:
            actions: [batch_size, n_samples, action_dim]
        """
        batch_size = s.shape[0]
        s = s.to(self.device)
        
        # Sample z ~ N(0,I)
        z = torch.randn(batch_size, n_samples, self.action_dim, device=self.device)
        
        # Reshape for batch processing: [batch_size * n_samples, action_dim]
        z_flat = z.reshape(batch_size * n_samples, self.action_dim)
        
        # Repeat state
        s_rep = s.unsqueeze(1).repeat(1, n_samples, 1)  # [batch_size, n_samples, state_dim]
        s_flat = s_rep.reshape(batch_size * n_samples, self.state_dim)
        
        # Forward through MAF
        a_flat = self.maf_net.sample(s_flat, z_flat)
        
        # Reshape back
        actions = a_flat.reshape(batch_size, n_samples, self.action_dim)
        
        return actions


class ActionConditionedTransitionKernel(nn.Module):
    """
    Learnable action-conditioned transition kernel q_θ(s'|s,a)
    
    Uses MDN with K mixture components to model distribution over next states
    given current state and action
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 n_components: int = 5, hidden_dim: int = 512):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            n_components: Number of mixture components in MDN
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_components = n_components
        
        # MDN for q(s'|s,a)
        self.mdn = MDNFullCov(
            state_dim=state_dim,
            action_dim=action_dim,
            K=n_components,
            hidden=hidden_dim
        )
    
    def forward(self, s: torch.Tensor, a: torch.Tensor, 
                n_samples: int = 1) -> torch.Tensor:
        """
        Sample next states from q_θ(s'|s,a)
        
        Args:
            s: Current states [batch_size, state_dim]
            a: Actions [batch_size, action_dim] or [batch_size, n_actions, action_dim]
            n_samples: Number of samples per (s,a) pair
        
        Returns:
            s_next: Sampled next states [batch_size, n_actions*n_samples, state_dim]
                    or [batch_size*n_actions, n_samples, state_dim]
        """
        
        batch_size = s.shape[0]
        
        # Handle different action input shapes
        if a.dim() == 2:
            # Single action per state: [batch_size, action_dim]
            n_actions = 1
            a_flat = a  # [batch_size, action_dim]
        elif a.dim() == 3:
            # Multiple actions per state: [batch_size, n_actions, action_dim]
            n_actions = a.shape[1]
            a_flat = a.reshape(batch_size * n_actions, self.action_dim)
        else:
            raise ValueError(f"Action tensor must be 2D or 3D, got {a.dim()}D")
        
        # Replicate states for each action
        if n_actions > 1:
            s_rep = s.unsqueeze(1).repeat(1, n_actions, 1)
            s_flat = s_rep.reshape(batch_size * n_actions, self.state_dim)
        else:
            s_flat = s
        
        # Get MDN parameters: log_pi [B*na, K], mu [B*na, K, d], L [B*na, K, d, d]
        log_pi, mu, L = self.mdn.forward_params(s_flat, a_flat)
        
        # Sample from mixture
        s_next_samples = self._sample_from_mdn(log_pi, mu, L, n_samples)
        # [B*na, n_samples, state_dim]
        
        if n_actions > 1:
            # Reshape back to [batch_size, n_actions, n_samples, state_dim]
            s_next = s_next_samples.reshape(batch_size, n_actions, n_samples, self.state_dim)
            # Flatten last two dims: [batch_size, n_actions*n_samples, state_dim]
            s_next = s_next.reshape(batch_size, n_actions * n_samples, self.state_dim)
        else:
            # [batch_size, n_samples, state_dim]
            s_next = s_next_samples
        
        return s_next
    
    def _sample_from_mdn(self, log_pi: torch.Tensor, mu: torch.Tensor, 
                        L: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample from mixture of Gaussians
        
        Args:
            log_pi: [B, K] log mixture weights
            mu: [B, K, d] means
            L: [B, K, d, d] Cholesky factors
            n_samples: Number of samples
        
        Returns:
            samples: [B, n_samples, d]
        """
        
        B, K, d = mu.shape
        device = mu.device
        
        # Sample mixture component per sample: [B, n_samples]
        pi = torch.exp(log_pi)  # [B, K]
        k_indices = torch.multinomial(pi, n_samples, replacement=True)
        # k_indices: [B, n_samples]
        
        # Gather means and L's for sampled components
        k_flat = k_indices.reshape(B * n_samples)  # [B*n_samples]
        b_indices = torch.arange(B, device=device).unsqueeze(1).repeat(1, n_samples).reshape(B * n_samples)
        
        mu_sampled = mu[b_indices, k_flat]  # [B*n_samples, d]
        L_sampled = L[b_indices, k_flat]    # [B*n_samples, d, d]
        
        # Sample epsilon ~ N(0,I)
        epsilon = torch.randn(B * n_samples, d, device=device)
        
        # s' = mu + L @ epsilon
        s_next = mu_sampled + torch.matmul(L_sampled, epsilon.unsqueeze(-1)).squeeze(-1)
        
        # Reshape back: [B, n_samples, d]
        s_next = s_next.reshape(B, n_samples, d)
        
        return s_next
    
    def log_prob(self, s: torch.Tensor, a: torch.Tensor, 
                 s_next: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability log q_θ(s'|s,a)
        
        Args:
            s: Current states [batch_size, state_dim]
            a: Actions [batch_size, action_dim]
            s_next: Next states [batch_size, state_dim]
        
        Returns:
            log_prob: [batch_size] log probabilities
        """
        return self.mdn.log_prob(s, a, s_next)


class PolicyAugmentedTransitionModel(nn.Module):
    """
    Combined policy-augmented model: q_θ(s'|s, π(s))
    
    Integration point for Kernel OT learning:
    - Frozen policy π(a|s) sampler
    - Learnable action-conditioned kernel q_θ(s'|s,a)
    - Compatible with existing Kernel OT losses
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 policy_ckpt: Optional[str] = None, n_components: int = 5, 
                 hidden_dim: int = 512, device: torch.device = None):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            policy_ckpt: Path to pretrained MAFNet policy (optional)
            n_components: MDN mixture components
            hidden_dim: Hidden layer dimension
            device: Computation device
        """
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Frozen policy model
        self.policy = FrozenPolicyModel(
            state_dim=state_dim,
            action_dim=action_dim,
            ckpt_path=policy_ckpt,
            device=self.device
        )
        
        # Learnable transition kernel
        self.kernel = ActionConditionedTransitionKernel(
            state_dim=state_dim,
            action_dim=action_dim,
            n_components=n_components,
            hidden_dim=hidden_dim
        ).to(self.device)
    
    def forward(self, s: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generate next-state samples: q_θ(s'|s, π(s))
        
        Compatible with Kernel OT interface: forward(s, n_samples) -> s_next
        
        Args:
            s: States [batch_size, state_dim]
            n_samples: Samples per state
        
        Returns:
            s_next: [batch_size, n_samples, state_dim]
                   Multi-sample transitions from fixed policy + learnable kernel
        """
        
        batch_size = s.shape[0]
        s = s.to(self.device)
        
        # Sample actions from policy: [batch_size, n_samples, action_dim]
        actions = self.policy.sample_actions(s, n_samples=n_samples)
        
        # For each state, we have n_samples actions
        # Generate next states for each (s, a) pair
        # Shape: [batch_size, n_samples, state_dim]
        s_next = self.kernel.forward(s, actions, n_samples=1)
        
        # Note: kernel.forward will expand to [batch_size, n_samples, state_dim]
        # by treating each action as separate
        
        return s_next
    
    def log_prob(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        Compute log prob: log q_θ(s'|s, π(s))
        
        Args:
            s: States [batch_size, state_dim]
            s_next: Next states [batch_size, state_dim]
        
        Returns:
            log_prob: [batch_size] log probabilities
        """
        
        s = s.to(self.device)
        s_next = s_next.to(self.device)
        
        # Note: For proper probabilistic interpretation, would need to marginalize
        # over policy samples. For Kernel OT, we use kernel embeddings instead.
        # This is a placeholder for compatibility.
        
        batch_size = s.shape[0]
        
        # Sample single action per state as representative
        with torch.no_grad():
            actions = self.policy.sample_actions(s, n_samples=1).squeeze(1)
        
        return self.kernel.log_prob(s, actions, s_next)
    
    def get_learnable_parameters(self):
        """Return only the learnable parameters (kernel, not policy)"""
        return self.kernel.parameters()
    
    def get_frozen_parameters(self):
        """Return frozen parameters (policy)"""
        return self.policy.parameters()
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.policy.to(device)
        self.kernel.to(device)
        return super().to(device)


if __name__ == "__main__":
    print("Policy-Augmented Transition Model Test")
    print("=" * 60)
    
    state_dim = 29  # Ant-maze
    action_dim = 8
    
    # Test model initialization
    print("\n1. Initializing model...")
    model = PolicyAugmentedTransitionModel(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_ckpt="/data1/sandesh/COT/OTFlowRL/ckpt_maf/best.pt",
        n_components=5,
        hidden_dim=512
    )
    print(f"✓ Model initialized")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    s = torch.randn(4, state_dim)
    s_next = model(s, n_samples=3)
    print(f"✓ Forward pass: {s.shape} → {s_next.shape}")
    
    # Verify output shape
    assert s_next.shape == (4, 3, state_dim), f"Expected (4, 3, {state_dim}), got {s_next.shape}"
    print(f"✓ Output shape correct: [batch_size, n_samples, state_dim]")
    
    print("\n✓ All tests passed!")
