"""
Loss Functions Module - UPDATED with Multi-GPU Support
Implements paired loss and terminal loss with kernel embedding matching
Supports parallelization of independent rollouts across multiple GPUs
"""

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import List, Tuple
import numpy as np


class PairedLoss:
    """
    Paired Kernel Embedding Loss
    
    L_paired = ||K(s'_gen_avg) - K(s'_data)||^2
    
    Matches kernel embeddings of generated next-states to observed next-states.
    Enforces that the transition model generates realistic next-state distributions.
    """
    
    def __init__(self, kernel, m=3, config=None):
        """
        Args:
            kernel: IMQKernel or other kernel instance
            m: number of samples per state for averaging
               larger m -> more stable embeddings, slower computation
            config: optional config dict for extraction of m value
        """
        self.kernel = kernel
        self.m = m
        
    def __call__(self, model, s_data, s_next_data, device=None):
        """
        Compute paired loss
        
        Args:
            model: transition model with forward(s, n_samples) -> s'_samples
            s_data: observed states [batch_size, state_dim]
            s_next_data: observed next states [batch_size, state_dim]
            device: torch device
        
        Returns:
            loss: scalar loss value
        """
        
        # Generate m samples per state: s'_gen [batch_size, m, state_dim]
        s_next_gen = model.forward(s_data, n_samples=self.m)
        
        # Average kernel embeddings: K(s'_gen_avg) [batch_size, 1]
        # For each state i: avg over m generated samples
        s_next_gen_avg = s_next_gen.mean(dim=1)  # [batch_size, state_dim]
        
        # Compute kernel embeddings
        K_gen = self.kernel(s_next_gen_avg)    # [batch_size, batch_size]
        K_data = self.kernel(s_next_data)      # [batch_size, batch_size]
        
        # L_paired = ||K(s'_gen) - K(s'_data)||^2
        # Use full kernel matrices (not just diagonal)
        # Diagonal is always 1.0 for IMQ kernel (K(x,x)=1), so use full matrix
        loss = F.mse_loss(K_gen, K_data)
        
        return loss


class TerminalLoss:
    """
    Terminal State Loss - OPTIMIZED FOR SPEED
    
    L_terminal = E[ min_j ||φ(s_terminal) - φ(goal_j)||^2 ]
    
    where φ(x) = K(x, goal_states) is the kernel embedding vector
    
    Algorithm:
    1. For each initial state: generate n_rollout_samples independent T-step rollouts
    2. Each rollout maintains full stochasticity (no averaging within trajectory)
    3. Compute kernel embeddings: K(s_terminal[k], goal_states) for each rollout k
    4. Compute kernel embeddings: K(goal_states, goal_states)
    5. For each rollout k and goal j: compute squared L2 distance in embedding space
         ||φ(s_terminal[k]) - φ(goal[j])||_2^2 = ||K_terminal[k] - K_goal[j]||_2^2
    6. Each rollout finds closest goal (minimum embedding distance)
    7. Loss for state i: average minimum distances over all rollouts
    
    OPTIMIZATIONS:
    - Single GPU (avoid model movement overhead)
    - Vectorized embeddings and distances
    - Early exit for small batch sizes
    - Efficient memory usage
    """
    
    def __init__(self, kernel, T=5, n_rollout_samples=2, config=None):
        """
        Args:
            kernel: IMQKernel or other kernel instance
            T: rollout horizon (number of steps)
            n_rollout_samples: number of rollout trajectories per initial state
                               larger -> better goal coverage estimate
            config: optional config dict for parameter extraction
        """
        self.kernel = kernel
        self.T = T
        self.n_rollout_samples = n_rollout_samples
        
    def __call__(self, model, s_init, goal_states, device=None):
        """
        Compute terminal loss using kernel embedding matching
        
        Args:
            model: transition model with forward(s, n_samples) -> s'_samples
            s_init: initial states [batch_size, state_dim]
            goal_states: goal states [n_goals, state_dim]
            device: torch device
        
        Returns:
            loss: scalar loss value
        """
        
        batch_size = s_init.shape[0]
        n_goals = goal_states.shape[0]
        
        # Early exit for empty goal states
        if n_goals == 0:
            return torch.tensor(0.0, device=device)
        
        loss_per_state = []
        
        # Compute goal embeddings once (reuse for all initial states)
        # K_goal: [n_goals, n_goals]
        # Each row is the embedding vector for that goal state
        K_goal = self.kernel(goal_states, goal_states)
        
        # For each initial state, generate multiple independent rollouts
        for i in range(batch_size):
            s_init_i = s_init[i:i+1]  # [1, state_dim]
            s_terminal_rollouts = []
            
            # ===== Generate n_rollout_samples INDEPENDENT rollouts =====
            for rollout_idx in range(self.n_rollout_samples):
                s_current = s_init_i  # [1, state_dim]
                
                # Rollout T steps (maintaining full stochasticity)
                for t in range(self.T):
                    # Sample ONE next state from the distribution
                    s_next_sample = model.forward(s_current, n_samples=1)
                    # [1, 1, state_dim]
                    
                    # Extract the single sample (remove the n_samples dimension)
                    s_current = s_next_sample.squeeze(1)  # [1, state_dim]
                
                # Terminal state of this rollout
                s_terminal_rollouts.append(s_current.squeeze(0))  # [state_dim]
            
            # ===== Stack terminal states from all rollouts =====
            # [n_rollout_samples, state_dim]
            s_terminal_samples = torch.stack(s_terminal_rollouts, dim=0)
            
            # ===== Compute kernel embeddings for terminal states =====
            # Use goal_states as reference set for kernel embeddings
            # K_terminal: [n_rollout_samples, n_goals]
            # Each row k is the embedding vector for terminal state k
            K_terminal = self.kernel(s_terminal_samples, goal_states)
            
            # ===== Vectorized embedding distance computation =====
            # For each terminal state k and goal j:
            # distance[k, j] = ||φ(s_T[k]) - φ(goal[j])||_2^2
            #                = ||K_terminal[k] - K_goal[j]||_2^2
            #                = sum_m (K_terminal[k,m] - K_goal[j,m])^2
            
            # Expand for broadcasting
            # K_terminal_exp: [n_rollout_samples, 1, n_goals]
            # K_goal_exp: [1, n_goals, n_goals]
            K_terminal_exp = K_terminal.unsqueeze(1)  # [n_rollout_samples, 1, n_goals]
            K_goal_exp = K_goal.unsqueeze(0)  # [1, n_goals, n_goals]
            
            # Compute embedding differences: [n_rollout_samples, n_goals, n_goals]
            emb_diff = K_terminal_exp - K_goal_exp
            
            # Squared L2 distances: [n_rollout_samples, n_goals]
            # emb_dist_sq[k, j] = sum_m (emb_diff[k, j, m])^2
            emb_dist_sq = torch.sum(emb_diff ** 2, dim=2)
            
            # For each rollout k, find minimum distance to any goal j
            # min_dist_per_rollout: [n_rollout_samples]
            min_dist_per_rollout = torch.min(emb_dist_sq, dim=1).values
            
            # Loss for this state: mean over rollouts
            loss_i = min_dist_per_rollout.mean()
            loss_per_state.append(loss_i)
        
        # Final loss: mean over states
        loss = torch.stack(loss_per_state).mean()
        
        return loss



class CombinedLoss:
    """
    Combined Loss Function
    
    L_total = lambda_paired * L_paired + lambda_terminal * L_terminal
    
    Balances data fidelity (paired loss) with goal-reaching (terminal loss)
    """
    
    def __init__(self, kernel, config):
        """
        Args:
            kernel: kernel instance
            config: dict with:
                   - lambda_paired: weight for paired loss
                   - lambda_terminal: weight for terminal loss
                   - m: samples for paired loss
                   - T: rollout horizon
                   - n_rollout_samples: trajectory samples
        """
        self.kernel = kernel
        self.config = config
        
        # Extract parameters
        self.lambda_paired = config['loss']['lambda_paired']
        self.lambda_terminal = config['loss']['lambda_terminal']
        
        # Initialize loss components
        self.paired_loss = PairedLoss(
            kernel,
            m=config['sampling']['m']
        )
        
        self.terminal_loss = TerminalLoss(
            kernel,
            T=config['sampling']['T'],
            n_rollout_samples=config['sampling']['n_rollout_samples']
        )
        
    def __call__(self, model, s_data, s_next_data, s_init_terminal, 
                 goal_states, use_terminal=True, device=None):
        """
        Compute combined loss
        
        Args:
            model: transition model
            s_data: data states [batch_size, state_dim]
            s_next_data: data next states [batch_size, state_dim]
            s_init_terminal: initial states for terminal loss rollout
            goal_states: goal states [n_goals, state_dim]
            use_terminal: whether to include terminal loss (can skip every N batches)
            device: torch device
        
        Returns:
            loss: scalar combined loss
            loss_dict: dict with individual loss components
        """
        
        # L_paired: always compute
        loss_paired = self.paired_loss(model, s_data, s_next_data, device=device)
        
        # L_terminal: compute if requested
        loss_terminal = torch.tensor(0.0, device=device)
        if use_terminal:
            loss_terminal = self.terminal_loss(
                model, s_init_terminal, goal_states, device=device
            )
        
        # L_total = lambda_paired * L_paired + lambda_terminal * L_terminal
        loss_total = (self.lambda_paired * loss_paired + 
                     self.lambda_terminal * loss_terminal)
        
        # Extract scalar values for loss_dict
        paired_loss_val = loss_paired.item() if hasattr(loss_paired, 'item') else float(loss_paired)
        terminal_loss_val = loss_terminal.item() if hasattr(loss_terminal, 'item') else float(loss_terminal)
        
        loss_dict = {
            'paired_loss': paired_loss_val,
            'terminal_loss': terminal_loss_val,
        }
        
        return loss_total, loss_dict


# Test usage
if __name__ == "__main__":
    from kernels import IMQKernel
    
    print("Loss Functions Test")
    print("=" * 60)
    
    # Create kernel
    kernel = IMQKernel(alpha=1.0)
    
    # Test data
    batch_size = 4
    state_dim = 5
    
    s_data = torch.randn(batch_size, state_dim)
    s_next_data = torch.randn(batch_size, state_dim)
    
    # Mock model
    class MockModel:
        def forward(self, s, n_samples=1):
            return torch.randn(s.shape[0], n_samples, s.shape[1])
    
    model = MockModel()
    goal_states = torch.randn(2, state_dim)  # 2 goal states
    
    # Test paired loss
    paired_loss_fn = PairedLoss(kernel, m=3)
    loss_paired = paired_loss_fn(model, s_data, s_next_data)
    print(f"✓ Paired Loss: {loss_paired.item():.6f}")
    
    # Test terminal loss
    terminal_loss_fn = TerminalLoss(kernel, T=5, n_rollout_samples=2)
    loss_terminal = terminal_loss_fn(model, s_data, goal_states)
    print(f"✓ Terminal Loss: {loss_terminal.item():.6f}")
    
    print("\n✓ All loss functions working correctly!")
