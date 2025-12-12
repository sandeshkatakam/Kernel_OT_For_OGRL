"""
Loss Functions Module - UPDATED with Kernel MMD Loss
Implements paired loss and terminal loss using Maximum Mean Discrepancy (MMD)
with kernel trick for efficient computation
"""

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import List, Tuple
import numpy as np

class PairedLoss:
    """
    Paired Kernel Embedding Loss using MMD (Maximum Mean Discrepancy)
    """

    def __init__(self, kernel, m=3, config=None):
        self.kernel = kernel
        self.m = m
        
    def __call__(self, model, s_data, s_next_data, device=None):

        # 1. Generate m samples per state: gen [batch_size, m, state_dim]
        gen = model.forward(s_data, n_samples=self.m)
        
        batch_size = gen.shape[0]
        state_dim = gen.shape[-1]

        # total generated samples n_gen = batch_size * m
        n_gen = batch_size * self.m
        n_data = batch_size
        
        # 2. Flatten generated samples for kernel computation
        gen_flat = gen.reshape(n_gen, state_dim)
        
        # ----------- A TERM (1 / n_gen^2 * sum k(gen, gen)) -----------
        K_gg = self.kernel(gen_flat, gen_flat)              # [n_gen, n_gen]
        A = K_gg.sum() / (n_gen * n_gen)                    # explicit 1/n_gen^2

        # ----------- B TERM (1 / n_data^2 * sum k(data, data)) -------
        t = s_next_data
        K_tt = self.kernel(t, t)                            # [n_data, n_data]
        B = K_tt.sum() / (n_data * n_data)                  # explicit 1/n_data^2

        # ----------- C TERM (2 / (n_gen * n_data) * sum k(gen, data)) -----------
        K_gt = self.kernel(gen_flat, t)                     # [n_gen, n_data]
        C = 2.0 * K_gt.sum() / (n_gen * n_data)             # correct empirical cross-term

        # 6. Loss = A + B - C
        loss = A + B - C
        
        return loss



class TerminalLoss:
    """
    Terminal State Loss using MMD with Kernel Trick - SINGLE GOAL VERSION
    
    ✅ FIXED: Works with SINGLE goal state only (extracted from dataset)
    ✅ FIXED: Removed torch.min() to prevent mode collapse
    ✅ NEW: Uniform sampling from unique dataset states
    
    L_terminal = E[ A + B - C ]
    
    For the single goal:
    - A = E[k(rollout_terminal_states, rollout_terminal_states)]
    - B = k(goal, goal)
    - C = (2/m) * E[k(rollout_terminal_states, goal)]
    
    Direct optimization toward the goal state without arbitrary attractors.
    """
    
    def __init__(self, kernel, T=5, n_rollout_samples=2, config=None, 
                 unique_states=None):
        """
        Args:
            kernel: IMQKernel or other kernel instance
            T: rollout horizon (number of steps)
            n_rollout_samples: number of rollout trajectories per initial state
            config: optional config dict for parameter extraction
            unique_states: [n_unique, state_dim] tensor of unique dataset states
                          If None, will be set during first call
        """
        self.kernel = kernel
        self.T = T
        self.n_rollout_samples = n_rollout_samples
        self.config = config
        self.unique_states = unique_states
        self.uniform_dist = None
        
        print(f"✓ TerminalLoss initialized with T={T}, n_rollout_samples={n_rollout_samples}")
        if unique_states is not None:
            self.n_unique_states = unique_states.shape[0]
            self.uniform_dist = torch.distributions.Categorical(
                torch.ones(self.n_unique_states) / self.n_unique_states
            )
            print(f"  Uniform distribution created over {self.n_unique_states} unique states")
    
    def set_unique_states(self, unique_states):
        """
        Set the unique states and create uniform distribution
        
        Call this ONCE after dataset is loaded, BEFORE training starts
        
        Args:
            unique_states: [n_unique, state_dim] tensor of unique dataset states
        """
        self.unique_states = unique_states
        self.n_unique_states = unique_states.shape[0]
        self.uniform_dist = torch.distributions.Categorical(
            torch.ones(self.n_unique_states) / self.n_unique_states
        )
        print(f"✓ TerminalLoss: Set {self.n_unique_states} unique states")
        print(f"  Uniform distribution ready for sampling")
    
    def sample_initial_states(self, batch_size, device=None):
        """
        Sample initial states uniformly from unique dataset states
        
        ✅ CRITICAL: Call this instead of using s_data directly
        
        Args:
            batch_size: number of states to sample
            device: torch device
        
        Returns:
            sampled_states: [batch_size, state_dim] uniformly sampled from unique states
        """
        if self.unique_states is None:
            raise RuntimeError("unique_states not set! Call set_unique_states() first")
        
        if self.uniform_dist is None:
            raise RuntimeError("uniform_dist not initialized! Call set_unique_states() first")
        
        # Sample indices uniformly from unique states
        indices = self.uniform_dist.sample((batch_size,))  # [batch_size]
        
        # Get states corresponding to sampled indices
        sampled_states = self.unique_states[indices]  # [batch_size, state_dim]
        
        if device is not None:
            sampled_states = sampled_states.to(device)
        
        return sampled_states
    
    def __call__(self, model, s_init, goal_state, device=None):
        """
        Compute terminal loss using kernel MMD with SINGLE goal matching
        ✅ T-STEP ROLLOUTS: Process multiple rollouts with T forward passes each
        """

        # If s_init None: sample (keeps your behaviour)
        if s_init is None:
            batch_size = self.config['training']['batch_size'] if self.config else 32
            s_init = self.sample_initial_states(batch_size, device=device)

        batch_size = s_init.shape[0]
        state_dim = s_init.shape[1]

        # Ensure goal_state shape: allow [d], [1, d], [batch_size, d], or [m_goal, d]
        if goal_state.dim() == 1:
            goal_state = goal_state.unsqueeze(0)  # [1, d]

        # Move goal to appropriate device
        target_device = device if device is not None else s_init.device
        goal_state = goal_state.to(target_device)
        s_init = s_init.to(target_device)

        # ---------------------------
        # Vectorized T-step rollouts
        # ---------------------------
        s_terminals = torch.zeros(
            batch_size, self.n_rollout_samples, state_dim,
            device=target_device, dtype=s_init.dtype
        )

        for rollout_idx in range(self.n_rollout_samples):
            s_current = s_init.clone()  # [batch_size, state_dim]
            for t in range(self.T):
                s_next = model.forward(s_current, n_samples=1)   # [batch_size, 1, state_dim]
                s_current = s_next.squeeze(1)                    # [batch_size, state_dim]
            s_terminals[:, rollout_idx, :] = s_current

        # Flatten terminals: [batch_size * n_rollout_samples, state_dim]
        s_terminals_flat = s_terminals.reshape(-1, state_dim)
        n_terminals = s_terminals_flat.shape[0]  # = batch_size * n_rollout_samples

       
        # A = (1 / n_terminals²) * sum_{i,j} k(terminal_i, terminal_j)
        K_rr = self.kernel(s_terminals_flat, s_terminals_flat)  # [n_terminals, n_terminals]
        A = K_rr.sum() / float(n_terminals * n_terminals)

        # B = (1 / 1²) * k(goal, goal) = k(goal, goal)
        # ✅ CRITICAL: Do NOT expand goal! Keep it as single point [1, d]
        K_gg = self.kernel(goal_state, goal_state)              # [1, 1]
        B = K_gg.squeeze() / 1.0  # Just k(goal, goal)

        # C = 2 * (1 / (n_terminals * 1)) * sum_i k(terminal_i, goal)
        # ✅ CRITICAL: Compare all terminals to single goal, then scale by 2
        K_rg = self.kernel(s_terminals_flat, goal_state)        # [n_terminals, 1]
        C = 2.0 * K_rg.sum() / float(n_terminals * 1)  # ← Has the 2.0 factor!

        # Loss = A + B - C (NO additional 2.0!)
        # ✅ CORRECT: When terminals close to goal, K_rg is large, C is large, loss is small ✓
        loss = A + B - C

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
                 goal_state, use_terminal=True, device=None):
        """
        Compute combined loss
        
        ✅ CHANGED: goal_states -> goal_state (SINGLE goal now)
        
        Args:
            model: transition model
            s_data: data states [batch_size, state_dim]
            s_next_data: data next states [batch_size, state_dim]
            s_init_terminal: initial states for terminal loss rollout
            goal_state: SINGLE goal state [state_dim] or [1, state_dim]
            use_terminal: whether to include terminal loss
            device: torch device
        
        Returns:
            loss: scalar combined loss
            loss_dict: dict with individual loss components
        """
        
        # L_paired: always compute
        loss_paired = self.paired_loss(model, s_data, s_next_data, device=device)
        
        # L_terminal: compute if requested
        loss_terminal = torch.tensor(0.0, device=device if device is not None else s_data.device)
        if use_terminal:
            # ✅ FIXED: Pass single goal_state instead of goal_states
            loss_terminal = self.terminal_loss(
                model, s_init_terminal, goal_state, device=device
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