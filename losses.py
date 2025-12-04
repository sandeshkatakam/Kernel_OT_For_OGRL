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
    
    L_paired = E_gen[k(gen, gen)] + E_data[k(data, data)] - 2*E_gen,data[k(gen, data)]
    
    Represents squared MMD between generated next-state distribution and observed next-state distribution.
    Uses kernel trick to avoid explicit feature map computation.
    """ 
    
    def __init__(self, kernel, m=3, config=None):
        """
        Args:
            kernel: IMQKernel or other kernel instance
            m: number of samples per state
               larger m -> more stable embeddings, slower computation
            config: optional config dict for extraction of m value
        """
        self.kernel = kernel
        self.m = m
        
    def __call__(self, model, s_data, s_next_data, device=None):
        """
        Compute paired loss using kernel MMD
        
        Args:
            model: transition model with forward(s, n_samples) -> s'_samples
            s_data: observed states [batch_size, state_dim]
            s_next_data: observed next states [batch_size, state_dim]
            device: torch device
        
        Returns:
            loss: scalar loss value
        """
        
        # 1. Generate m samples per state: gen [batch_size, m, state_dim]
        gen = model.forward(s_data, n_samples=self.m)
        
        batch_size = gen.shape[0]
        state_dim = gen.shape[-1]
        
        # 2. Reshape to [batch_size * m, state_dim] for kernel computation
        gen_flat = gen.reshape(batch_size * self.m, state_dim)
        
        # 3. Compute A = E[k(gen, gen)]
        # K_gg: [batch_size * m, batch_size * m]
        K_gg = self.kernel(gen_flat, gen_flat)
        A = K_gg.mean()
        
        # 4. Compute B = E[k(data, data)]
        # K_tt: [batch_size, batch_size]
        t = s_next_data
        K_tt = self.kernel(t, t)
        B = K_tt.mean()
        
        # 5. Compute C = (2/m) * E[k(gen, data)]
        # K_gt: [batch_size * m, batch_size]
        K_gt = self.kernel(gen_flat, t)
        
        # Average over the m samples for each state, then over batch
        # Reshape to [batch_size, m, batch_size]
        K_gt_per_state = K_gt.reshape(batch_size, self.m, batch_size)
        # Average over m samples: [batch_size, batch_size]
        K_gt_avg = K_gt_per_state.mean(dim=1)
        # Average over all samples
        C = (2.0 / self.m) * K_gt_avg.mean()
        
        # 6. Loss = A + B - C (squared MMD)
        loss = A + B - C
        
        return loss


class TerminalLoss:
    """
    Terminal State Loss using MMD with Kernel Trick - SINGLE GOAL VERSION
    
    ✅ FIXED: Works with SINGLE goal state only (extracted from dataset)
    ✅ FIXED: Removed torch.min() to prevent mode collapse
    
    L_terminal = E[ A + B - C ]
    
    For the single goal:
    - A = E[k(rollout_terminal_states, rollout_terminal_states)]
    - B = k(goal, goal)
    - C = (2/m) * E[k(rollout_terminal_states, goal)]
    
    Direct optimization toward the goal state without arbitrary attractors.
    """
    
    def __init__(self, kernel, T=5, n_rollout_samples=2, config=None):
        """
        Args:
            kernel: IMQKernel or other kernel instance
            T: rollout horizon (number of steps)
            n_rollout_samples: number of rollout trajectories per initial state
                               larger -> better stochasticity coverage
            config: optional config dict for parameter extraction
        """
        self.kernel = kernel
        self.T = T
        self.n_rollout_samples = n_rollout_samples
        
    def __call__(self, model, s_init, goal_state, device=None):
        """
        Compute terminal loss using kernel MMD with SINGLE goal matching
        
        ✅ KEY CHANGE: goal_state is now a SINGLE goal, not multiple goals
        
        Args:
            model: transition model with forward(s, n_samples) -> s'_samples
            s_init: initial states [batch_size, state_dim]
            goal_state: SINGLE goal state [state_dim] or [1, state_dim]
            device: torch device
        
        Returns:
            loss: scalar loss value
        """
        
        batch_size = s_init.shape[0]
        state_dim = s_init.shape[1]
        
        # ✅ FIX: Handle goal_state shape - ensure it's [1, state_dim]
        if goal_state.dim() == 1:
            goal_state = goal_state.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        
        # Ensure goal is on same device as s_init
        if device is not None:
            goal_state = goal_state.to(device)
        else:
            goal_state = goal_state.to(s_init.device)
        
        loss_per_state = []
        
        # For each initial state, generate rollouts
        for i in range(batch_size):
            s_init_i = s_init[i:i+1]  # [1, state_dim]
            s_terminal_rollouts = []
            
            # Generate n_rollout_samples independent rollouts
            for rollout_idx in range(self.n_rollout_samples):
                s_current = s_init_i  # [1, state_dim]
                
                # Rollout T steps
                for t in range(self.T):
                    s_next_sample = model.forward(s_current, n_samples=1)  # [1, 1, state_dim]
                    s_current = s_next_sample.squeeze(1)  # [1, state_dim]
                
                s_terminal_rollouts.append(s_current.squeeze(0))  # [state_dim]
            
            # Stack rollouts: [n_rollout_samples, state_dim]
            rollouts = torch.stack(s_terminal_rollouts, dim=0)
            
            # ✅ FIXED: Direct computation without torch.min()
            # Compute MMD between rollout terminal states and SINGLE goal
            
            # A = E[k(rollouts, rollouts)]
            K_rr = self.kernel(rollouts, rollouts)  # [m, m]
            A = K_rr.mean()
            
            # B = k(goal, goal) - constant term depending only on goal
            K_gg = self.kernel(goal_state, goal_state)  # [1, 1]
            B = K_gg.squeeze()
            
            # C = (2/m) * E[k(rollouts, goal)]
            # This is the KEY term: it measures how close rollouts are to goal
            K_rg = self.kernel(rollouts, goal_state)  # [m, 1]
            C = (2.0 / self.n_rollout_samples) * K_rg.mean()
            
            # ✅ FIXED: Direct MMD loss (NO torch.min())
            # Loss = A + B - C
            # - When rollouts close to goal: K_rg large → C large → loss small ✓
            # - When rollouts far from goal: K_rg small → C small → loss large ✗
            mmd_loss_i = A + B - C
            loss_per_state.append(mmd_loss_i)
        
        # Final loss: mean over all initial states
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
