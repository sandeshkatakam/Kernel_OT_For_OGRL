"""
Evaluation Module
Metrics and analysis for goal-conditioned trajectory generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Evaluator:
    """
    Evaluation framework for kernel OT goal-conditioned rollouts
    """
    
    def __init__(self, model, goal_states, config, device=None):
        """
        Args:
            model: trained TransitionModel
            goal_states: goal state tensor [n_goals, state_dim]
            config: configuration dictionary
            device: torch device
        """
        self.model = model
        self.goal_states = goal_states
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.model.eval()
        
    def generate_rollout(self, s_init, T, n_samples=1):
        """
        Generate T-step rollout from initial state
        
        Args:
            s_init: initial state [state_dim] or [batch_size, state_dim]
            T: horizon (number of steps)
            n_samples: number of rollout samples per initial state
        
        Returns:
            trajectory: [batch_size, T, state_dim] if batch_size>1 else [T, state_dim]
            or
            trajectories: [batch_size, n_samples, T, state_dim] if saving all samples
        """
        
        if s_init.dim() == 1:
            s_init = s_init.unsqueeze(0)
        
        batch_size = s_init.shape[0]
        s_init = s_init.to(self.device)
        
        with torch.no_grad():
            # Replicate for n_samples
            s_current = s_init.unsqueeze(1).repeat(1, n_samples, 1)
            # [batch_size, n_samples, state_dim]
            
            trajectory = [s_current]
            
            for t in range(T):
                # Flatten batch and samples
                s_flat = s_current.reshape(batch_size * n_samples, -1)
                
                # Generate next states (1 sample per current state)
                s_next_samples = self.model(s_flat, n_samples=1)
                # [batch_size * n_samples, 1, state_dim]
                
                s_next = s_next_samples.squeeze(1)
                # [batch_size * n_samples, state_dim]
                
                # Reshape back
                s_current = s_next.reshape(batch_size, n_samples, -1)
                trajectory.append(s_current)
            
            # Stack trajectory: [batch_size, n_samples, T, state_dim]
            trajectory = torch.stack(trajectory, dim=2)
            
        return trajectory
    
    def goal_reaching_distance(self, trajectory, goal_state):
        """
        Compute distance from final state to goal
        
        Args:
            trajectory: [batch_size, n_samples, T, state_dim]
            goal_state: [state_dim]
        
        Returns:
            distances: [batch_size, n_samples]
        """
        
        # Extract terminal state
        s_terminal = trajectory[:, :, -1, :]  # [batch_size, n_samples, state_dim]
        goal_state = goal_state.to(self.device)
        
        # L2 distance
        distances = torch.norm(s_terminal - goal_state, dim=-1)
        # [batch_size, n_samples]
        
        return distances
    
    def success_rate(self, initial_states, T, distance_threshold=0.5, n_samples=10):
        """
        Compute success rate: fraction of rollouts reaching goal within threshold
        
        Args:
            initial_states: [batch_size, state_dim]
            T: horizon
            distance_threshold: success if distance < threshold
            n_samples: number of rollout samples per initial state
        
        Returns:
            success_rates: [n_goals]
        """
        
        initial_states = initial_states.to(self.device)
        n_goals = self.goal_states.shape[0]
        
        # Generate rollouts
        trajectories = self.generate_rollout(initial_states, T, n_samples=n_samples)
        # [batch_size, n_samples, T, state_dim]
        
        success_rates = []
        
        for goal_idx in range(n_goals):
            goal_state = self.goal_states[goal_idx]
            distances = self.goal_reaching_distance(trajectories, goal_state)
            # [batch_size, n_samples]
            
            # Success if any sample reaches goal
            success = (distances < distance_threshold).any(dim=1).float()
            success_rate = success.mean().item()
            
            success_rates.append(success_rate)
        
        return np.array(success_rates)
    
    def trajectory_statistics(self, initial_states, T, n_samples=10):
        """
        Compute statistics of generated trajectories
        
        Args:
            initial_states: [batch_size, state_dim]
            T: horizon
            n_samples: rollout samples
        
        Returns:
            stats: dict with statistics
        """
        
        initial_states = initial_states.to(self.device)
        trajectories = self.generate_rollout(initial_states, T, n_samples=n_samples)
        # [batch_size, n_samples, T, state_dim]
        
        # Compute statistics
        mean_trajectory = trajectories.mean(dim=(0, 1))
        std_trajectory = trajectories.std(dim=(0, 1))
        
        # Compute distances at each timestep
        distances_to_goal = []
        for goal_idx in range(self.goal_states.shape[0]):
            goal_state = self.goal_states[goal_idx]
            dist_t = []
            
            for t in range(T + 1):
                s_t = trajectories[:, :, t, :]
                d_t = torch.norm(s_t - goal_state, dim=-1).mean().item()
                dist_t.append(d_t)
            
            distances_to_goal.append(dist_t)
        
        stats = {
            'mean_trajectory': mean_trajectory.cpu().numpy(),
            'std_trajectory': std_trajectory.cpu().numpy(),
            'distances_to_goal': np.array(distances_to_goal),
        }
        
        return stats
    
    def evaluate_dataset(self, test_states, T, n_goals=None, n_samples=10, 
                        distance_threshold=0.5):
        """
        Comprehensive evaluation on test dataset
        
        Args:
            test_states: [n_test, state_dim]
            T: horizon
            n_goals: number of goals to evaluate (default: all)
            n_samples: rollout samples
            distance_threshold: success threshold
        
        Returns:
            eval_results: dict with comprehensive metrics
        """
        
        print("=" * 80)
        print("EVALUATION")
        print("=" * 80)
        
        if n_goals is None:
            n_goals = self.goal_states.shape[0]
        
        goal_subset = self.goal_states[:n_goals]
        test_states = test_states.to(self.device)
        
        # Sample subset for evaluation
        n_test = min(100, len(test_states))
        test_indices = np.random.choice(len(test_states), n_test, replace=False)
        test_subset = test_states[test_indices]
        
        print(f"Test states: {n_test}")
        print(f"Horizon: {T}")
        print(f"Rollout samples: {n_samples}")
        print(f"Goals: {n_goals}")
        print()
        
        # Generate rollouts
        print("Generating rollouts...")
        trajectories = self.generate_rollout(test_subset, T, n_samples=n_samples)
        
        # Evaluate success rates
        print("Computing success rates...")
        success_rates = []
        distances_list = []
        
        for goal_idx in tqdm(range(n_goals), desc="Evaluating goals"):
            goal = self.goal_states[goal_idx]
            distances = self.goal_reaching_distance(trajectories, goal)
            
            success = (distances < distance_threshold).any(dim=1).float()
            success_rate = success.mean().item()
            success_rates.append(success_rate)
            
            distances_list.append(distances.cpu().numpy())
        
        success_rates = np.array(success_rates)
        
        # Aggregate metrics
        eval_results = {
            'success_rates': success_rates,
            'mean_success': success_rates.mean(),
            'std_success': success_rates.std(),
            'distances_to_goals': distances_list,
            'distance_threshold': distance_threshold,
        }
        
        # Print results
        print("\n" + "-" * 80)
        print("RESULTS")
        print("-" * 80)
        print(f"Mean Success Rate: {eval_results['mean_success']:.4f} ± {eval_results['std_success']:.4f}")
        print(f"Success Rates by Goal:")
        for i, sr in enumerate(success_rates):
            print(f"  Goal {i}: {sr:.4f}")
        
        print("=" * 80)
        
        return eval_results
    
    def visualize_rollout(self, s_init, goal_idx=0, T=10, n_samples=5, figsize=(12, 4)):
        """
        Visualize rollout trajectories
        
        Args:
            s_init: initial state [state_dim]
            goal_idx: which goal to target
            T: horizon
            n_samples: number of samples to show
            figsize: figure size
        
        Returns:
            fig: matplotlib figure
        """
        
        if s_init.dim() == 1:
            s_init = s_init.unsqueeze(0)
        
        # Generate rollout
        trajectory = self.generate_rollout(s_init, T, n_samples=n_samples)
        # [1, n_samples, T, state_dim]
        
        trajectory = trajectory.squeeze(0).cpu().numpy()  # [n_samples, T, state_dim]
        goal = self.goal_states[goal_idx].cpu().numpy()
        s_init = s_init.squeeze(0).cpu().numpy()
        
        # For visualization, use first 2 dimensions (if possible)
        traj_2d = trajectory[:, :, :2]
        goal_2d = goal[:2]
        init_2d = s_init[:2]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Trajectories
        for i in range(n_samples):
            ax.plot(traj_2d[i, :, 0], traj_2d[i, :, 1], 
                   alpha=0.5, linewidth=1.5, label=f'Sample {i+1}')
        
        # Initial state
        ax.plot(init_2d[0], init_2d[1], 'go', markersize=10, label='Initial')
        
        # Goal
        ax.plot(goal_2d[0], goal_2d[1], 'r*', markersize=20, label='Goal')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Rollout Trajectories (Goal {goal_idx})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig


# Test usage
if __name__ == "__main__":
    
    print("Evaluator Test")
    print("=" * 60)
    
    # Mock setup
    state_dim = 5
    T = 3
    
    goal_states = torch.randn(2, state_dim)
    
    # Mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, s, n_samples=1):
            batch_size = s.shape[0]
            return torch.randn(batch_size, n_samples, s.shape[1])
    
    model = MockModel()
    config = {}
    
    evaluator = Evaluator(model, goal_states, config)
    
    # Test rollout generation
    s_init = torch.randn(2, state_dim)
    trajectory = evaluator.generate_rollout(s_init, T, n_samples=2)
    print(f"✓ Generated rollout: {trajectory.shape}")
    
    # Test distance computation
    distances = evaluator.goal_reaching_distance(trajectory, goal_states[0])
    print(f"✓ Computed distances: {distances.shape}")
    
    print("\n✓ Evaluator working correctly!")
