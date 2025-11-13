"""
Data Preparation Module
Generates paired transition dataset and selects fixed goal states from antmaze
"""

import numpy as np
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/data1/sandesh/COT')

try:
    import ogbench
except ImportError:
    print("Warning: ogbench not found. Install with: pip install ogbench")

class DataPreparator:
    """
    Loads antmaze environment data and prepares:
    1. Paired transition dataset: {(s_i, s_j)}
    2. Fixed goal states: {g_1, g_2, ..., g_K}
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary with data section
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
    def prepare_data(self):
        """
        Main pipeline:
        1. Load offline RL dataset from antmaze
        2. Extract transition pairs
        3. Select and fix goal states
        4. Return as torch tensors
        
        Returns:
            transition_data: dict with keys 's', 's_next'
            goal_states: torch tensor of shape (n_goals, state_dim)
        """
        
        print("="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        # Step 1: Load antmaze environment and dataset
        print(f"\n[1/3] Loading {self.config['environment']} environment...")
        try:
            env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
                self.config['environment']
            )
            print(f"✓ Environment loaded")
        except Exception as e:
            print(f"✗ Failed to load environment: {e}")
            raise
        
        # Step 2: Extract transition pairs from offline data
        print(f"\n[2/3] Extracting transition pairs...")
        transition_data = self._extract_transitions(train_dataset)
        
        # Step 3: Select and fix goal states
        print(f"\n[3/3] Selecting fixed goal states...")
        goal_states = self._select_goal_states(transition_data['s'])
        
        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE")
        print("="*80)
        
        return transition_data, goal_states
    
    def _extract_transitions(self, dataset):
        """
        Extract paired transitions from offline RL dataset
        
        Args:
            dataset: dict with keys 'observations', 'next_observations', etc.
        
        Returns:
            transition_data: dict with torch tensors
                's': current states [n_samples, state_dim]
                's_next': next states [n_samples, state_dim]
        """
        
        observations = dataset['observations']
        next_observations = dataset['next_observations']
        
        # Use only subset for faster training
        num_samples = min(self.config['num_transitions'], len(observations))
        indices = np.random.choice(len(observations), num_samples, replace=False)
        
        s = torch.from_numpy(observations[indices]).to(self.device).to(self.dtype)
        s_next = torch.from_numpy(next_observations[indices]).to(self.device).to(self.dtype)
        
        print(f"  • Extracted {num_samples} transition pairs")
        print(f"  • State shape: {s.shape}")
        print(f"  • State range: [{s.min():.3f}, {s.max():.3f}]")
        
        transition_data = {
            's': s,
            's_next': s_next,
        }
        
        return transition_data
    
    def _select_goal_states(self, states):
        """
        Select fixed goal states from the state space
        
        Args:
            states: tensor of shape [n_states, state_dim]
        
        Returns:
            goal_states: tensor of shape [n_goals, state_dim]
        """
        
        n_goals = self.config['num_goal_states']
        
        # Randomly select diverse goal states
        indices = np.random.choice(len(states), n_goals, replace=False)
        goal_states = states[indices].clone()
        
        print(f"  • Selected {n_goals} goal states")
        print(f"  • Goal shape: {goal_states.shape}")
        print(f"  • Goal range: [{goal_states.min():.3f}, {goal_states.max():.3f}]")
        
        return goal_states
    
    def save_data(self, transition_data, goal_states, save_dir):
        """
        Save prepared data to disk
        
        Args:
            transition_data: dict with transition tensors
            goal_states: tensor of goal states
            save_dir: directory to save to
        """
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save transitions
        torch.save(transition_data, 
                  os.path.join(save_dir, self.config['transition_data_save_path']))
        
        # Save goals
        torch.save(goal_states,
                  os.path.join(save_dir, self.config['goal_state_save_path']))
        
        print(f"\n✓ Data saved to {save_dir}")
        
    def load_data(self, save_dir):
        """
        Load prepared data from disk
        
        Args:
            save_dir: directory to load from
        
        Returns:
            transition_data: dict
            goal_states: tensor
        """
        
        transition_data = torch.load(
            os.path.join(save_dir, self.config['transition_data_save_path'])
        )
        
        goal_states = torch.load(
            os.path.join(save_dir, self.config['goal_state_save_path'])
        )
        
        print(f"\n✓ Data loaded from {save_dir}")
        
        return transition_data, goal_states


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare data
    preparator = DataPreparator(config['data'])
    transition_data, goal_states = preparator.prepare_data()
    
    # Save for later use
    preparator.save_data(transition_data, goal_states, config['output']['save_dir'])
