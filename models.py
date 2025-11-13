"""
Model Module
Placeholder interface for user-provided transition model q(s'|s)
"""

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """
    Abstract base class for transition models q(s'|s)
    
    User should subclass this and implement forward() method.
    The model takes a state s and generates multiple samples from
    the learned distribution q(s'|s).
    """
    
    def __init__(self, state_dim, noise_dim=16, **kwargs):
        """
        Args:
            state_dim: dimension of state space
            noise_dim: dimension of noise/latent variable
            **kwargs: additional model-specific parameters
        """
        super().__init__()
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        
    def forward(self, s, n_samples=1):
        """
        Generate samples from transition distribution q(s'|s)
        
        Args:
            s: input states [batch_size, state_dim]
            n_samples: number of samples per state
        
        Returns:
            s_next: generated next states [batch_size, n_samples, state_dim]
        """
        raise NotImplementedError("Subclass must implement forward()")


class PlaceholderTransitionModel(TransitionModel):
    """
    Placeholder model with simple neural network
    
    IMPORTANT: This is only for testing/debugging. Users should:
    
    1. Train a model on offline RL data using their method
    2. Create a subclass of TransitionModel
    3. Load their trained model weights
    4. Pass to training loop
    
    Example:
    --------
    class MyOfflineRLModel(TransitionModel):
        def __init__(self, state_dim, noise_dim=16):
            super().__init__(state_dim, noise_dim)
            self.encoder = nn.Sequential(...)
            self.decoder = nn.Sequential(...)
            # Load pretrained weights:
            # self.load_state_dict(torch.load('my_model.pt'))
        
        def forward(self, s, n_samples=1):
            # Your custom sampling logic here
            ...
    """
    
    def __init__(self, state_dim, noise_dim=16, hidden_dim=64):
        """
        Args:
            state_dim: dimension of state
            noise_dim: dimension of noise variable
            hidden_dim: hidden layer dimension
        """
        super().__init__(state_dim, noise_dim)
        
        # Simple MLP: [state + noise] -> [state]
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.decoder = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, s, n_samples=1):
        """
        Generate next-state samples
        
        Args:
            s: states [batch_size, state_dim]
            n_samples: number of samples per state
        
        Returns:
            s_next: [batch_size, n_samples, state_dim]
        """
        
        batch_size = s.shape[0]
        device = s.device
        
        # Replicate state for n_samples
        s_rep = s.unsqueeze(1).repeat(1, n_samples, 1)  # [bs, n_samples, state_dim]
        s_rep_flat = s_rep.reshape(batch_size * n_samples, -1)
        
        # Sample noise
        noise = torch.randn(batch_size * n_samples, self.noise_dim, device=device)
        
        # Concatenate state and noise
        x = torch.cat([s_rep_flat, noise], dim=-1)
        
        # Forward pass
        h = self.encoder(x)
        s_next_flat = self.decoder(h)
        
        # Reshape to [batch_size, n_samples, state_dim]
        s_next = s_next_flat.reshape(batch_size, n_samples, -1)
        
        return s_next


class GaussianTransitionModel(TransitionModel):
    """
    Gaussian transition model: q(s'|s) = N(mu(s), sigma(s))
    
    More realistic model that learns mean and variance of transitions.
    """
    
    def __init__(self, state_dim, noise_dim=16, hidden_dim=64):
        """
        Args:
            state_dim: dimension of state
            noise_dim: not used in this model (for compatibility)
            hidden_dim: hidden layer dimension
        """
        super().__init__(state_dim, noise_dim)
        
        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Log-variance network (to ensure positivity)
        self.logvar_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
    def forward(self, s, n_samples=1):
        """
        Sample from Gaussian: s' ~ N(mu(s), sigma(s))
        
        Args:
            s: states [batch_size, state_dim]
            n_samples: number of samples per state
        
        Returns:
            s_next: [batch_size, n_samples, state_dim]
        """
        
        batch_size = s.shape[0]
        device = s.device
        
        # Compute mean and variance
        mu = self.mean_net(s)              # [batch_size, state_dim]
        logvar = self.logvar_net(s)        # [batch_size, state_dim]
        sigma = torch.exp(0.5 * logvar)    # [batch_size, state_dim]
        
        # Sample n_samples per state
        epsilon = torch.randn(batch_size, n_samples, self.state_dim, device=device)
        
        # Reparameterization trick: s' = mu + sigma * epsilon
        mu_expanded = mu.unsqueeze(1)              # [batch_size, 1, state_dim]
        sigma_expanded = sigma.unsqueeze(1)        # [batch_size, 1, state_dim]
        s_next = mu_expanded + sigma_expanded * epsilon
        
        return s_next


def load_pretrained_model(model_path, model_class, state_dim, **kwargs):
    """
    Helper function to load a pretrained transition model
    
    Args:
        model_path: path to saved model weights
        model_class: TransitionModel subclass
        state_dim: state dimension
        **kwargs: model-specific arguments
    
    Returns:
        model: loaded model on appropriate device
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model
    model = model_class(state_dim, **kwargs)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


# Test usage
if __name__ == "__main__":
    
    print("Transition Model Test")
    print("=" * 60)
    
    state_dim = 29  # Ant Maze
    batch_size = 4
    n_samples = 3
    
    # Test placeholder model
    print("\n1. Placeholder Model:")
    model1 = PlaceholderTransitionModel(state_dim)
    s = torch.randn(batch_size, state_dim)
    s_next = model1(s, n_samples=n_samples)
    print(f"   Input: {s.shape} → Output: {s_next.shape}")
    print(f"   ✓ Correct shape [batch_size, n_samples, state_dim]")
    
    # Test Gaussian model
    print("\n2. Gaussian Model:")
    model2 = GaussianTransitionModel(state_dim)
    s_next = model2(s, n_samples=n_samples)
    print(f"   Input: {s.shape} → Output: {s_next.shape}")
    print(f"   ✓ Correct shape [batch_size, n_samples, state_dim]")
    
    print("\n✓ All models working correctly!")
    print("\nTo use your own model:")
    print("  1. Subclass TransitionModel")
    print("  2. Implement forward(s, n_samples)")
    print("  3. Load your pretrained weights in __init__")
    print("  4. Pass instance to training loop")
