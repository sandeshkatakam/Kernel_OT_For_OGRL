"""
Training Module
Main training loop following COT patterns
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json
from datetime import datetime


class Trainer:
    """
    Training loop for kernel OT optimization
    
    Combines paired loss and terminal loss with configurable scheduling
    """
    
    def __init__(self, model, kernel, loss_fn, config, device=None):
        """
        Args:
            model: TransitionModel instance (user-provided)
            kernel: IMQKernel instance
            loss_fn: CombinedLoss instance
            config: configuration dictionary
            device: torch device
        """
        self.model = model
        self.kernel = kernel
        self.loss_fn = loss_fn
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move models to device
        self.model.to(self.device)
        self.kernel.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.history = {
            'loss_total': [],
            'loss_paired': [],
            'loss_terminal': [],
        }
        
    def _setup_optimizer(self):
        """Setup optimizer from config"""
        
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 0.0)
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        return optimizer
    
    def train_epoch(self, train_loader, val_loader=None):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader with batches of (s, s_next)
            val_loader: optional validation DataLoader
        
        Returns:
            epoch_metrics: dict with averaged metrics
        """
        
        self.model.train()
        epoch_loss_total = 0.0
        epoch_loss_paired = 0.0
        epoch_loss_terminal = 0.0
        num_batches = 0
        
        # Get parameters from config
        terminal_loss_freq = self.config['training'].get(
            'terminal_loss_frequency', 1
        )
        batch_size = self.config['training']['batch_size']
        
        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config['training']['num_epochs']}",
            leave=True
        )
        
        for batch_idx, (s_data, s_next_data) in enumerate(pbar):
            
            # Move to device
            s_data = s_data.to(self.device)
            s_next_data = s_next_data.to(self.device)
            
            # Use same states for terminal loss (or could use different batch)
            s_init_terminal = s_data.clone()
            goal_states = train_loader.dataset.goal_states.to(self.device)
            
            # Determine if we compute terminal loss this batch
            use_terminal = (batch_idx % terminal_loss_freq) == 0
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_total, loss_dict = self.loss_fn(
                self.model,
                s_data,
                s_next_data,
                s_init_terminal,
                goal_states,
                use_terminal=use_terminal,
                device=self.device
            )
            
            # Backward pass
            loss_total.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss_total += loss_dict['loss_total']
            epoch_loss_paired += loss_dict['loss_paired']
            epoch_loss_terminal += loss_dict['loss_terminal']
            num_batches += 1
            
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['loss_total'],
                'paired': loss_dict['loss_paired'],
                'terminal': loss_dict['loss_terminal'],
            })
        
        # Average metrics over epoch
        epoch_metrics = {
            'loss_total': epoch_loss_total / num_batches,
            'loss_paired': epoch_loss_paired / num_batches,
            'loss_terminal': epoch_loss_terminal / num_batches,
        }
        
        # Store in history
        self.history['loss_total'].append(epoch_metrics['loss_total'])
        self.history['loss_paired'].append(epoch_metrics['loss_paired'])
        self.history['loss_terminal'].append(epoch_metrics['loss_terminal'])
        
        # Validation
        if val_loader is not None:
            val_metrics = self.validate(val_loader)
            epoch_metrics.update(val_metrics)
        
        return epoch_metrics
    
    def validate(self, val_loader):
        """
        Validation step
        
        Args:
            val_loader: DataLoader with validation batches
        
        Returns:
            val_metrics: dict with validation metrics
        """
        
        self.model.eval()
        val_loss_total = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for s_data, s_next_data in val_loader:
                
                s_data = s_data.to(self.device)
                s_next_data = s_next_data.to(self.device)
                s_init_terminal = s_data.clone()
                goal_states = val_loader.dataset.goal_states.to(self.device)
                
                # Compute loss
                loss_total, loss_dict = self.loss_fn(
                    self.model,
                    s_data,
                    s_next_data,
                    s_init_terminal,
                    goal_states,
                    use_terminal=True,
                    device=self.device
                )
                
                val_loss_total += loss_dict['loss_total']
                num_batches += 1
        
        val_metrics = {
            'val_loss': val_loss_total / num_batches,
        }
        
        return val_metrics
    
    def train(self, train_loader, val_loader=None):
        """
        Complete training loop
        
        Args:
            train_loader: training DataLoader
            val_loader: optional validation DataLoader
        
        Returns:
            history: training history
        """
        
        num_epochs = self.config['training']['num_epochs']
        
        print("=" * 80)
        print("TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch(train_loader, val_loader)
            
            # Print metrics
            msg = f"[Epoch {epoch+1}/{num_epochs}] "
            msg += f"Loss: {epoch_metrics['loss_total']:.6f} | "
            msg += f"L_paired: {epoch_metrics['loss_paired']:.6f} | "
            msg += f"L_terminal: {epoch_metrics['loss_terminal']:.6f}"
            
            if 'val_loss' in epoch_metrics:
                msg += f" | Val Loss: {epoch_metrics['val_loss']:.6f}"
            
            print(msg)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_freq', 1) == 0:
                self.save_checkpoint(epoch)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        return self.history
    
    def save_checkpoint(self, epoch):
        """
        Save model checkpoint
        
        Args:
            epoch: epoch number
        """
        
        save_dir = self.config['output']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config,
        }
        
        save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
        torch.save(checkpoint, save_path)
        
        print(f"  ✓ Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load from checkpoint
        
        Args:
            checkpoint_path: path to checkpoint file
        """
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Resumed from epoch {self.epoch}, step {self.step}")


class TransitionDataset(torch.utils.data.Dataset):
    """
    Dataset for transition pairs
    """
    
    def __init__(self, transition_data, goal_states):
        """
        Args:
            transition_data: dict with 's' and 's_next' tensors
            goal_states: goal state tensor
        """
        self.s = transition_data['s']
        self.s_next = transition_data['s_next']
        self.goal_states = goal_states
        
    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, idx):
        return self.s[idx], self.s_next[idx]


# Test usage
if __name__ == "__main__":
    
    print("Trainer Test")
    print("=" * 60)
    
    # Note: Full training test requires data and models
    # This just checks that imports and setup work
    
    print("✓ Trainer module imports successfully")
    print("✓ Ready for training!")
