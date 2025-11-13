#!/usr/bin/env python3
"""
Standalone Training Script for Kernel OT Learning on Offline Goal Conditioned RL
Converts notebook training pipeline to command-line executable script

Usage:
    python train_standalone.py --config config_experiments.yaml
    python train_standalone.py --config config_experiments.yaml --wandb-run-name "my_exp_v1"
    python train_standalone.py --config config_experiments.yaml --wandb-run-name "my_exp_v1" --checkpoint-dir custom_checkpoints
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# ============================================================================
# Setup and Configuration Loading
# ============================================================================

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Kernel OT Training Script for Offline Goal Conditioned RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_standalone.py --config config_experiments.yaml
  python train_standalone.py --config config_experiments.yaml --wandb-run-name "exp_v1"
  python train_standalone.py --config config_experiments.yaml --checkpoint-dir checkpoints_phase3
        """
    )
    
    parser.add_argument('--config', type=str, default='config_experiments.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Override checkpoint directory from config')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Override WandB run name from config')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=None,
                       help='Override WandB tags from config')
    parser.add_argument('--wandb-notes', type=str, default=None,
                       help='Override WandB notes from config')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (verbose logging)')
    
    return parser.parse_args()

class ExperimentLogger:
    """Logger class for both console and file output"""
    
    def __init__(self, output_dir, filename="experiment_log.txt"):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, filename)
        
    def log(self, message):
        """Log to both console and file"""
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")
    
    def log_dict(self, title, data_dict):
        """Log a dictionary with title"""
        self.log(f"\n{title}:")
        for key, val in data_dict.items():
            self.log(f"  {key}: {val}")

def setup_device(config):
    """Setup GPU device"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu']['cuda_visible_devices'])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            device = torch.device(f"cuda:{config['gpu']['cuda_device']}")
            print(f"✓ GPU Available: {torch.cuda.get_device_name(config['gpu']['cuda_device'])}")
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
            print(f"  CUDA Capability: {torch.cuda.get_device_capability(config['gpu']['cuda_device'])}")
            
            # GPU Memory Optimization
            torch.cuda.set_per_process_memory_fraction(config['gpu']['per_process_memory_fraction'])
            torch.backends.cuda.matmul.allow_tf32 = config['gpu']['enable_tf32']
            
            print(f"✓ GPU Memory Optimization ENABLED:")
            print(f"  - Per-process memory fraction: {config['gpu']['per_process_memory_fraction']*100}%")
            print(f"  - TF32 precision: {'ENABLED' if config['gpu']['enable_tf32'] else 'DISABLED'}")
            
        else:
            device = torch.device("cpu")
            print(f"⚠ GPU Not available - using CPU (PyTorch version: {torch.__version__})")
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        device = torch.device("cpu")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}\n")
    
    return device

def setup_output_directories(config):
    """Create output directories"""
    base_output_dir = config['output']['base_output_dir']
    checkpoint_dir_name = config['output']['checkpoint_dir']
    
    output_dir = base_output_dir
    checkpoint_dir = os.path.join(output_dir, checkpoint_dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return output_dir, checkpoint_dir

# ============================================================================
# Data Loading
# ============================================================================

def load_data(config, device):
    """Load data using DataPreparator"""
    print("[Step 2] Data Loading")
    print("="*80)
    
    # Import custom modules
    sys.path.insert(0, '/data1/sandesh/COT/Kernel_OT_for_OGRL')
    sys.path.insert(0, '/data1/sandesh/COT')
    
    from data_prep import DataPreparator
    from train import TransitionDataset
    
    # Create config dict compatible with DataPreparator
    data_prep_config = {
        'environment': config['environment']['env_name'],  # ← DataPreparator expects string
        'num_goal_states': config['environment']['num_goal_states'],
        'num_transitions': config['environment']['num_transitions'],
        'transition_data_save_path': config['data']['transition_data_save_path'],
        'goal_state_save_path': config['data']['goal_state_save_path'],
    }
    
    preparator = DataPreparator(data_prep_config)
    transition_data, goal_states_prepared = preparator.prepare_data()
    
    print(f"✓ Data loaded from {config['environment']['env_name']}")
    print(f"  States shape: {transition_data['s'].shape}")
    print(f"  Next states shape: {transition_data['s_next'].shape}")
    
    # Extract and process
    s_all = transition_data['s'].cpu().float() if transition_data['s'].is_cuda else transition_data['s'].float()
    s_next_all = transition_data['s_next'].cpu().float() if transition_data['s_next'].is_cuda else transition_data['s_next'].float()
    STATE_DIM = s_all.shape[1]
    
    GOAL_STATE = goal_states_prepared[0].cpu().float() if goal_states_prepared[0].is_cuda else goal_states_prepared[0].float()
    
    print(f"✓ Goal state extracted (state_dim={STATE_DIM})")
    
    # Train/val split
    split_idx = int(config['data']['train_val_split'] * len(s_all))
    s_train = s_all[:split_idx]
    s_next_train = s_next_all[:split_idx]
    s_val = s_all[split_idx:]
    s_next_val = s_next_all[split_idx:]
    
    # Create datasets
    train_transition_data = {
        's': s_train.numpy(),
        's_next': s_next_train.numpy(),
    }
    val_transition_data = {
        's': s_val.numpy(),
        's_next': s_next_val.numpy(),
    }
    
    train_dataset = TransitionDataset(train_transition_data, goal_states=GOAL_STATE.unsqueeze(0))
    val_dataset = TransitionDataset(val_transition_data, goal_states=GOAL_STATE.unsqueeze(0))
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Data split: train={len(s_train)}, val={len(s_val)}")
    print(f"✓ Dataloaders created (batch_size={batch_size})")
    print(f"✓ Data loading complete!\n")
    
    return (train_loader, val_loader, train_dataset, val_dataset, 
            STATE_DIM, GOAL_STATE, s_train, s_next_train, s_val, s_next_val)

# ============================================================================
# Model Initialization
# ============================================================================

def initialize_model(config, STATE_DIM, device):
    """Initialize transition model and kernel"""
    print("[Step 3] Model Initialization")
    print("="*80)
    
    sys.path.insert(0, '/data1/sandesh/COT/Kernel_OT_for_OGRL')
    sys.path.insert(0, '/data1/sandesh/COT')
    
    from kernels import IMQKernel
    from models_advanced.policy_aq_model import PolicyAugmentedTransitionModel
    
    # Initialize kernel
    kernel = IMQKernel(alpha=config['kernel']['alpha'])
    print(f"✓ {config['kernel']['type']} initialized (α={config['kernel']['alpha']}, fixed)")
    
    # Initialize PolicyAugmentedTransitionModel
    maf_ckpt_path = config['environment']['maf_checkpoint']
    n_components = config['model']['n_mdn_components']
    hidden_dim = config['model']['hidden_dim']
    action_dim = config['model']['action_dim']
    
    print(f"\nInitializing PolicyAugmentedTransitionModel...")
    print(f"  State dim: {STATE_DIM}")
    print(f"  Action dim: {action_dim}")
    print(f"  MDN components: {n_components}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  MAF checkpoint: {maf_ckpt_path}")
    
    transition_model = PolicyAugmentedTransitionModel(
        state_dim=STATE_DIM,
        action_dim=action_dim,
        policy_ckpt=maf_ckpt_path,
        n_components=n_components,
        hidden_dim=hidden_dim,
        device=device
    )
    
    transition_model = transition_model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in transition_model.parameters())
    learnable_params = sum(p.numel() for p in transition_model.parameters() if p.requires_grad)
    
    print(f"\n✓ PolicyAugmentedTransitionModel initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Learnable parameters: {learnable_params:,}")
    print(f"  Device: {str(device)}")
    print(f"✓ Model initialization complete!\n")
    
    return transition_model, kernel

# ============================================================================
# Loss and Trainer Setup
# ============================================================================

def setup_training(config, transition_model, kernel, device):
    """Setup loss functions and trainer"""
    print("[Step 4] Loss Functions and Trainer")
    print("="*80)
    
    sys.path.insert(0, '/data1/sandesh/COT/Kernel_OT_for_OGRL')
    sys.path.insert(0, '/data1/sandesh/COT')
    
    from losses import CombinedLoss
    from train import Trainer
    
    # Create loss function
    loss_fn = CombinedLoss(kernel=kernel, config=config)
    print("✓ CombinedLoss initialized")
    print(f"  Lambda paired: {config['loss']['lambda_paired']}")
    print(f"  Lambda terminal: {config['loss']['lambda_terminal']}")
    print(f"  Rollout horizon T: {config['sampling']['T']}")
    
    # Initialize trainer
    trainer = Trainer(
        model=transition_model,
        kernel=kernel,
        loss_fn=loss_fn,
        config=config,
        device=device
    )
    print("✓ Trainer initialized")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Device: {str(device)}")
    print(f"✓ Loss and trainer setup complete!\n")
    
    # Initialize training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_loss_paired': [],
        'train_loss_terminal': [],
        'val_loss_paired': [],
    }
    
    return loss_fn, trainer, training_history

# ============================================================================
# Weights & Biases Setup
# ============================================================================

def setup_wandb(config, args, logger):
    """Setup Weights & Biases"""
    print("[Step 4.5] Weights & Biases - Interactive Run Configuration")
    print("="*80)
    
    if args.no_wandb:
        logger.log("⚠ WandB disabled via --no-wandb flag")
        return None
    
    if not config['wandb']['enabled']:
        logger.log("⚠ WandB disabled in config")
        return None
    
    try:
        import wandb
        print(f"✓ wandb already installed (version: {wandb.__version__})")
    except ImportError:
        print("Installing wandb...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb", "-q"])
        import wandb
        print(f"✓ wandb installed (version: {wandb.__version__})")
    
    # Login to wandb
    api_key = config['wandb']['api_key']
    try:
        import wandb
        wandb.login(key=api_key, relogin=False)
        print(f"✓ Weights & Biases logged in successfully!")
    except Exception as e:
        print(f"⚠️  wandb login error: {e}")
        wandb.login(key=api_key, relogin=True)
    
    # Get run configuration
    run_name = args.wandb_run_name or config['wandb']['run_name']
    if not run_name:
        run_name = f"kernel_ot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_description = config['wandb']['run_description']
    tags = args.wandb_tags or config['wandb']['tags']
    notes = args.wandb_notes or config['wandb']['notes']
    
    print("\n" + "-"*80)
    print("WANDB RUN CONFIGURATION:")
    print("-"*80)
    print(f"Run Name:     {run_name}")
    print(f"Description:  {run_description}")
    print(f"Tags:         {', '.join(tags)}")
    print(f"Notes:        {notes if notes else '(none)'}")
    print(f"Project:      {config['wandb']['project_name']}")
    print("-"*80)
    
    # Initialize wandb run
    try:
        import wandb
        wandb_run = wandb.init(
            project=config['wandb']['project_name'],
            name=run_name,
            tags=tags,
            notes=notes,
            config={
                'architecture': 'PolicyAugmentedTransitionModel',
                'epochs': config['training']['num_epochs'],
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'lambda_paired': config['loss']['lambda_paired'],
                'lambda_terminal': config['loss']['lambda_terminal'],
                'rollout_horizon_T': config['sampling']['T'],
                'n_rollout_samples': config['sampling']['n_rollout_samples'],
                'kernel_type': config['kernel']['type'],
            },
        )
        
        print(f"\n✓ Weights & Biases run initialized!")
        print(f"  Project: {wandb_run.project_name}")
        print(f"  Run ID: {wandb_run.id}")
        print(f"  Run URL: {wandb_run.get_url()}")
        
        logger.log(f"✓ W&B Run initialized: {run_name}")
        logger.log(f"  Run URL: {wandb_run.get_url()}")
        
        return wandb_run
        
    except Exception as e:
        print(f"⚠️  Error initializing W&B run: {e}")
        logger.log(f"⚠️  W&B initialization failed: {e}")
        return None

# ============================================================================
# Training Loop
# ============================================================================

def train(config, trainer, loss_fn, train_loader, val_loader, train_dataset, 
          val_dataset, device, logger, output_dir, wandb_run):
    """Main training loop"""
    
    logger.log("\n[Step 5] SINGLE-GPU Training Loop (Terminal Loss Only)")
    logger.log("="*80)
    logger.log("⚠️  IMPORTANT: TERMINAL LOSS ONLY - NO paired loss")
    logger.log("   Lambda_paired = 0.0 (disabled)")
    logger.log("   Lambda_terminal = 1.0 (enabled)")
    logger.log("="*80)
    
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "policy_augmented_transition_kernel_best_terminal_only.pt")
    
    # Gradient monitoring
    max_grad_norms = []
    mean_grad_norms = []
    terminal_loss_tracker = []
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_loss_terminal': [],
    }
    
    for epoch in range(num_epochs):
        # Training phase
        trainer.model.train()
        epoch_train_loss = 0.0
        epoch_train_loss_terminal = 0.0
        num_batches = 0
        epoch_grad_norms = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", disable=False)
        
        for batch_idx, (s_batch, s_next_batch) in enumerate(pbar):
            # Move to device
            s_batch = s_batch.to(device)
            s_next_batch = s_next_batch.to(device)
            goal_states_batch = train_dataset.goal_states.to(device)
            
            # Compute terminal loss only
            use_terminal_loss = True
            
            try:
                loss_total, loss_dict = loss_fn(
                    trainer.model, 
                    s_batch, 
                    s_next_batch, 
                    s_batch,
                    goal_states_batch,
                    use_terminal=use_terminal_loss,
                    device=device
                )
            except Exception as e:
                logger.log(f"❌ ERROR in loss computation (Epoch {epoch+1}, Batch {batch_idx}): {str(e)}")
                raise
            
            # Check for NaN/Inf
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                logger.log(f"❌ ERROR: Loss is NaN/Inf at Epoch {epoch+1}, Batch {batch_idx}")
                raise ValueError("Loss is NaN/Inf - training unstable!")
            
            # Backward pass
            trainer.optimizer.zero_grad()
            loss_total.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), 
                max_norm=config['training']['gradient_clip_norm']
            )
            epoch_grad_norms.append(grad_norm.item())
            torch.nn.utils.clip_grad_value_(
                trainer.model.parameters(), 
                clip_value=config['training']['gradient_clip_value']
            )
            
            if grad_norm > 1.0:
                logger.log(f"⚠ WARNING Epoch {epoch+1}, Batch {batch_idx}: High gradient norm {grad_norm:.4f}")
            
            trainer.optimizer.step()
            
            # Extract loss values
            loss_terminal_val = loss_dict.get('terminal_loss', 0.0)
            loss_total_val = loss_total.item()
            grad_norm_val = grad_norm.item()
            
            epoch_train_loss += loss_total_val
            epoch_train_loss_terminal += loss_terminal_val
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_total_val:.4f}",
                'terminal': f"{loss_terminal_val:.4f}",
                'grad_norm': f"{grad_norm_val:.4f}"
            })
            
            # Log to wandb every batch
            if wandb_run is not None:
                import wandb
                wandb.log({
                    'batch': (epoch * len(train_loader)) + batch_idx,
                    'epoch': epoch + 1,
                    'train/loss_total': loss_total_val,
                    'train/loss_terminal': loss_terminal_val,
                    'train/gradient_norm': grad_norm_val,
                    'train/learning_rate': config['training']['learning_rate'],
                })
            
            # Log terminal loss periodically
            if batch_idx == 0 or (batch_idx + 1) % config['logging']['log_every_n_batches'] == 0:
                logger.log(f"[Epoch {epoch+1}, Batch {batch_idx+1}] Terminal Loss: {loss_terminal_val:.6f}")
            
            # Clear GPU memory every 10 batches
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Average losses
        epoch_train_loss /= num_batches
        epoch_train_loss_terminal /= max(1, num_batches)
        
        # Clear GPU memory after training phase
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Gradient statistics
        max_grad_norm = max(epoch_grad_norms) if epoch_grad_norms else 0.0
        mean_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        max_grad_norms.append(max_grad_norm)
        mean_grad_norms.append(mean_grad_norm)
        terminal_loss_tracker.append(epoch_train_loss_terminal)
        
        # Validation phase
        trainer.model.eval()
        val_loss_total = 0.0
        val_loss_terminal = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for s_val, s_next_val in val_loader:
                s_val = s_val.to(device)
                s_next_val = s_next_val.to(device)
                goal_states_val = val_dataset.goal_states.to(device)
                
                loss_val_total, loss_val_dict = loss_fn(
                    trainer.model,
                    s_val,
                    s_next_val,
                    s_val,
                    goal_states_val,
                    use_terminal=True,
                    device=device
                )
                
                val_loss_total += loss_val_total.item()
                val_loss_terminal += loss_val_dict.get('terminal_loss', 0.0)
                num_val_batches += 1
        
        # Average validation losses
        val_loss_total /= max(1, num_val_batches)
        val_loss_terminal /= max(1, num_val_batches)
        
        # Clear GPU memory after validation
        torch.cuda.empty_cache()
        
        # Save best model
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(trainer.model.state_dict(), best_model_path)
            logger.log(f"✓ Saved best model at epoch {epoch+1} (val_loss={val_loss_total:.6f})")
        
        # Store training history
        training_history['train_loss'].append(epoch_train_loss)
        training_history['val_loss'].append(val_loss_total)
        training_history['train_loss_terminal'].append(epoch_train_loss_terminal)
        
        # Log epoch metrics to wandb
        if wandb_run is not None:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'train/loss_total_avg': epoch_train_loss,
                'train/loss_terminal_avg': epoch_train_loss_terminal,
                'val/loss_total': val_loss_total,
                'val/loss_terminal': val_loss_terminal,
                'train/gradient_max': max_grad_norm,
                'train/gradient_mean': mean_grad_norm,
            })
        
        # Epoch summary
        logger.log(f"\n[Epoch {epoch+1}/{num_epochs}] Training Summary:")
        logger.log(f"  Train Loss (terminal only): {epoch_train_loss:.6f}")
        logger.log(f"  Val Loss (terminal only): {val_loss_total:.6f}")
        logger.log(f"  Gradient (max: {max_grad_norm:.6f}, mean: {mean_grad_norm:.6f})")
        
        # Check terminal loss progression
        if epoch > 0:
            prev_terminal = training_history['train_loss_terminal'][-2]
            curr_terminal = training_history['train_loss_terminal'][-1]
            if abs(prev_terminal - curr_terminal) < 1e-6:
                logger.log(f"  ⚠️  Terminal loss not changing! Previous: {prev_terminal:.6f}, Current: {curr_terminal:.6f}")
    
    logger.log("\n" + "="*80)
    logger.log("✓ Training Complete!")
    logger.log("="*80)
    
    return training_history, best_model_path

# ============================================================================
# WandB Finalization
# ============================================================================

def finalize_wandb(wandb_run, training_history, output_dir, logger):
    """Finalize WandB run with summary metrics"""
    
    logger.log("\n[Step 5.5] Weights & Biases - Training Summary & Artifacts")
    logger.log("="*80)
    
    if wandb_run is None:
        logger.log("⚠️  W&B not initialized - skipping artifact logging")
        return
    
    try:
        import wandb
        
        # Calculate final metrics
        initial_terminal_loss = training_history['train_loss_terminal'][0]
        final_terminal_loss = training_history['train_loss_terminal'][-1]
        best_val_idx = np.argmin(training_history['val_loss'])
        best_val_loss_final = training_history['val_loss'][best_val_idx]
        
        # Log summary metrics
        wandb.log({
            'final/train_loss': training_history['train_loss'][-1],
            'final/best_val_loss': best_val_loss_final,
            'final/best_epoch': best_val_idx + 1,
            'final/terminal_loss': final_terminal_loss,
            'metrics/terminal_loss_reduction': (1 - final_terminal_loss / max(initial_terminal_loss, 1e-8)) * 100,
        })
        
        # Save best model to wandb artifacts
        best_model_path = os.path.join(output_dir, "policy_augmented_transition_kernel_best_terminal_only.pt")
        if os.path.exists(best_model_path):
            logger.log(f"✓ Saving best model to W&B artifacts...")
            artifact = wandb.Artifact('model-best-terminal-only', type='model')
            artifact.add_file(best_model_path, name='best_model.pt')
            wandb_run.log_artifact(artifact)
        
        # Create and log training curves
        logger.log(f"✓ Creating training curves for W&B...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Total loss
        ax = axes[0]
        ax.plot(training_history['train_loss'], 'o-', label='Train', linewidth=2, markersize=5)
        ax.plot(training_history['val_loss'], 's-', label='Val', linewidth=2, markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss (Terminal Loss Only)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Terminal loss only
        ax = axes[1]
        ax.plot(training_history['train_loss_terminal'], 'o-', label='Train Terminal', linewidth=2, markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Terminal Loss')
        ax.set_title('Terminal Loss Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"training_curves": wandb.Image(fig)})
        plt.close()
        
        logger.log(f"✓ Training curves logged to W&B!")
        
        # Finish the run
        wandb_run.finish()
        logger.log(f"\n✓ W&B run finished successfully!")
        logger.log(f"  View results: {wandb_run.get_url()}")
        
    except Exception as e:
        logger.log(f"⚠️  Error finalizing W&B run: {e}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override config values from command line
    if args.checkpoint_dir:
        config['output']['checkpoint_dir'] = args.checkpoint_dir
    
    # Print header
    print("\n" + "="*80)
    print("Kernel OT Learning for Offline Goal Conditioned RL")
    print("Training Script (Standalone)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {args.config}")
    print(f"Output dir: {config['output']['base_output_dir']}")
    print("="*80 + "\n")
    
    # Setup
    device = setup_device(config)
    output_dir, checkpoint_dir = setup_output_directories(config)
    
    # Update config with detected values
    config['gpu']['device'] = str(device)
    config['output']['checkpoint_dir'] = checkpoint_dir
    
    # Initialize logger
    logger = ExperimentLogger(output_dir, config['logging']['log_file'])
    
    logger.log(f"\n{'='*80}")
    logger.log(f"Kernel OT Learning for Offline Goal Conditioned RL (TRAINING)")
    logger.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"{'='*80}")
    logger.log(f"✓ Device: {device}")
    logger.log(f"✓ Output directory: {output_dir}")
    logger.log(f"✓ Checkpoint directory: {checkpoint_dir}")
    
    # Set random seeds
    np.random.seed(config['training']['random_seed'])
    torch.manual_seed(config['training']['random_seed'])
    
    # Step 1: Data Loading
    (train_loader, val_loader, train_dataset, val_dataset, 
     STATE_DIM, GOAL_STATE, s_train, s_next_train, s_val, s_next_val) = load_data(config, device)
    
    # Update config with detected state dim
    config['model']['state_dim'] = STATE_DIM
    
    # Step 2: Model Initialization
    transition_model, kernel = initialize_model(config, STATE_DIM, device)
    
    # Step 3: Loss and Trainer Setup
    loss_fn, trainer, _ = setup_training(config, transition_model, kernel, device)
    
    # Step 4: WandB Setup
    wandb_run = setup_wandb(config, args, logger)
    
    # Step 5: Training Loop
    training_history, best_model_path = train(
        config, trainer, loss_fn, train_loader, val_loader, 
        train_dataset, val_dataset, device, logger, output_dir, wandb_run
    )
    
    # Step 6: WandB Finalization
    finalize_wandb(wandb_run, training_history, output_dir, logger)
    
    # Save training history
    np.save(os.path.join(output_dir, 'training_history.npy'), training_history)
    logger.log(f"✓ Training history saved")
    
    # Save configuration used
    config_save_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.log(f"✓ Configuration saved: {config_save_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults:")
    print(f"  Output Directory: {output_dir}")
    print(f"  Best Model: {best_model_path}")
    print(f"  Training Log: {logger.log_path}")
    print(f"  Training History: {os.path.join(output_dir, 'training_history.npy')}")
    if wandb_run:
        print(f"  WandB Run: {wandb_run.get_url()}")
    print("="*80 + "\n")
    
    logger.log(f"\n✓ Training pipeline complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
