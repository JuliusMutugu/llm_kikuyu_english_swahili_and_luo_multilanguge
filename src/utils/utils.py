"""
Utilities and helper functions for modern LLM training
"""

import torch
import torch.nn as nn
import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union
import numpy as np
from pathlib import Path


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: Dict[str, Any], filename: str, checkpoint_dir: str = "checkpoints"):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, device: str = 'cpu') -> Dict[str, Any]:
    """Load training checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    logging.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """Get human-readable model size"""
    param_count = count_parameters(model)
    if param_count >= 1e9:
        return f"{param_count / 1e9:.1f}B"
    elif param_count >= 1e6:
        return f"{param_count / 1e6:.1f}M"
    elif param_count >= 1e3:
        return f"{param_count / 1e3:.1f}K"
    else:
        return str(param_count)


def setup_logging(log_file: str = "training.log", level: int = logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return np.exp(loss)


def print_model_info(model: nn.Module):
    """Print detailed model information"""
    total_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,} ({model_size})")
    print(f"Model size: {model_size}")
    
    # Calculate memory usage (approximate)
    param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32 parameter
    print(f"Approximate memory (parameters only): {param_memory:.2f} GB")
    print("=" * 50)


class LRScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(self, optimizer, schedule_type: str = 'cosine', **kwargs):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.base_lr = kwargs.get('base_lr', 1e-4)
        self.max_steps = kwargs.get('max_steps', 100000)
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        self.min_lr = kwargs.get('min_lr', 1e-6)
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Main schedule
            if self.schedule_type == 'cosine':
                progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            elif self.schedule_type == 'linear':
                progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                lr = self.base_lr * (1 - progress) + self.min_lr * progress
            else:
                lr = self.base_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0):
    """Apply gradient clipping to model parameters"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


def ensure_dir(directory: Union[str, Path]):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


class ConfigValidator:
    """Validate configuration parameters"""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        required_fields = [
            'vocab_size', 'hidden_size', 'num_layers', 
            'num_heads', 'max_position_embeddings'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in model config: {field}")
        
        # Validate dimensions
        if config['hidden_size'] % config['num_heads'] != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        
        if 'num_kv_heads' in config:
            if config['num_heads'] % config['num_kv_heads'] != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")
        
        return True

    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration"""
        required_fields = [
            'batch_size', 'learning_rate', 'max_steps'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in training config: {field}")
        
        # Validate ranges
        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        
        if config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        
        return True


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully!")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test time formatting
    print(f"Time format test: {format_time(3665)}")
    
    # Test perplexity calculation
    loss = 2.5
    perplexity = calculate_perplexity(loss)
    print(f"Loss: {loss}, Perplexity: {perplexity:.2f}")
