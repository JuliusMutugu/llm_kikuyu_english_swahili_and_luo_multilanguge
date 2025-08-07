#!/usr/bin/env python3
"""
Laptop-Optimized LLM Training Script

This script is specifically designed for training modern LLMs on laptop hardware.
It includes memory optimizations, efficient data loading, and automatic hyperparameter tuning.

Usage:
    python train_laptop.py
    python train_laptop.py --data_path ./data/sample.txt
    python train_laptop.py --resume ./checkpoints/model_latest.pt
"""

import os
import argparse
import time
import math
import psutil
import GPUtil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from models.modern_llm import ModernLLM, ModelConfig, create_model
try:
    from data.data_loader import TextDataset, create_data_loaders
except ImportError:
    print("⚠️ Using simple data loader (no external dependencies)")
    from data.simple_data_loader import SimpleTextDataset, create_simple_data_loaders

class LaptopTrainer:
    """Memory-efficient trainer optimized for laptop hardware."""
    
    def __init__(
        self,
        model: ModernLLM,
        config: ModelConfig,
        device: str = "auto",
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
    ):
        self.config = config
        self.device = self._get_optimal_device(device)
        self.mixed_precision = mixed_precision and self.device != "cpu"
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Enable optimizations
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        # Compile model for PyTorch 2.0+ (if available)
        try:
            if hasattr(torch, 'compile') and self.device != "cpu":
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ Model compiled with PyTorch 2.0")
        except Exception as e:
            print(f"⚠️ Could not compile model: {e}")
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        print(f"🚀 Trainer initialized on {self.device}")
        if self.mixed_precision:
            print("✅ Mixed precision training enabled")
    
    def _get_optimal_device(self, device: str) -> str:
        """Determine the best device for training."""
        if device != "auto":
            return device
        
        if torch.cuda.is_available():
            # Check GPU memory
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    available_memory = gpu.memoryFree  # MB
                    if available_memory > 1000:  # At least 1GB
                        print(f"🎮 Using CUDA (GPU memory: {available_memory:.0f}MB)")
                        return "cuda"
            except:
                pass
        
        if torch.backends.mps.is_available():
            print("🍎 Using MPS (Apple Silicon)")
            return "mps"
        
        print("💻 Using CPU")
        return "cpu"
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage."""
        info = {}
        
        # System memory
        mem = psutil.virtual_memory()
        info['system_memory_used'] = mem.used / 1024**2  # MB
        info['system_memory_total'] = mem.total / 1024**2  # MB
        info['system_memory_percent'] = mem.percent
        
        # GPU memory
        if self.device == "cuda" and torch.cuda.is_available():
            info['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**2  # MB
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        
        return info
    
    def setup_optimizer(self, learning_rate: float = 3e-4, weight_decay: float = 0.1) -> optim.Optimizer:
        """Create memory-efficient optimizer."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'layernorm', 'norm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Use AdamW with 8-bit optimizer for memory efficiency
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            print("✅ Using 8-bit AdamW optimizer")
        except ImportError:
            print("⚠️ bitsandbytes not available, using standard AdamW")
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            print("✅ Using standard AdamW optimizer")
        except Exception as e:
            print(f"⚠️ Could not use 8-bit optimizer: {e}")
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            print("✅ Using standard AdamW optimizer")
        
        return optimizer
    
    def calculate_batch_size(self, sequence_length: int = 256) -> int:
        """Automatically calculate optimal batch size based on available memory."""
        if self.device == "cpu":
            return 1  # CPU training is slow, use minimal batch
        
        # Get available memory
        if self.device == "cuda":
            available_mem = torch.cuda.get_device_properties(0).total_memory
            # Reserve 20% for other processes
            available_mem = int(available_mem * 0.8)
        else:
            # MPS or other devices - be conservative
            available_mem = 4 * 1024**3  # Assume 4GB
        
        # Estimate memory per sample (very rough)
        params = sum(p.numel() for p in self.model.parameters())
        estimated_per_sample = (
            params * 4 +  # Forward pass activations
            sequence_length * self.config.hidden_size * 4 +  # Input embeddings
            sequence_length * self.config.vocab_size * 4  # Output logits
        )
        
        # Add overhead for gradients and optimizer
        estimated_per_sample *= 3
        
        max_batch_size = max(1, available_mem // estimated_per_sample)
        recommended_batch = min(max_batch_size, 8)  # Cap at 8 for stability
        
        return recommended_batch
    
    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: optim.Optimizer,
        accumulation_steps: int = 1,
    ) -> float:
        """Perform one training step with memory optimizations."""
        # Clear cache periodically
        if self.step % 100 == 0 and self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Split batch for gradient accumulation
        batch_size = batch.size(0)
        micro_batch_size = max(1, batch_size // accumulation_steps)
        total_loss = 0.0
        
        optimizer.zero_grad()
        
        for i in range(0, batch_size, micro_batch_size):
            micro_batch = batch[i:i + micro_batch_size]
            
            # Mixed precision forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(micro_batch, labels=micro_batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / accumulation_steps  # Scale for accumulation
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(micro_batch, labels=micro_batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                loss = loss / accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
        
        # Optimizer step
        if self.mixed_precision:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        self.step += 1
        return total_loss
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        accumulation_steps: int = 4,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 500,
        eval_every: int = 200,
    ):
        """Main training loop optimized for laptops."""
        
        # Setup
        Path(save_dir).mkdir(exist_ok=True)
        Path(log_dir).mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        optimizer = self.setup_optimizer()
        
        # Calculate steps
        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        
        print(f"\n🏋️ Starting Laptop Training")
        print(f"📊 Training Configuration:")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.mixed_precision}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total steps: {total_steps}")
        print(f"   Gradient accumulation: {accumulation_steps}")
        
        # Memory info
        mem_info = self.get_memory_info()
        print(f"\n💾 Memory Status:")
        print(f"   System RAM: {mem_info['system_memory_used']:.0f}MB / {mem_info['system_memory_total']:.0f}MB ({mem_info['system_memory_percent']:.1f}%)")
        if 'gpu_memory_total' in mem_info:
            print(f"   GPU VRAM: {mem_info['gpu_memory_used']:.0f}MB / {mem_info['gpu_memory_total']:.0f}MB")
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(train_dataloader):
                batch = batch.to(self.device)
                
                # Training step
                step_loss = self.train_step(batch, optimizer, accumulation_steps)
                epoch_loss += step_loss
                
                # Logging
                if self.step % 50 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = self.step * batch.numel() / elapsed
                    
                    print(f"   Step {self.step:6d} | Loss: {step_loss:.4f} | Tokens/sec: {tokens_per_sec:.0f}")
                    
                    writer.add_scalar('Loss/Train', step_loss, self.step)
                    writer.add_scalar('Throughput/TokensPerSec', tokens_per_sec, self.step)
                    
                    # Memory tracking
                    if self.step % 200 == 0:
                        mem_info = self.get_memory_info()
                        writer.add_scalar('Memory/SystemRAM_MB', mem_info['system_memory_used'], self.step)
                        if 'gpu_memory_used' in mem_info:
                            writer.add_scalar('Memory/GPU_MB', mem_info['gpu_memory_used'], self.step)
                
                # Validation
                if val_dataloader and self.step % eval_every == 0:
                    val_loss = self.evaluate(val_dataloader)
                    writer.add_scalar('Loss/Validation', val_loss, self.step)
                    print(f"   📊 Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(save_dir, "best_model.pt")
                        print("   💾 Saved best model!")
                
                # Regular checkpointing
                if self.step % save_every == 0:
                    self.save_checkpoint(save_dir, "latest_model.pt")
                    print(f"   💾 Checkpoint saved at step {self.step}")
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"   📊 Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")
            writer.add_scalar('Loss/EpochAvg', avg_epoch_loss, epoch)
        
        # Final save
        self.save_checkpoint(save_dir, "final_model.pt")
        writer.close()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"   Final loss: {avg_epoch_loss:.4f}")
        print(f"   Best validation loss: {self.best_loss:.4f}")
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(batch, labels=batch)
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                else:
                    outputs = self.model(batch, labels=batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, save_dir: str, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = Path(save_dir) / filename
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Loaded checkpoint from step {self.step}")


def main():
    parser = argparse.ArgumentParser(description='Laptop-optimized LLM training')
    parser.add_argument('--data_path', type=str, default='./data/sample_data.txt',
                       help='Path to training data')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0 for auto)')
    parser.add_argument('--sequence_length', type=int, default=256, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no_gradient_checkpointing', action='store_true', help='Disable gradient checkpointing')
    
    args = parser.parse_args()
    
    print("🚀 Laptop LLM Trainer")
    print("=" * 50)
    
    # Create laptop-optimized model config
    config = ModelConfig(
        vocab_size=8000,              # Smaller vocab for faster training
        hidden_size=512,              # Laptop-friendly size
        num_layers=6,                 # Fewer layers for memory
        num_heads=8,
        num_kv_heads=4,               # Multi-Query Attention
        intermediate_size=1024,       # Smaller FFN
        max_position_embeddings=1024, # Shorter sequences
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_mixed_precision=not args.no_mixed_precision,
    )
    
    # Create model
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model: {total_params:,} parameters (~{total_params * 4 / 1024**2:.1f}MB)")
    
    # Create trainer
    trainer = LaptopTrainer(
        model=model,
        config=config,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    
    # Auto-calculate batch size if not specified
    if args.batch_size == 0:
        batch_size = trainer.calculate_batch_size(args.sequence_length)
        print(f"🔧 Auto-calculated batch size: {batch_size}")
    else:
        batch_size = args.batch_size
    
    # Load data
    try:
        if 'create_data_loaders' in globals():
            train_loader, val_loader = create_data_loaders(
                data_path=args.data_path,
                batch_size=batch_size,
                sequence_length=args.sequence_length,
                vocab_size=config.vocab_size,
                split_ratio=0.9,
            )
        else:
            train_loader, val_loader = create_simple_data_loaders(
                data_path=args.data_path,
                batch_size=batch_size,
                sequence_length=args.sequence_length,
                vocab_size=config.vocab_size,
                split_ratio=0.9,
            )
        print(f"📚 Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    except Exception as e:
        print(f"❌ Could not load data from {args.data_path}: {e}")
        print("🔧 Creating dummy data for testing...")
        
        # Create dummy dataset for testing
        dummy_data = torch.randint(0, config.vocab_size, (1000, args.sequence_length))
        train_dataset = torch.utils.data.TensorDataset(dummy_data[:800])
        val_dataset = torch.utils.data.TensorDataset(dummy_data[800:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print("✅ Using dummy data for testing")
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.epochs,
            accumulation_steps=4,  # Simulate larger batch sizes
        )
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        trainer.save_checkpoint("./checkpoints", "interrupted_model.pt")
        print("💾 Saved interrupted model")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Training session completed!")


if __name__ == "__main__":
    main()
