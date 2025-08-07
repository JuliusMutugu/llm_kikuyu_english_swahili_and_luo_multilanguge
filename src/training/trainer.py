"""
Training loop for Modern LLM with latest optimizations
Includes: Mixed precision, gradient checkpointing, distributed training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm
import yaml
import os
from typing import Dict, Any, Optional
import logging

from ..models.modern_llm import ModernLLM, ModelConfig
from ..data.data_loader import create_dataloader
from ..utils.utils import AverageMeter, save_checkpoint, load_checkpoint


class ModernTrainer:
    """Modern training loop with latest optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['system']['device'] if config['system']['device'] != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.world_size = config['system'].get('num_gpus', 1)
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize distributed training if needed
        if self.world_size > 1:
            self._init_distributed()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup data loaders
        self.train_loader, self.eval_loader = self._create_dataloaders()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['use_mixed_precision'] else None
        
        # Initialize experiment tracking
        if self.rank == 0 and config.get('wandb', {}).get('project'):
            self._init_wandb()

    def _init_distributed(self):
        """Initialize distributed training"""
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_model(self) -> ModernLLM:
        """Create and setup the model"""
        # Create model config
        model_config = ModelConfig(**self.config['model'])
        model = ModernLLM(model_config)
        
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['training']['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        
        # Compile model for faster training (PyTorch 2.0)
        if self.config['system'].get('compile_model', False):
            model = torch.compile(model)
        
        # Wrap with DDP for distributed training
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model

    def _create_optimizer(self):
        """Create optimizer with modern settings"""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay'],
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            eps=1e-8,
        )
        
        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=self.config['training']['max_steps'],
        )

    def _create_dataloaders(self):
        """Create training and evaluation data loaders"""
        train_loader = create_dataloader(
            self.config['data'],
            split='train',
            batch_size=self.config['training']['batch_size'],
            world_size=self.world_size,
            rank=self.rank,
        )
        
        eval_loader = create_dataloader(
            self.config['evaluation'],
            split='validation',
            batch_size=self.config['evaluation']['eval_batch_size'],
            world_size=self.world_size,
            rank=self.rank,
        )
        
        return train_loader, eval_loader

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb'].get('entity'),
            name=self.config['wandb'].get('name'),
            config=self.config,
            tags=self.config['wandb'].get('tags', []),
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs['logits']
                
                # Compute loss (next token prediction)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs['logits']
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Normalize loss for gradient accumulation
        loss = loss / self.config['training']['gradient_accumulation_steps']
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {'loss': loss.item() * self.config['training']['gradient_accumulation_steps']}

    def update_parameters(self):
        """Update model parameters"""
        # Gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        
        total_loss = AverageMeter()
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", disable=self.rank != 0):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            with torch.cuda.amp.autocast() if self.scaler is not None else torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs['logits']
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            total_loss.update(loss.item(), input_ids.size(0))
        
        # Compute perplexity
        perplexity = torch.exp(torch.tensor(total_loss.avg))
        
        return {
            'eval_loss': total_loss.avg,
            'perplexity': perplexity.item(),
        }

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        accumulation_steps = 0
        
        while self.step < self.config['training']['max_steps']:
            for batch in self.train_loader:
                # Training step
                metrics = self.train_step(batch)
                accumulation_steps += 1
                
                # Update parameters after gradient accumulation
                if accumulation_steps >= self.config['training']['gradient_accumulation_steps']:
                    self.update_parameters()
                    self.step += 1
                    accumulation_steps = 0
                    
                    # Logging
                    if self.step % self.config['training']['logging_steps'] == 0 and self.rank == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        self.logger.info(
                            f"Step {self.step}: Loss = {metrics['loss']:.4f}, LR = {lr:.2e}"
                        )
                        
                        if hasattr(self, 'wandb') and wandb.run is not None:
                            wandb.log({
                                'train/loss': metrics['loss'],
                                'train/learning_rate': lr,
                                'train/step': self.step,
                            })
                    
                    # Evaluation
                    if self.step % self.config['training']['eval_steps'] == 0:
                        eval_metrics = self.evaluate()
                        
                        if self.rank == 0:
                            self.logger.info(
                                f"Step {self.step}: Eval Loss = {eval_metrics['eval_loss']:.4f}, "
                                f"Perplexity = {eval_metrics['perplexity']:.2f}"
                            )
                            
                            if hasattr(self, 'wandb') and wandb.run is not None:
                                wandb.log({
                                    'eval/loss': eval_metrics['eval_loss'],
                                    'eval/perplexity': eval_metrics['perplexity'],
                                    'train/step': self.step,
                                })
                    
                    # Save checkpoint
                    if self.step % self.config['training']['save_steps'] == 0 and self.rank == 0:
                        self.save_checkpoint()
                    
                    # Check if training is complete
                    if self.step >= self.config['training']['max_steps']:
                        break

    def save_checkpoint(self):
        """Save training checkpoint"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_checkpoint(checkpoint, f"checkpoint_step_{self.step}.pt")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = ModernTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
