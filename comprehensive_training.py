#!/usr/bin/env python3
"""
Comprehensive Training Script for Enhanced Trilingual LLM
Improved model architecture, enhanced dataset, and better training techniques
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Add models directory to path
sys.path.append(str(Path(__file__).parent))

# Import our enhanced model
try:
    from models.simple_llm import SimpleLLM, SimpleConfig, create_model_and_tokenizer
except ImportError:
    print("❌ Could not import enhanced model. Creating fallback...")
    class SimpleLLM(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(config.hidden_size, 8, batch_first=True),
                num_layers=6
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            x = self.transformer(x, x)
            return type('', (), {'last_hidden_state': self.lm_head(x)})()

class EnhancedTokenizer:
    """Enhanced character-level tokenizer with better handling of multilingual text"""
    
    def __init__(self, texts: List[str], min_freq: int = 2):
        self.min_freq = min_freq
        self._build_vocab(texts)
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts with frequency filtering"""
        # Count character frequencies
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Filter by minimum frequency
        frequent_chars = [char for char, freq in char_freq.items() if freq >= self.min_freq]
        
        # Sort for consistent ordering
        frequent_chars = sorted(frequent_chars)
        
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
        
        # Build vocabulary
        self.vocab = special_tokens + frequent_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        
        self.vocab_size = len(self.vocab)
        
        print(f"📝 Vocabulary built with {self.vocab_size} tokens")
        print(f"🔢 Special tokens: {special_tokens}")
        print(f"📊 Character frequency threshold: {self.min_freq}")
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        # Add BOS token
        tokens = [self.bos_token_id]
        
        # Encode characters
        for char in text:
            token_id = self.char_to_idx.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        # Add EOS token
        tokens.append(self.eos_token_id)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        chars = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            char = self.idx_to_char.get(token_id, '<UNK>')
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)
    
    def batch_encode(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode a batch of texts"""
        encoded = []
        attention_masks = []
        
        for text in texts:
            tokens = self.encode(text, max_length)
            
            # Pad to max_length
            attention_mask = [1] * len(tokens)
            while len(tokens) < max_length:
                tokens.append(self.pad_token_id)
                attention_mask.append(0)
            
            encoded.append(tokens)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }

class EnhancedTextDataset(Dataset):
    """Enhanced dataset with better data processing and augmentation"""
    
    def __init__(self, texts: List[str], tokenizer: EnhancedTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-encode all texts for faster training
        print("🔄 Pre-encoding dataset...")
        self.encoded_texts = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Encoded {i}/{len(texts)} texts")
            
            tokens = self.tokenizer.encode(text, max_length)
            
            # Pad to max_length
            attention_mask = [1] * len(tokens)
            while len(tokens) < max_length:
                tokens.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)
            
            self.encoded_texts.append({
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            })
        
        print(f"✅ Pre-encoded {len(self.encoded_texts)} examples")
    
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        item = self.encoded_texts[idx]
        
        # For language modeling: input and target are shifted by 1
        input_ids = item['input_ids'][:-1]
        labels = item['input_ids'][1:]
        attention_mask = item['attention_mask'][:-1]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

class ComprehensiveTrainer:
    """Comprehensive trainer with state-of-the-art training techniques"""
    
    def __init__(self, model, tokenizer, device='auto'):
        # Device selection
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = 'cpu'
                print("💻 Using CPU")
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        # Mixed precision training
        self.use_amp = self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training hyperparameters (optimized for performance)
        self.learning_rate = 3e-4  # Increased for faster learning
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8
        self.max_grad_norm = 1.0
        
        # Learning rate scheduling
        self.warmup_steps = 500    # Reduced warmup
        self.total_steps = 5000    # More training steps
        
        # Optimizer with better settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
            div_factor=10,    # Start with lower lr
            final_div_factor=100  # End with much lower lr
        )
        
        # Logging setup
        self.setup_logging()
        
        # Training metrics
        self.step = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.eval_losses = []
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"Model Memory: ~{total_params * 4 / 1024**2:.1f} MB")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.use_amp}")
    
    def compute_loss(self, batch):
        """Compute loss with better handling"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        if self.use_amp:
            with autocast():
                outputs = self.model(input_ids)
                logits = outputs.last_hidden_state
                
                # Compute cross-entropy loss with label smoothing
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    label_smoothing=0.1
                )
        else:
            outputs = self.model(input_ids)
            logits = outputs.last_hidden_state
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
                label_smoothing=0.1
            )
        
        return loss, logits
    
    def train_step(self, batch):
        """Enhanced training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        loss, logits = self.compute_loss(batch)
        
        # Backward pass with proper gradient handling
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        self.step += 1
        return loss.item()
    
    def evaluate(self, eval_dataloader):
        """Comprehensive evaluation"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                loss, _ = self.compute_loss(batch)
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def generate_sample(self, prompt: str = "Habari", max_length: int = 100, temperature: float = 0.8):
        """Generate a sample to check model progress"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, max_length=50), 
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) >= 400:  # Prevent too long sequences
                    break
                
                outputs = self.model(generated)
                logits = outputs.last_hidden_state[:, -1, :] / temperature
                
                # Top-k sampling for better quality
                top_k = 30
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                else:
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        return generated_text
    
    def save_checkpoint(self, eval_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'tokenizer': {
                'vocab': self.tokenizer.vocab,
                'char_to_idx': self.tokenizer.char_to_idx,
                'idx_to_char': self.tokenizer.idx_to_char,
                'vocab_size': self.tokenizer.vocab_size,
                'special_tokens': {
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'unk_token_id': self.tokenizer.unk_token_id,
                    'bos_token_id': self.tokenizer.bos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'mask_token_id': self.tokenizer.mask_token_id
                }
            },
            'eval_loss': eval_loss,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{self.step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 New best model saved! Loss: {eval_loss:.4f}")
        
        self.logger.info(f"📁 Checkpoint saved: {checkpoint_path}")
    
    def train(self, train_dataloader, eval_dataloader, num_epochs: int = 3):
        """Main training loop with comprehensive logging"""
        self.logger.info("🚀 Starting Enhanced Training")
        self.logger.info(f"📊 Training samples: {len(train_dataloader.dataset)}")
        self.logger.info(f"📊 Evaluation samples: {len(eval_dataloader.dataset)}")
        self.logger.info(f"🎯 Target steps: {self.total_steps}")
        self.logger.info(f"🔄 Epochs: {num_epochs}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.logger.info(f"\n🌟 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_losses = []
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Training step
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Logging
                if self.step % 50 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    avg_loss = np.mean(epoch_losses[-50:])  # Last 50 steps
                    
                    self.logger.info(
                        f"Step {self.step:5d} | "
                        f"Loss: {loss:.4f} | "
                        f"Avg: {avg_loss:.4f} | "
                        f"LR: {current_lr:.6f}"
                    )
                
                # Evaluation and checkpointing
                if self.step % 250 == 0:
                    eval_loss, perplexity = self.evaluate(eval_dataloader)
                    self.eval_losses.append(eval_loss)
                    
                    self.logger.info(f"📊 Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
                    
                    # Save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(eval_loss, is_best=True)
                    
                    # Generate sample
                    sample = self.generate_sample("Habari", max_length=50)
                    self.logger.info(f"🎭 Sample: {sample}")
                
                # Stop if we've reached target steps
                if self.step >= self.total_steps:
                    break
            
            # End of epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            epoch_time = time.time() - epoch_start
            
            self.logger.info(f"✅ Epoch {epoch + 1} completed in {epoch_time:.1f}s")
            self.logger.info(f"📈 Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            eval_loss, perplexity = self.evaluate(eval_dataloader)
            self.save_checkpoint(eval_loss, is_best=False)
            
            if self.step >= self.total_steps:
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"\n🎉 Training completed in {total_time:.1f}s")
        self.logger.info(f"🏆 Best evaluation loss: {self.best_loss:.4f}")

def load_training_data():
    """Load and prepare training data"""
    print("📚 Loading training data...")
    
    # Try to load enhanced dataset first
    enhanced_data_path = Path("data/enhanced_training_data.txt")
    if enhanced_data_path.exists():
        with open(enhanced_data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(texts)} examples from enhanced dataset")
    else:
        # Fallback to original data
        original_data_path = Path("data/sample_data.txt")
        if original_data_path.exists():
            with open(original_data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(texts)} examples from original dataset")
        else:
            # Create minimal dataset
            texts = [
                "Hello, how are you? Habari yako? Wĩ atĩa? Inadi?",
                "I love you. Nakupenda. Nĩngũkwenda. Aheri.",
                "Good morning! Habari za asubuhi! Rũciinĩ rũega! Okinyi maber!",
                "Thank you very much. Asante sana. Nĩngũgũcookeria ngaatho. Erokamano ahinya."
            ]
            print(f"⚠️ Using minimal dataset with {len(texts)} examples")
    
    return texts

def main():
    """Main training function"""
    print("🚀 Enhanced Trilingual LLM Training")
    print("=" * 60)
    
    # Load data
    texts = load_training_data()
    
    # Create tokenizer
    print("🔧 Building tokenizer...")
    tokenizer = EnhancedTokenizer(texts, min_freq=1)  # Lower freq threshold for small dataset
    
    # Create dataset
    print("📦 Creating dataset...")
    max_length = 256  # Smaller for faster training
    dataset = EnhancedTextDataset(texts, tokenizer, max_length)
    
    # Split dataset
    train_size = max(1, int(0.8 * len(dataset)))  # Ensure at least 1 sample
    eval_size = len(dataset) - train_size
    
    if eval_size == 0:
        eval_size = 1
        train_size = len(dataset) - 1
    
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Eval samples: {len(eval_dataset)}")
    
    # Create data loaders
    batch_size = 4  # Small batch size for memory efficiency
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("🧠 Creating model...")
    try:
        from models.simple_llm import SimpleConfig, SimpleLLM
        config = SimpleConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,  # Smaller for faster training
            num_layers=6,     # Fewer layers
            num_heads=8,
            num_kv_heads=4,
            intermediate_size=1024,
            max_length=max_length,
            dropout=0.1,
            use_rotary_emb=True,
            tie_word_embeddings=True
        )
        model = SimpleLLM(config)
    except ImportError:
        # Fallback model
        class SimpleConfig:
            vocab_size = tokenizer.vocab_size
            hidden_size = 256
            num_layers = 4
        
        config = SimpleConfig()
        model = SimpleLLM(config)
    
    print(f"🎯 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ComprehensiveTrainer(model, tokenizer)
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train(train_dataloader, eval_dataloader, num_epochs=5)
    
    print("\n🎉 Training completed!")
    print("📁 Models saved in 'checkpoints/' directory")
    print("📋 Logs saved in 'logs/' directory")
    
    # Test generation
    print("\n🎭 Testing final model:")
    final_sample = trainer.generate_sample("Habari yako", max_length=80)
    print(f"Generated: {final_sample}")

if __name__ == "__main__":
    main()
