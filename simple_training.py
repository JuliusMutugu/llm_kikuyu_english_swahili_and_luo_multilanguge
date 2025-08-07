#!/usr/bin/env python3
"""
Simple Training Script for Enhanced Trilingual LLM
Streamlined version that works reliably on Windows
"""

import os
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    print("✅ PyTorch and dependencies loaded successfully")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install with: pip install torch numpy")
    sys.exit(1)

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, texts):
        # Get all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Sort for consistent ordering
        self.chars = sorted(list(all_chars))
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab = special_tokens + self.chars
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        self.vocab_size = len(self.vocab)
        print(f"📝 Tokenizer created with {self.vocab_size} tokens")
    
    def encode(self, text, max_length=128):
        """Encode text to token IDs"""
        tokens = [self.bos_token_id]
        
        for char in text:
            token_id = self.char_to_idx.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        tokens.append(self.eos_token_id)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        # Pad if too short
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        chars = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            char = self.idx_to_char.get(token_id, '<UNK>')
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)

class SimpleLLM(nn.Module):
    """Simple but effective language model"""
    
    def __init__(self, vocab_size, hidden_size=256, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights properly"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        # Create attention mask for padding
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Convert to boolean and invert (transformer expects True for positions to ignore)
        src_key_padding_mask = ~attention_mask.bool()
        
        # Embeddings
        x = self.embedding(input_ids)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        logits = self.lm_head(x)
        
        return type('ModelOutput', (), {'logits': logits})()

class SimpleTrainer:
    """Simple but effective trainer"""
    
    def __init__(self, model, tokenizer, device='auto'):
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🖥️ Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # Higher learning rate for faster training
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )
        
        self.step = 0
        self.losses = []
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Move to device
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long).to(self.device)
        
        # Create labels (shifted by 1)
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs.logits
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        return loss.item()
    
    def generate(self, prompt, max_length=50):
        """Generate text"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, max_length=20),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) >= 100:  # Prevent infinite generation
                    break
                
                # Forward pass
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]
                
                # Sample next token
                probs = F.softmax(logits / 0.8, dim=-1)  # Temperature sampling
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        return self.tokenizer.decode(generated[0].cpu().tolist())
    
    def save_model(self, path):
        """Save model and tokenizer"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_vocab': self.tokenizer.char_to_idx,
            'tokenizer_chars': self.tokenizer.chars,
            'step': self.step,
            'losses': self.losses
        }
        torch.save(checkpoint, path)
        print(f"💾 Model saved to {path}")

def load_data():
    """Load training data"""
    data_file = Path("data/enhanced_training_data.txt")
    
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"📚 Loaded {len(texts)} examples from enhanced dataset")
    else:
        # Fallback data
        texts = [
            "Hello, how are you? Habari yako? Wĩ atĩa? Inadi?",
            "I love you very much. Nakupenda sana. Nĩngũkwenda mũno. Aheri ahinya.",
            "Good morning, my friend! Habari za asubuhi, rafiki! Rũciinĩ rũega, mũrata! Okinyi maber, osiepna!",
            "Thank you for your help. Asante kwa msaada wako. Nĩngũgũcookeria ngaatho nĩ ũrĩa wandeithirie. Erokamano kuom konyruok.",
            "How is your family? Familia yako hali gani? Nyũmba yaku ĩrĩ atĩa? Joodu to nade?",
            "I am very happy today. Nifurahi sana leo. Ndĩ na gĩkeno kĩnene ũmũthĩ. Amor maduong' kawuono.",
            "Where are you going? Unaenda wapi? Ũrathii kũ? Idhi kanye?",
            "Welcome to our home. Karibu nyumbani kwetu. Wamũkĩire gũcoki mũciĩ witũ. Marima e dalawa.",
            "The weather is beautiful today. Hali ya anga ni nzuri leo. Kĩrĩa kĩrĩ rĩega ũmũthĩ. Yore ber kawuono.",
            "I want to learn more languages. Nataka kujifunza lugha zaidi. Nĩndĩrenda gũthoma ciũgano ingĩ. Adwaro puonjora dhok moko."
        ]
        print(f"⚠️ Using fallback dataset with {len(texts)} examples")
    
    return texts

def main():
    """Main training function"""
    print("🚀 Simple Trilingual LLM Training")
    print("=" * 50)
    
    # Load data
    texts = load_data()
    
    # Create tokenizer
    print("🔧 Creating tokenizer...")
    tokenizer = SimpleTokenizer(texts)
    
    # Create model
    print("🧠 Creating model...")
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,  # Smaller for faster training
        num_layers=3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = SimpleTrainer(model, tokenizer)
    
    # Prepare data
    print("📦 Preparing training data...")
    encoded_texts = []
    for text in texts:
        encoded = tokenizer.encode(text, max_length=64)  # Shorter sequences
        encoded_texts.append(encoded)
    
    # Training loop
    print("\n🎯 Starting training...")
    batch_size = 2  # Small batch size
    num_steps = 500  # Fewer steps for demo
    
    for step in range(num_steps):
        # Create random batch
        batch_indices = np.random.choice(len(encoded_texts), batch_size, replace=True)
        batch = {
            'input_ids': [encoded_texts[i] for i in batch_indices]
        }
        
        # Training step
        loss = trainer.train_step(batch)
        trainer.losses.append(loss)
        
        # Logging
        if step % 50 == 0:
            lr = trainer.scheduler.get_last_lr()[0]
            print(f"Step {step:3d} | Loss: {loss:.4f} | LR: {lr:.6f}")
            
            # Generate sample
            if step % 100 == 0:
                sample = trainer.generate("Habari", max_length=30)
                print(f"Sample: {sample}")
        
        # Save checkpoint
        if step % 200 == 0 and step > 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            trainer.save_model(f"checkpoints/model_step_{step}.pt")
    
    # Final save
    print("\n💾 Saving final model...")
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    trainer.save_model("checkpoints/final_model.pt")
    
    # Final test
    print("\n🎭 Final generation test:")
    test_prompts = ["Habari", "Hello", "Wĩ atĩa"]
    for prompt in test_prompts:
        result = trainer.generate(prompt, max_length=40)
        print(f"'{prompt}' -> '{result}'")
    
    # Training summary
    if trainer.losses:
        avg_loss = np.mean(trainer.losses[-50:])  # Last 50 steps
        print(f"\n📊 Training completed!")
        print(f"📈 Final average loss: {avg_loss:.4f}")
        print(f"📁 Model saved in 'checkpoints/' directory")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Training completed successfully!")
        else:
            print("\n❌ Training failed!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
