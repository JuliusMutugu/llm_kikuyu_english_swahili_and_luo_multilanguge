#!/usr/bin/env python3
"""
Quick Training Script for Trilingual LLM
Fast training to get a working model for the API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTokenizer:
    """Simple tokenizer for quick training"""
    
    def __init__(self, texts):
        # Get all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Basic vocabulary
        self.chars = sorted(list(all_chars))
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab = self.special_tokens + self.chars
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = len(self.vocab)
        
        logger.info(f"Tokenizer created with {self.vocab_size} tokens")
    
    def encode(self, text, max_length=64):
        tokens = [self.bos_token_id]
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.unk_token_id))
        tokens.append(self.eos_token_id)
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        return tokens
    
    def decode(self, token_ids):
        chars = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            char = self.idx_to_char.get(token_id, '<UNK>')
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)

class QuickLLM(nn.Module):
    """Simple transformer for quick training"""
    
    def __init__(self, vocab_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Simple LSTM for faster training
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.lm_head(x)
        return type('ModelOutput', (), {'logits': logits})()

def load_training_data():
    """Load training data"""
    data_file = Path("data/enhanced_training_data.txt")
    
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} examples from enhanced dataset")
    else:
        # Quick fallback dataset
        texts = [
            "Hello, how are you? Habari yako? Wĩ atĩa? Inadi?",
            "I love you. Nakupenda. Nĩngũkwenda. Aheri.",
            "Good morning! Habari za asubuhi! Rũciinĩ rũega! Okinyi maber!",
            "Thank you very much. Asante sana. Nĩngũgũcookeria ngaatho. Erokamano ahinya.",
            "How is your family? Familia yako hali gani? Nyũmba yaku ĩrĩ atĩa? Joodu to nade?",
            "I am very happy today. Nifurahi sana leo. Ndĩ na gĩkeno kĩnene ũmũthĩ. Amor maduong' kawuono.",
            "Where are you going? Unaenda wapi? Ũrathii kũ? Idhi kanye?",
            "Welcome to our home. Karibu nyumbani kwetu. Wamũkĩire gũcoki mũciĩ witũ. Marima e dalawa."
        ]
        logger.info(f"Using fallback dataset with {len(texts)} examples")
    
    return texts

def quick_train():
    """Quick training function"""
    logger.info("Starting Quick Trilingual Training")
    
    # Load data
    texts = load_training_data()
    
    # Create tokenizer
    tokenizer = QuickTokenizer(texts)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QuickLLM(vocab_size=tokenizer.vocab_size, hidden_size=64, num_layers=2)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters on {device}")
    
    # Prepare training data
    encoded_texts = []
    for text in texts:
        encoded = tokenizer.encode(text, max_length=32)  # Short sequences
        encoded_texts.append(encoded)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    # Quick training loop
    logger.info("Starting training...")
    for epoch in range(10):  # Quick training
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(encoded_texts), 4):  # Batch size 4
            batch = encoded_texts[i:i+4]
            
            # Pad batch
            max_len = max(len(seq) for seq in batch)
            padded_batch = []
            for seq in batch:
                padded = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
                padded_batch.append(padded)
            
            # Convert to tensors
            input_ids = torch.tensor(padded_batch, dtype=torch.long).to(device)
            
            # Create targets (shifted by 1)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
    
    # Save model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'tokenizer': {
            'vocab': tokenizer.vocab,
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': tokenizer.idx_to_char,
            'vocab_size': tokenizer.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        },
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'hidden_size': 64,
            'num_layers': 2
        }
    }
    
    model_path = checkpoint_dir / "quick_trilingual_model.pt"
    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Test generation
    model.eval()
    test_prompt = "Habari"
    input_ids = torch.tensor(
        tokenizer.encode(test_prompt, max_length=16),
        dtype=torch.long
    ).unsqueeze(0).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(20):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    logger.info(f"Test generation: '{test_prompt}' -> '{generated_text}'")
    
    logger.info("Quick training completed successfully!")
    return True

def load_model_for_inference(model_path):
    """Load a trained model for inference"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {model_path} on {device}")
        
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Reconstruct tokenizer
        tokenizer_data = checkpoint['tokenizer']
        tokenizer = QuickTokenizer([])  # Empty init
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.char_to_idx = tokenizer_data['char_to_idx']
        tokenizer.idx_to_char = tokenizer_data['idx_to_char']
        tokenizer.vocab_size = tokenizer_data['vocab_size']
        tokenizer.pad_token_id = tokenizer_data['pad_token_id']
        tokenizer.unk_token_id = tokenizer_data['unk_token_id']
        tokenizer.bos_token_id = tokenizer_data['bos_token_id']
        tokenizer.eos_token_id = tokenizer_data['eos_token_id']
        
        # Reconstruct model
        model_config = checkpoint['model_config']
        model = QuickLLM(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info("✅ Original model loaded successfully")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device,
            'config': model_config
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


if __name__ == "__main__":
    try:
        quick_train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
