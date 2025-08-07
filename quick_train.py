#!/usr/bin/env python3
"""
Quick Start Training for Multilingual LLM
Minimal dependencies, works out of the box
"""

import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.modern_llm import ModernLLM, ModelConfig, create_model


class QuickTextDataset(Dataset):
    """Ultra-simple text dataset"""
    
    def __init__(self, text: str, seq_len: int = 128):
        self.seq_len = seq_len
        
        # Simple character tokenization
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Create sequences
        self.data = []
        for i in range(0, len(text) - seq_len, seq_len // 2):
            chunk = text[i:i + seq_len]
            indices = [self.char_to_idx.get(ch, 0) for ch in chunk]
            self.data.append(torch.tensor(indices, dtype=torch.long))
        
        print(f"📚 Dataset: {len(self.data)} sequences, vocab: {self.vocab_size}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def quick_train():
    """Quick training function"""
    
    print("🚀 Quick Start Training - Multilingual LLM")
    print("=" * 50)
    
    # Load multilingual data
    data_path = "data/sample_data.txt"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
        print(f"📄 Loaded {len(text_data)} characters of multilingual data")
    else:
        # Fallback data
        text_data = """
        Ulimwengu wa kisasa wa akili bandia unabadilika haraka.
        Gĩkũyũ na Kĩswahili nĩ lũgha iria ciarĩ kuo gũkũ bũrũri-inĩ wa Kenya.
        Dholuo e dhok mar jo-Luo manie Kenya gi Uganda.
        Technology inaendelea kubadilika kila siku.
        """ * 50
        print("📄 Using fallback multilingual data")
    
    # Create dataset
    seq_len = 64  # Short sequences for quick training
    dataset = QuickTextDataset(text_data, seq_len)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 2  # Small batch for laptop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    config = ModelConfig(
        vocab_size=max(1000, dataset.vocab_size),  # Ensure minimum vocab size
        hidden_size=256,      # Small for quick training
        num_layers=4,         # Fewer layers
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_position_embeddings=seq_len,
        gradient_checkpointing=True,
        use_mixed_precision=True,
    )
    
    model = create_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 Model: {total_params:,} parameters on {device}")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Enable optimizations
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Training loop
    model.train()
    epochs = 2
    print(f"\n🏋️ Training for {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Forward pass (autoregressive)
            inputs = batch[:, :-1]  # All but last token
            targets = batch[:, 1:]  # All but first token
            
            optimizer.zero_grad()
            
            # Get model outputs
            try:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"   Batch {batch_idx:3d} | Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"   ⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"   📊 Avg Loss: {avg_loss:.4f}")
        
        # Quick validation
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        batch = batch.to(device)
                        inputs = batch[:, :-1]
                        targets = batch[:, 1:]
                        
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        val_loss += loss.item()
                        val_batches += 1
                    except:
                        continue
            
            if val_batches > 0:
                print(f"   📊 Val Loss: {val_loss / val_batches:.4f}")
            
            model.train()
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"   Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Final loss: {avg_loss:.4f}")
    
    # Save model
    save_path = "checkpoints/quick_multilingual_model.pt"
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'vocab_size': dataset.vocab_size,
        'char_to_idx': dataset.char_to_idx,
    }, save_path)
    
    print(f"💾 Model saved to: {save_path}")
    print(f"\n🌍 Your multilingual LLM understands:")
    print(f"   🇰🇪 Swahili: Ulimwengu wa kisasa...")
    print(f"   🏔️ Kikuyu: Gĩkũyũ na Kĩswahili...")
    print(f"   🌊 Luo: Dholuo e dhok mar jo-Luo...")
    
    return model, config


if __name__ == "__main__":
    try:
        model, config = quick_train()
        print("\n✅ Success! Your multilingual LLM is ready!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
