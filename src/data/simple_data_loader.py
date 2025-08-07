"""
Simple data loader for multilingual text training
No external dependencies except PyTorch
"""

import os
import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleTextDataset(Dataset):
    """Simple text dataset for character/word level training"""
    
    def __init__(
        self,
        text_data: str,
        sequence_length: int = 256,
        vocab_size: int = 8000,
        overlap: int = 64
    ):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.overlap = overlap
        
        # Simple character-level tokenization
        self.chars = sorted(list(set(text_data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Pad vocab to desired size
        while len(self.chars) < vocab_size:
            self.chars.append(f"<unk_{len(self.chars)}>")
            self.char_to_idx[self.chars[-1]] = len(self.chars) - 1
            self.idx_to_char[len(self.chars) - 1] = self.chars[-1]
        
        # Create sequences with overlap
        self.sequences = []
        step = sequence_length - overlap
        
        for i in range(0, len(text_data) - sequence_length, step):
            sequence = text_data[i:i + sequence_length]
            # Convert to indices
            indices = [self.char_to_idx.get(ch, 0) for ch in sequence]
            self.sequences.append(torch.tensor(indices, dtype=torch.long))
        
        print(f"📚 Dataset created:")
        print(f"   Characters: {len(self.char_to_idx)}")
        print(f"   Sequences: {len(self.sequences)}")
        print(f"   Sequence length: {sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def load_text_file(file_path: str) -> str:
    """Load text from file with encoding handling"""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # Fallback: read as bytes and decode with error handling
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')


def create_simple_data_loaders(
    data_path: str,
    batch_size: int = 4,
    sequence_length: int = 256,
    vocab_size: int = 8000,
    split_ratio: float = 0.9,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    print(f"📂 Loading data from: {data_path}")
    
    # Load text data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    text_data = load_text_file(data_path)
    print(f"📄 Loaded {len(text_data)} characters")
    
    # Split data
    split_idx = int(len(text_data) * split_ratio)
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]
    
    # Create datasets
    train_dataset = SimpleTextDataset(
        train_text, 
        sequence_length=sequence_length,
        vocab_size=vocab_size
    )
    
    val_dataset = SimpleTextDataset(
        val_text,
        sequence_length=sequence_length, 
        vocab_size=vocab_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def preview_data(data_loader: DataLoader, dataset: SimpleTextDataset, num_samples: int = 3):
    """Preview some data samples"""
    print(f"\n🔍 Data Preview:")
    
    # Get a batch
    batch = next(iter(data_loader))
    
    for i in range(min(num_samples, len(batch))):
        sequence = batch[i]
        # Convert back to text for preview
        text = ''.join([dataset.idx_to_char[idx.item()] for idx in sequence[:50]])
        print(f"   Sample {i+1}: '{text}...'")
        print(f"   Shape: {sequence.shape}")


if __name__ == "__main__":
    # Test the data loader
    data_path = "../../data/sample_data.txt"
    
    try:
        train_loader, val_loader = create_simple_data_loaders(
            data_path=data_path,
            batch_size=2,
            sequence_length=128,
            vocab_size=1000
        )
        
        print(f"✅ Data loaders created successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Preview data
        preview_data(train_loader, train_loader.dataset)
        
    except Exception as e:
        print(f"❌ Error: {e}")
