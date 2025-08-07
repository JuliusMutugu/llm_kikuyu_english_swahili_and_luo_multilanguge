#!/usr/bin/env python3
"""
Quick Test and Training Script
Tests the environment and runs a short training session
"""

import sys
import os

def test_environment():
    """Test if environment is ready"""
    print("🔧 Testing environment...")
    
    # Test Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Test device
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name()
            print(f"🚀 GPU: {device}")
        else:
            print("💻 Using CPU")
        
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    return True

def load_data():
    """Load training data"""
    from pathlib import Path
    
    data_file = Path("data/enhanced_training_data.txt")
    
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"📚 Loaded {len(lines)} training examples")
        return lines[:50]  # Use first 50 for quick training
    else:
        print("❌ Training data not found")
        return None

def quick_training():
    """Run a quick training session"""
    print("\n🎯 Starting Quick Training...")
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    
    # Load data
    texts = load_data()
    if not texts:
        return False
    
    # Simple tokenizer
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    
    chars = sorted(list(all_chars))
    vocab_size = len(chars) + 4  # Add special tokens
    char_to_idx = {char: idx + 4 for idx, char in enumerate(chars)}
    char_to_idx.update({'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3})
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    print(f"📝 Vocabulary size: {vocab_size}")
    
    def encode_text(text, max_len=32):
        tokens = [2]  # BOS
        for char in text[:max_len-2]:
            tokens.append(char_to_idx.get(char, 1))
        tokens.append(3)  # EOS
        while len(tokens) < max_len:
            tokens.append(0)  # PAD
        return tokens
    
    def decode_tokens(tokens):
        chars = []
        for token in tokens:
            if token in [0, 2, 3]:  # Skip special tokens
                continue
            char = idx_to_char.get(token, '<UNK>')
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size, hidden_size=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.output(lstm_out)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare data
    encoded_texts = [encode_text(text) for text in texts]
    
    # Training loop
    print("🔄 Training...")
    model.train()
    
    for step in range(100):  # Quick 100 steps
        # Random batch
        batch_indices = np.random.choice(len(encoded_texts), min(4, len(encoded_texts)), replace=True)
        batch = torch.tensor([encoded_texts[i] for i in batch_indices], dtype=torch.long).to(device)
        
        # Forward pass
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1), ignore_index=0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n🎭 Testing generation...")
    model.eval()
    
    test_input = encode_text("Hello", max_len=10)
    input_tensor = torch.tensor([test_input], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = input_tensor.clone()
        for _ in range(20):
            outputs = model(generated)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 3:  # EOS
                break
    
    result = decode_tokens(generated[0].cpu().tolist())
    print(f"Generated: '{result}'")
    
    # Save model
    from pathlib import Path
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }, 'checkpoints/quick_model.pt')
    
    print("💾 Model saved to checkpoints/quick_model.pt")
    return True

def main():
    """Main function"""
    print("🧪 Quick Test and Training")
    print("=" * 40)
    
    # Test environment
    if not test_environment():
        print("❌ Environment test failed")
        return False
    
    print("✅ Environment ready")
    
    # Run quick training
    try:
        success = quick_training()
        if success:
            print("\n✅ Quick training completed successfully!")
            print("📁 Model saved in 'checkpoints/' directory")
            print("🚀 You can now run the full training script")
        else:
            print("\n❌ Quick training failed")
        return success
    except Exception as e:
        print(f"\n💥 Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
