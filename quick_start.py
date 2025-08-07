"""
Quick start script for modern LLM development
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.modern_llm import ModernLLM, ModelConfig
from src.utils.utils import print_model_info, get_device
from transformers import AutoTokenizer


def main():
    print("🚀 Modern LLM Quick Start")
    print("=" * 40)
    
    # Set device
    device = get_device()
    
    # Create a small model for demonstration
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,   # Small for demo
        num_layers=8,      # Small for demo
        num_heads=8,
        num_kv_heads=4,    # Multi-Query Attention
        intermediate_size=2048,
        max_position_embeddings=2048,
    )
    
    print("\n📋 Model Configuration:")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Number of layers: {config.num_layers}")
    print(f"   Number of heads: {config.num_heads}")
    print(f"   KV heads: {config.num_kv_heads}")
    print(f"   Vocabulary size: {config.vocab_size}")
    
    # Create model
    print("\n🏗️ Creating modern LLM...")
    model = ModernLLM(config)
    model = model.to(device)
    
    # Print model info
    print_model_info(model)
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    
    # Test generation (simple greedy)
    print("\n🎯 Testing text generation...")
    model.eval()
    
    with torch.no_grad():
        # Start with a simple prompt
        prompt_ids = torch.randint(0, 1000, (1, 10)).to(device)  # Random prompt
        generated = prompt_ids.clone()
        
        # Generate 20 tokens
        for _ in range(20):
            outputs = model(generated)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
    
    print(f"✅ Generation successful!")
    print(f"   Generated sequence length: {generated.shape[1]}")
    print(f"   Token IDs: {generated[0][:10].tolist()}...")
    
    # Test with tokenizer (if available)
    try:
        print("\n📝 Testing with tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_text = "The future of artificial intelligence"
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        
        print(f"✅ Tokenizer test successful!")
        print(f"   Input text: '{test_text}'")
        print(f"   Tokenized length: {input_ids.shape[1]}")
        print(f"   Output logits shape: {logits.shape}")
        
    except Exception as e:
        print(f"⚠️ Tokenizer test failed: {e}")
        print("   This is normal if transformers library is not installed")
    
    print("\n🎉 Quick start completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Explore the Jupyter notebook for detailed examples")
    print("   2. Try training on your own data")
    print("   3. Experiment with different model configurations")
    print("   4. Implement additional optimizations")
    
    return model, config


if __name__ == "__main__":
    model, config = main()
