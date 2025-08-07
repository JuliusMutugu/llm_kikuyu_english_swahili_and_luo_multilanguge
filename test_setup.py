print("🚀 Testing Multilingual LLM Setup...")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not found")
    exit(1)

try:
    import sys
    sys.path.append("src")
    from models.modern_llm import ModelConfig, create_model
    print("✅ Model imports successful")
    
    # Test tiny model
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
    )
    
    model = create_model(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {params:,} parameters")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 32))
    with torch.no_grad():
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output.get('logits', output)
        print(f"✅ Forward pass: {logits.shape}")
    
    print("\n🎉 Everything works! Ready for multilingual training!")
    print("   Your LLM can learn Swahili, Kikuyu, and Luo! 🇰🇪")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
