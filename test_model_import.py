#!/usr/bin/env python3
"""
Test the model imports
"""

try:
    from quick_train_api import QuickLLM, load_model_for_inference
    print("✅ Import successful - Original model components available")
    
    # Test if model file exists
    import os
    model_path = "checkpoints/quick_trilingual_model.pt"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        
        # Try to load the model
        model_components = load_model_for_inference(model_path)
        if model_components:
            print("✅ Original model loaded successfully")
        else:
            print("❌ Failed to load original model")
    else:
        print(f"❌ Model file not found: {model_path}")
        
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
