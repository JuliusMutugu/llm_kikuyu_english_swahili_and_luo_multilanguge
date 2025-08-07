#!/usr/bin/env python3
"""
Setup and Configuration Script for Enhanced Trilingual LLM
Installs dependencies and prepares the environment
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU Available: {gpu_name}")
            print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("💻 No GPU available, will use CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models',
        'checkpoints',
        'logs',
        'static',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_data():
    """Setup training data"""
    data_dir = Path('data')
    
    # Copy enhanced training data if it exists
    enhanced_data_source = Path('enhanced_training_data.txt')
    enhanced_data_target = data_dir / 'enhanced_training_data.txt'
    
    if enhanced_data_source.exists() and not enhanced_data_target.exists():
        import shutil
        shutil.copy2(enhanced_data_source, enhanced_data_target)
        print(f"📋 Copied enhanced training data to {enhanced_data_target}")
    
    # Create sample data if no data exists
    if not enhanced_data_target.exists():
        sample_data = [
            "Hello, how are you? Habari yako? Wĩ atĩa? Inadi?",
            "I love you very much. Nakupenda sana. Nĩngũkwenda mũno. Aheri ahinya.",
            "Good morning, my friend! Habari za asubuhi, rafiki! Rũciinĩ rũega, mũrata! Okinyi maber, osiepna!",
            "Thank you for your help. Asante kwa msaada wako. Nĩngũgũcookeria ngaatho nĩ ũrĩa wandeithirie. Erokamano kuom konyruok.",
            "How is your family? Familia yako hali gani? Nyũmba yaku ĩrĩ atĩa? Joodu to nade?",
            "I am very happy today. Nifurahi sana leo. Ndĩ na gĩkeno kĩnene ũmũthĩ. Amor maduong' kawuono.",
            "Where are you going? Unaenda wapi? Ũrathii kũ? Idhi kanye?",
            "Welcome to our home. Karibu nyumbani kwetu. Wamũkĩire gũcoki mũciĩ witũ. Marima e dalawa."
        ]
        
        with open(enhanced_data_target, 'w', encoding='utf-8') as f:
            for line in sample_data:
                f.write(line + '\n')
        
        print(f"📝 Created sample training data with {len(sample_data)} examples")

def create_config():
    """Create configuration file"""
    config = {
        "model": {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 8,
            "num_kv_heads": 4,
            "intermediate_size": 1024,
            "max_length": 256,
            "dropout": 0.1,
            "use_rotary_emb": True,
            "tie_word_embeddings": True
        },
        "training": {
            "learning_rate": 3e-4,
            "batch_size": 4,
            "num_epochs": 5,
            "max_steps": 5000,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_steps": 250,
            "eval_steps": 250,
            "logging_steps": 50
        },
        "data": {
            "max_length": 256,
            "train_split": 0.8,
            "min_char_freq": 1
        },
        "generation": {
            "max_length": 100,
            "temperature": 0.8,
            "top_k": 30,
            "top_p": 0.9
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("⚙️ Created configuration file: config.json")

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'torch',
        'numpy',
        'fastapi',
        'uvicorn',
        'jinja2',
        'python-multipart'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return missing_packages

def main():
    """Main setup function"""
    print("🔧 Enhanced Trilingual LLM Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Check for missing packages
    print("\n📦 Checking requirements...")
    missing_packages = check_requirements()
    
    if missing_packages:
        print(f"\n🔄 Installing {len(missing_packages)} missing packages...")
        
        # Install PyTorch first (with appropriate version)
        if 'torch' in missing_packages:
            print("🔥 Installing PyTorch...")
            torch_success = install_package("torch")
            if torch_success:
                missing_packages.remove('torch')
        
        # Install other packages
        for package in missing_packages:
            install_package(package)
    
    # Check GPU after PyTorch installation
    print("\n🚀 Checking GPU availability...")
    check_gpu()
    
    # Setup data
    print("\n📚 Setting up training data...")
    setup_data()
    
    # Create config
    print("\n⚙️ Creating configuration...")
    create_config()
    
    print("\n✅ Setup completed!")
    print("\n🚀 Next steps:")
    print("1. Run training: python comprehensive_training.py")
    print("2. Start server: python app.py")
    print("3. Open browser: http://localhost:8000")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
