#!/usr/bin/env python3
"""
Free Hosting Deployment Helper
Guides you through deploying your Trilingual AI system
"""

import os
import subprocess
import sys
from pathlib import Path

class DeploymentHelper:
    def __init__(self):
        self.project_dir = Path.cwd()
        
    def check_requirements(self):
        """Check if all required files exist"""
        required_files = {
            'multi_model_api.py': 'API server file',
            'streamlit_app.py': 'Streamlit interface',
            'api_requirements.txt': 'API dependencies',
            'requirements.txt': 'Streamlit dependencies',
            'railway.toml': 'Railway config',
            'render.yaml': 'Render config'
        }
        
        missing = []
        for file, desc in required_files.items():
            if not (self.project_dir / file).exists():
                missing.append(f"{file} ({desc})")
        
        if missing:
            print("❌ Missing required files:")
            for file in missing:
                print(f"   - {file}")
            return False
        
        print("✅ All required files present")
        return True
    
    def setup_git(self):
        """Initialize git repository if needed"""
        if not (self.project_dir / '.git').exists():
            print("📁 Initializing git repository...")
            subprocess.run(['git', 'init'], cwd=self.project_dir)
            subprocess.run(['git', 'add', '.'], cwd=self.project_dir)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.project_dir)
            print("✅ Git repository initialized")
        else:
            print("✅ Git repository already exists")
    
    def show_deployment_options(self):
        """Show available deployment options"""
        print("\n🚀 Free Hosting Options:")
        print("=" * 50)
        
        options = {
            "1": {
                "name": "Railway (API) + Streamlit Cloud (UI)",
                "description": "Best combo: Railway for API, Streamlit Cloud for UI",
                "steps": [
                    "1. Push code to GitHub",
                    "2. Deploy API on Railway: https://railway.app",
                    "3. Deploy UI on Streamlit Cloud: https://share.streamlit.io",
                    "4. Update API_URL in Streamlit app settings"
                ]
            },
            "2": {
                "name": "Render (Both API + UI)",
                "description": "Single platform for both services",
                "steps": [
                    "1. Push code to GitHub", 
                    "2. Create API service on Render: https://render.com",
                    "3. Create Streamlit service on Render",
                    "4. Link services via environment variables"
                ]
            },
            "3": {
                "name": "Hugging Face Spaces (UI only)",
                "description": "Quick demo deployment (UI only)",
                "steps": [
                    "1. Create space on: https://huggingface.co/spaces",
                    "2. Upload streamlit_app.py and requirements.txt",
                    "3. Note: API will use fallback responses only"
                ]
            }
        }
        
        for key, option in options.items():
            print(f"\n{key}. {option['name']}")
            print(f"   {option['description']}")
            for step in option['steps']:
                print(f"   {step}")
    
    def create_github_instructions(self):
        """Create GitHub setup instructions"""
        instructions = """
# GitHub Repository Setup

## Step 1: Create Repository
1. Go to https://github.com
2. Click "New repository"
3. Name: trilingual-ai-assistant
4. Make it public (required for free hosting)
5. Click "Create repository"

## Step 2: Push Your Code
```bash
# In your project directory
git remote add origin https://github.com/YOUR_USERNAME/trilingual-ai-assistant.git
git branch -M main
git push -u origin main
```

## Step 3: Verify Upload
Make sure these files are in your GitHub repo:
- multi_model_api.py
- streamlit_app.py
- api_requirements.txt
- requirements.txt
- railway.toml
- render.yaml

## Next: Choose hosting platform from the options above!
"""
        
        with open(self.project_dir / 'GITHUB_SETUP.md', 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        print("Created GITHUB_SETUP.md with detailed instructions")
    
    def run(self):
        """Main deployment helper"""
        print("Trilingual AI - Free Hosting Deployment Helper")
        print("=" * 55)
        
        # Check requirements
        if not self.check_requirements():
            print("\nPlease create missing files before deploying")
            return False
        
        # Setup git
        self.setup_git()
        
        # Create GitHub instructions
        self.create_github_instructions()
        
        # Show deployment options
        self.show_deployment_options()
        
        print(f"\nNext Steps:")
        print("1. Push your code to GitHub (see GITHUB_SETUP.md)")
        print("2. Choose a hosting option from above")
        print("3. Your AI assistant will be live!")
        
        print(f"\nTips:")
        print("- Railway: Best for API hosting (500 hours/month free)")
        print("- Streamlit Cloud: Perfect for UI (unlimited free)")
        print("- Render: Good all-in-one option")
        print("- Hugging Face: Quick demo deployment")
        
        return True

if __name__ == "__main__":
    helper = DeploymentHelper()
    helper.run()
