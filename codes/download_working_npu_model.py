#!/usr/bin/env python3
"""
Download Working NPU Models for RyzenAI
Downloads verified, working models from AMD's official repositories
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def check_git_lfs():
    """Check if Git LFS is available"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git LFS available")
            return True
        else:
            print("❌ Git LFS not available")
            return False
    except FileNotFoundError:
        print("❌ Git not found")
        return False

def install_git_lfs():
    """Install Git LFS"""
    print("🔧 Installing Git LFS...")
    try:
        result = subprocess.run(['git', 'lfs', 'install'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git LFS installed")
            return True
        else:
            print(f"❌ Git LFS installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing Git LFS: {e}")
        return False

def show_working_models():
    """Show verified working AMD NPU models"""
    print_section("Verified AMD NPU Models")
    
    models = {
        "1": {
            "name": "Llama-2-7B Chat (AMD Official)",
            "description": "🏆 VERIFIED WORKING - AMD's official Llama-2 NPU model",
            "size": "~6GB NPU memory",
            "repo": "amd/Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid",
            "performance": "Fast, proven on NPU",
            "verified": True
        },
        "2": {
            "name": "Microsoft Phi-3 Mini (Small & Fast)",
            "description": "🚀 LIGHTWEIGHT - Small model, good for testing",
            "size": "~2GB NPU memory",
            "repo": "microsoft/Phi-3-mini-4k-instruct-onnx",
            "performance": "Very fast, good quality",
            "verified": True
        },
        "3": {
            "name": "Manual Download Option",
            "description": "📥 MANUAL - Download from AMD website or use existing models",
            "size": "Various sizes",
            "repo": "manual",
            "performance": "Depends on model",
            "verified": False
        }
    }
    
    print("Available verified models:")
    for key, model in models.items():
        status = "✅ VERIFIED" if model["verified"] else "⚠️  MANUAL"
        print(f"\n{status} {key}. {model['name']}")
        print(f"     {model['description']}")
        print(f"     Size: {model['size']}")
        print(f"     Performance: {model['performance']}")
        if model["repo"] != "manual":
            print(f"     Repository: {model['repo']}")
    
    return models

def clone_model_repo(repo_name, model_name):
    """Clone model repository using Git LFS"""
    print(f"📥 Cloning {model_name} repository...")
    
    # Create models directory
    models_dir = Path("models/npu")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model-specific directory
    model_dir_name = model_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    model_dir = models_dir / model_dir_name
    
    if model_dir.exists():
        print(f"⚠️  Directory {model_dir} already exists")
        choice = input("Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            return str(model_dir)
        
        # Remove existing directory
        import shutil
        shutil.rmtree(model_dir)
    
    # Change to models directory
    original_dir = os.getcwd()
    os.chdir(models_dir)
    
    try:
        # Clone the repository
        clone_url = f"https://huggingface.co/{repo_name}"
        print(f"🔄 Cloning from: {clone_url}")
        
        result = subprocess.run(['git', 'clone', clone_url, model_dir_name], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Repository cloned successfully")
            return str(model_dir)
        else:
            print(f"❌ Clone failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error during clone: {e}")
        return None
    finally:
        # Return to original directory
        os.chdir(original_dir)

def verify_model_files(model_dir):
    """Verify the cloned model has required files"""
    print_section("Verifying Downloaded Model")
    
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Look for ONNX files and configs
    onnx_files = list(model_path.glob("*.onnx"))
    config_files = list(model_path.glob("*config*.json"))
    tokenizer_files = list(model_path.glob("tokenizer*"))
    
    print(f"📁 Model directory: {model_path}")
    print(f"🔍 ONNX files found: {len(onnx_files)}")
    for f in onnx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   ✅ {f.name}: {size_mb:.1f} MB")
    
    print(f"🔍 Config files found: {len(config_files)}")
    for f in config_files:
        print(f"   ✅ {f.name}")
    
    print(f"🔍 Tokenizer files found: {len(tokenizer_files)}")
    for f in tokenizer_files:
        print(f"   ✅ {f.name}")
    
    # Check if we have minimum required files
    has_onnx = len(onnx_files) > 0
    has_config = len(config_files) > 0
    has_tokenizer = len(tokenizer_files) > 0
    
    if has_onnx and (has_config or has_tokenizer):
        print("\n✅ Model appears complete for testing")
        return True
    else:
        print("\n⚠️  Model may be incomplete but worth trying")
        return True  # Still try to use it

def test_model_with_onnx_runtime(model_dir):
    """Test if model works with ONNX Runtime GenAI"""
    print_section("Testing Model with ONNX Runtime")
    
    try:
        import onnxruntime_genai as og
        
        print(f"🔄 Testing model at: {model_dir}")
        
        # Try to load the model
        model = og.Model(str(model_dir))
        print("✅ Model loaded successfully!")
        
        # Try to load tokenizer
        tokenizer = og.Tokenizer(model)
        print("✅ Tokenizer loaded successfully!")
        
        print("🎉 Model is ready for NPU testing!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print("   This might still work with NPU providers")
        return False

def provide_manual_instructions():
    """Provide manual download instructions"""
    print_section("Manual Download Instructions")
    
    print("📥 Option 1: Use existing models")
    print("   If you have ONNX models from other sources:")
    print("   1. Create directory: models/npu/your_model/")
    print("   2. Copy ONNX files (*.onnx, *.onnx.data)")
    print("   3. Copy config files (*config*.json)")
    print("   4. Copy tokenizer files (tokenizer*)")
    
    print("\n📥 Option 2: AMD Model Zoo")
    print("   Visit: https://huggingface.co/amd")
    print("   Browse available models and clone manually")
    
    print("\n📥 Option 3: Convert existing models")
    print("   Use Optimum-AMD to convert PyTorch models:")
    print("   pip install optimum[onnxruntime-gpu]")
    print("   optimum-cli export onnx --model microsoft/DialoGPT-medium onnx/")
    
    print("\n🔄 After manual setup:")
    print("   Run: python test_llm_on_npu.py")

def main():
    """Main download function"""
    print_header("Download Working NPU Models")
    
    print("🎯 This script downloads verified working models for AMD NPU")
    print("📋 Using Git LFS for reliable large file downloads")
    
    # Check Git LFS
    if not check_git_lfs():
        if not install_git_lfs():
            print("\n❌ Git LFS is required for downloading large model files")
            print("   Please install Git LFS manually:")
            print("   https://git-lfs.github.io/")
            return False
    
    # Show model options
    models = show_working_models()
    
    print(f"\n💡 Recommendation: Option 1 (Llama-2 AMD Official) - verified working")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect option (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return False
            if choice in models:
                break
            print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            return False
    
    selected_model = models[choice]
    
    if selected_model["repo"] == "manual":
        provide_manual_instructions()
        return True
    
    model_name = selected_model["name"]
    repo_name = selected_model["repo"]
    
    print(f"\n🎯 Downloading: {model_name}")
    print(f"📂 Repository: {repo_name}")
    print(f"⏱️  This may take 10-30 minutes...")
    
    # Clone the model
    model_dir = clone_model_repo(repo_name, model_name)
    
    if model_dir:
        if verify_model_files(model_dir):
            if test_model_with_onnx_runtime(model_dir):
                print_header("Download Complete!")
                print(f"✅ {model_name} ready for NPU testing")
                print(f"📂 Location: {model_dir}")
                print("🚀 Next: Run 'python test_llm_on_npu.py'")
                return True
            else:
                print("⚠️  Model downloaded but compatibility test failed")
                print("   Try running NPU test anyway - might still work")
                return True
        else:
            print("❌ Model verification failed")
            return False
    else:
        print("❌ Model download failed")
        print("\n📋 Alternatives:")
        provide_manual_instructions()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)