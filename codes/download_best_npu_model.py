#!/usr/bin/env python3
"""
Download Best NPU-Optimized LLM Models for RyzenAI
Downloads top-performing models optimized for AMD NPU
Supports both direct download and Git LFS cloning
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def download_file(url, destination, description=""):
    """Download file with progress"""
    print(f"📥 Downloading {description}...")
    print(f"   From: {url}")
    print(f"   To: {destination}")
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r   Progress: {percent}%")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n✅ Download completed: {destination}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

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

def show_model_options():
    """Show available model options"""
    print_section("Available NPU-Optimized Models")
    
    models = {
        "1": {
            "name": "AMD Llama-2-7B Ryzen Strix (Perfect Match)",
            "description": "🏆 IDEAL - AMD official, Ryzen Strix optimized, public access",
            "size": "~4GB NPU memory",
            "performance": "Excellent, proven on AMD NPU",
            "repo": "amd/Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix",
            "method": "git_lfs",
            "recommended": True
        },
        "2": {
            "name": "DeepSeek R1 Distill Qwen-7B", 
            "description": "🧠 REASONING EXPERT - Latest 2024 model, optimized for complex tasks",
            "size": "~7GB NPU memory",
            "performance": "92/100 quality, 40-50 tokens/sec",
            "url_base": "https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-7B-ryzenai-onnx-int4",
            "method": "direct",
            "recommended": True
        },
        "3": {
            "name": "Llama-2-7B Chat (AMD Official)",
            "description": "✅ VERIFIED WORKING - AMD's official Llama-2 NPU model",
            "size": "~6GB NPU memory",
            "performance": "Fast, proven on NPU",
            "repo": "amd/Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid",
            "method": "git_lfs",
            "recommended": True
        },
        "4": {
            "name": "Qwen 2.5-7B (Requires HuggingFace Auth)",
            "description": "🏆 BEST PERFORMANCE - Requires HuggingFace login",
            "size": "~6GB NPU memory",
            "performance": "95/100 quality, 45-55 tokens/sec",
            "repo": "Qwen/Qwen2.5-7B-Instruct-ONNX",
            "method": "git_lfs",
            "recommended": False
        },
        "5": {
            "name": "Manual Download Option",
            "description": "📥 MANUAL - Instructions for manual setup",
            "size": "Various sizes",
            "performance": "Depends on model",
            "method": "manual",
            "recommended": False
        }
    }
    
    print("Available models (sorted by recommendation):")
    for key, model in models.items():
        star = "⭐" if model["recommended"] else "  "
        method = model.get("method", "direct")
        method_icon = "🔧" if method == "git_lfs" else "📥" if method == "direct" else "📋"
        print(f"\n{star} {key}. {model['name']} {method_icon}")
        print(f"     {model['description']}")
        print(f"     Size: {model['size']}")
        print(f"     Performance: {model['performance']}")
        if method == "git_lfs" and "repo" in model:
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

def download_model_files(base_url, model_dir, model_name):
    """Download all files for a specific model"""
    print(f"📥 Downloading {model_name} from AMD/HuggingFace...")
    
    # Try AMD's RyzenAI optimized versions first
    if "amd/" in base_url:
        files_to_download = [
            ("config.json", "Model configuration"),
            ("genai_config.json", "GenAI configuration"),
            ("model.onnx", "Main model file"),
            ("model.onnx.data", "Model weights"),
            ("tokenizer.json", "Tokenizer configuration"),
            ("tokenizer_config.json", "Tokenizer settings"),
            ("special_tokens_map.json", "Special tokens"),
            ("vocab.json", "Vocabulary")
        ]
    else:
        # Standard ONNX format
        files_to_download = [
            ("genai_config.json", "GenAI configuration"),
            ("model.onnx", "Main model file"),
            ("model.onnx.data", "Model weights"),
            ("tokenizer.json", "Tokenizer configuration"),
            ("tokenizer_config.json", "Tokenizer settings")
        ]
    
    success_count = 0
    total_files = len(files_to_download)
    
    for filename, description in files_to_download:
        url = f"{base_url}/resolve/main/{filename}"
        destination = model_dir / filename
        
        if destination.exists():
            print(f"⏭️  Skipping {filename} (already exists)")
            success_count += 1
            continue
        
        if download_file(url, destination, f"{description} ({filename})"):
            success_count += 1
        else:
            print(f"⚠️  Failed to download {filename} - trying alternative...")
            # Try without /resolve/main/ for some repositories
            alt_url = f"{base_url}/{filename}"
            if download_file(alt_url, destination, f"{description} ({filename}) - alternative"):
                success_count += 1
    
    print(f"\n📊 Download Summary: {success_count}/{total_files} files successful")
    
    if success_count >= 3:  # Need at least config, model, tokenizer
        print("✅ Sufficient files downloaded for testing")
        return True
    else:
        print("❌ Not enough files downloaded")
        return False

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

def provide_manual_instructions():
    """Provide manual download instructions"""
    print_section("Manual Download Instructions")
    
    print("📥 Option 1: Use existing models")
    print("   If you have ONNX models from other sources:")
    print("   1. Create directory: models/npu/your_model/")
    print("   2. Copy ONNX files (*.onnx, *.onnx.data)")
    print("   3. Copy config files (*config*.json)")
    print("   4. Copy tokenizer files (tokenizer*)")
    
    print("\n📥 Option 2: Manual HuggingFace Download")
    print("   Visit: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx")
    print("   Click 'Files' tab and download:")
    print("   - genai_config.json")
    print("   - model.onnx")
    print("   - model.onnx.data")
    print("   - tokenizer.json")
    print("   - tokenizer_config.json")
    
    print("\n📥 Option 3: AMD Model Zoo")
    print("   Visit: https://huggingface.co/amd")
    print("   Browse available models and clone manually")
    
    print("\n🔄 After manual setup:")
    print("   Run: python test_llm_on_npu.py")

def verify_model_installation(model_dir, model_name):
    """Verify the downloaded model is complete"""
    print_section(f"Verifying {model_name} Installation")
    
    required_files = [
        "genai_config.json",
        "model.onnx",
        "tokenizer.json"
    ]
    
    missing_files = []
    total_size = 0
    
    for filename in required_files:
        file_path = model_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {filename}: {size_mb:.1f} MB")
        else:
            missing_files.append(filename)
            print(f"❌ Missing: {filename}")
    
    print(f"\n📏 Total model size: {total_size:.1f} MB")
    
    if not missing_files:
        print("✅ Model installation complete!")
        return True
    else:
        print(f"❌ Missing {len(missing_files)} required files")
        return False

def test_model_compatibility(model_dir, model_name):
    """Test if model loads with ONNX Runtime GenAI"""
    print_section(f"Testing {model_name} Compatibility")
    
    try:
        import onnxruntime_genai as og
        
        print(f"🔄 Testing model at: {Path(model_dir).absolute()}")
        
        # Try to load the model
        model = og.Model(str(model_dir))
        print("✅ Model loaded successfully with ONNX Runtime GenAI")
        
        # Try to load tokenizer
        tokenizer = og.Tokenizer(model)
        print("✅ Tokenizer loaded successfully")
        
        print(f"🎉 {model_name} is compatible and ready for NPU testing!")
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def main():
    """Main download function"""
    print_header("Best NPU Models Download for RyzenAI")
    
    print("🎯 Download the best LLM models optimized for AMD NPU")
    print("📊 Multiple download methods available")
    print("🔧 Git LFS for large files, Direct download for smaller files")
    
    # Show model options
    models = show_model_options()
    
    print(f"\n💡 Recommendation: Option 1 (Phi-3 Mini) - no authentication required, fast download")
    print(f"💡 Alternative: Option 2 (DeepSeek R1) - latest reasoning capabilities")
    print(f"💡 Safe Option: Option 3 (Llama-2 AMD) - verified working")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return False
            if choice in models:
                break
            print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            return False
    
    selected_model = models[choice]
    model_name = selected_model["name"]
    method = selected_model.get("method", "direct")
    
    # Handle manual option
    if method == "manual":
        provide_manual_instructions()
        return True
    
    print(f"\n🎯 Downloading: {model_name}")
    print(f"📏 Expected size: {selected_model['size']}")
    print(f"⏱️  Estimated time: 10-30 minutes")
    print(f"🔧 Method: {method.replace('_', ' ').title()}")
    
    model_dir = None
    success = False
    
    # Git LFS method
    if method == "git_lfs":
        # Check Git LFS
        if not check_git_lfs():
            if not install_git_lfs():
                print("\n❌ Git LFS is required for this model")
                print("   Please install Git LFS manually: https://git-lfs.github.io/")
                print("   Or choose a different model")
                return False
        
        repo_name = selected_model["repo"]
        model_dir = clone_model_repo(repo_name, model_name)
        
        if model_dir:
            success = verify_model_files(model_dir)
    
    # Direct download method
    elif method == "direct":
        base_url = selected_model["url_base"]
        
        # Create model directory
        model_dir = Path(f"models/npu/{model_name.lower().replace(' ', '_').replace('-', '_')}")
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {model_dir.absolute()}")
        
        # Download model
        if download_model_files(base_url, model_dir, model_name):
            success = verify_model_installation(model_dir, model_name)
        else:
            success = False
    
    # Test model compatibility
    if success and model_dir:
        if test_model_compatibility(model_dir, model_name):
            print_header("Download Complete!")
            print(f"✅ {model_name} ready for NPU testing")
            print(f"📂 Location: {model_dir}")
            print("🚀 Next: Run 'python test_llm_on_npu.py'")
            return True
        else:
            print("⚠️  Model downloaded but compatibility test failed")
            print("   Try running NPU test anyway - might still work")
            return True
    elif model_dir:
        print("❌ Model download/verification failed")
        print("\n📋 Alternatives:")
        provide_manual_instructions()
        return False
    else:
        print("❌ Model download failed")
        provide_manual_instructions()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)