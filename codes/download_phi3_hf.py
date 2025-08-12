#!/usr/bin/env python3
"""
Download Phi-3 Mini using HuggingFace Hub
More reliable than direct HTTP downloads
"""

import os
import sys
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def install_huggingface_hub():
    """Install HuggingFace Hub if not available"""
    print_section("Installing HuggingFace Hub")
    
    try:
        import huggingface_hub
        print("✅ HuggingFace Hub already installed")
        return True
    except ImportError:
        print("📦 Installing HuggingFace Hub...")
        import subprocess
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'], check=True)
            print("✅ HuggingFace Hub installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install HuggingFace Hub")
            return False

def download_phi3_mini():
    """Download Phi-3 Mini ONNX model using HuggingFace Hub"""
    print_section("Downloading Phi-3 Mini ONNX Model")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Create models directory
        model_dir = Path("./models/npu/phi3_mini")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Target directory: {model_dir.absolute()}")
        print("📥 Downloading Phi-3 Mini from HuggingFace...")
        print("⏱️  This may take 5-15 minutes...")
        
        # Download the model
        downloaded_path = snapshot_download(
            repo_id="microsoft/Phi-3-mini-4k-instruct-onnx",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Model downloaded to: {downloaded_path}")
        return model_dir
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def verify_download(model_dir):
    """Verify the downloaded model"""
    print_section("Verifying Download")
    
    if not model_dir or not model_dir.exists():
        print("❌ Model directory not found")
        return False
    
    # Check for essential files
    required_files = [
        "model.onnx",
        "genai_config.json", 
        "tokenizer.json"
    ]
    
    found_files = []
    missing_files = []
    
    for filename in required_files:
        file_path = model_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename}: {size_mb:.1f} MB")
            found_files.append(filename)
        else:
            print(f"❌ Missing: {filename}")
            missing_files.append(filename)
    
    # List all downloaded files
    all_files = list(model_dir.glob("*"))
    print(f"\n📂 Total files downloaded: {len(all_files)}")
    for file in all_files[:10]:  # Show first 10 files
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   {file.name}: {size_mb:.1f} MB")
    
    if len(found_files) >= 2:  # At least model and one config
        print("\n✅ Download appears successful!")
        return True
    else:
        print(f"\n⚠️  Only {len(found_files)} essential files found")
        return len(found_files) > 0

def main():
    """Main download function"""
    print_header("Phi-3 Mini Download via HuggingFace Hub")
    
    print("🎯 Downloading Microsoft Phi-3 Mini 4K Instruct ONNX")
    print("📏 Expected size: ~2-4 GB")
    print("🔧 Method: HuggingFace Hub (more reliable)")
    
    # Install HuggingFace Hub
    if not install_huggingface_hub():
        return False
    
    # Download model
    model_dir = download_phi3_mini()
    
    # Verify download
    if verify_download(model_dir):
        print_header("Download Complete!")
        print(f"✅ Phi-3 Mini ready for NPU testing")
        print(f"📂 Location: {model_dir.absolute()}")
        print("🚀 Next: Run 'python codes/test_llm_on_npu.py'")
        return True
    else:
        print("❌ Download verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)