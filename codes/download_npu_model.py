#!/usr/bin/env python3
"""
Download NPU-Optimized LLM Model for RyzenAI
Downloads a small, compatible model for testing NPU functionality
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

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

def create_model_directory():
    """Create model directory structure"""
    print_section("Creating Model Directory")
    
    model_dir = Path("models/npu")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created directory: {model_dir.absolute()}")
    return model_dir

def download_phi3_mini_model(model_dir):
    """Download Phi-3 Mini model optimized for NPU"""
    print_section("Downloading Phi-3 Mini INT4 Model")
    
    # Microsoft's official Phi-3 Mini 4K Instruct ONNX model
    base_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main"
    
    files_to_download = [
        ("genai_config.json", "GenAI configuration"),
        ("model.onnx", "Main model file"),
        ("model.onnx.data", "Model weights"),
        ("tokenizer.json", "Tokenizer configuration"),
        ("tokenizer_config.json", "Tokenizer settings"),
        ("special_tokens_map.json", "Special tokens"),
        ("vocab.json", "Vocabulary")
    ]
    
    success_count = 0
    total_files = len(files_to_download)
    
    for filename, description in files_to_download:
        url = f"{base_url}/{filename}"
        destination = model_dir / filename
        
        if destination.exists():
            print(f"⏭️  Skipping {filename} (already exists)")
            success_count += 1
            continue
        
        if download_file(url, destination, f"{description} ({filename})"):
            success_count += 1
        else:
            print(f"⚠️  Failed to download {filename}")
    
    print(f"\n📊 Download Summary: {success_count}/{total_files} files successful")
    
    if success_count >= 4:  # Need at least config, model, tokenizer files
        print("✅ Sufficient files downloaded for testing")
        return True
    else:
        print("❌ Not enough files downloaded")
        return False

def download_alternative_small_model(model_dir):
    """Download a smaller alternative model if Phi-3 fails"""
    print_section("Downloading Alternative Small Model")
    
    print("🔄 Attempting to download TinyLlama model...")
    print("   This is a smaller model that might work better for testing")
    
    # Note: This is a placeholder - actual implementation would need 
    # a confirmed NPU-compatible small model
    print("⚠️  Alternative download not implemented yet")
    print("   Please try manual download or use Phi-3 Mini")
    
    return False

def verify_model_installation(model_dir):
    """Verify the downloaded model is complete"""
    print_section("Verifying Model Installation")
    
    required_files = [
        "genai_config.json",
        "model.onnx", 
        "tokenizer.json"
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = model_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename}: {size_mb:.1f} MB")
        else:
            missing_files.append(filename)
            print(f"❌ Missing: {filename}")
    
    if not missing_files:
        print("✅ Model installation complete!")
        return True
    else:
        print(f"❌ Missing {len(missing_files)} required files")
        return False

def test_model_compatibility():
    """Quick test to see if model loads with ONNX Runtime GenAI"""
    print_section("Testing Model Compatibility")
    
    try:
        import onnxruntime_genai as og
        
        model_dir = Path("models/npu")
        if not model_dir.exists():
            print("❌ Model directory not found")
            return False
        
        print(f"🔄 Testing model at: {model_dir.absolute()}")
        
        # Try to load the model
        model = og.Model(str(model_dir))
        print("✅ Model loaded successfully with ONNX Runtime GenAI")
        
        # Try to load tokenizer
        tokenizer = og.Tokenizer(model)
        print("✅ Tokenizer loaded successfully")
        
        print("🎉 Model is compatible and ready for NPU testing!")
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        print("   This might be due to:")
        print("   - Incomplete model download")
        print("   - Model format incompatibility")
        print("   - Missing dependencies")
        return False

def main():
    """Main download function"""
    print_header("NPU Model Download for RyzenAI")
    
    print("🎯 This will download a small LLM model optimized for NPU inference")
    print("📏 Model size: ~2-4 GB (Phi-3 Mini INT4)")
    print("⏱️  Download time: 5-15 minutes depending on connection")
    
    print("\nPress Enter to continue, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        return False
    
    # Create model directory
    model_dir = create_model_directory()
    
    # Download Phi-3 Mini model
    phi3_success = download_phi3_mini_model(model_dir)
    
    # If Phi-3 fails, try alternative
    if not phi3_success:
        print("🔄 Trying alternative model...")
        alt_success = download_alternative_small_model(model_dir)
        if not alt_success:
            print("❌ All download attempts failed")
            return False
    
    # Verify installation
    if verify_model_installation(model_dir):
        # Test compatibility
        if test_model_compatibility():
            print_header("Download Complete!")
            print("✅ NPU-optimized model ready for testing")
            print("🚀 Next step: Run 'python test_llm_on_npu.py'")
            return True
        else:
            print("⚠️  Model downloaded but compatibility test failed")
            print("   Try running the NPU test anyway")
            return True
    else:
        print("❌ Model installation incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)