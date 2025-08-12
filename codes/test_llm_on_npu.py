#!/usr/bin/env python3
"""
Simple LLM Test on AMD NPU using RyzenAI v1.5.1
Tests basic LLM inference capabilities on NPU hardware
"""

import sys
import time
import os
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def check_prerequisites():
    """Check if all prerequisites are available"""
    print_section("Prerequisites Check")
    
    try:
        import onnxruntime as ort
        import onnxruntime_genai as og
        
        print(f"✅ ONNX Runtime: {ort.__version__}")
        print(f"✅ ONNX Runtime GenAI: {og.__version__}")
        
        # Check NPU provider
        providers = ort.get_available_providers()
        npu_available = "VitisAIExecutionProvider" in providers
        
        print(f"✅ NPU Provider: {'Available' if npu_available else 'Not Available'}")
        
        return npu_available
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def download_test_model():
    """Download or locate a test model for NPU"""
    print_section("Model Setup")
    
    # Expanded model paths for RyzenAI
    model_paths = [
        "C:/Program Files/RyzenAI/models",
        "C:/Program Files/RyzenAI/share/models",
        "C:/Program Files/RyzenAI/examples/models",
        "C:/RyzenAI/models", 
        "./models/npu",
        "./models",
        "../models/npu"
    ]
    
    print("🔍 Searching for pre-installed NPU models...")
    
    # First check if any model directories exist
    found_model_dirs = []
    for base_path in model_paths:
        if Path(base_path).exists():
            found_model_dirs.append(base_path)
            print(f"📁 Found model directory: {base_path}")
            
            # List contents
            try:
                contents = list(Path(base_path).iterdir())
                if contents:
                    print(f"   Contents: {[p.name for p in contents[:5]]}")
                    
                    # Look for ONNX models in subdirectories
                    for item in contents:
                        if item.is_dir():
                            # Check subdirectory for ONNX files
                            try:
                                sub_contents = list(item.iterdir())
                                onnx_files = [p for p in sub_contents if p.suffix in ['.onnx', '.ort']]
                                if onnx_files:
                                    print(f"✅ Found ONNX models in {item.name}: {[p.name for p in onnx_files]}")
                                    return str(item)  # Return subdirectory path
                            except Exception:
                                continue
                    
                    # Also check current directory for ONNX files
                    onnx_files = [p for p in contents if p.suffix in ['.onnx', '.ort']]
                    if onnx_files:
                        print(f"✅ Found ONNX models: {[p.name for p in onnx_files]}")
                        return str(onnx_files[0].parent)  # Return directory path
                else:
                    print("   (Empty directory)")
            except Exception as e:
                print(f"   (Cannot list contents: {e})")
    
    if found_model_dirs:
        print(f"\n📁 Found {len(found_model_dirs)} model directories but no ONNX models")
    else:
        print("❌ No model directories found")
    
    print("\n📥 To download NPU-optimized models:")
    print("   1. Visit: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx")
    print("   2. Or check: C:/Program Files/RyzenAI/ for model zoo")
    print("   3. Place models in: ./models/npu/")
    
    # Create a simple mock model path for testing
    print("\n🧪 Proceeding with provider configuration test (no inference)")
    return None

def test_model_loading(model_path=None):
    """Test loading model with NPU provider using proper VitisAI configuration"""
    print_section("Model Loading Test")
    
    if not model_path:
        print("⚠️  No model available - testing provider configuration only")
        return test_provider_configuration()
    
    try:
        import onnxruntime as ort
        from pathlib import Path
        
        print(f"📂 Loading model from: {model_path}")
        
        # Get provider configuration first
        providers, provider_options = test_provider_configuration()
        if not providers:
            print("❌ Cannot configure NPU providers")
            return None, None
        
        # Find the ONNX model file
        model_file = Path(model_path) / "model.onnx"
        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return None, None
        
        # Create session options
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        
        # Create inference session with NPU provider
        print("🔄 Creating NPU inference session...")
        session = ort.InferenceSession(
            str(model_file),
            providers=providers,
            sess_options=session_options,
            provider_options=provider_options
        )
        
        print("✅ NPU inference session created successfully!")
        print(f"✅ Model loaded with providers: {session.get_providers()}")
        
        # For LLM inference, we'd still need ONNX Runtime GenAI
        # But this proves the NPU provider works with the model
        return session, None
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   This may be due to:")
        print("   - Model not compatible with NPU")
        print("   - Missing model files")
        print("   - NPU driver/VitisAI configuration issues")
        return None, None

def test_provider_configuration():
    """Test NPU provider configuration with proper VitisAI setup"""
    print_section("NPU Provider Configuration Test")
    
    try:
        import onnxruntime as ort
        import subprocess
        import os
        
        # Check for RYZEN_AI_INSTALLATION_PATH environment variable
        install_dir = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
        if not install_dir:
            print("❌ RYZEN_AI_INSTALLATION_PATH environment variable not set")
            print("   Please set this variable to your RyzenAI installation directory")
            return None, None
        
        print(f"✅ RyzenAI Installation: {install_dir}")
        
        # Detect NPU type (from working hello_world.py)
        command = r'pnputil /enum-devices /bus PCI /deviceids '
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        npu_type = ''
        if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): npu_type = 'PHX/HPT'
        if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): npu_type = 'STX'
        if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): npu_type = 'STX'
        if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): npu_type = 'STX'
        
        if not npu_type:
            print("❌ No supported NPU hardware detected")
            return None, None
            
        print(f"✅ NPU Type: {npu_type}")
        
        # Set xclbin file based on NPU type
        xclbin_file = ''
        if npu_type == 'PHX/HPT':
            xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
        elif npu_type == 'STX':
            xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')
        
        if not os.path.exists(xclbin_file):
            print(f"❌ xclbin file not found: {xclbin_file}")
            return None, None
            
        print(f"✅ xclbin file: {xclbin_file}")
        
        # Get config file path
        config_file_path = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')
        if not os.path.exists(config_file_path):
            print(f"❌ Config file not found: {config_file_path}")
            return None, None
            
        print(f"✅ Config file: {config_file_path}")
        
        # Prepare provider options - must match number of providers
        provider_options = [
            {  # VitisAIExecutionProvider options
                'config_file': config_file_path,
                'cacheDir': './cache',
                'cacheKey': 'npu_test_cache',
                'xclbin': xclbin_file
            },
            {}  # CPUExecutionProvider options (empty dict)
        ]
        
        print("✅ NPU provider configuration ready")
        return ["VitisAIExecutionProvider", "CPUExecutionProvider"], provider_options
        
    except Exception as e:
        print(f"❌ Provider configuration failed: {e}")
        return None, None

def test_simple_inference(model, tokenizer):
    """Test simple text generation"""
    print_section("Simple Inference Test")
    
    if not model or not tokenizer:
        print("⚠️  Skipping inference test - no model loaded")
        return False
    
    try:
        import onnxruntime_genai as og
        
        # Simple test prompt
        test_prompt = "What is artificial intelligence?"
        print(f"📝 Test prompt: '{test_prompt}'")
        
        # Tokenize input
        print("🔄 Tokenizing input...")
        tokens = tokenizer.encode(test_prompt)
        print(f"✅ Tokenized to {len(tokens)} tokens")
        
        # Create generator
        print("🔄 Creating generator...")
        generator = og.Generator(model, tokens)
        print("✅ Generator created")
        
        # Generate response
        print("🔄 Generating response...")
        start_time = time.time()
        
        generated_tokens = []
        max_tokens = 50  # Keep it short for test
        
        while not generator.is_done() and len(generated_tokens) < max_tokens:
            generator.compute_logits()
            generator.generate_next_token()
            generated_tokens.extend(generator.get_next_tokens())
        
        generation_time = time.time() - start_time
        
        # Decode response
        response = tokenizer.decode(generated_tokens)
        
        print(f"✅ Generation completed in {generation_time:.2f} seconds")
        print(f"📊 Generated {len(generated_tokens)} tokens")
        print(f"🚀 Speed: {len(generated_tokens)/generation_time:.1f} tokens/second")
        
        print(f"\n💬 Generated Response:")
        print(f"   '{response}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def test_performance_benchmark():
    """Simple performance benchmark"""
    print_section("Performance Benchmark")
    
    print("📊 NPU Performance Expectations:")
    print("   - Hybrid Mode: 40-60 tokens/second")
    print("   - NPU-Only Mode: 25-35 tokens/second")
    print("   - Time to First Token: 0.2-0.5 seconds")
    print("   - Memory Usage: <8GB NPU memory")
    
    # Simple timing test
    import time
    
    print("\n⏱️  Basic Timing Test:")
    start_time = time.time()
    
    # Simulate some computation
    for i in range(1000000):
        _ = i * 2
    
    end_time = time.time()
    print(f"   Python computation: {(end_time - start_time)*1000:.2f} ms")
    
    return True

def main():
    """Main test function"""
    print_header("Simple LLM Test on AMD NPU")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required packages:")
        print("   pip install onnxruntime onnxruntime-genai")
        return False
    
    # Setup model
    model_path = download_test_model()
    
    # Test model loading
    model, tokenizer = test_model_loading(model_path)
    
    # Test inference
    inference_success = test_simple_inference(model, tokenizer)
    
    # Performance benchmark
    test_performance_benchmark()
    
    # Summary
    print_header("Test Summary")
    
    if model_path and inference_success:
        print("✅ NPU LLM TEST PASSED!")
        print("   - Model loaded successfully")
        print("   - Inference completed on NPU")
        print("   - Ready for agent testing framework")
    elif model_path:
        print("⚠️  NPU LLM TEST PARTIAL:")
        print("   - Model found but inference failed")
        print("   - Check NPU drivers and model compatibility")
    else:
        print("⚠️  NPU LLM TEST SKIPPED:")
        print("   - No NPU-compatible model found")
        print("   - Provider configuration tested only")
        
    print(f"\n📋 Next Steps:")
    if inference_success:
        print("   ✅ Ready to integrate with DeepEval")
        print("   ✅ Can proceed with agent testing framework")
    else:
        print("   📥 Download NPU-optimized model")
        print("   🔧 Verify RyzenAI installation")
        print("   🔄 Re-run test after model setup")
    
    return inference_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)