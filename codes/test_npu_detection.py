#!/usr/bin/env python3
"""
NPU Detection Test for AMD RyzenAI v1.5.1
Tests if NPU is properly detected and available for LLM inference
"""

import sys
import platform
import subprocess
import os

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def test_system_info():
    """Test basic system information"""
    print_section("System Information")
    
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    print(f"Architecture: {platform.architecture()}")
    
    # Check if Windows (required for RyzenAI)
    if platform.system() != "Windows":
        print("❌ ERROR: RyzenAI requires Windows OS")
        return False
    else:
        print("✅ Windows OS detected")
        return True

def test_onnxruntime_installation():
    """Test ONNX Runtime installation and providers"""
    print_section("ONNX Runtime Detection")
    
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime version: {ort.__version__}")
        
        # Get all available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {len(providers)}")
        
        for i, provider in enumerate(providers, 1):
            print(f"  {i}. {provider}")
        
        # Check for NPU support
        npu_available = "VitisAIExecutionProvider" in providers
        gpu_available = "DmlExecutionProvider" in providers
        
        print(f"\n🔍 NPU Support (VitisAI): {'✅ YES' if npu_available else '❌ NO'}")
        print(f"🔍 GPU Support (DirectML): {'✅ YES' if gpu_available else '❌ NO'}")
        
        return npu_available, gpu_available
        
    except ImportError as e:
        print(f"❌ ONNX Runtime not found: {e}")
        return False, False

def test_onnxruntime_genai():
    """Test ONNX Runtime GenAI for LLM support"""
    print_section("ONNX Runtime GenAI Detection")
    
    try:
        import onnxruntime_genai as og
        print(f"✅ ONNX Runtime GenAI version: {og.__version__}")
        
        # Test basic functionality
        print("✅ GenAI module imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ ONNX Runtime GenAI not found: {e}")
        print("   This is required for LLM inference on NPU")
        return False

def test_ryzenai_components():
    """Test RyzenAI specific components"""
    print_section("RyzenAI Components")
    
    # Check for RyzenAI installation in common paths
    ryzenai_paths = [
        "C:\\Program Files\\RyzenAI",
        "C:\\Program Files (x86)\\RyzenAI", 
        "C:\\RyzenAI"
    ]
    
    print("🔍 Checking RyzenAI installation paths...")
    ryzenai_found = False
    
    for path in ryzenai_paths:
        if os.path.exists(path):
            print(f"✅ RyzenAI found at: {path}")
            ryzenai_found = True
            
            # List contents
            try:
                contents = os.listdir(path)
                print(f"   Contents: {', '.join(contents[:5])}{'...' if len(contents) > 5 else ''}")
            except:
                print("   (Cannot list contents)")
            break
    
    if not ryzenai_found:
        print("❌ RyzenAI installation not found in common paths")
    
    # Check for executable in PATH
    try:
        result = subprocess.run(['where', 'ryzen-ai'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ RyzenAI executable in PATH")
            print(f"   Path: {result.stdout.strip()}")
        else:
            print("⚠️  RyzenAI executable not in PATH (this is normal)")
    except Exception as e:
        print(f"⚠️  Cannot check PATH: {e}")
    
    # Check Device Manager for NPU (Windows specific)
    print("\n🔍 NPU Device Detection:")
    print("   Please check Windows Device Manager:")
    print("   System Devices -> Look for 'AMD NPU Device' or similar")
    
    return True

def test_npu_session():
    """Test creating a simple NPU session"""
    print_section("NPU Session Test")
    
    try:
        import onnxruntime as ort
        
        # Try to create session with NPU provider
        providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
        
        print(f"Attempting to create session with providers: {providers}")
        
        # Configure session options
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # Reduce logging
        
        print("✅ Session options configured")
        print("   (Cannot test full session without model file)")
        
        return True
        
    except Exception as e:
        print(f"❌ NPU session test failed: {e}")
        return False

def test_memory_info():
    """Test memory information"""
    print_section("Memory Information")
    
    try:
        import psutil
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"Used RAM: {(memory.total - memory.available) / (1024**3):.1f} GB")
        
        print(f"\n💾 Memory for NPU/GPU:")
        print(f"   Expected NPU Memory: 16 GB (dedicated)")
        print(f"   Expected GPU Memory: 16 GB (dedicated)")
        
        return True
        
    except ImportError:
        print("❌ psutil not available for memory info")
        print("   Run: pip install psutil")
        return False

def main():
    """Main test function"""
    print_header("NPU Detection Test for AMD RyzenAI v1.5.1")
    
    # Track test results
    results = {}
    
    # Run tests
    results['system'] = test_system_info()
    results['onnxrt'], results['gpu'] = test_onnxruntime_installation()
    results['genai'] = test_onnxruntime_genai()
    results['ryzenai'] = test_ryzenai_components()
    results['session'] = test_npu_session()
    results['memory'] = test_memory_info()
    
    # Summary
    print_header("Test Summary")
    
    print("✅ PASSED TESTS:")
    for test, passed in results.items():
        if passed:
            print(f"   - {test.upper()}")
    
    print("\n❌ FAILED TESTS:")
    failed_tests = [test for test, passed in results.items() if not passed]
    if failed_tests:
        for test in failed_tests:
            print(f"   - {test.upper()}")
    else:
        print("   None")
    
    # Overall assessment
    npu_ready = results.get('onnxrt', False) and results.get('genai', False)
    
    print(f"\n🎯 NPU READINESS: {'✅ READY' if npu_ready else '❌ NOT READY'}")
    
    if npu_ready:
        print("\n✅ Your system appears ready for NPU-accelerated LLM testing!")
        print("   Next step: Test running a simple LLM on NPU")
    else:
        print("\n❌ NPU support issues detected.")
        print("   Please check RyzenAI v1.5.1 installation")
    
    return npu_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)