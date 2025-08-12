#!/usr/bin/env python3
"""
Test NPU with Pure ONNX Runtime (like hello_world.py)
This approach should actually use NPU hardware
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def configure_npu_like_hello_world():
    """Configure NPU exactly like the working hello_world.py"""
    print("🔧 Configuring NPU like working hello_world.py...")
    
    # Get installation directory
    install_dir = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
    if not install_dir:
        raise Exception("RYZEN_AI_INSTALLATION_PATH not set")
    
    # Detect NPU type (copied from hello_world.py)
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    npu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): npu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): npu_type = 'STX'
    
    print(f"✅ NPU Type: {npu_type}")
    
    # Set xclbin file (copied from hello_world.py)
    xclbin_file = ''
    if npu_type == 'PHX/HPT':
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    elif npu_type == 'STX':
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')
    
    if not os.path.exists(xclbin_file):
        raise Exception(f"xclbin file not found: {xclbin_file}")
    
    print(f"✅ xclbin file: {xclbin_file}")
    
    # Config file path (copied from hello_world.py)
    config_file_path = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')
    if not os.path.exists(config_file_path):
        raise Exception(f"Config file not found: {config_file_path}")
    
    print(f"✅ Config file: {config_file_path}")
    
    # Clear cache (like hello_world.py)
    cache_directory = "./cache"
    Path(cache_directory).mkdir(exist_ok=True)
    
    return {
        "xclbin_file": xclbin_file,
        "config_file_path": config_file_path,
        "cache_directory": cache_directory
    }

def test_npu_with_phi3_model():
    """Test NPU using Phi-3 model with pure ONNX Runtime"""
    print_header("Pure ONNX Runtime NPU Test with Phi-3")
    
    try:
        import onnxruntime as ort
        import onnx
        
        # Configure NPU
        npu_config = configure_npu_like_hello_world()
        
        # Find Phi-3 ONNX model
        model_paths = [
            "./models/npu/phi3_mini/directml/directml-int4-awq-block-128/model.onnx",
            "./models/npu/phi3_mini/cpu_and_mobile/cpu-int4-rtn-block-32/phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx",
            "./models/npu/phi3_mini/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"
        ]
        
        model_file = None
        for path in model_paths:
            if Path(path).exists():
                model_file = path
                break
        
        if not model_file:
            raise Exception("No ONNX model file found")
        
        print(f"📂 Using model: {model_file}")
        
        # Load model (like hello_world.py)
        print("📥 Loading ONNX model...")
        model = onnx.load(model_file)
        
        # Create session options (like hello_world.py)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        
        # Create NPU session with VitisAI provider (EXACTLY like hello_world.py)
        print("🚀 Creating NPU session with VitisAI provider...")
        print("👀 WATCH TASK MANAGER -> PERFORMANCE -> NPU NOW!")
        
        npu_session = ort.InferenceSession(
            model.SerializeToString(),
            providers=['VitisAIExecutionProvider'],
            sess_options=session_options,
            provider_options=[{
                'config_file': npu_config["config_file_path"],
                'cacheDir': npu_config["cache_directory"], 
                'cacheKey': 'phi3_npu_test',
                'xclbin': npu_config["xclbin_file"]
            }]
        )
        
        print(f"✅ NPU session created!")
        print(f"📊 Providers: {npu_session.get_providers()}")
        
        # Get model inputs
        inputs = npu_session.get_inputs()
        outputs = npu_session.get_outputs()
        
        print(f"📝 Model inputs: {[inp.name for inp in inputs]}")
        print(f"📤 Model outputs: {[out.name for out in outputs]}")
        
        # Create test input data
        print("🔄 Creating test input data...")
        
        # This is tricky - we need to match Phi-3's expected input format
        # Phi-3 typically expects input_ids as the main input
        
        # Define consistent dimensions
        batch_size = 1
        sequence_length = 20  # Shorter sequence for testing
        past_sequence_length = 0  # No past for initial inference
        
        input_data = {}
        
        for inp in inputs:
            shape = inp.shape
            print(f"   🔍 Input {inp.name}: shape {shape}, type {inp.type}")
            
            if inp.name == 'input_ids':
                # Main input tokens
                test_data = np.random.randint(1, 32000, size=(batch_size, sequence_length), dtype=np.int64)
                input_data[inp.name] = test_data
                print(f"   📝 {inp.name}: {test_data.shape}")
                
            elif inp.name == 'position_ids':
                # Position indices - must match input_ids sequence length
                test_data = np.arange(sequence_length, dtype=np.int64).reshape(1, sequence_length)
                input_data[inp.name] = test_data
                print(f"   📝 {inp.name}: {test_data.shape}")
                
            elif inp.name == 'attention_mask':
                # Attention mask - must match total sequence length  
                test_data = np.ones((batch_size, sequence_length), dtype=np.int64)
                input_data[inp.name] = test_data
                print(f"   📝 {inp.name}: {test_data.shape}")
                
            elif 'past_key_values' in inp.name:
                # Past key/value states - empty for initial inference
                # Shape: [batch_size, num_heads, past_sequence_length, head_dim]
                concrete_shape = []
                for dim in shape:
                    if isinstance(dim, str):
                        if 'batch_size' in dim:
                            concrete_shape.append(batch_size)
                        elif 'past_sequence_length' in dim:
                            concrete_shape.append(past_sequence_length)  # Empty past
                        else:
                            concrete_shape.append(1)
                    elif dim == -1:
                        concrete_shape.append(batch_size)
                    else:
                        concrete_shape.append(dim)
                
                if inp.type == 'tensor(float16)':
                    test_data = np.zeros(concrete_shape, dtype=np.float16)
                else:
                    test_data = np.zeros(concrete_shape, dtype=np.float32)
                    
                input_data[inp.name] = test_data
                print(f"   📝 {inp.name}: {test_data.shape}")
                
            else:
                # Handle any other inputs
                concrete_shape = []
                for dim in shape:
                    if isinstance(dim, str):
                        if 'batch_size' in dim:
                            concrete_shape.append(batch_size)
                        elif 'sequence_length' in dim:
                            concrete_shape.append(sequence_length)
                        elif 'total_sequence_length' in dim:
                            concrete_shape.append(sequence_length + past_sequence_length)
                        else:
                            concrete_shape.append(1)
                    elif dim == -1:
                        concrete_shape.append(batch_size)
                    else:
                        concrete_shape.append(dim)
                
                if inp.type == 'tensor(int64)':
                    if 'mask' in inp.name.lower():
                        test_data = np.ones(concrete_shape, dtype=np.int64)
                    else:
                        test_data = np.zeros(concrete_shape, dtype=np.int64)
                elif inp.type == 'tensor(float16)':
                    test_data = np.zeros(concrete_shape, dtype=np.float16)
                else:
                    test_data = np.zeros(concrete_shape, dtype=np.float32)
                
                input_data[inp.name] = test_data
                print(f"   📝 {inp.name}: {test_data.shape}")
        
        if not input_data:
            raise Exception("Could not create input data for model")
        
        # Run multiple inferences to trigger NPU activity
        print("\n🔥 RUNNING INTENSIVE NPU INFERENCE")
        print("👀 WATCH TASK MANAGER NPU TAB NOW!")
        
        input("Press Enter to start NPU inference test...")
        
        inference_count = 0
        
        try:
            for i in range(20):  # Multiple runs to trigger NPU
                print(f"🔄 NPU Inference {i+1}/20...")
                
                # Run inference (this should use NPU if configured correctly)
                start_time = time.time() if 'time' in dir() else 0
                
                npu_results = npu_session.run(None, input_data)
                
                inference_count += 1
                
                if i == 0:
                    print(f"✅ First inference successful!")
                    print(f"📤 Output shapes: {[r.shape for r in npu_results]}")
                
                # Brief pause to see NPU spikes
                import time
                time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ NPU inference failed: {e}")
            return False
        
        print(f"\n🏁 NPU Test Complete!")
        print(f"✅ Successfully ran {inference_count} NPU inferences")
        
        # Ask user about NPU activity
        user_response = input("Did you see NPU activity spikes in Task Manager? (y/n): ").strip().lower()
        
        if user_response == 'y':
            print("🎉 SUCCESS: NPU is working with Phi-3 model!")
            return True
        else:
            print("⚠️  No NPU activity observed")
            print("   This suggests the VitisAI provider may not be routing to NPU")
            return False
            
    except Exception as e:
        print(f"❌ NPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main pure ONNX Runtime NPU test"""
    print_header("Pure ONNX Runtime NPU Test")
    
    print("🎯 Using the same approach as working hello_world.py")
    print("🔧 Pure ONNX Runtime + VitisAI provider")
    print("📊 This should trigger actual NPU usage")
    
    try:
        success = test_npu_with_phi3_model()
        
        if success:
            print("\n🎉 NPU TEST SUCCESSFUL!")
            print("✅ Phi-3 model running on AMD NPU")
        else:
            print("\n❌ NPU TEST INCONCLUSIVE")
            print("💡 Possible solutions:")
            print("   1. Model may need NPU-specific compilation")
            print("   2. Try quantized model specifically for NPU")
            print("   3. Check VAI EP documentation for Phi-3 support")
        
        return success
        
    except Exception as e:
        print(f"❌ Main test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)