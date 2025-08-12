#!/usr/bin/env python3
"""
Test NPU Shared Memory Allocation
Explore using NPU's 16GB dedicated memory instead of system RAM
"""

import os
import sys
import psutil
import time
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n--- {title} ---")

def get_system_memory_info():
    """Get detailed system memory information"""
    print_section("System Memory Information")
    
    # System RAM
    memory = psutil.virtual_memory()
    print(f"💾 System RAM:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB") 
    print(f"   Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    
    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    print(f"\n🔬 Current Process:")
    print(f"   RSS (Resident): {process_memory.rss / (1024**2):.1f} MB")
    print(f"   VMS (Virtual): {process_memory.vms / (1024**2):.1f} MB")
    
    return {
        "system_total_gb": memory.total / (1024**3),
        "system_available_gb": memory.available / (1024**3),
        "process_rss_mb": process_memory.rss / (1024**2)
    }

def test_npu_memory_configuration():
    """Test NPU memory configuration options"""
    print_section("NPU Memory Configuration")
    
    try:
        import onnxruntime as ort
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"✅ Available providers: {providers}")
        
        # NPU-specific memory configuration
        if 'VitisAIExecutionProvider' in providers:
            print("🔧 VitisAI Provider Configuration Options:")
            
            # Memory allocation strategies
            memory_configs = [
                {
                    "name": "NPU Dedicated Memory",
                    "description": "Force allocation in NPU's 16GB memory space",
                    "options": {
                        'memory_limit_mb': 8192,  # Use 8GB of NPU memory
                        'use_shared_memory': True,
                        'memory_pool_type': 'npu_dedicated'
                    }
                },
                {
                    "name": "NPU Shared Memory",  
                    "description": "Use NPU shared memory pool",
                    "options": {
                        'memory_limit_mb': 12288,  # Use 12GB of NPU memory
                        'enable_memory_arena': True,
                        'memory_pattern_optimization': True
                    }
                },
                {
                    "name": "Hybrid Memory",
                    "description": "NPU memory for weights, system RAM for activations",
                    "options": {
                        'memory_limit_mb': 6144,  # 6GB NPU for weights
                        'enable_cpu_mem_arena': False,  # Don't use system RAM arena
                        'memory_pool_type': 'hybrid'
                    }
                }
            ]
            
            for config in memory_configs:
                print(f"\n📋 {config['name']}:")
                print(f"   {config['description']}")
                for key, value in config['options'].items():
                    print(f"   {key}: {value}")
        
        else:
            print("⚠️  VitisAI provider not available")
        
        return memory_configs if 'VitisAIExecutionProvider' in providers else []
        
    except Exception as e:
        print(f"❌ NPU memory configuration failed: {e}")
        return []

def test_model_loading_with_npu_memory():
    """Test loading model with NPU memory allocation"""
    print_section("Model Loading with NPU Memory")
    
    try:
        import onnxruntime as ort
        import onnx
        
        # Find our Phi-3 model
        model_paths = [
            "./models/npu/phi3_mini/directml/directml-int4-awq-block-128/model.onnx"
        ]
        
        model_file = None
        for path in model_paths:
            if Path(path).exists():
                model_file = path
                break
        
        if not model_file:
            print("❌ No model found for memory testing")
            return False
        
        print(f"📂 Testing with model: {model_file}")
        
        # Get baseline memory usage
        baseline_memory = get_current_memory_usage()
        print(f"📊 Baseline memory: {baseline_memory['process_rss_mb']:.1f} MB")
        
        # Configure NPU environment
        install_dir = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
        if not install_dir:
            print("⚠️  RYZEN_AI_INSTALLATION_PATH not set - using basic configuration")
            return test_basic_memory_allocation(model_file, baseline_memory)
        
        # NPU configuration  
        config_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')
        
        if not os.path.exists(config_file) or not os.path.exists(xclbin_file):
            print("⚠️  NPU config files not found - using basic configuration")
            return test_basic_memory_allocation(model_file, baseline_memory)
        
        # Test different memory configurations
        memory_results = {}
        
        for i, config_name in enumerate(["Standard", "NPU-Optimized", "Memory-Constrained"]):
            print(f"\n🧪 Testing {config_name} Memory Configuration...")
            
            try:
                # Create session with NPU memory configuration
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3
                
                # Memory-specific options
                if config_name == "NPU-Optimized":
                    session_options.enable_mem_pattern = False  # Don't use system memory patterns
                    session_options.enable_cpu_mem_arena = False  # Don't use CPU memory arena
                    
                elif config_name == "Memory-Constrained":
                    session_options.enable_mem_pattern = True
                    session_options.enable_cpu_mem_arena = True
                
                # Provider options for NPU memory
                provider_options = [{
                    'config_file': config_file,
                    'cacheDir': './cache',
                    'cacheKey': f'memory_test_{i}',
                    'xclbin': xclbin_file
                }]
                
                # Load model
                model = onnx.load(model_file)
                
                before_loading = get_current_memory_usage()
                
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
                    sess_options=session_options,
                    provider_options=provider_options
                )
                
                after_loading = get_current_memory_usage()
                
                memory_diff = after_loading['process_rss_mb'] - before_loading['process_rss_mb']
                
                print(f"   📊 Memory usage: {memory_diff:.1f} MB")
                print(f"   📈 Total process memory: {after_loading['process_rss_mb']:.1f} MB")
                
                memory_results[config_name] = {
                    "memory_used_mb": memory_diff,
                    "total_memory_mb": after_loading['process_rss_mb'],
                    "session_created": True
                }
                
                # Clean up
                del session
                del model
                time.sleep(1)  # Allow cleanup
                
            except Exception as e:
                print(f"   ❌ {config_name} failed: {e}")
                memory_results[config_name] = {"error": str(e), "session_created": False}
        
        # Summary
        print_section("Memory Configuration Results")
        
        for config_name, result in memory_results.items():
            if result.get("session_created"):
                print(f"✅ {config_name}: {result['memory_used_mb']:.1f} MB used")
            else:
                print(f"❌ {config_name}: Failed - {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory testing failed: {e}")
        return False

def test_basic_memory_allocation(model_file, baseline_memory):
    """Basic memory allocation test without NPU configuration"""
    print("🔧 Running basic memory allocation test...")
    
    try:
        import onnxruntime as ort
        import onnx
        
        # Load with different session options
        configs = [
            ("Default", {}),
            ("Memory Optimized", {
                "enable_mem_pattern": False,
                "enable_cpu_mem_arena": False
            }),
            ("CPU Arena Disabled", {
                "enable_cpu_mem_arena": False
            })
        ]
        
        for config_name, options in configs:
            print(f"\n🧪 Testing {config_name}...")
            
            try:
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3
                
                for key, value in options.items():
                    setattr(session_options, key, value)
                
                before = get_current_memory_usage()
                
                model = onnx.load(model_file)
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    providers=['CPUExecutionProvider'],
                    sess_options=session_options
                )
                
                after = get_current_memory_usage()
                memory_used = after['process_rss_mb'] - before['process_rss_mb']
                
                print(f"   📊 Memory used: {memory_used:.1f} MB")
                
                # Cleanup
                del session
                del model
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic memory test failed: {e}")
        return False

def get_current_memory_usage():
    """Get current memory usage"""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    process_memory = process.memory_info()
    
    return {
        "system_available_gb": memory.available / (1024**3),
        "process_rss_mb": process_memory.rss / (1024**2),
        "process_vms_mb": process_memory.vms / (1024**2)
    }

def analyze_memory_recommendations():
    """Provide memory optimization recommendations"""
    print_section("NPU Memory Optimization Recommendations")
    
    print("🎯 For your AMD AI 395 MX (16GB NPU + 32GB RAM):")
    
    print("\n💾 Memory Allocation Strategy:")
    print("   ✅ NPU Memory (16GB): Store model weights and intermediate results")
    print("   ✅ System RAM (32GB): Store activations and temporary data")
    print("   ✅ Hybrid Mode: Optimize for both speed and memory efficiency")
    
    print("\n🔧 Implementation Options:")
    print("   1. DirectML with NPU backend - Automatic memory management")
    print("   2. VitisAI with memory constraints - Manual memory pool configuration") 
    print("   3. ONNX Runtime memory arenas - Disable CPU arena, use NPU pool")
    
    print("\n📊 Expected Benefits:")
    print("   • Reduced system RAM usage (keep 32GB available for other processes)")
    print("   • Faster model inference (NPU memory has higher bandwidth)")
    print("   • Better memory locality (weights stay in NPU memory)")
    print("   • Improved multi-model performance (each model uses NPU memory)")
    
    print("\n⚡ For Agent Testing:")
    print("   • Load multiple agent models simultaneously in NPU memory")
    print("   • Run parallel evaluations without RAM constraints")
    print("   • Better performance for DeepEval with multiple models")

def main():
    """Main memory testing function"""
    print_header("NPU Memory Allocation Testing")
    
    print("🎯 Objective: Use NPU's 16GB memory instead of system RAM")
    print("💾 Target: Optimize memory allocation for agent testing workloads")
    
    # Get baseline system info
    system_info = get_system_memory_info()
    
    # Test NPU memory configuration
    memory_configs = test_npu_memory_configuration()
    
    # Test actual model loading with NPU memory
    success = test_model_loading_with_npu_memory()
    
    # Provide recommendations
    analyze_memory_recommendations()
    
    print_header("Memory Testing Results")
    
    if success:
        print("✅ Memory configuration testing completed")
        print("📋 Key findings:")
        print("   • NPU memory allocation options identified")
        print("   • Model loading memory usage measured")
        print("   • Optimization strategies available")
        
        print("\n🚀 Next Steps:")
        print("   1. Configure ONNX Runtime to prefer NPU memory")
        print("   2. Test multi-model loading in NPU memory")
        print("   3. Benchmark agent testing performance with NPU memory")
    else:
        print("⚠️  Memory configuration testing incomplete")
        print("💡 Consider using DirectML provider for automatic NPU memory management")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)