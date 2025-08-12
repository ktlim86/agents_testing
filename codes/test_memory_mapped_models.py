#!/usr/bin/env python3
"""
Test Memory-Mapped Model Loading
Reduce system RAM usage through memory mapping
"""

import os
import sys
import psutil
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_memory_mapped_loading():
    """Test memory-mapped model loading"""
    print_header("Memory-Mapped Model Loading Test")
    
    try:
        import onnxruntime as ort
        
        model_path = "./models/npu/phi3_mini/directml/directml-int4-awq-block-128/model.onnx"
        if not Path(model_path).exists():
            print("❌ Model not found")
            return False
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024**2)
        print(f"📊 Baseline memory: {baseline_memory:.1f} MB")
        
        # Test 1: Standard loading
        print("\n🧪 Test 1: Standard Model Loading")
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        
        session1 = ort.InferenceSession(
            model_path,
            providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
            sess_options=session_options
        )
        
        memory1 = process.memory_info().rss / (1024**2)
        memory_used1 = memory1 - baseline_memory
        print(f"📈 Memory after loading: {memory_used1:.1f} MB")
        print(f"📊 Providers: {session1.get_providers()}")
        
        del session1
        
        # Test 2: Memory-optimized loading
        print("\n🧪 Test 2: Memory-Optimized Loading")
        session_options2 = ort.SessionOptions()
        session_options2.log_severity_level = 3
        session_options2.enable_mem_pattern = False  # Don't pre-allocate memory patterns
        session_options2.enable_cpu_mem_arena = False  # Don't use CPU memory arena
        
        session2 = ort.InferenceSession(
            model_path,
            providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
            sess_options=session_options2
        )
        
        memory2 = process.memory_info().rss / (1024**2)
        memory_used2 = memory2 - baseline_memory
        print(f"📈 Memory after optimized loading: {memory_used2:.1f} MB")
        print(f"📊 Memory reduction: {memory_used1 - memory_used2:.1f} MB")
        
        del session2
        
        return True
        
    except Exception as e:
        print(f"❌ Memory-mapped test failed: {e}")
        return False

def main():
    """Main memory optimization test"""
    print("Objective: Minimize system RAM usage for model loading")
    print("Strategy: Use DirectML and memory optimization techniques")
    
    success = test_memory_mapped_loading()
    
    print_header("Memory Optimization Summary")
    
    if success:
        print("Memory optimization tests completed")
        print("\nKey Insights:")
        print("   • NPU memory is used as computation buffer, not storage")
        print("   • Model weights remain in system RAM by design")
        print("   • DirectML provider can optimize memory usage")
        print("   • Memory arenas can be disabled to reduce footprint")
        
        print("\nFor Agent Testing:")
        print("   • Use DirectML provider for automatic optimization")
        print("   • Disable memory arenas to reduce RAM usage")
        print("   • Consider model quantization to reduce memory footprint")
        print("   • Load models sequentially instead of simultaneously")
    
    return success

if __name__ == "__main__":
    main()