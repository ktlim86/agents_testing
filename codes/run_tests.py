#!/usr/bin/env python3
"""
Test Runner for NPU Detection and LLM Testing
Runs both NPU detection and LLM inference tests
"""

import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def run_test_script(script_name, description):
    """Run a test script and return success status"""
    print_header(f"Running {description}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Test script not found: {script_path}")
        return False
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, text=True)
        
        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {description}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main test runner"""
    print_header("AMD NPU Testing Suite")
    
    print("🔍 This will run two tests:")
    print("   1. NPU Detection Test")
    print("   2. Simple LLM on NPU Test")
    print("\nPress Enter to continue, or Ctrl+C to exit...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        return False
    
    # Track test results
    results = {}
    
    # Run NPU detection test
    results['npu_detection'] = run_test_script(
        'test_npu_detection.py', 
        'NPU Detection Test'
    )
    
    # Run LLM test (even if NPU detection failed, for troubleshooting)
    results['llm_test'] = run_test_script(
        'test_llm_on_npu.py',
        'LLM on NPU Test'
    )
    
    # Final summary
    print_header("Final Test Summary")
    
    print("📊 Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    # Overall assessment
    all_passed = all(results.values())
    any_passed = any(results.values())
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   Your NPU is ready for agent testing framework")
        print("   Next: Integrate with DeepEval and run agent tests")
        
    elif any_passed:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print("   Some tests passed, some failed")
        print("   Check individual test outputs for details")
        
    else:
        print(f"\n❌ ALL TESTS FAILED")
        print("   NPU may not be properly configured")
        print("   Check RyzenAI v1.5.1 installation")
    
    print(f"\n📋 Troubleshooting:")
    print("   - Ensure RyzenAI v1.5.1 is properly installed")
    print("   - Check Windows Device Manager for NPU device")
    print("   - Verify ONNX Runtime and GenAI packages")
    print("   - Try running tests individually for more details")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)