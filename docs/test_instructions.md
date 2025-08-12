# NPU Testing Instructions

## Quick Start

Run the comprehensive test suite:

```bash
cd agents_testing/codes
python run_tests.py
```

Or run individual tests:

```bash
# Test 1: NPU Detection
python test_npu_detection.py

# Test 2: LLM on NPU
python test_llm_on_npu.py
```

## Test Overview

### Test 1: NPU Detection (`test_npu_detection.py`)

**Purpose**: Verify NPU hardware and software stack detection

**What it checks:**
- ✅ Windows OS compatibility
- ✅ ONNX Runtime installation and version
- ✅ VitisAI Execution Provider (NPU support)
- ✅ DirectML Execution Provider (GPU support)
- ✅ ONNX Runtime GenAI availability
- ✅ RyzenAI installation detection
- ✅ Memory information (32GB RAM expected)
- ✅ NPU session configuration

**Expected Results:**
- All checks should pass if RyzenAI v1.5.1 is properly installed
- NPU Provider should show "✅ YES"
- Memory should show ~32GB total RAM

### Test 2: LLM on NPU (`test_llm_on_npu.py`)

**Purpose**: Test actual LLM inference on NPU hardware

**What it checks:**
- ✅ Prerequisites (ONNX Runtime + GenAI)
- ✅ NPU-optimized model detection
- ✅ Model loading with NPU provider
- ✅ Tokenizer functionality
- ✅ Simple text generation
- ✅ Performance benchmarking

**Expected Results:**
- Model loading may fail if no NPU models are available
- Provider configuration should succeed
- If model available: 25-60 tokens/second generation speed

## Prerequisites

### Required Packages
```bash
pip install onnxruntime onnxruntime-genai psutil
```

### Optional: Install Missing Dependencies
```bash
# If tests show missing packages
pip install -r ../requirements.txt
```

## Interpreting Results

### ✅ All Tests Pass
- **Meaning**: NPU is fully functional
- **Next Step**: Ready for DeepEval integration
- **Action**: Proceed with agent testing framework

### ⚠️ NPU Detection Passes, LLM Test Fails
- **Meaning**: NPU hardware detected but no models available
- **Next Step**: Download NPU-optimized models
- **Action**: Get models from RyzenAI model zoo or HuggingFace

### ❌ NPU Detection Fails
- **Meaning**: RyzenAI installation issue
- **Next Step**: Reinstall RyzenAI v1.5.1
- **Action**: Check Windows Device Manager for NPU device

## Troubleshooting

### Common Issues

**1. "VitisAIExecutionProvider not found"**
```bash
# Solution: Reinstall RyzenAI
# Download from: AMD Developer Portal
# Install: ryzen-ai-1.5.1.msi
```

**2. "onnxruntime_genai not found"**
```bash
# Solution: Install GenAI package
pip install onnxruntime-genai
```

**3. "Model not found"**
```bash
# Solution: Download NPU model
# Option 1: Use RyzenAI model zoo
# Option 2: Download from HuggingFace
# Place in: ./models/npu/ directory
```

**4. Memory allocation issues**
```bash
# Check memory usage
# Expected: 32GB RAM + 16GB NPU + 16GB GPU
# If different, adjust model sizes accordingly
```

### Verification Commands

**Manual NPU check:**
```python
import onnxruntime as ort
print("NPU Available:", "VitisAIExecutionProvider" in ort.get_available_providers())
```

**Manual GenAI check:**
```python
import onnxruntime_genai as og
print("GenAI Version:", og.__version__)
```

**Device Manager check:**
1. Open Device Manager (Windows)
2. Look under "System Devices" 
3. Find "AMD NPU Device" or similar

## Next Steps After Testing

### If Tests Pass
1. **Download NPU Models**: Get phi-3-mini-4k-int4 or similar
2. **Test DeepEval Integration**: Create simple evaluation test
3. **Run Agent Tests**: Start with single agent evaluation
4. **Scale Up**: Test full 38-agent suite

### If Tests Fail
1. **Check Installation**: Verify RyzenAI v1.5.1 installed correctly
2. **Update Drivers**: Ensure latest AMD chipset drivers
3. **Restart System**: NPU drivers may need system restart
4. **Contact Support**: Check AMD RyzenAI documentation

## Performance Expectations

### Successful NPU Setup Should Show:
- **NPU Detection**: ✅ VitisAIExecutionProvider available
- **Model Loading**: ✅ Models load in 2-5 seconds
- **Inference Speed**: 25-60 tokens/second depending on model
- **Memory Usage**: <8GB NPU memory for small models
- **Power Efficiency**: Lower power than CPU-only inference

### Benchmark Targets:
- **Single Agent Test**: 30-45 seconds
- **Domain Test (5 agents)**: 3-8 minutes
- **Full Suite (38 agents)**: 20-30 minutes

This represents excellent performance for local, private LLM evaluation without any API costs.

## Support Resources

- **AMD RyzenAI Documentation**: https://ryzenai.docs.amd.com/
- **ONNX Runtime GenAI**: https://github.com/microsoft/onnxruntime-genai
- **Model Downloads**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
- **Issue Reporting**: Create issue in this repository with test output