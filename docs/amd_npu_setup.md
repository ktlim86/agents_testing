# AMD NPU Setup Guide for Agent Testing

## System Specifications
- **Hardware**: AMD AI 395 MX with NPU
- **GPU**: Radeon 8060S integrated graphics  
- **Memory**: 32GB RAM + 16GB NPU + 16GB GPU (dedicated allocations)
- **Software**: RyzenAI v1.5.1 (Pre-installed)
- **Architecture**: Hybrid NPU + iGPU + CPU

## Setup Options

### Option 1: RyzenAI v1.5.1 (Official NPU Support) ⭐ **RECOMMENDED**

**Why RyzenAI v1.5.1:**
- Official AMD NPU software stack (already installed)
- ONNX Runtime GenAI integration
- Hybrid NPU + iGPU execution modes
- Optimized for 16GB NPU allocation
- Direct Python API support

**Setup:**
```bash
# RyzenAI is already installed - verify NPU detection
python -c "import onnxruntime as ort; print([p.device_type for p in ort.get_available_providers()])"

# Check for VitisAI EP (NPU support)
python -c "import onnxruntime as ort; print('VitisAIExecutionProvider' in ort.get_available_providers())"

# Verify hybrid mode support
python -c "from ryzenai import get_npu_info; print(get_npu_info())"
```

**Configuration:**
```python
# GAIA configuration for DeepEval integration
from gaia import GaiaModel

# NPU-optimized model
gaia_model = GaiaModel(
    model_name="phi3-mini-int4",
    device="npu",
    hybrid_mode=True,  # Use NPU + iGPU
    memory_limit="16GB"  # Allocate for model
)
```

### Option 2: Ollama GPU Fallback

**If NPU setup fails:**
```bash
# Install Ollama (will use Radeon 8060S)
curl -fsSL https://ollama.ai/install.sh | sh

# Install memory-efficient models
ollama pull phi3.5:mini      # 2-4GB allocation
ollama pull gemma2:9b        # 6-8GB allocation (if memory allows)

# Test GPU detection
ollama run phi3.5:mini "Test message"
```

## Memory Management Strategy

### Optimal Memory Allocation (Dedicated Allocation)
```
├── System RAM: 32GB (Windows + applications + DeepEval)
├── NPU Memory: 16GB (Dedicated NPU model execution)
└── GPU Memory: 16GB (iGPU for hybrid mode)
```

### RyzenAI Execution Modes
```
├── Hybrid Mode: NPU (prefill) + iGPU (decode) - BEST PERFORMANCE
├── NPU-Only Mode: Pure NPU execution - POWER EFFICIENT  
└── GPU Fallback: iGPU-only execution - COMPATIBILITY
```

### Model Size Recommendations
| Model | NPU Support | Memory Usage | Quality | Speed | Mode |
|-------|-------------|--------------|---------|-------|-------|
| phi-3-mini-4k-int4 | ✅ RyzenAI | 2-4GB NPU | Good | Very Fast | Hybrid |
| llama-2-7b-int4 | ✅ RyzenAI | 8-12GB NPU | Better | Fast | Hybrid |
| mistral-7b-int4 | ✅ RyzenAI | 8-12GB NPU | Better | Fast | NPU-Only |
| phi3.5:mini | ❌ Ollama | 4GB GPU | Good | Medium | GPU |

## DeepEval Integration

### RyzenAI + DeepEval Configuration
```python
# agents_testing/codes/config.py
import onnxruntime as ort
from onnxruntime_genai import Model, Generator

# NPU-optimized configuration with RyzenAI
AMD_NPU_CONFIG = {
    "primary_model": {
        "model_path": "phi-3-mini-4k-int4-onnx",
        "providers": ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        "execution_mode": "hybrid",  # NPU + iGPU
        "npu_memory": "16GB"
    },
    "fallback_model": "ollama/phi3.5:mini",
    "batch_size": 3,     # Can handle more with dedicated NPU memory
    "timeout": 30,       # NPU is faster than expected
    "cache_enabled": True # VitisAI EP caching
}

# Use in DeepEval metrics
from deepeval.metrics import AnswerRelevancyMetric

correctness_metric = AnswerRelevancyMetric(
    threshold=0.8,
    model=AMD_NPU_CONFIG["primary_model"]
)
```

### Performance Optimization
```python
# Memory-conscious testing approach
def run_agent_test_amd_npu(agent_name, test_scenarios):
    # Process in smaller batches
    batch_size = 2
    results = []
    
    for i in range(0, len(test_scenarios), batch_size):
        batch = test_scenarios[i:i+batch_size]
        
        # Clear cache between batches
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        batch_results = evaluate_batch(batch)
        results.extend(batch_results)
        
        # Brief pause to prevent memory overflow
        time.sleep(1)
    
    return results
```

## Troubleshooting

### NPU Detection Issues
```bash
# Check RyzenAI NPU availability
python -c "import onnxruntime as ort; print('VitisAIExecutionProvider' in ort.get_available_providers())"

# Verify RyzenAI installation
where ryzen-ai

# Check NPU device in Windows
# Device Manager -> System Devices -> "AMD NPU Device"

# Test ONNX Runtime GenAI
python -c "import onnxruntime_genai; print(onnxruntime_genai.__version__)"
```

### Memory Issues
```bash
# Monitor memory usage
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value

# Clear model cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Performance Optimization
```python
# Reduce context window for shared memory
MAX_CONTEXT_LENGTH = 2048  # Instead of 4096
BATCH_SIZE = 1            # Process one at a time
CONCURRENT_TESTS = 1      # No parallel testing
```

## Expected Performance

### NPU Mode (GAIA)
- **Throughput**: 15-25 tokens/second
- **Memory**: 2-4GB model allocation
- **Power**: Low power consumption
- **Quality**: Good for evaluation tasks

### GPU Fallback (Ollama)
- **Throughput**: 10-20 tokens/second  
- **Memory**: 4-8GB allocation
- **Power**: Higher consumption
- **Quality**: Similar evaluation quality

## Testing Strategy Adjustments

### Modified Test Approach
1. **Smaller batches**: Process 1-2 agents at a time
2. **Sequential testing**: Avoid parallel execution
3. **Memory monitoring**: Check allocation between tests
4. **Fallback ready**: Switch to GPU if NPU fails

### Success Metrics
- **NPU utilization**: Monitor via Task Manager
- **Memory efficiency**: Stay under 80% allocation
- **Test completion**: All 38 agents tested successfully
- **Quality maintenance**: Evaluation scores remain consistent

This setup maximizes your AMD AI 395 MX capabilities while working within the shared memory constraints.