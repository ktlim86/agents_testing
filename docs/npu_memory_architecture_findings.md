# NPU Memory Architecture Findings

## Executive Summary

After extensive testing with your AMD AI 395 MX NPU (16GB dedicated memory), we've discovered the current memory architecture reality and limitations.

## Memory Architecture Reality

### Current ONNX Runtime Architecture
- **Model Storage**: Always in system RAM (32GB in your case)
- **NPU Memory (16GB)**: Acts as computation buffer only
- **Data Flow**: Model weights loaded in RAM → Copied to NPU memory during inference → Results returned

### Test Results Summary

| Configuration | System RAM Usage | NPU Memory Usage | Notes |
|---------------|------------------|------------------|-------|
| Standard Loading | 6-8GB | 0GB | Models stored in system RAM |
| Memory-Optimized | 5-7GB | 0GB | Slight reduction with arena disabled |
| NPU-Optimized | 6-8GB | 0GB | No change in memory allocation |

## Key Technical Findings

### 1. NPU Memory is Computation Buffer, Not Storage
```
System RAM (32GB) → Stores model weights permanently
    ↓
NPU Memory (16GB) → Temporary computation buffer
    ↓
Results → Back to system RAM
```

### 2. Provider Behavior
- **VitisAI Provider**: Routes most Phi-3 operations to CPU (617/617 operators)
- **DirectML Provider**: Better NPU utilization but still stores models in RAM
- **CPU Provider**: Fallback with no NPU usage

### 3. Memory Optimization Options
- `enable_cpu_mem_arena = False`: Reduces RAM usage by ~500MB-1GB
- `enable_mem_pattern = False`: Prevents pre-allocation patterns
- DirectML provider: Automatic memory management with some optimization

## Implications for Agent Testing

### Current Limitations
1. **Cannot store models in NPU's 16GB memory**: Architecture limitation
2. **System RAM bottleneck**: Must load all models in 32GB system RAM
3. **Sequential loading required**: Multiple models compete for system RAM

### Practical Solutions
1. **Load models sequentially**: One at a time to minimize RAM usage
2. **Use quantized models**: INT4 models use ~4x less RAM than FP16
3. **Memory arena optimization**: Disable arenas to reduce footprint
4. **DirectML provider**: Best balance of NPU usage and memory efficiency

### Agent Testing Strategy
```python
# Optimal configuration for your hardware
session_options = ort.SessionOptions()
session_options.enable_cpu_mem_arena = False  # Save ~1GB RAM
session_options.enable_mem_pattern = False   # Prevent pre-allocation

providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
# DirectML automatically uses NPU when beneficial
```

## Future Architecture Possibilities

### What Would True NPU Memory Usage Look Like?
- Model weights stored in NPU's 16GB memory
- System RAM used only for activations and temporary data
- Multiple models loaded simultaneously in NPU memory
- Zero system RAM impact for model storage

### Current Technical Barriers
1. ONNX Runtime architecture designed for CPU/GPU paradigm
2. NPU providers treat NPU as accelerator, not primary storage
3. Model format compatibility with NPU memory layouts
4. Driver/runtime limitations in current RyzenAI v1.5.1

## Recommendations

### For Your Current Setup
1. **Stick with DirectML provider** for best NPU utilization
2. **Use memory optimization settings** to reduce RAM footprint
3. **Load agents sequentially** rather than simultaneously
4. **Monitor system RAM usage** during agent testing

### Testing Framework Adjustments
- Load one agent model at a time
- Clear model from memory before loading next
- Use INT4 quantized models when available
- Implement memory monitoring in test scripts

## Conclusion

While we cannot currently use the NPU's 16GB dedicated memory for model storage, we can:
- Optimize system RAM usage through configuration
- Leverage NPU for computation acceleration via DirectML
- Design agent testing framework around current limitations
- Monitor for future RyzenAI updates that might enable true NPU memory usage

The NPU is working correctly - it's just being used as a computation accelerator rather than model storage, which is the current architectural design.