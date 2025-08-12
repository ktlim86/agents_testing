# Agent Testing Implementation Guide

## Quick Start

### 1. Environment Setup

#### Option A: AMD NPU Setup (Recommended for AMD AI 395 MX)
```bash
# Install AMD GAIA for NPU support
# Download from: https://github.com/amd/gaia
git clone https://github.com/amd/gaia.git
cd gaia
pip install -r requirements.txt

# Configure NPU + iGPU hybrid mode
python setup.py install
```

#### Option B: Ollama GPU Setup (Fallback)
```bash
# Install Ollama (will use Radeon 8060S GPU, not NPU)
curl -fsSL https://ollama.ai/install.sh | sh

# Download lightweight models for shared memory
ollama pull phi3.5:mini      # 2-4GB VRAM
ollama pull gemma2:9b        # 6-8GB VRAM

# Verify installation
ollama list
```

#### Option B: OpenAI Setup (Paid)
```bash
# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

#### Common Setup
```bash
# Navigate to agents_testing directory
cd agents_testing

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
# Using Local LLM (Ollama)
python codes/test_runner.py --agent data-scientist --category correctness --model ollama/llama3.1:8b

# Using OpenAI
python codes/test_runner.py --agent data-scientist --category correctness --model gpt-4

# Run all tests for a domain
python codes/test_runner.py --domain data_science --model ollama/gemma2:9b

# Run full evaluation suite
python codes/test_runner.py --all --model ollama/llama3.1:8b
```

## Testing Categories Configuration

### Correctness Testing
```python
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel

# Local LLM Configuration
ollama_model = OllamaModel(model="llama3.1:8b")

correctness_metrics = [
    AnswerRelevancyMetric(threshold=0.8, model=ollama_model),
    FaithfulnessMetric(threshold=0.8, model=ollama_model),
    # Custom domain-specific metrics
]
```

### Safety Testing
```python
from deepeval.metrics import BiasMetric, ToxicityMetric

safety_metrics = [
    BiasMetric(threshold=0.9),
    ToxicityMetric(threshold=0.9),
    # Custom safety checks
]
```

### Performance Testing
```python
performance_metrics = [
    # Response time, quality, efficiency metrics
    CoherenceMetric(threshold=0.7),
    ConcisenessMetric(threshold=0.7),
]
```

### Usability Testing
```python
usability_metrics = [
    # User experience, clarity metrics
    ReadabilityMetric(threshold=0.8),
    HelpfulnessMetric(threshold=0.8),
]
```

## Test Scenarios Structure

Each domain has specific test scenarios in `codes/scenarios/`:

```
scenarios/
├── engineering/
│   ├── code_review_scenarios.json
│   ├── architecture_scenarios.json
│   └── debugging_scenarios.json
├── data_science/
│   ├── modeling_scenarios.json
│   ├── analysis_scenarios.json
│   └── validation_scenarios.json
└── [other_domains]/
```

### Scenario Format
```json
{
  "scenario_id": "eng_001",
  "domain": "engineering",
  "agent": "backend-developer",
  "category": "correctness",
  "input": "Review this Python code for performance issues: [code]",
  "expected_criteria": [
    "Identifies performance bottlenecks",
    "Suggests specific optimizations",
    "Explains reasoning clearly"
  ],
  "difficulty": "medium"
}
```

## Custom Metrics for Each Domain

### Engineering Domain
```python
from deepeval import evaluate
from deepeval.metrics import GEval

code_quality_metric = GEval(
    name="Code Quality",
    criteria="Evaluate the technical accuracy and best practices adherence",
    evaluation_steps=[
        "Check for security vulnerabilities",
        "Assess performance considerations", 
        "Verify coding standards compliance"
    ],
    threshold=0.8
)
```

### Data Science Domain
```python
statistical_accuracy_metric = GEval(
    name="Statistical Accuracy",
    criteria="Evaluate the correctness of statistical methods and approaches",
    evaluation_steps=[
        "Verify statistical methodology",
        "Check for data science best practices",
        "Assess model validation approaches"
    ],
    threshold=0.8
)
```

## Integration with Human Resource Specialist

### Automated Improvement Trigger
```python
def check_improvement_needed(test_results):
    """Trigger HR specialist when agent performance degrades"""
    failed_tests = [r for r in test_results if r.score < threshold]
    
    if len(failed_tests) > improvement_threshold:
        trigger_hr_specialist_improvement(
            agent=test_results.agent,
            failed_categories=failed_tests,
            improvement_suggestions=analyze_failures(failed_tests)
        )
```

### Results Integration
```python
def update_agent_history(agent_name, test_results):
    """Update improvement history with test results"""
    improvement_data = {
        "test_date": datetime.now(),
        "scores": test_results.scores,
        "improvements_needed": test_results.failed_categories,
        "validation_status": "requires_improvement" if test_results.failed else "passed"
    }
    
    update_history_file(agent_name, improvement_data)
```

## Report Generation

### Individual Agent Report
```python
python codes/generate_report.py --agent data-scientist --output reports/
```

### Domain Summary Report
```python
python codes/generate_report.py --domain engineering --summary
```

### Full System Report
```python
python codes/generate_report.py --all --format html
```

## Hardware Requirements

### Minimum System Requirements
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for models
- **CPU**: Modern multi-core processor

### AMD AI 395 MX Specific Requirements
- **Shared Memory**: 64GB total (RAM + NPU + GPU)
- **NPU**: AMD AI 395 MX with hybrid NPU+iGPU support
- **GPU**: Radeon 8060S integrated graphics

### Recommended Models for Shared Memory Architecture
- **NPU-optimized**: 1-3B parameter models (GAIA)
- **GPU fallback**: phi3.5:mini (2-4GB), gemma2:9b (6-8GB)
- **Avoid**: Large 8B+ models due to shared memory constraints

### Model Selection for AMD AI 395 MX
```python
# Model selection optimized for shared memory architecture
MODEL_RECOMMENDATIONS = {
    "npu_optimized": "gaia/phi3-mini-int4",      # AMD GAIA NPU
    "gpu_efficient": "ollama/phi3.5:mini",      # 2-4GB shared
    "balanced": "ollama/gemma2:9b",             # 6-8GB shared
    "fallback_cpu": "ollama/phi3.5:mini"        # CPU-only mode
}

# Memory allocation strategy for 64GB shared
MEMORY_ALLOCATION = {
    "system_ram": "40GB",     # Windows + applications
    "model_cache": "16GB",    # Model weights
    "inference": "8GB"        # Active inference
}
```

## Configuration Files

### DeepEval Configuration
```python
# codes/config.py
DEEPEVAL_CONFIG = {
    # Local LLM Options (Recommended)
    "local_models": {
        "primary": "ollama/llama3.1:8b",      # Best overall
        "efficient": "ollama/gemma2:9b",      # Memory efficient
        "lightweight": "ollama/phi3.5:mini"  # Fastest
    },
    
    # OpenAI Options (Paid)
    "openai_models": {
        "primary": "gpt-4",
        "fallback": "gpt-3.5-turbo"
    },
    
    "default_model": "ollama/llama3.1:8b",  # Use local by default
    "threshold_correctness": 0.8,
    "threshold_safety": 0.9,
    "threshold_performance": 0.7,
    "threshold_usability": 0.8,
    "max_retries": 3,
    "timeout": 30
}

# Model Selection Helper
def get_model(preference="local"):
    if preference == "local":
        return DEEPEVAL_CONFIG["local_models"]["primary"]
    elif preference == "efficient":
        return DEEPEVAL_CONFIG["local_models"]["efficient"]
    else:
        return DEEPEVAL_CONFIG["openai_models"]["primary"]
```

### Test Environment Settings
```python
# codes/test_config.py
TEST_SETTINGS = {
    "batch_size": 5,  # Number of concurrent tests
    "output_format": "json",  # json, html, markdown
    "save_detailed_logs": True,
    "generate_improvement_suggestions": True
}
```

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Implement retry logic and batch processing
2. **Memory Usage**: Process agents in batches for large test suites
3. **Test Consistency**: Use fixed random seeds for reproducible results

### Debug Mode
```bash
python codes/test_runner.py --agent data-scientist --debug --verbose
```

## Next Steps

1. Run initial baseline tests across all agents
2. Analyze results and identify improvement areas
3. Use Human Resource Specialist for targeted improvements
4. Implement continuous testing pipeline
5. Monitor agent performance trends over time

This implementation provides a comprehensive, simple, and scalable testing framework focused on the 4 core evaluation categories using DeepEval.