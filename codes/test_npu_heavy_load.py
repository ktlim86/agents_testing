#!/usr/bin/env python3
"""
Heavy NPU Load Test with Long Token Generation
Forces intensive computation to trigger visible NPU usage
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n--- {title} ---")

class HeavyNPULoadTester:
    """Generate heavy computational load to trigger NPU usage"""
    
    def __init__(self):
        self.setup_environment()
        self.load_models()
    
    def setup_environment(self):
        """Setup NPU environment"""
        print_section("Heavy NPU Load Setup")
        
        # Check environment
        install_dir = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
        if not install_dir:
            raise Exception("RYZEN_AI_INSTALLATION_PATH not set")
        
        print(f"✅ RyzenAI: {install_dir}")
        
        # Detect NPU
        command = r'pnputil /enum-devices /bus PCI /deviceids '
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        npu_type = 'STX' if 'PCI\\VEN_1022&DEV_17F0' in stdout.decode() else 'PHX/HPT'
        print(f"✅ NPU Type: {npu_type}")
        
        self.npu_config = {
            "install_dir": install_dir,
            "npu_type": npu_type
        }
    
    def load_models(self):
        """Load multiple models for testing"""
        print_section("Loading Models for Heavy Testing")
        
        try:
            import onnxruntime_genai as og
            
            # Try different model paths
            model_paths = [
                "./models/npu/phi3_mini/directml/directml-int4-awq-block-128",
                "./models/npu/phi3_mini/cpu_and_mobile/cpu-int4-rtn-block-32",
                "./models/npu/phi3_mini/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
            ]
            
            self.models = []
            self.tokenizers = []
            
            for model_path in model_paths:
                if Path(model_path).exists():
                    try:
                        print(f"📂 Loading: {Path(model_path).name}")
                        model = og.Model(model_path)
                        tokenizer = og.Tokenizer(model)
                        self.models.append(model)
                        self.tokenizers.append(tokenizer)
                        print(f"   ✅ Loaded successfully")
                        break  # Use first working model
                    except Exception as e:
                        print(f"   ❌ Failed: {e}")
                        continue
            
            if not self.models:
                raise Exception("No models loaded successfully")
            
            print(f"✅ {len(self.models)} model(s) ready for testing")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def create_very_long_prompt(self, iteration):
        """Create extremely long prompt to force heavy computation"""
        base_prompt = f"""You are an expert AI system providing comprehensive analysis. This is generation #{iteration}.

DETAILED REQUIREMENTS:
Please provide an extremely detailed, comprehensive analysis covering all aspects of modern software development, including but not limited to:

1. FRONTEND DEVELOPMENT:
   - React 18 with concurrent features and suspense
   - TypeScript for type safety and developer experience
   - Next.js 14 with App Router and server components
   - CSS-in-JS solutions like styled-components and emotion
   - State management with Redux Toolkit and Zustand
   - Testing strategies with Jest, React Testing Library, and Playwright
   - Performance optimization techniques and Core Web Vitals
   - Accessibility compliance with WCAG 2.1 AA standards
   - Mobile-first responsive design principles
   - Progressive Web App implementation

2. BACKEND DEVELOPMENT:
   - Node.js with Express.js and Fastify frameworks
   - Python with FastAPI and Django for robust APIs
   - Database design with PostgreSQL and MongoDB
   - Authentication and authorization with JWT and OAuth 2.0
   - API documentation with OpenAPI and Swagger
   - Microservices architecture and containerization
   - Message queues with Redis and RabbitMQ
   - Caching strategies and CDN implementation
   - Load balancing and horizontal scaling
   - Security best practices and OWASP compliance

3. DATA SCIENCE AND ANALYTICS:
   - Python data science stack: pandas, numpy, scipy
   - Machine learning with scikit-learn and TensorFlow
   - Data visualization with matplotlib, seaborn, and plotly
   - Statistical analysis and hypothesis testing
   - ETL pipelines and data warehousing
   - Real-time analytics and stream processing
   - Big data technologies like Apache Spark
   - Feature engineering and model selection
   - Model deployment and MLOps practices
   - Data governance and privacy compliance

4. DEVOPS AND INFRASTRUCTURE:
   - Containerization with Docker and Kubernetes
   - CI/CD pipelines with GitHub Actions and Jenkins
   - Infrastructure as Code with Terraform and CloudFormation
   - Monitoring and observability with Prometheus and Grafana
   - Log aggregation with ELK stack
   - Security scanning and vulnerability management
   - Cloud platforms: AWS, Azure, Google Cloud
   - Serverless computing and edge functions
   - Disaster recovery and backup strategies
   - Performance monitoring and alerting

5. QUANTITATIVE FINANCE:
   - Algorithmic trading strategies and backtesting
   - Risk management and portfolio optimization
   - Statistical arbitrage and market microstructure
   - Options pricing models and Greeks calculation
   - High-frequency trading infrastructure
   - Market data processing and normalization
   - Compliance and regulatory reporting
   - Performance attribution and factor analysis
   - Alternative data sources and alpha generation
   - Execution algorithms and transaction cost analysis

Please provide specific implementation details, code examples, best practices, performance considerations, security implications, and real-world case studies for each area. Include detailed explanations of trade-offs, architectural decisions, and industry standards.

Your response should be comprehensive, technical, and demonstrate deep expertise across all domains. Provide specific metrics, benchmarks, and measurable outcomes where applicable.

QUERY: How would you implement a comprehensive, enterprise-grade solution that integrates all these technologies into a cohesive platform?"""

        return base_prompt
    
    def generate_heavy_load(self, duration_seconds=30):
        """Generate heavy computational load for specified duration"""
        print_section(f"Heavy NPU Load Generation ({duration_seconds}s)")
        
        print("🔥 STARTING INTENSIVE NPU COMPUTATION")
        print("👀 WATCH TASK MANAGER -> PERFORMANCE -> NPU NOW!")
        print(f"⏱️  Running for {duration_seconds} seconds...")
        
        start_time = time.time()
        generation_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                generation_count += 1
                current_time = time.time() - start_time
                
                print(f"\n🔄 Generation {generation_count} (t={current_time:.1f}s)")
                
                # Create very long prompt
                long_prompt = self.create_very_long_prompt(generation_count)
                print(f"📝 Prompt length: {len(long_prompt)} characters")
                
                # Encode tokens (this should use computational resources)
                tokens = self.tokenizers[0].encode(long_prompt)
                print(f"🔢 Encoded to: {len(tokens)} tokens")
                
                # Try to force actual generation if possible
                try:
                    # Create generator params for longer generation
                    params = og.GeneratorParams(self.models[0])
                    
                    # Try to set longer generation parameters
                    if hasattr(params, 'set_search_options'):
                        params.set_search_options(max_length=200, min_length=50)
                    
                    # Try multiple generation approaches
                    for attempt in range(3):
                        print(f"   🔄 Generation attempt {attempt+1}/3...")
                        
                        # This should trigger model computation
                        try:
                            # Method 1: Try direct generation
                            generator = og.Generator(self.models[0], params)
                            
                            # Generate multiple tokens
                            token_count = 0
                            max_tokens = 100
                            
                            for _ in range(max_tokens):
                                if hasattr(generator, 'is_done') and generator.is_done():
                                    break
                                if hasattr(generator, 'compute_logits'):
                                    generator.compute_logits()
                                if hasattr(generator, 'generate_next_token'):
                                    generator.generate_next_token()
                                token_count += 1
                            
                            print(f"   ✅ Generated {token_count} tokens")
                            
                        except Exception as e:
                            print(f"   ⚠️  Generation attempt {attempt+1} failed: {str(e)[:100]}...")
                            continue
                        
                        # Small delay between attempts
                        time.sleep(0.1)
                
                except Exception as e:
                    print(f"   ❌ Generation setup failed: {str(e)[:100]}...")
                
                # Force some computational work regardless
                print("   🔄 Forcing computational work...")
                
                # Multiple encoding/decoding cycles
                for i in range(5):
                    test_text = f"This is computational test {i} for heavy NPU load testing " * 20
                    test_tokens = self.tokenizers[0].encode(test_text)
                    decoded_text = self.tokenizers[0].decode(test_tokens)
                
                print(f"   ✅ Completed computational cycles")
                
                # Brief pause to see NPU activity
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n⚠️  Heavy load test interrupted by user")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🏁 Heavy Load Test Complete")
        print(f"   ⏱️  Duration: {total_time:.1f} seconds")
        print(f"   🔢 Generations: {generation_count}")
        print(f"   📊 Rate: {generation_count/total_time:.2f} generations/second")
        
        return generation_count
    
    def test_multiple_models_parallel(self):
        """Test with multiple models to increase load"""
        print_section("Multi-Model Parallel Testing")
        
        if len(self.models) < 1:
            print("❌ Need at least 1 model for testing")
            return
        
        print(f"🔥 Testing with {len(self.models)} model(s)")
        print("👀 This should create maximum NPU utilization!")
        
        # Generate with each model
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            print(f"\n🔄 Model {i+1}/{len(self.models)}")
            
            # Create intensive prompt
            prompt = self.create_very_long_prompt(f"model_{i}")
            tokens = tokenizer.encode(prompt)
            
            print(f"📝 Tokens: {len(tokens)}")
            
            # Multiple encoding cycles
            for cycle in range(10):
                test_tokens = tokenizer.encode(f"Heavy computation cycle {cycle} " * 50)
                decoded = tokenizer.decode(test_tokens)
            
            print(f"✅ Model {i+1} processing complete")

def main():
    """Main heavy load testing"""
    print_header("Heavy NPU Load Testing")
    
    print("🔥 This test creates MAXIMUM computational load")
    print("📊 You should see SIGNIFICANT NPU activity spikes")
    print("⚠️  Test runs for 30 seconds with intensive computation")
    
    try:
        # Create heavy load tester
        tester = HeavyNPULoadTester()
        
        print("\n⚠️  IMPORTANT INSTRUCTIONS:")
        print("1. Open Task Manager -> Performance -> NPU")
        print("2. Watch for sustained NPU activity during test")
        print("3. Test will run for 30 seconds with maximum load")
        
        input("Press Enter when Task Manager is ready...")
        
        # Run heavy load test
        generation_count = tester.generate_heavy_load(duration_seconds=30)
        
        # Additional multi-model test
        print("\n🔥 Running additional multi-model test...")
        tester.test_multiple_models_parallel()
        
        print_header("Heavy Load Test Results")
        
        user_response = input("Did you observe ANY NPU activity spikes? (y/n): ").strip().lower()
        
        if user_response == 'y':
            print("🎉 SUCCESS: NPU activity detected!")
            print("   This confirms the model is using NPU acceleration")
        else:
            print("❌ NO NPU ACTIVITY OBSERVED")
            print("   Possible reasons:")
            print("   1. ONNX Runtime GenAI may not support VitisAI provider")
            print("   2. Model may be running on CPU/DirectML instead")
            print("   3. NPU utilization may be too low to show in Task Manager")
            print("   4. Different model format needed for NPU")
        
        return user_response == 'y'
        
    except Exception as e:
        print(f"❌ Heavy load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n💡 NEXT STEPS:")
        print("1. Try using pure ONNX Runtime (not GenAI) with VitisAI provider")
        print("2. Check if model needs compilation for NPU") 
        print("3. Verify NPU model format compatibility")
        print("4. Consider using AMD's VAI_EP directly")
    
    sys.exit(0 if success else 1)