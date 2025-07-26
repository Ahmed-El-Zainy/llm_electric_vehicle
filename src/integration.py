import os 
import sys

# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Try to import custom logger, fallback to standard logging
try:
    from logger.custom_logger import CustomLoggerTracker
    logger_tracker = CustomLoggerTracker()
    logger = logger_tracker.get_logger("main")
    logger.info("Custom logger initialized")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("main")
    logger.info("Using standard logger - custom logger not available")



def create_directory_structure(project_path: Path):
    """Create the complete directory structure."""
    directories = [
        "config",
        "src/config",
        "src/data", 
        "src/training",
        "src/evaluation",
        "src/deployment",
        "src/orchestration",
        "src/utils",
        "data/raw",
        "data/processed", 
        "data/synthetic",
        "data/benchmarks",
        "models/base",
        "models/fine_tuned",
        "models/checkpoints",
        "models/registry",
        "logs",
        "outputs/evaluations",
        "outputs/reports",
        "scripts",
        "tests",
        "docker",
        "docs"
    ]
    
    for directory in directories:
        dir_path = project_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith('src/'):
            init_file = dir_path / '__init__.py'
            init_file.write_text('"""LLM Pipeline module."""\n')

def create_main_files(project_path: Path):
    """Create main application files."""
    
    # main.py - copy from our existing main pipeline file
    main_content = '''#!/usr/bin/env python3
"""
End-to-End Small Language Model Fine-tuning and Serving Pipeline
Based on Meta Synthetic Data Kit and Unsloth for domain-specific QA systems
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import pipeline modules
from src.config.config_manager import ConfigManager
from src.data.collector import DataCollector
from src.data.processor import DataProcessor
from src.training.synthetic_data_generator import SyntheticDataGenerator
from src.training.fine_tuner import FineTuner
from src.evaluation.evaluator import ModelEvaluator
from src.deployment.server import ModelServer
from src.orchestration.pipeline_manager import PipelineManager
from src.utils.logging_utils import setup_logging
from src.utils.monitoring import PipelineMonitor

class LLMPipeline:
    """Main pipeline orchestrator for end-to-end LLM fine-tuning and deployment."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logging(self.config.get('logging', {}))
        self.monitor = PipelineMonitor(self.config.get('monitoring', {}))
        
        # Initialize pipeline components
        self.data_collector = DataCollector(self.config.get('data_collection', {}))
        self.data_processor = DataProcessor(self.config.get('data_processing', {}))
        self.synthetic_generator = SyntheticDataGenerator(self.config.get('synthetic_data', {}))
        self.fine_tuner = FineTuner(self.config.get('fine_tuning', {}))
        self.evaluator = ModelEvaluator(self.config.get('evaluation', {}))
        self.server = ModelServer(self.config.get('deployment', {}))
        self.pipeline_manager = PipelineManager(self.config.get('orchestration', {}))
        
        self.logger.info("LLM Pipeline initialized successfully")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete end-to-end pipeline."""
        self.logger.info("Starting full pipeline execution...")
        
        try:
            with self.monitor.track_pipeline():
                # Step 1: Data Collection
                self.logger.info("Stage 1/6: Data Collection")
                raw_data = self.run_data_collection()
                
                # Step 2: Data Processing
                self.logger.info("Stage 2/6: Data Processing")
                processed_data = self.run_data_processing(raw_data)
                
                # Step 3: Synthetic Data Generation
                self.logger.info("Stage 3/6: Synthetic Data Generation")
                training_dataset = self.run_synthetic_data_generation(processed_data)
                
                # Step 4: Fine-tuning
                self.logger.info("Stage 4/6: Fine-tuning")
                model_info = self.run_fine_tuning(training_dataset)
                
                # Step 5: Evaluation
                self.logger.info("Stage 5/6: Evaluation")
                evaluation_results = self.run_evaluation(model_info)
                
                # Step 6: Deployment
                self.logger.info("Stage 6/6: Deployment")
                deployment_info = self.run_deployment(model_info)
                
                # Pipeline results
                pipeline_results = {
                    'status': 'success',
                    'data_stats': {
                        'raw_sources': raw_data['metadata']['sources_count'],
                        'processed_chunks': len(processed_data['chunks']),
                        'qa_pairs': len(training_dataset)
                    },
                    'model_info': model_info,
                    'evaluation_results': evaluation_results,
                    'deployment_info': deployment_info,
                    'execution_time': self.monitor.get_total_execution_time(),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save pipeline results
                self.pipeline_manager.save_pipeline_results(pipeline_results)
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        
        self.logger.info("Full pipeline execution completed successfully")
        return pipeline_results
    
    def run_data_collection(self) -> Dict[str, Any]:
        """Step 1: Collect domain-specific data from various sources."""
        with self.monitor.track_stage("data_collection"):
            # Implementation here - call actual methods
            pass
    
    def run_data_processing(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Clean, deduplicate, and process collected data."""
        with self.monitor.track_stage("data_processing"):
            # Implementation here
            pass
    
    def run_synthetic_data_generation(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate synthetic QA pairs using Meta's synthetic-data-kit."""
        with self.monitor.track_stage("synthetic_data_generation"):
            # Implementation here
            pass
    
    def run_fine_tuning(self, training_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Fine-tune the model using Unsloth with LoRA/QLoRA."""
        with self.monitor.track_stage("fine_tuning"):
            # Implementation here
            pass
    
    def run_evaluation(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Evaluate the fine-tuned model against baseline."""
        with self.monitor.track_stage("evaluation"):
            # Implementation here
            pass
    
    def run_deployment(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Deploy the model for production serving."""
        with self.monitor.track_stage("deployment"):
            # Implementation here
            pass

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    parser.add_argument("--stage", "-s", choices=[
        "data_collection", "data_processing", "synthetic_data", 
        "fine_tuning", "evaluation", "deployment", "full"
    ], default="full", help="Pipeline stage to run")
    parser.add_argument("--resume", "-r", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LLMPipeline(args.config)
    
    try:
        if args.stage == "full":
            results = pipeline.run_full_pipeline()
        else:
            # Run individual stages
            stage_methods = {
                "data_collection": pipeline.run_data_collection,
                "data_processing": pipeline.run_data_processing,
                "synthetic_data": pipeline.run_synthetic_data_generation,
                "fine_tuning": pipeline.run_fine_tuning,
                "evaluation": pipeline.run_evaluation,
                "deployment": pipeline.run_deployment
            }
            
            if args.stage in stage_methods:
                results = stage_methods[args.stage]()
            else:
                raise ValueError(f"Unknown stage: {args.stage}")
        
        print(f"✅ Pipeline stage '{args.stage}' completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(project_path / "main.py", "w") as f:
        f.write(main_content)
    
    # requirements.txt
    requirements_content = '''# Core ML Libraries
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.7.0
trl>=0.7.0
bitsandbytes>=0.41.0

# Unsloth and Synthetic Data Kit
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
synthetic-data-kit>=0.0.3
vllm>=0.3.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
nltk>=3.8.0
spacy>=3.7.0
beautifulsoup4>=4.12.0
requests>=2.31.0
PyPDF2>=3.0.0
pdfplumber>=0.9.0
python-docx>=0.8.11

# Evaluation Metrics
rouge-score>=0.1.2
sacrebleu>=2.3.0
bert-score>=0.3.13
evaluate>=0.4.0

# Web Framework & API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0
starlette>=0.27.0

# Database & Storage
sqlalchemy>=2.0.0
redis>=5.0.0

# Monitoring & Logging
prometheus-client>=0.18.0
wandb>=0.16.0
loguru>=0.7.0

# Utilities
pyyaml>=6.0.0
python-dotenv>=1.0.0
click>=8.1.0
tqdm>=4.66.0
rich>=13.6.0
typer>=0.9.0
PyJWT>=2.8.0
psutil>=5.9.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.25.0

# Development
black>=23.9.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.6.0
pre-commit>=3.5.0
'''
    
    with open(project_path / "requirements.txt", "w") as f:
        f.write(requirements_content)

def create_source_modules(project_path: Path):
    """Create all source module files."""
    
    # We'll create simplified versions that import from our existing artifacts
    modules = {
        "src/config/config_manager.py": '''"""Configuration management module."""
from src.utils.orchestration import ConfigManager
''',
        
        "src/data/collector.py": '''"""Data collection module."""
# Import the complete DataCollector implementation
import os
import re
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from datetime import datetime
import hashlib

# [Include the complete DataCollector class from the provided paste.txt]
# This would be the full implementation from the data collector artifact

class DataCollector:
    """Data collection implementation - see full code in artifacts."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Full implementation would go here
        pass
''',
        
        "src/data/processor.py": '''"""Data processing module."""
from src.utils.orchestration import DataProcessor
''',
        
        "src/training/synthetic_data_generator.py": '''"""Synthetic data generation module."""
# Full implementation from synthetic_data_generator artifact
import os
import json
import time
import yaml
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset
import pandas as pd

class SyntheticDataGenerator:
    """Synthetic data generation using Meta's SDK."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Full implementation would go here
        pass
''',
        
        "src/training/fine_tuner.py": '''"""Fine-tuning module with Unsloth."""
# Full implementation from fine_tuner artifact
import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datasets import Dataset

class FineTuner:
    """Fine-tuning with Unsloth and LoRA."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Full implementation would go here
        pass
''',
        
        "src/evaluation/evaluator.py": '''"""Model evaluation module."""
# Full implementation from evaluator_deployment artifact
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

class ModelEvaluator:
    """Model evaluation with multiple metrics."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Full implementation would go here
        pass
''',
        
        "src/deployment/server.py": '''"""Model serving and deployment."""
# Full implementation from evaluator_deployment artifact
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ModelServer:
    """Production model serving with FastAPI."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Full implementation would go here
        pass
''',
        
        "src/orchestration/pipeline_manager.py": '''"""Pipeline orchestration."""
from src.utils.orchestration import PipelineManager
''',
        
        "src/utils/logging_utils.py": '''"""Logging utilities."""
from src.utils.orchestration import setup_logging
''',
        
        "src/utils/monitoring.py": '''"""Monitoring utilities."""
from src.utils.orchestration import PipelineMonitor
''',
        
        "src/utils/orchestration.py": '''"""Complete orchestration utilities."""
# This would contain the full orchestration_utils content
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import yaml

# Full implementation from orchestration_utils artifact would go here
class ConfigManager:
    def __init__(self, config_path: str):
        pass

class PipelineManager:
    def __init__(self, config: Dict[str, Any]):
        pass

class PipelineMonitor:
    def __init__(self, config: Dict[str, Any]):
        pass

def setup_logging(config: Dict[str, Any]):
    pass
'''
    }
    
    for module_path, content in modules.items():
        file_path = project_path / module_path
        with open(file_path, "w") as f:
            f.write(content)

def create_config_files(project_path: Path):
    """Create configuration files."""
    
    # Main config
    config_content = '''# LLM Fine-tuning Pipeline Configuration

domain_topic: "electric vehicle charging stations"
use_case: "qa"

environment:
  name: "llm_pipeline"
  gpu_enabled: true
  mixed_precision: true

data_collection:
  web_sources:
    - "https://www.energy.gov/eere/electricvehicles"
    - "https://www.nrel.gov/transportation/"
  pdf_sources: []
  output_dir: "data/raw"
  max_pages_per_source: 10
  scraping_delay: 1.0

data_processing:
  cleaning:
    remove_html_tags: true
    normalize_whitespace: true
    min_text_length: 50
  deduplication:
    method: "fuzzy"
    threshold: 0.85
  quality_filtering:
    min_quality_score: 0.7
  chunking:
    max_chunk_size: 2048
    overlap: 64
  output_dir: "data/processed"

synthetic_data:
  base_model: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  temperature: 0.7
  top_p: 0.95
  pairs_per_chunk: 25
  max_generation_tokens: 512
  output_dir: "data/synthetic"

fine_tuning:
  base_model: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true
  
  lora_r: 16
  lora_alpha: 16
  lora_dropout: 0.0
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  
  training_params:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    warmup_steps: 5
    max_steps: 60
    learning_rate: 2.0e-4
    logging_steps: 1
    optim: "adamw_8bit"
    weight_decay: 0.01
    lr_scheduler_type: "linear"
    seed: 3407
    
  output_path: "models/fine_tuned"
  save_method: "lora"

evaluation:
  baseline_model: "unsloth/Llama-3.2-3B-Instruct"
  benchmark_size: 50
  metrics:
    - "rouge"
    - "bleu"
    - "bert_score"
  output_dir: "outputs/evaluations"

deployment:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  version_tag: "latest"
  authentication: true
  metrics_port: 9090
  model_registry_path: "models/registry"

orchestration:
  checkpoints_dir: "models/checkpoints"
  enable_resume: true
  save_intermediate: true
  cleanup_temp: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"

monitoring:
  enabled: true
  metrics_backend: "prometheus"
  alerts_enabled: true
'''
    
    with open(project_path / "config/config.yaml", "w") as f:
        f.write(config_content)
    
    # Environment template
    env_content = '''# Environment Variables Template
# Copy this file to environment.env and fill in your values

# API Keys
HUGGINGFACE_TOKEN=hf_your_token_here
OPENAI_API_KEY=sk_your_openai_key_here
WANDB_API_KEY=your_wandb_key_here

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
API_TOKEN=your_secure_api_token_here

# Model Configuration
BASE_MODEL_PATH=/models/base
CACHE_DIR=/cache/huggingface

# Resource Limits
MAX_MEMORY_GB=16
MAX_GPU_MEMORY_GB=12
MAX_WORKERS=4

# Feature Flags
ENABLE_DISTRIBUTED_TRAINING=false
ENABLE_MIXED_PRECISION=true
ENABLE_GRADIENT_CHECKPOINTING=true
'''
    
    with open(project_path / "config/environment.env.example", "w") as f:
        f.write(env_content)

def create_docker_setup(project_path: Path):
    """Create Docker configuration files."""
    
    # Dockerfile
    dockerfile_content = '''FROM nvidia/cuda:12.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3.10-dev \\
    python3-pip \\
    git \\
    wget \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/{raw,processed,synthetic,benchmarks} \\
    models/{base,fine_tuned,checkpoints,registry} \\
    logs \\
    outputs/{evaluations,reports}

EXPOSE 8000 9090 3000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py", "--config", "config/config.yaml", "--stage", "full"]
'''
    
    with open(project_path / "docker/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # docker-compose.yml
    compose_content = '''version: '3.8'

services:
  llm-pipeline:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: llm-pipeline
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./outputs:/app/outputs
      - ./config:/app/config
    ports:
      - "8000:8000"
      - "9090:9090"
      - "3000:3000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - API_TOKEN=${API_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    container_name: llm-pipeline-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
'''
    
    with open(project_path / "docker-compose.yml", "w") as f:
        f.write(compose_content)

def create_documentation(project_path: Path):
    """Create documentation files."""
    
    # README.md (comprehensive)
    readme_content = '''# LLM Fine-tuning Pipeline

An end-to-end pipeline for collecting domain-specific data, fine-tuning small language models, and deploying them for production use. Built with Meta's Synthetic Data Kit and Unsloth for memory-efficient training.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd llm-finetuning-pipeline
pip install -r requirements.txt

# 2. Configure environment
cp config/environment.env.example config/environment.env
# Edit environment.env with your API keys

# 3. Configure for your domain
# Edit config/config.yaml with your domain topic and data sources

# 4. Run full pipeline
python main.py --config config/config.yaml --stage full

# 5. Deploy model
python main.py --config config/config.yaml --stage deployment
```

## 📋 Features

- **Data Collection**: Web scraping, PDF extraction, metadata attribution
- **Data Processing**: Cleaning, deduplication, quality filtering
- **Synthetic Data**: Meta's Synthetic Data Kit for QA generation  
- **Memory-Efficient Training**: Unsloth with LoRA/QLoRA
- **Evaluation**: ROUGE, BLEU, BERT Score metrics
- **Production Deployment**: FastAPI server with authentication
- **MLOps**: Monitoring, checkpointing, orchestration

## 🏗️ Architecture

```
Data Collection → Data Processing → Synthetic Data → Fine-tuning → Evaluation → Deployment
```

## 📁 Project Structure

```
llm-finetuning-pipeline/
├── main.py                 # Main pipeline orchestrator
├── requirements.txt        # Dependencies
├── config/                 # Configuration files
├── src/                    # Source code modules
├── data/                   # Data storage
├── models/                 # Model storage
├── docker/                 # Docker configuration
└── docs/                   # Documentation
```

## ⚙️ Configuration

Configure your domain in `config/config.yaml`:

```yaml
domain_topic: "your domain here"
data_collection:
  web_sources:
    - "https://example.com"
  pdf_sources:
    - "data/raw/document.pdf"
```

## 🔄 Pipeline Stages

1. **Data Collection**: Scrape web sources and extract PDF content
2. **Data Processing**: Clean, deduplicate, and chunk data
3. **Synthetic Data**: Generate QA pairs using Meta's SDK
4. **Fine-tuning**: Train with Unsloth LoRA optimizations
5. **Evaluation**: Compare against baseline with multiple metrics
6. **Deployment**: Serve model with FastAPI

## 🧪 Usage Examples

### Run Full Pipeline
```bash
python main.py --config config/config.yaml --stage full
```

### Run Individual Stages
```bash
python main.py --config config/config.yaml --stage data_collection
python main.py --config config/config.yaml --stage fine_tuning
```

### Resume from Checkpoint
```bash
python main.py --config config/config.yaml --stage fine_tuning --resume data_processing
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual components
docker build -t llm-pipeline .
docker run --gpus all -p 8000:8000 llm-pipeline
```

## 🌐 API Usage

Once deployed:

```python
import requests

response = requests.post(
    "http://localhost:8000/inference",
    headers={"Authorization": "Bearer your_token"},
    json={
        "prompt": "Your question here?",
        "max_tokens": 256
    }
)
```

## 📊 Monitoring

- Health check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Logs: `logs/pipeline.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Meta AI for Synthetic Data Kit
- Unsloth for training optimizations
- Hugging Face for transformers ecosystem
'''
    
    with open(project_path / "README.md", "w") as f:
        f.write(readme_content)

def create_test_suite(project_path: Path):
    """Create test suite."""
    
    test_content = '''"""Test suite for LLM Pipeline."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that core modules can be imported."""
    try:
        from src.config.config_manager import ConfigManager
        from src.data.collector import DataCollector
        from src.training.fine_tuner import FineTuner
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config_loading():
    """Test configuration loading."""
    from src.config.config_manager import ConfigManager
    
    # Test with example config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists():
        config = ConfigManager(str(config_path))
        assert config.get("domain_topic") is not None

def test_data_collector():
    """Test data collector initialization."""
    from src.data.collector import DataCollector
    
    config = {
        'output_dir': 'data/raw',
        'web_sources': [],
        'pdf_sources': []
    }
    
    collector = DataCollector(config)
    assert hasattr(collector, 'config')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    with open(project_path / "tests/test_pipeline.py", "w") as f:
        f.write(test_content)

def create_utility_scripts(project_path: Path):
    """Create utility scripts."""
    
    # Setup script
    setup_script = '''#!/bin/bash

echo "🚀 Setting up LLM Fine-tuning Pipeline..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,synthetic,benchmarks}
mkdir -p models/{base,fine_tuned,checkpoints,registry}
mkdir -p logs outputs/{evaluations,reports}

# Copy config template
echo "⚙️ Setting up configuration..."
if [ ! -f config/environment.env ]; then
    cp config/environment.env.example config/environment.env
    echo "✏️ Please edit config/environment.env with your API keys"
fi

echo "✅ Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. source venv/bin/activate"
echo "2. Edit config/environment.env with your API keys"
echo "3. Edit config/config.yaml with your domain"
echo "4. python main.py --config config/config.yaml --stage full"
'''
    
    setup_path = project_path / "scripts/setup.sh"
    with open(setup_path, "w") as f:
        f.write(setup_script)
    
"""
Complete Integration Script for LLM Fine-tuning Pipeline
This script creates the entire project structure and integrates all components.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any

def create_complete_pipeline_project(project_name: str = "llm-finetuning-pipeline"):
    """Create the complete LLM fine-tuning pipeline project."""
    
    print("🚀 Creating Complete LLM Fine-tuning Pipeline Project")
    print("=" * 60)
    
    project_path = Path(project_name)
    if project_path.exists():
        response = input(f"Directory {project_name} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return
        shutil.rmtree(project_path)
    
    project_path.mkdir()
    
    # Step 1: Create directory structure
    print("📁 Creating directory structure...")
    create_directory_structure(project_path)
    
    # Step 2: Create main application files
    print("📝 Creating main application files...")
    create_main_files(project_path)
    
    # Step 3: Create source modules
    print("🔧 Creating source modules...")
    create_source_modules(project_path)
    
    # Step 4: Create configuration files
    print("⚙️ Creating configuration files...")
    create_config_files(project_path)
    
    # Step 5: Create Docker setup
    print("🐳 Creating Docker setup...")
    create_docker_setup(project_path)
    
    # Step 6: Create documentation
    print("📚 Creating documentation...")
    create_documentation(project_path)
    
    # Step 7: Create tests
    print("🧪 Creating test suite...")
    create_test_suite(project_path)
    
    # Step 8: Create utility scripts
    print("🛠️ Creating utility scripts...")
    create_utility_scripts(project_path)
    
    print("\n✅ Project created successfully!")
    print_next_steps(project_name)

def create_directory_structure(project_path: Path):
    """Create the complete directory structure."""