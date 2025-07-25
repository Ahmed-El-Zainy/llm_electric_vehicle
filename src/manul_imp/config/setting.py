# config/settings.py
import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class DomainConfig:
    topic: str
    description: str

@dataclass
class DataSourceConfig:
    web_scraping: Dict[str, Any]
    pdf_sources: Dict[str, Any]

@dataclass
class ModelConfig:
    base_model: str
    max_length: int
    use_lora: bool
    lora_config: Dict[str, Any]

@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    save_steps: int
    eval_steps: int
    max_grad_norm: float
    fp16: bool

@dataclass
class DataProcessingConfig:
    min_text_length: int
    max_text_length: int
    similarity_threshold: float
    quality_threshold: float

@dataclass
class EvaluationConfig:
    metrics: List[str]
    benchmark_size: int
    test_split: float

@dataclass
class DeploymentConfig:
    model_name: str
    version: str
    max_concurrent_requests: int
    timeout_seconds: int
    port: int

@dataclass
class StorageConfig:
    data_dir: str
    model_dir: str
    logs_dir: str
    cache_dir: str

@dataclass
class LoggingConfig:
    level: str
    format: str

@dataclass
class MonitoringConfig:
    mlflow_tracking_uri: str
    experiment_name: str

class Settings:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._load_config()
        self._load_env_variables()
        self._create_directories()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.domain = DomainConfig(**config['domain'])
        self.use_case = config['use_case']
        self.data_sources = DataSourceConfig(**config['data_sources'])
        self.model = ModelConfig(**config['model'])
        self.training = TrainingConfig(**config['training'])
        self.data_processing = DataProcessingConfig(**config['data_processing'])
        self.evaluation = EvaluationConfig(**config['evaluation'])
        self.deployment = DeploymentConfig(**config['deployment'])
        self.storage = StorageConfig(**config['storage'])
        self.logging = LoggingConfig(**config['logging'])
        self.monitoring = MonitoringConfig(**config['monitoring'])
    
    def _load_env_variables(self):
        """Load environment variables with defaults"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        self.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', self.monitoring.mlflow_tracking_uri)
        self.cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        
        # Database settings
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///pipeline.db')
        
        # Storage settings
        self.s3_bucket = os.getenv('S3_BUCKET')
        self.s3_access_key = os.getenv('S3_ACCESS_KEY')
        self.s3_secret_key = os.getenv('S3_SECRET_KEY')
        
        # API settings
        self.api_key = os.getenv('API_KEY', 'default-api-key')
        self.allowed_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.storage.data_dir,
            self.storage.model_dir,
            self.storage.logs_dir,
            self.storage.cache_dir,
            f"{self.storage.data_dir}/raw",
            f"{self.storage.data_dir}/processed",
            f"{self.storage.data_dir}/datasets",
            f"{self.storage.model_dir}/checkpoints",
            f"{self.storage.model_dir}/final"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, checkpoint: Optional[str] = None) -> str:
        """Get model path for saving/loading"""
        if checkpoint:
            return f"{self.storage.model_dir}/checkpoints/{checkpoint}"
        return f"{self.storage.model_dir}/final/{self.deployment.model_name}"
    
    def get_data_path(self, data_type: str) -> str:
        """Get data path for different data types"""
        return f"{self.storage.data_dir}/{data_type}"
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        required_env_vars = []
        
        if not self.openai_api_key and self.data_sources.web_scraping['enabled']:
            required_env_vars.append('OPENAI_API_KEY')
        
        if not self.huggingface_token:
            required_env_vars.append('HUGGINGFACE_TOKEN')
        
        if required_env_vars:
            raise ValueError(f"Missing required environment variables: {required_env_vars}")
        
        return True

# Global settings instance
settings = Settings()