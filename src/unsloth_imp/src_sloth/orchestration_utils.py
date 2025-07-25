def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _load_environment_variables(self):
        """Load environment variables and override config values."""
        env_file = self.config_path.parent / "environment.env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'data_collection.web_sources')."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        output_path = output_path or self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

# ============================================================================
# Pipeline Manager
# ============================================================================

@dataclass
class PipelineCheckpoint:
    """Data class for pipeline checkpoints."""
    stage: str
    timestamp: str
    status: str
    data_path: str
    metadata: Dict[str, Any]

class PipelineManager:
    """
    Pipeline orchestration and checkpoint management.
    Handles workflow automation, manual/scheduled triggers, and state management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.checkpoints_dir = Path(config.get('checkpoints_dir', 'models/checkpoints'))
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_resume = config.get('enable_resume', True)
        self.save_intermediate = config.get('save_intermediate', True)
        self.cleanup_temp = config.get('cleanup_temp', True)
        
        # Pipeline state
        self.current_stage = None
        self.pipeline_id = None
        self.start_time = None
        
    def start_pipeline(self, pipeline_id: Optional[str] = None) -> str:
        """Start a new pipeline execution."""
        if pipeline_id is None:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.pipeline_id = pipeline_id
        self.start_time = datetime.now()
        
        # Create pipeline directory
        pipeline_dir = self.checkpoints_dir / pipeline_id
        pipeline_dir.mkdir(exist_ok=True)
        
        # Save pipeline metadata
        metadata = {
            'pipeline_id': pipeline_id,
            'start_time': self.start_time.isoformat(),
            'status': 'started',
            'stages_completed': [],
            'current_stage': None
        }
        
        metadata_path = pipeline_dir / 'pipeline_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Pipeline started: {pipeline_id}")
        return pipeline_id
    
    def save_checkpoint(self, stage: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint for a pipeline stage."""
        if not self.save_intermediate:
            return ""
        
        checkpoint_id = f"{self.pipeline_id}_{stage}_{int(time.time())}"
        checkpoint_dir = self.checkpoints_dir / self.pipeline_id / stage
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = checkpoint_dir / f"{stage}_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Create checkpoint record
        checkpoint = PipelineCheckpoint(
            stage=stage,
            timestamp=datetime.now().isoformat(),
            status='completed',
            data_path=str(data_path),
            metadata=metadata or {}
        )
        
        # Save checkpoint metadata
        checkpoint_path = checkpoint_dir / 'checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)
        
        # Update pipeline metadata
        self._update_pipeline_metadata(stage, 'completed')
        
        self.logger.info(f"Checkpoint saved for stage: {stage}")
        return checkpoint_id
    
    def load_checkpoint(self, stage: str, pipeline_id: Optional[str] = None) -> Any:
        """Load checkpoint data for a stage."""
        pipeline_id = pipeline_id or self.pipeline_id
        
        if not pipeline_id:
            raise ValueError("No pipeline ID specified")
        
        checkpoint_dir = self.checkpoints_dir / pipeline_id / stage
        checkpoint_path = checkpoint_dir / 'checkpoint.json'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for stage: {stage}")
        
        # Load checkpoint metadata
        with open(checkpoint_path, 'r') as f:
            checkpoint_metadata = json.load(f)
        
        # Load data
        data_path = checkpoint_metadata['data_path']
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.logger.info(f"Checkpoint loaded for stage: {stage}")
        return data
    
    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available checkpoints for a pipeline."""
        pipeline_id = pipeline_id or self.pipeline_id
        
        if not pipeline_id:
            return []
        
        pipeline_dir = self.checkpoints_dir / pipeline_id
        if not pipeline_dir.exists():
            return []
        
        checkpoints = []
        for stage_dir in pipeline_dir.iterdir():
            if stage_dir.is_dir():
                checkpoint_file = stage_dir / 'checkpoint.json'
                if checkpoint_file.exists():
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    checkpoints.append(checkpoint_data)
        
        return sorted(checkpoints, key=lambda x: x['timestamp'])
    
    def _update_pipeline_metadata(self, stage: str, status: str):
        """Update pipeline metadata."""
        if not self.pipeline_id:
            return
        
        metadata_path = self.checkpoints_dir / self.pipeline_id / 'pipeline_metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'pipeline_id': self.pipeline_id,
                'start_time': datetime.now().isoformat(),
                'stages_completed': [],
                'current_stage': None
            }
        
        # Update metadata
        metadata['current_stage'] = stage
        metadata['last_updated'] = datetime.now().isoformat()
        
        if status == 'completed' and stage not in metadata['stages_completed']:
            metadata['stages_completed'].append(stage)
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_pipeline_results(self, results: Dict[str, Any]) -> str:
        """Save final pipeline results."""
        if not self.pipeline_id:
            raise ValueError("No active pipeline")
        
        results_path = self.checkpoints_dir / self.pipeline_id / 'final_results.json'
        
        # Add execution summary
        results['execution_summary'] = {
            'pipeline_id': self.pipeline_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now().isoformat(),
            'total_duration': str(datetime.now() - self.start_time) if self.start_time else None,
            'checkpoints_saved': len(self.list_checkpoints())
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Pipeline results saved: {results_path}")
        return str(results_path)
    
    def cleanup_pipeline(self, pipeline_id: Optional[str] = None, keep_final_results: bool = True):
        """Clean up pipeline files and temporary data."""
        if not self.cleanup_temp:
            return
        
        pipeline_id = pipeline_id or self.pipeline_id
        pipeline_dir = self.checkpoints_dir / pipeline_id
        
        if not pipeline_dir.exists():
            return
        
        # Remove intermediate checkpoints but keep final results
        for item in pipeline_dir.iterdir():
            if item.is_dir():
                # Remove stage directories
                import shutil
                shutil.rmtree(item, ignore_errors=True)
            elif not keep_final_results or item.name != 'final_results.json':
                item.unlink(missing_ok=True)
        
        self.logger.info(f"Pipeline cleanup completed: {pipeline_id}")

# ============================================================================
# Monitoring and Logging
# ============================================================================

class PipelineMonitor:
    """
    Pipeline monitoring system with metrics collection and alerts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline monitor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.enabled = config.get('enabled', True)
        self.metrics_backend = config.get('metrics_backend', 'prometheus')
        self.alerts_enabled = config.get('alerts_enabled', True)
        
        # Metrics storage
        self.stage_metrics = {}
        self.pipeline_start_time = None
        self.current_stage_start = None
    
    @contextmanager
    def track_stage(self, stage_name: str):
        """Context manager to track stage execution time and metrics."""
        if not self.enabled:
            yield
            return
        
        self.current_stage_start = time.time()
        self.logger.info(f"Starting stage: {stage_name}")
        
        try:
            yield
            
            # Record successful completion
            duration = time.time() - self.current_stage_start
            self.stage_metrics[stage_name] = {
                'status': 'success',
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'memory_usage': self._get_memory_usage()
            }
            
            self.logger.info(f"Stage completed: {stage_name} ({duration:.2f}s)")
            
        except Exception as e:
            # Record failure
            duration = time.time() - self.current_stage_start if self.current_stage_start else 0
            self.stage_metrics[stage_name] = {
                'status': 'failed',
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'memory_usage': self._get_memory_usage()
            }
            
            self.logger.error(f"Stage failed: {stage_name} ({duration:.2f}s) - {str(e)}")
            raise
    
    @contextmanager
    def track_pipeline(self):
        """Context manager to track entire pipeline execution."""
        if not self.enabled:
            yield
            return
        
        self.pipeline_start_time = time.time()
        self.logger.info("Pipeline execution started")
        
        try:
            yield
            
            total_duration = time.time() - self.pipeline_start_time
            self.logger.info(f"Pipeline execution completed ({total_duration:.2f}s)")
            
        except Exception as e:
            total_duration = time.time() - self.pipeline_start_time
            self.logger.error(f"Pipeline execution failed ({total_duration:.2f}s) - {str(e)}")
            raise
    
    def get_stage_metrics(self) -> Dict[str, Any]:
        """Get metrics for all tracked stages."""
        return self.stage_metrics.copy()
    
    def get_total_execution_time(self) -> float:
        """Get total pipeline execution time."""
        if self.pipeline_start_time:
            return time.time() - self.pipeline_start_time
        return 0.0
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration for the pipeline.
    """
    log_level = config.get('level', 'INFO')
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('file', 'logs/pipeline.log')
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Configure specific loggers
    logger = logging.getLogger('llm_pipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger

# ============================================================================
# Utility Functions
# ============================================================================

def generate_content_hash(content: str) -> str:
    """Generate a hash for content deduplication."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text."""
    # Simple estimation: ~1.3 tokens per word
    word_count = len(text.split())
    return int(word_count * 1.3)

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_bytes(bytes_count: int) -> str:
    """Format byte count in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}TB"

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate pipeline configuration and return list of issues."""
    issues = []
    
    # Required sections
    required_sections = [
        'domain_topic',
        'data_collection',
        'fine_tuning',
        'evaluation',
        'deployment'
    ]
    
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required configuration section: {section}")
    
    # Validate data collection
    if 'data_collection' in config:
        data_config = config['data_collection']
        if not data_config.get('web_sources') and not data_config.get('pdf_sources'):
            issues.append("No data sources specified in data_collection")
    
    # Validate fine-tuning
    if 'fine_tuning' in config:
        ft_config = config['fine_tuning']
        if not ft_config.get('base_model'):
            issues.append("No base_model specified in fine_tuning")
    
    return issues

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {
        'unsloth': False,
        'synthetic_data_kit': False,
        'transformers': False,
        'datasets': False,
        'torch': False,
        'fastapi': False,
        'rouge_score': False,
        'sacrebleu': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'synthetic_data_kit':
                import subprocess
                result = subprocess.run(['synthetic-data-kit', '--version'], 
                                      capture_output=True, timeout=10)
                dependencies[dep] = result.returncode == 0
            else:
                __import__(dep)
                dependencies[dep] = True
        except:
            dependencies[dep] = False
    
    return dependencies

def create_project_structure(base_path: str):
    """Create the complete project directory structure."""
    base_path = Path(base_path)
    
    directories = [
        'config',
        'src/config',
        'src/data',
        'src/training',
        'src/evaluation',
        'src/deployment',
        'src/orchestration',
        'src/utils',
        'data/raw',
        'data/processed',
        'data/synthetic',
        'data/benchmarks',
        'models/base',
        'models/fine_tuned',
        'models/checkpoints',
        'models/registry',
        'logs',
        'outputs/evaluations',
        'outputs/reports',
        'scripts',
        'tests',
        'docker',
        'docs'
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith('src/'):
            init_file = dir_path / '__init__.py'
            if not init_file.exists():
                init_file.write_text('"""Pipeline module."""\n')
    
    print(f"Project structure created at: {base_path}")

# ============================================================================
# Authentication Utilities
# ============================================================================

import jwt
import hashlib
from datetime import datetime, timedelta

class AuthManager:
    """Simple authentication manager for API endpoints."""
    
    def __init__(self, secret_key: str):
        """Initialize with secret key."""
        self.secret_key = secret_key
        self.algorithm = 'HS256'
    
    def generate_token(self, user_id: str, expires_in_hours: int = 24) -> str:
        """Generate JWT token for user."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password for storage."""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt + pwdhash
    
    def verify_password(self, stored_password: bytes, provided_password: str) -> bool:
        """Verify password against stored hash."""
        salt = stored_password[:32]
        stored_hash = stored_password[32:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return pwdhash == stored_hash

# ============================================================================
# Data Processing Utilities
# ============================================================================

class DataProcessor:
    """
    Data processing utilities for cleaning, deduplication, and quality filtering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing configuration
        self.cleaning_config = config.get('cleaning', {})
        self.dedup_config = config.get('deduplication', {})
        self.quality_config = config.get('quality_filtering', {})
        self.chunking_config = config.get('chunking', {})
        
        self.output_dir = Path(config.get('output_dir', 'data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean raw data by removing HTML, normalizing text, etc."""
        self.logger.info("Starting data cleaning...")
        
        cleaned_data = {
            'web_content': [],
            'pdf_content': [],
            'metadata': raw_data.get('metadata', {})
        }
        
        # Clean web content
        for item in raw_data.get('web_content', []):
            cleaned_item = self._clean_text_item(item)
            if cleaned_item:
                cleaned_data['web_content'].append(cleaned_item)
        
        # Clean PDF content
        for item in raw_data.get('pdf_content', []):
            cleaned_item = self._clean_text_item(item)
            if cleaned_item:
                cleaned_data['pdf_content'].append(cleaned_item)
        
        self.logger.info(f"Data cleaning completed. {len(cleaned_data['web_content']) + len(cleaned_data['pdf_content'])} items processed")
        return cleaned_data
    
    def _clean_text_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean individual text item."""
        content = item.get('content', '')
        
        if not content or len(content.strip()) < self.cleaning_config.get('min_text_length', 50):
            return None
        
        # Clean content
        if self.cleaning_config.get('remove_html_tags', True):
            import re
            content = re.sub(r'<[^>]+>', '', content)
        
        if self.cleaning_config.get('normalize_whitespace', True):
            import re
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
        
        # Update item
        cleaned_item = item.copy()
        cleaned_item['content'] = content
        cleaned_item['cleaned_at'] = datetime.now().isoformat()
        
        return cleaned_item
    
    def deduplicate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate content using fuzzy matching."""
        self.logger.info("Starting deduplication...")
        
        threshold = self.dedup_config.get('threshold', 0.85)
        method = self.dedup_config.get('method', 'fuzzy')
        
        all_items = data.get('web_content', []) + data.get('pdf_content', [])
        
        if method == 'fuzzy':
            unique_items = self._fuzzy_deduplicate(all_items, threshold)
        else:
            # Simple hash-based deduplication
            unique_items = self._hash_deduplicate(all_items)
        
        # Separate back into web and pdf content
        deduplicated_data = {
            'web_content': [item for item in unique_items if item.get('source_type') == 'web'],
            'pdf_content': [item for item in unique_items if item.get('source_type') == 'pdf'],
            'metadata': data.get('metadata', {})
        }
        
        original_count = len(all_items)
        final_count = len(unique_items)
        duplicates_removed = original_count - final_count
        
        self.logger.info(f"Deduplication completed. Removed {duplicates_removed} duplicates from {original_count} items")
        return deduplicated_data
    
    def _fuzzy_deduplicate(self, items: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Fuzzy deduplication using content similarity."""
        try:
            from difflib import SequenceMatcher
        except ImportError:
            self.logger.warning("difflib not available, falling back to hash deduplication")
            return self._hash_deduplicate(items)
        
        unique_items = []
        
        for item in items:
            content = item.get('content', '')
            is_duplicate = False
            
            for unique_item in unique_items:
                unique_content = unique_item.get('content', '')
                similarity = SequenceMatcher(None, content, unique_content).ratio()
                
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_items.append(item)
        
        return unique_items
    
    def _hash_deduplicate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hash-based deduplication."""
        seen_hashes = set()
        unique_items = []
        
        for item in items:
            content_hash = item.get('content_hash')
            if not content_hash:
                content_hash = generate_content_hash(item.get('content', ''))
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_items.append(item)
        
        return unique_items
    
    def quality_filter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quality filters to remove low-quality content."""
        self.logger.info("Starting quality filtering...")
        
        min_quality_score = self.quality_config.get('min_quality_score', 0.7)
        filters = self.quality_config.get('filters', ['language_detection', 'content_relevance'])
        
        all_items = data.get('web_content', []) + data.get('pdf_content', [])
        filtered_items = []
        
        for item in all_items:
            quality_score = self._calculate_quality_score(item, filters)
            
            if quality_score >= min_quality_score:
                item['quality_score'] = quality_score
                filtered_items.append(item)
        
        # Separate back into categories
        filtered_data = {
            'web_content': [item for item in filtered_items if item.get('source_type') == 'web'],
            'pdf_content': [item for item in filtered_items if item.get('source_type') == 'pdf'],
            'metadata': data.get('metadata', {})
        }
        
        original_count = len(all_items)
        final_count = len(filtered_items)
        filtered_out = original_count - final_count
        
        self.logger.info(f"Quality filtering completed. Filtered out {filtered_out} items from {original_count}")
        return filtered_data
    
    def _calculate_quality_score(self, item: Dict[str, Any], filters: List[str]) -> float:
        """Calculate quality score for an item."""
        scores = []
        content = item.get('content', '')
        
        if 'language_detection' in filters:
            # Simple English language detection
            english_score = self._detect_english(content)
            scores.append(english_score)
        
        if 'content_relevance' in filters:
            # Check content relevance (basic length and structure checks)
            relevance_score = self._check_relevance(content)
            scores.append(relevance_score)
        
        if 'readability' in filters:
            # Basic readability check
            readability_score = self._check_readability(content)
            scores.append(readability_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _detect_english(self, content: str) -> float:
        """Simple English language detection."""
        # Basic check for English words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
        words = content.lower().split()
        
        if not words:
            return 0.0
        
        english_count = sum(1 for word in words if word in english_words)
        return min(english_count / len(words) * 4, 1.0)  # Scale up the score
    
    def _check_relevance(self, content: str) -> float:
        """Check content relevance based on structure and length."""
        if len(content) < 100:
            return 0.3
        elif len(content) < 500:
            return 0.7
        else:
            return 1.0
    
    def _check_readability(self, content: str) -> float:
        """Basic readability check."""
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Prefer moderate sentence lengths
        if 10 <= avg_sentence_length <= 25:
            return 1.0
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 40:
            return 0.7
        else:
            return 0.4
    
    def tokenize_and_chunk(self, data: Dict[str, Any], max_chunk_size: int = 2048,
                          overlap: int = 64) -> Dict[str, Any]:
        """Tokenize and chunk data for training."""
        self.logger.info("Starting tokenization and chunking...")
        
        all_items = data.get('web_content', []) + data.get('pdf_content', [])
        chunks = []
        
        for item in all_items:
            content = item.get('content', '')
            if len(content) < 100:  # Skip very short content
                continue
            
            # Simple text chunking (can be enhanced with proper tokenization)
            item_chunks = self._chunk_text(content, max_chunk_size, overlap)
            
            for i, chunk_text in enumerate(item_chunks):
                chunk = {
                    'chunk_id': f"{item.get('source_type', 'unknown')}_{item.get('content_hash', 'unknown')}_{i}",
                    'content': chunk_text,
                    'source_metadata': item.get('metadata', {}),
                    'source_type': item.get('source_type'),
                    'chunk_index': i,
                    'total_chunks': len(item_chunks),
                    'char_count': len(chunk_text),
                    'estimated_tokens': estimate_tokens(chunk_text)
                }
                chunks.append(chunk)
        
        processed_data = {
            'chunks': chunks,
            'metadata': {
                'total_chunks': len(chunks),
                'total_items_processed': len(all_items),
                'avg_chunk_size': sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        self.logger.info(f"Tokenization and chunking completed. Created {len(chunks)} chunks")
        return processed_data
    
    def _chunk_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence end within the last 20% of the chunk
                search_start = end - int(max_size * 0.2)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > search_start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def save_processed_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save processed data to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.json"
        
        save_path = self.output_dir / filename
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processed data saved to: {save_path}")
        return str(save_path)
    
    def load_processed_data(self, filename: str) -> Dict[str, Any]:
        """Load processed data from disk."""
        load_path = self.output_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Processed data loaded from: {load_path}")
        return data

# ============================================================================
# Complete Setup Script
# ============================================================================

def create_complete_project(project_name: str, base_path: str = "."):
    """Create a complete project with all files and configurations."""
    
    project_path = Path(base_path) / project_name
    project_path.mkdir(exist_ok=True)
    
    print(f"Creating complete LLM fine-tuning pipeline project: {project_path}")
    
    # Create directory structure
    create_project_structure(str(project_path))
    
    # Create configuration files
    _create_config_files(project_path)
    
    # Create main application files
    _create_application_files(project_path)
    
    # Create requirements files
    _create_requirements_files(project_path)
    
    # Create setup files
    _create_setup_files(project_path)
    
    print(f"✅ Project created successfully at: {project_path}")
    print("\n📋 Next steps:")
    print("1. cd", project_name)
    print("2. pip install -r requirements.txt")
    print("3. cp config/environment.env.example config/environment.env")
    print("4. Edit config/environment.env with your API keys")
    print("5. Edit config/config.yaml with your domain and data sources")
    print("6. python main.py --config config/config.yaml --stage full")

def _create_config_files(project_path: Path):
    """Create configuration files."""
    config_dir = project_path / "config"
    
    # Main configuration
    config_content = """# LLM Fine-tuning Pipeline Configuration

# Domain-specific configuration
domain_topic: "electric vehicle charging stations"
use_case: "qa"

# Environment settings
environment:
  name: "llm_pipeline"
  gpu_enabled: true
  mixed_precision: true
  distributed: false

# Data Collection Configuration
data_collection:
  web_sources:
    - "https://www.energy.gov/eere/electricvehicles"
    - "https://www.nrel.gov/transportation/electric-vehicle-charging-infrastructure.html"
  pdf_sources:
    - "data/raw/ev_charging_standards.pdf"
    - "data/raw/ev_infrastructure_report.pdf"
  output_dir: "data/raw"
  max_pages_per_source: 10
  scraping_delay: 1.0

# Data Processing Configuration
data_processing:
  cleaning:
    remove_html_tags: true
    normalize_whitespace: true
    min_text_length: 50
    language: "en"
  deduplication:
    method: "fuzzy"
    threshold: 0.85
  quality_filtering:
    min_quality_score: 0.7
    filters:
      - "language_detection"
      - "content_relevance"
      - "readability"
  chunking:
    max_chunk_size: 2048
    overlap: 64
    method: "sliding_window"
  output_dir: "data/processed"

# Synthetic Data Generation Configuration
synthetic_data:
  base_model: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  temperature: 0.7
  top_p: 0.95
  pairs_per_chunk: 25
  max_generation_tokens: 512
  output_dir: "data/synthetic"

# Fine-tuning Configuration
fine_tuning:
  base_model: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true
  load_in_8bit: false
  full_finetuning: false
  
  # LoRA Configuration
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
  bias: "none"
  use_rslora: false
  
  # Training Parameters
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

# Evaluation Configuration
evaluation:
  baseline_model: "unsloth/Llama-3.2-3B-Instruct"
  benchmark_size: 50
  metrics:
    - "rouge"
    - "bleu"
    - "bert_score"
  output_dir: "outputs/evaluations"

# Deployment Configuration
deployment:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  version_tag: "latest"
  authentication: true
  metrics_port: 9090
  model_registry_path: "models/registry"
  
  api:
    max_tokens: 512
    temperature: 0.1
    top_p: 0.9
    timeout: 30
    rate_limit: 100

# Orchestration Configuration
orchestration:
  checkpoints_dir: "models/checkpoints"
  enable_resume: true
  save_intermediate: true
  cleanup_temp: true

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"
  max_size: "10MB"
  backup_count: 5

# Monitoring Configuration
monitoring:
  enabled: true
  metrics_backend: "prometheus"
  alerts_enabled: true
  dashboard_port: 3000
"""
    
    with open(config_dir / "config.yaml", "w") as f:
        f.write(config_content)
    
    # Environment file template
    env_content = """# Environment variables for the LLM Pipeline

# API Keys
HUGGINGFACE_TOKEN=hf_your_token_here
OPENAI_API_KEY=sk_your_openai_key_here
WANDB_API_KEY=your_wandb_key_here

# Model Configuration
BASE_MODEL_PATH=/models/base
CACHE_DIR=/cache/huggingface
TORCH_HOME=/cache/torch

# Database Configuration
DATABASE_URL=sqlite:///pipeline.db
REDIS_URL=redis://localhost:6379

# Deployment Configuration
MODEL_REGISTRY_URL=http://localhost:5000
PROMETHEUS_URL=http://localhost:9090

# Resource Limits
MAX_MEMORY_GB=16
MAX_GPU_MEMORY_GB=12
MAX_WORKERS=4

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
API_KEY_SALT=your_api_key_salt_here
API_TOKEN=your_secure_api_token_here

# Feature Flags
ENABLE_DISTRIBUTED_TRAINING=false
ENABLE_MIXED_PRECISION=true
ENABLE_GRADIENT_CHECKPOINTING=true
"""
    
    with open(config_dir / "environment.env.example", "w") as f:
        f.write(env_content)

def _create_requirements_files(project_path: Path):
    """Create requirements files."""
    
    requirements_content = """# Core ML Libraries
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
alembic>=1.12.0
redis>=5.0.0

# Monitoring & Logging
prometheus-client>=0.18.0
wandb>=0.16.0
mlflow>=2.8.0
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
"""
    
    with open(project_path / "requirements.txt", "w") as f:
        f.write(requirements_content)

def _create_setup_files(project_path: Path):
    """Create setup and installation files."""
    
    # setup.py
    setup_content = '''"""Setup script for LLM Fine-tuning Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-finetuning-pipeline",
    version="1.0.0",
    author="EnergyAI Team",
    author_email="contact@energyai.berlin",
    description="End-to-end pipeline for fine-tuning small language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/energyai/llm-finetuning-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-pipeline=main:main",
        ],
    },
)
'''
    
    with open(project_path / "setup.py", "w") as f:
        f.write(setup_content)
    
    # Installation script
    install_script = '''#!/bin/bash

# LLM Fine-tuning Pipeline Installation Script

echo "🚀 Installing LLM Fine-tuning Pipeline..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8 or higher required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,synthetic,benchmarks}
mkdir -p models/{base,fine_tuned,checkpoints,registry}
mkdir -p logs
mkdir -p outputs/{evaluations,reports}

# Copy configuration template
echo "⚙️ Setting up configuration..."
cp config/environment.env.example config/environment.env

echo "✅ Installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. source venv/bin/activate"
echo "2. Edit config/environment.env with your API keys"
echo "3. Edit config/config.yaml with your domain and data sources"
echo "4. python main.py --config config/config.yaml --stage full"
'''
    
    scripts_dir = project_path / "scripts"
    with open(scripts_dir / "setup_environment.sh", "w") as f:
        f.write(install_script)
    
    # Make script executable
    import stat
    script_path = scripts_dir / "setup_environment.sh"
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

def _create_application_files(project_path: Path):
    """Create core application files."""
    
    # Create __init__.py files for all src subdirectories
    src_dirs = [
        "src",
        "src/config", 
        "src/data",
        "src/training",
        "src/evaluation", 
        "src/deployment",
        "src/orchestration",
        "src/utils"
    ]
    
    for src_dir in src_dirs:
        init_path = project_path / src_dir / "__init__.py"
        with open(init_path, "w") as f:
            f.write('"""LLM Fine-tuning Pipeline module."""\n')
    
    # Create a simple test file
    test_content = '''"""Basic tests for the LLM pipeline."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_loading():
    """Test configuration loading."""
    from src.config.config_manager import ConfigManager
    
    # This will pass if config file exists
    try:
        config = ConfigManager("config/config.yaml")
        assert config.get("domain_topic") is not None
    except FileNotFoundError:
        pytest.skip("Config file not found")

def test_data_collector_init():
    """Test data collector initialization."""
    from src.data.collector import DataCollector
    
    config = {
        'output_dir': 'data/raw',
        'web_sources': [],
        'pdf_sources': []
    }
    
    collector = DataCollector(config)
    assert collector.output_dir.name == 'raw'

def test_pipeline_dependencies():
    """Test that key dependencies are available."""
    dependencies = [
        'torch',
        'transformers', 
        'datasets',
        'pandas',
        'numpy'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            pytest.fail(f"Required dependency {dep} not available")

if __name__ == "__main__":
    pytest.main([__file__])
'''
    
    with open(project_path / "tests" / "test_basic.py", "w") as f:
        f.write(test_content)

# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Pipeline Utilities")
    parser.add_argument("--create-project", help="Create new project")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--validate-config", help="Validate configuration file")
    
    args = parser.parse_args()
    
    if args.create_project:
        create_complete_project(args.create_project)
    elif args.check_deps:
        deps = check_dependencies()
        print("Dependency Status:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
    elif args.validate_config:
        import yaml
        with open(args.validate_config, 'r') as f:
            config = yaml.safe_load(f)
        issues = validate_config(config)
        if issues:
            print("Configuration Issues:")
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("✅ Configuration is valid")
    else:
        parser.print_help()"""
Pipeline Orchestration and Utility Modules
Includes pipeline management, monitoring, logging, and helper utilities.
"""

import os
import json
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import yaml
from dataclasses import dataclass, asdict

# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigManager:
    """
    Configuration management system for the LLM pipeline.
    Handles YAML config files and environment variables.
    """
    
    def __init__(self, config_path: str):
        """Initialize with configuration file path."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_environment_variables()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():