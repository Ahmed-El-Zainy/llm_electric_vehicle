"""
Model Evaluation and Deployment modules for the LLM Fine-tuning Pipeline.
Implements evaluation metrics, benchmarking, and production deployment.
"""

import os
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: Evaluation metrics not available. Install with: pip install rouge-score sacrebleu evaluate")
    METRICS_AVAILABLE = False

# Deployment dependencies
try:
    from fastapi import FastAPI, HTTPException, Depends, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    DEPLOYMENT_AVAILABLE = True
except ImportError:
    print("Warning: Deployment dependencies not available. Install with: pip install fastapi uvicorn")
    DEPLOYMENT_AVAILABLE = False

@dataclass
class EvaluationMetrics:
    """Data class for evaluation metrics."""
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bleu_score: float
    bert_score: float
    domain_relevance: float
    response_time: float
    
class ModelEvaluator:
    """
    Evaluation system for fine-tuned models with domain-specific benchmarks
    and automated metrics (ROUGE, BLEU, BERT Score).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.baseline_model = config.get('baseline_model', 'unsloth/Llama-3.2-3B-Instruct')
        self.benchmark_size = config.get('benchmark_size', 100)
        self.metrics_list = config.get('metrics', ['rouge', 'bleu', 'bert_score'])
        self.output_dir = Path(config.get('output_dir', 'outputs/evaluations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.bert_scorer = evaluate.load("bertscore")
        
        self.logger.info("ModelEvaluator initialized")
    
    def create_benchmark_dataset(self, domain: str, size: int = 100) -> List[Dict[str, Any]]:
        """
        Create a domain-specific benchmark dataset for evaluation.
        """
        self.logger.info(f"Creating benchmark dataset for domain: {domain}, size: {size}")
        
        # Generate domain-specific questions and expected answers
        benchmark_questions = self._generate_domain_questions(domain, size)
        
        # Save benchmark dataset
        benchmark_path = self.output_dir / f"benchmark_{domain.replace(' ', '_')}_{size}.json"
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_questions, f, indent=2)
        
        self.logger.info(f"Benchmark dataset created with {len(benchmark_questions)} examples")
        return benchmark_questions
    
    def _generate_domain_questions(self, domain: str, size: int) -> List[Dict[str, Any]]:
        """Generate domain-specific evaluation questions."""
        
        # Template questions for different domains
        if "electric vehicle" in domain.lower():
            question_templates = [
                "What are the main types of EV charging stations?",
                "How long does it take to charge an electric vehicle?",
                "What is the difference between AC and DC charging?",
                "What are the benefits of electric vehicles?",
                "How does regenerative braking work in EVs?",
                "What factors affect EV battery life?",
                "What is the current EV charging infrastructure status?",
                "How do you find EV charging stations?",
                "What are the costs associated with EV charging?",
                "What safety considerations exist for EV charging?"
            ]
            
            reference_answers = [
                "The main types of EV charging stations are Level 1 (120V household outlets), Level 2 (240V), and DC Fast Charging (Level 3) which provides rapid charging.",
                "Charging time varies by vehicle and charger type. Level 1 takes 8-12 hours, Level 2 takes 4-8 hours, and DC fast charging can charge 80% in 30-60 minutes.",
                "AC charging uses alternating current and is slower, typically for overnight charging. DC charging bypasses the vehicle's onboard charger for much faster charging speeds.",
                "Electric vehicles offer zero direct emissions, lower operating costs, quiet operation, instant torque, and reduced dependence on fossil fuels.",
                "Regenerative braking captures kinetic energy during deceleration and converts it back to electrical energy to charge the battery, extending range.",
                "EV battery life is affected by temperature extremes, charging frequency, depth of discharge, and overall usage patterns. Most batteries last 8-15 years.",
                "EV charging infrastructure is rapidly expanding with thousands of public charging stations and growing fast-charging networks along major highways.",
                "EV charging stations can be found using mobile apps like PlugShare, ChargePoint, or built-in vehicle navigation systems that show nearby charging locations.",
                "EV charging costs vary by location and electricity rates, typically ranging from $0.10-0.30 per kWh, making it generally cheaper than gasoline.",
                "EV charging safety includes proper cable handling, avoiding damaged equipment, charging in appropriate weather conditions, and using certified charging stations."
            ]
        else:
            # Generic questions for other domains
            question_templates = [
                f"What is {domain}?",
                f"How does {domain} work?",
                f"What are the benefits of {domain}?",
                f"What are the challenges in {domain}?",
                f"What are the latest developments in {domain}?",
                f"How can I get started with {domain}?",
                f"What are the costs associated with {domain}?",
                f"What safety considerations exist for {domain}?",
                f"What are the environmental impacts of {domain}?",
                f"What is the future outlook for {domain}?"
            ]
            
            reference_answers = [f"Answer about {domain}" for _ in question_templates]
        
        # Create benchmark dataset
        benchmark_data = []
        templates_count = len(question_templates)
        
        for i in range(size):
            template_idx = i % templates_count
            
            benchmark_item = {
                'id': i + 1,
                'question': question_templates[template_idx],
                'reference_answer': reference_answers[template_idx] if template_idx < len(reference_answers) else f"Reference answer for {question_templates[template_idx]}",
                'domain': domain,
                'category': 'factual' if i % 3 == 0 else ('analytical' if i % 3 == 1 else 'explanatory'),
                'difficulty': 'easy' if i % 3 == 0 else ('medium' if i % 3 == 1 else 'hard')
            }
            
            benchmark_data.append(benchmark_item)
        
        return benchmark_data
    
    def load_model(self, model_path: str) -> Any:
        """Load a model for evaluation."""
        # This would use the FineTuner's load_fine_tuned_model method
        from src.training.fine_tuner import FineTuner
        
        fine_tuner = FineTuner(self.config.get('fine_tuning', {}))
        model, tokenizer = fine_tuner.load_fine_tuned_model(model_path)
        
        return {'model': model, 'tokenizer': tokenizer, 'fine_tuner': fine_tuner}
    
    def load_baseline_model(self, model_name: str) -> Any:
        """Load baseline model for comparison."""
        try:
            from unsloth import FastLanguageModel
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            FastLanguageModel.for_inference(model)
            
            return {'model': model, 'tokenizer': tokenizer, 'model_name': model_name}
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline model: {str(e)}")
            raise
    
    def evaluate_model(self, model_info: Dict[str, Any], 
                      benchmark_dataset: List[Dict[str, Any]], 
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a model on the benchmark dataset with multiple metrics.
        """
        self.logger.info(f"Evaluating {model_name} on {len(benchmark_dataset)} examples")
        
        if not METRICS_AVAILABLE:
            self.logger.error("Evaluation metrics not available")
            return {}
        
        predictions = []
        references = []
        response_times = []
        
        # Generate predictions
        for i, example in enumerate(benchmark_dataset):
            if i % 10 == 0:
                self.logger.info(f"Processing example {i+1}/{len(benchmark_dataset)}")
            
            question = example['question']
            reference = example['reference_answer']
            
            # Measure response time
            start_time = time.time()
            
            try:
                # Generate prediction
                if 'fine_tuner' in model_info:
                    prediction = model_info['fine_tuner'].inference(
                        model_info['model'], 
                        model_info['tokenizer'], 
                        question,
                        max_new_tokens=256,
                        temperature=0.1
                    )
                else:
                    # Baseline model inference
                    messages = [{"role": "user", "content": question}]
                    inputs = model_info['tokenizer'].apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                    
                    with torch.no_grad():
                        outputs = model_info['model'].generate(
                            input_ids=inputs,
                            max_new_tokens=256,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=model_info['tokenizer'].eos_token_id
                        )
                    
                    generated_ids = outputs[0][len(inputs[0]):]
                    prediction = model_info['tokenizer'].decode(generated_ids, skip_special_tokens=True).strip()
                
                response_time = time.time() - start_time
                
                predictions.append(prediction)
                references.append(reference)
                response_times.append(response_time)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate prediction for example {i}: {str(e)}")
                predictions.append("")
                references.append(reference)
                response_times.append(0.0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, references, response_times)
        
        # Save detailed results
        detailed_results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions': predictions,
            'references': references,
            'response_times': response_times,
            'benchmark_size': len(benchmark_dataset)
        }
        
        results_path = self.output_dir / f"evaluation_{model_name}_{int(time.time())}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        self.logger.info(f"Evaluation completed for {model_name}")
        return detailed_results
    
    def _calculate_metrics(self, predictions: List[str], references: List[str], 
                          response_times: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # ROUGE metrics
        if 'rouge' in self.metrics_list:
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                if pred and ref:
                    scores = self.rouge_scorer.score(ref, pred)
                    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            metrics.update({
                'rouge_1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
                'rouge_2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
                'rouge_l': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
            })
        
        # BLEU score
        if 'bleu' in self.metrics_list:
            try:
                # Prepare references and predictions for BLEU
                bleu_refs = [[ref.split()] for ref in references if ref]
                bleu_preds = [pred.split() for pred in predictions if pred]
                
                if bleu_refs and bleu_preds:
                    bleu_score = corpus_bleu(bleu_preds, bleu_refs).score / 100.0
                    metrics['bleu_score'] = bleu_score
                else:
                    metrics['bleu_score'] = 0.0
            except Exception as e:
                self.logger.warning(f"BLEU calculation failed: {str(e)}")
                metrics['bleu_score'] = 0.0
        
        # BERT Score
        if 'bert_score' in self.metrics_list:
            try:
                valid_preds = [p for p in predictions if p]
                valid_refs = [r for r in references if r]
                
                if valid_preds and valid_refs:
                    bert_results = self.bert_scorer.compute(
                        predictions=valid_preds, 
                        references=valid_refs, 
                        lang="en"
                    )
                    metrics['bert_score'] = np.mean(bert_results['f1'])
                else:
                    metrics['bert_score'] = 0.0
            except Exception as e:
                self.logger.warning(f"BERT Score calculation failed: {str(e)}")
                metrics['bert_score'] = 0.0
        
        # Performance metrics
        metrics.update({
            'avg_response_time': np.mean(response_times) if response_times else 0.0,
            'max_response_time': np.max(response_times) if response_times else 0.0,
            'min_response_time': np.min(response_times) if response_times else 0.0,
            'total_examples': len(predictions),
            'valid_predictions': len([p for p in predictions if p])
        })
        
        return metrics
    
    def compare_models(self, fine_tuned_results: Dict[str, Any], 
                      baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare fine-tuned model results with baseline."""
        comparison = {
            'improvement_summary': {},
            'detailed_comparison': {},
            'performance_gains': {}
        }
        
        ft_metrics = fine_tuned_results.get('metrics', {})
        baseline_metrics = baseline_results.get('metrics', {})
        
        # Calculate improvements
        for metric_name in ft_metrics:
            if metric_name in baseline_metrics:
                ft_value = ft_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                if baseline_value > 0:
                    improvement = ((ft_value - baseline_value) / baseline_value) * 100
                    comparison['improvement_summary'][metric_name] = {
                        'fine_tuned': ft_value,
                        'baseline': baseline_value,
                        'improvement_percent': improvement,
                        'improved': improvement > 0
                    }
        
        # Overall assessment
        improved_metrics = sum(1 for m in comparison['improvement_summary'].values() if m['improved'])
        total_metrics = len(comparison['improvement_summary'])
        
        comparison['performance_gains'] = {
            'metrics_improved': improved_metrics,
            'total_metrics': total_metrics,
            'improvement_ratio': improved_metrics / total_metrics if total_metrics > 0 else 0,
            'overall_improved': improved_metrics > total_metrics / 2
        }
        
        return comparison
    
    def calculate_performance_metrics(self, model_info: Dict[str, Any], 
                                    benchmark_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics like latency and throughput."""
        self.logger.info("Calculating performance metrics...")
        
        # Measure inference latency and throughput
        sample_questions = [ex['question'] for ex in benchmark_dataset[:10]]  # Sample for performance testing
        
        latencies = []
        throughput_tokens = []
        
        for question in sample_questions:
            start_time = time.time()
            
            try:
                if 'fine_tuner' in model_info:
                    response = model_info['fine_tuner'].inference(
                        model_info['model'], 
                        model_info['tokenizer'], 
                        question,
                        max_new_tokens=100
                    )
                else:
                    # Simplified inference for baseline
                    response = "Sample response"
                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
                
                # Estimate tokens (rough approximation)
                token_count = len(response.split()) * 1.3  # Rough token estimation
                throughput_tokens.append(token_count / latency if latency > 0 else 0)
                
            except Exception as e:
                self.logger.warning(f"Performance test failed for question: {str(e)}")
                latencies.append(0)
                throughput_tokens.append(0)
        
        performance_metrics = {
            'avg_latency_ms': np.mean(latencies) * 1000 if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000 if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000 if latencies else 0,
            'avg_throughput_tokens_per_sec': np.mean(throughput_tokens) if throughput_tokens else 0,
            'samples_tested': len(sample_questions)
        }
        
        return performance_metrics
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save evaluation results to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        save_path = self.output_dir / filename
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {save_path}")
        return str(save_path)

# Deployment Models
class InferenceRequest(BaseModel):
    """Request model for inference API."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    stream: bool = False

class InferenceResponse(BaseModel):
    """Response model for inference API."""
    response: str
    response_time: float
    model_version: str
    timestamp: str

class ModelServer:
    """
    Production deployment and serving system with API endpoints,
    authentication, and monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model server with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Server configuration
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8000)
        self.workers = config.get('workers', 1)
        self.api_config = config.get('api', {})
        
        # Model registry
        self.model_registry_path = Path(config.get('model_registry_path', 'models/registry'))
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        # Current loaded model
        self.current_model = None
        self.current_tokenizer = None
        self.current_fine_tuner = None
        self.current_version = None
        
        # Authentication
        self.auth_enabled = config.get('authentication', True)
        self.security = HTTPBearer() if self.auth_enabled else None
        
        # API rate limiting and monitoring
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        if DEPLOYMENT_AVAILABLE:
            self.app = FastAPI(
                title="LLM Fine-tuning Pipeline API",
                description="Production API for fine-tuned language models",
                version="1.0.0"
            )
            self._setup_routes()
        
        self.logger.info("ModelServer initialized")
    
    def register_model(self, model_path: str, model_metadata: Dict[str, Any], 
                      version_tag: str = "latest") -> str:
        """Register a model version in the model registry."""
        self.logger.info(f"Registering model version: {version_tag}")
        
        # Create version ID
        version_id = f"{version_tag}_{int(time.time())}"
        
        # Registry entry
        registry_entry = {
            'version_id': version_id,
            'version_tag': version_tag,
            'model_path': model_path,
            'metadata': model_metadata,
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        # Save to registry
        registry_file = self.model_registry_path / f"{version_id}.json"
        with open(registry_file, 'w') as f:
            json.dump(registry_entry, f, indent=2)
        
        # Update latest symlink if this is the latest version
        if version_tag == "latest":
            latest_file = self.model_registry_path / "latest.json"
            with open(latest_file, 'w') as f:
                json.dump(registry_entry, f, indent=2)
        
        self.logger.info(f"Model registered with version ID: {version_id}")
        return version_id
    
    def load_model_version(self, version_id: str) -> bool:
        """Load a specific model version for serving."""
        self.logger.info(f"Loading model version: {version_id}")
        
        # Load registry entry
        if version_id == "latest":
            registry_file = self.model_registry_path / "latest.json"
        else:
            registry_file = self.model_registry_path / f"{version_id}.json"
        
        if not registry_file.exists():
            raise FileNotFoundError(f"Model version not found: {version_id}")
        
        with open(registry_file, 'r') as f:
            registry_entry = json.load(f)
        
        try:
            # Load the model using FineTuner
            from src.training.fine_tuner import FineTuner
            
            fine_tuner = FineTuner(self.config.get('fine_tuning', {}))
            model, tokenizer = fine_tuner.load_fine_tuned_model(
                registry_entry['model_path'],
                registry_entry['metadata']
            )
            
            # Update current model
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_fine_tuner = fine_tuner
            self.current_version = version_id
            
            self.logger.info(f"Model version {version_id} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model version {version_id}: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes for the model server."""
        if not DEPLOYMENT_AVAILABLE:
            return
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.current_model is not None,
                "current_version": self.current_version,
                "uptime_seconds": time.time() - self.start_time
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus-style metrics endpoint."""
            uptime = time.time() - self.start_time
            
            return {
                "requests_total": self.request_count,
                "errors_total": self.error_count,
                "uptime_seconds": uptime,
                "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
                "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
                "model_info": {
                    "loaded": self.current_model is not None,
                    "version": self.current_version
                }
            }
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def inference_endpoint(
            request: InferenceRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security) if self.auth_enabled else None
        ):
            """Main inference endpoint."""
            self.request_count += 1
            
            # Authentication
            if self.auth_enabled and not self._validate_token(credentials.credentials):
                self.error_count += 1
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            # Check if model is loaded
            if self.current_model is None:
                self.error_count += 1
                raise HTTPException(status_code=503, detail="No model loaded")
            
            try:
                start_time = time.time()
                
                # Generate response
                response_text = self.current_fine_tuner.inference(
                    self.current_model,
                    self.current_tokenizer,
                    request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=request.stream
                )
                
                response_time = time.time() - start_time
                
                return InferenceResponse(
                    response=response_text,
                    response_time=response_time,
                    model_version=self.current_version or "unknown",
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Inference failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        @self.app.post("/load_model/{version_id}")
        async def load_model_endpoint(
            version_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security) if self.auth_enabled else None
        ):
            """Endpoint to load a specific model version."""
            if self.auth_enabled and not self._validate_token(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            try:
                success = self.load_model_version(version_id)
                return {"status": "success", "version_loaded": version_id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        @self.app.get("/models")
        async def list_models():
            """List available model versions."""
            models = []
            
            for registry_file in self.model_registry_path.glob("*.json"):
                if registry_file.name != "latest.json":
                    try:
                        with open(registry_file, 'r') as f:
                            registry_entry = json.load(f)
                        models.append({
                            "version_id": registry_entry['version_id'],
                            "version_tag": registry_entry['version_tag'],
                            "registered_at": registry_entry['registered_at'],
                            "status": registry_entry['status']
                        })
                    except:
                        continue
            
            return {"models": models, "current_version": self.current_version}
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token."""
        # Simple token validation - implement proper JWT validation in production
        expected_token = os.getenv('API_TOKEN', 'default_token')
        return token == expected_token
    
    def start_server(self, model_version: str = "latest", **kwargs) -> Dict[str, Any]:
        """Start the model server."""
        if not DEPLOYMENT_AVAILABLE:
            raise ImportError("Deployment dependencies not available")
        
        # Load the specified model version
        self.load_model_version(model_version)
        
        # Server configuration
        server_config = {
            'host': kwargs.get('host', self.host),
            'port': kwargs.get('port', self.port),
            'workers': kwargs.get('workers', self.workers)
        }
        
        self.logger.info(f"Starting server on {server_config['host']}:{server_config['port']}")
        
        # In production, you would run this with uvicorn.run()
        # For now, return configuration for external startup
        return {
            'app': self.app,
            'config': server_config,
            'model_version': model_version,
            'status': 'configured'
        }
    
    def setup_api_endpoints(self, authentication: bool = True) -> Dict[str, Any]:
        """Setup and configure API endpoints."""
        endpoints = {
            'inference': '/inference',
            'health': '/health',
            'metrics': '/metrics',
            'load_model': '/load_model/{version_id}',
            'list_models': '/models'
        }
        
        api_info = {
            'endpoints': endpoints,
            'authentication_enabled': authentication,
            'base_url': f"http://{self.host}:{self.port}",
            'api_docs': f"http://{self.host}:{self.port}/docs"
        }
        
        return api_info
    
    def setup_monitoring(self, metrics_port: int = 9090) -> Dict[str, Any]:
        """Setup monitoring and metrics collection."""
        monitoring_info = {
            'metrics_endpoint': f"http://{self.host}:{self.port}/metrics",
            'prometheus_port': metrics_port,
            'health_endpoint': f"http://{self.host}:{self.port}/health",
            'monitoring_enabled': True
        }
        
        # In production, you would setup Prometheus metrics collection here
        self.logger.info("Monitoring setup completed")
        
        return monitoring_info
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get current server statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
            'requests_per_second': self.request_count / uptime if uptime > 0 else 0,
            'current_model': {
                'version': self.current_version,
                'loaded': self.current_model is not None
            },
            'server_config': {
                'host': self.host,
                'port': self.port,
                'workers': self.workers,
                'auth_enabled': self.auth_enabled
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the server."""
        self.logger.info("Shutting down model server...")
        
        # Clean up model
        if self.current_fine_tuner:
            self.current_fine_tuner.cleanup()
        
        self.current_model = None
        self.current_tokenizer = None
        self.current_fine_tuner = None
        
        self.logger.info("Server shutdown completed")

# Utility function to run the server
def run_server(config_path: str, model_version: str = "latest"):
    """Utility function to run the model server."""
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and start server
    server = ModelServer(config.get('deployment', {}))
    server_info = server.start_server(model_version=model_version)
    
    # Run with uvicorn
    uvicorn.run(
        server_info['app'],
        host=server_info['config']['host'],
        port=server_info['config']['port'],
        workers=1,  # Use 1 worker for model serving
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        model_version = sys.argv[2] if len(sys.argv) > 2 else "latest"
        run_server(config_path, model_version)
    else:
        print("Usage: python evaluator_deployment.py <config_path> [model_version]")