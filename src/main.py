#!/usr/bin/env python3
"""
End-to-End Small Language Model Fine-tuning and Serving Pipeline
Based on Meta Synthetic Data Kit and Unsloth for domain-specific QA systems

This pipeline implements:
- Data collection (web scraping, PDF extraction)
- Data processing and quality filtering
- Synthetic QA pair generation using Meta's synthetic-data-kit
- Memory-efficient fine-tuning with Unsloth (LoRA/QLoRA)
- Evaluation and benchmarking
- Production deployment and serving
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
from src_sloth.config.config_manager import ConfigManager
from unsloth_imp.src_sloth.collector import DataCollector
from src_sloth.data.processor import DataProcessor
from unsloth_imp.src_sloth.synthetic_data_generator import SyntheticDataGenerator
from unsloth_imp.src_sloth.fine_tuner import FineTuner
from unsloth_imp.src_sloth.evaluator import ModelEvaluator
from src_sloth.deployment.server import ModelServer
from src_sloth.orchestration.pipeline_manager import PipelineManager
# from src_sloth.utils.logging_utils import setup_logging
# from src_sloth.utils.monitoring import PipelineMonitor


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
    
    def run_data_collection(self) -> Dict[str, Any]:
        """Step 1: Collect domain-specific data from various sources."""
        self.logger.info("Starting data collection phase...")
        
        with self.monitor.track_stage("data_collection"):
            # Web scraping for domain-specific content
            web_data = self.data_collector.scrape_web_sources(
                self.config.get('data_collection.web_sources', [])
            )
            
            # PDF extraction with layout preservation
            pdf_data = self.data_collector.extract_pdf_content(
                self.config.get('data_collection.pdf_sources', [])
            )
            
            # Combine and store raw data
            raw_data = {
                'web_content': web_data,
                'pdf_content': pdf_data,
                'metadata': {
                    'collection_timestamp': datetime.now().isoformat(),
                    'domain': self.config.get('domain_topic'),
                    'sources_count': len(web_data) + len(pdf_data)
                }
            }
            
            # Save raw data
            self.data_collector.save_raw_data(raw_data)
            
        self.logger.info(f"Data collection completed. Collected {raw_data['metadata']['sources_count']} sources")
        return raw_data
    
    def run_data_processing(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Clean, deduplicate, and process collected data."""
        self.logger.info("Starting data processing phase...")
        
        with self.monitor.track_stage("data_processing"):
            # Clean and normalize data
            cleaned_data = self.data_processor.clean_data(raw_data)
            
            # Deduplicate content
            deduplicated_data = self.data_processor.deduplicate(cleaned_data)
            
            # Quality filtering
            filtered_data = self.data_processor.quality_filter(deduplicated_data)
            
            # Tokenization and chunking for training
            processed_data = self.data_processor.tokenize_and_chunk(
                filtered_data,
                max_chunk_size=self.config.get('data_processing.max_chunk_size', 2048),
                overlap=self.config.get('data_processing.overlap', 64)
            )
            
            # Store processed data
            self.data_processor.save_processed_data(processed_data)
            
        self.logger.info(f"Data processing completed. {len(processed_data['chunks'])} chunks created")
        return processed_data
    
    def run_synthetic_data_generation(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate synthetic QA pairs using Meta's synthetic-data-kit."""
        self.logger.info("Starting synthetic data generation phase...")
        
        with self.monitor.track_stage("synthetic_data_generation"):
            # Initialize synthetic data generator with base model
            self.synthetic_generator.initialize_model(
                model_name=self.config.get('synthetic_data.base_model', 'unsloth/Llama-3.2-3B-Instruct'),
                max_seq_length=self.config.get('synthetic_data.max_seq_length', 2048)
            )
            
            # Generate QA pairs from processed chunks
            qa_pairs = self.synthetic_generator.generate_qa_pairs(
                processed_data['chunks'],
                num_pairs_per_chunk=self.config.get('synthetic_data.pairs_per_chunk', 25),
                temperature=self.config.get('synthetic_data.temperature', 0.7),
                domain_topic=self.config.get('domain_topic')
            )
            
            # Curate and format for training
            training_dataset = self.synthetic_generator.prepare_training_dataset(qa_pairs)
            
            # Save training dataset
            self.synthetic_generator.save_dataset(training_dataset)
            
            # Cleanup memory
            self.synthetic_generator.cleanup()
            
        self.logger.info(f"Synthetic data generation completed. {len(training_dataset)} QA pairs generated")
        return training_dataset
    
    def run_fine_tuning(self, training_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Fine-tune the model using Unsloth with LoRA/QLoRA."""
        self.logger.info("Starting fine-tuning phase...")
        
        with self.monitor.track_stage("fine_tuning"):
            # Load base model with Unsloth optimizations
            model, tokenizer = self.fine_tuner.load_base_model(
                model_name=self.config.get('fine_tuning.base_model', 'unsloth/Llama-3.2-3B-Instruct'),
                max_seq_length=self.config.get('fine_tuning.max_seq_length', 2048),
                load_in_4bit=self.config.get('fine_tuning.load_in_4bit', True)
            )
            
            # Apply LoRA adapters
            model = self.fine_tuner.apply_lora(
                model,
                r=self.config.get('fine_tuning.lora_r', 16),
                target_modules=self.config.get('fine_tuning.target_modules'),
                lora_alpha=self.config.get('fine_tuning.lora_alpha', 16)
            )
            
            # Prepare dataset for training
            formatted_dataset = self.fine_tuner.format_dataset(training_dataset, tokenizer)
            
            # Train the model
            training_results = self.fine_tuner.train(
                model=model,
                tokenizer=tokenizer,
                dataset=formatted_dataset,
                training_config=self.config.get('fine_tuning.training_params', {})
            )
            
            # Save fine-tuned model
            model_info = self.fine_tuner.save_model(
                model, tokenizer,
                save_path=self.config.get('fine_tuning.output_path', 'models/fine_tuned'),
                save_method=self.config.get('fine_tuning.save_method', 'lora')
            )
            
        self.logger.info("Fine-tuning completed successfully")
        return {
            'model_info': model_info,
            'training_results': training_results,
            'model_path': model_info['path']
        }
    
    def run_evaluation(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Evaluate the fine-tuned model against baseline."""
        self.logger.info("Starting evaluation phase...")
        
        with self.monitor.track_stage("evaluation"):
            # Create domain-specific benchmark dataset
            benchmark_dataset = self.evaluator.create_benchmark_dataset(
                domain=self.config.get('domain_topic'),
                size=self.config.get('evaluation.benchmark_size', 100)
            )
            
            # Load fine-tuned model for evaluation
            fine_tuned_model = self.evaluator.load_model(model_info['model_path'])
            
            # Load baseline model for comparison
            baseline_model = self.evaluator.load_baseline_model(
                self.config.get('evaluation.baseline_model', 'unsloth/Llama-3.2-3B-Instruct')
            )
            
            # Run evaluations
            fine_tuned_results = self.evaluator.evaluate_model(
                fine_tuned_model, benchmark_dataset, "fine_tuned"
            )
            
            baseline_results = self.evaluator.evaluate_model(
                baseline_model, benchmark_dataset, "baseline"
            )
            
            # Compare results
            comparison = self.evaluator.compare_models(
                fine_tuned_results, baseline_results
            )
            
            # Performance metrics
            performance_metrics = self.evaluator.calculate_performance_metrics(
                fine_tuned_model, benchmark_dataset
            )
            
            evaluation_results = {
                'fine_tuned_results': fine_tuned_results,
                'baseline_results': baseline_results,
                'comparison': comparison,
                'performance_metrics': performance_metrics,
                'benchmark_dataset_size': len(benchmark_dataset)
            }
            
            # Save evaluation results
            self.evaluator.save_evaluation_results(evaluation_results)
            
        self.logger.info("Evaluation completed successfully")
        return evaluation_results
    
    def run_deployment(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Deploy the model for production serving."""
        self.logger.info("Starting deployment phase...")
        
        with self.monitor.track_stage("deployment"):
            # Register model version
            model_version = self.server.register_model(
                model_path=model_info['model_path'],
                model_metadata=model_info,
                version_tag=self.config.get('deployment.version_tag', 'latest')
            )
            
            # Start inference server
            server_info = self.server.start_server(
                model_version=model_version,
                port=self.config.get('deployment.port', 8000),
                host=self.config.get('deployment.host', '0.0.0.0'),
                workers=self.config.get('deployment.workers', 1)
            )
            
            # Setup API endpoints with authentication
            api_endpoints = self.server.setup_api_endpoints(
                authentication=self.config.get('deployment.authentication', True)
            )
            
            # Initialize monitoring
            monitoring_info = self.server.setup_monitoring(
                metrics_port=self.config.get('deployment.metrics_port', 9090)
            )
            
            deployment_info = {
                'model_version': model_version,
                'server_info': server_info,
                'api_endpoints': api_endpoints,
                'monitoring': monitoring_info
            }
            
        self.logger.info(f"Model deployed successfully on {server_info['host']}:{server_info['port']}")
        return deployment_info
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete end-to-end pipeline."""
        self.logger.info("Starting full pipeline execution...")
        
        try:
            with self.monitor.track_pipeline():
                # Step 1: Data Collection
                raw_data = self.run_data_collection()
                
                # Step 2: Data Processing
                processed_data = self.run_data_processing(raw_data)
                
                # Step 3: Synthetic Data Generation
                training_dataset = self.run_synthetic_data_generation(processed_data)
                
                # Step 4: Fine-tuning
                model_info = self.run_fine_tuning(training_dataset)
                
                # Step 5: Evaluation
                evaluation_results = self.run_evaluation(model_info)
                
                # Step 6: Deployment
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
            pipeline_results = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise
        
        self.logger.info("Full pipeline execution completed successfully")
        return pipeline_results

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
        elif args.stage == "data_collection":
            results = pipeline.run_data_collection()
        elif args.stage == "data_processing":
            # Load previous data if resuming
            raw_data = pipeline.pipeline_manager.load_checkpoint("data_collection") if args.resume else {}
            results = pipeline.run_data_processing(raw_data)
        elif args.stage == "synthetic_data":
            processed_data = pipeline.pipeline_manager.load_checkpoint("data_processing") if args.resume else {}
            results = pipeline.run_synthetic_data_generation(processed_data)
        elif args.stage == "fine_tuning":
            training_dataset = pipeline.pipeline_manager.load_checkpoint("synthetic_data") if args.resume else {}
            results = pipeline.run_fine_tuning(training_dataset)
        elif args.stage == "evaluation":
            model_info = pipeline.pipeline_manager.load_checkpoint("fine_tuning") if args.resume else {}
            results = pipeline.run_evaluation(model_info)
        elif args.stage == "deployment":
            model_info = pipeline.pipeline_manager.load_checkpoint("fine_tuning") if args.resume else {}
            results = pipeline.run_deployment(model_info)
        
        print(f"Pipeline stage '{args.stage}' completed successfully!")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()