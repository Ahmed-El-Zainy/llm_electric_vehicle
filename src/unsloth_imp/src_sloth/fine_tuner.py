"""
Fine-tuning implementation using Unsloth for memory-efficient training with LoRA/QLoRA.
Based on the Unsloth approach from Meta_Synthetic_Data_Llama3_2_(3B).ipynb
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datasets import Dataset
import pandas as pd

try:
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, TextStreamer
    from trl import SFTTrainer, SFTConfig
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Warning: Unsloth not available. Please install with: pip install unsloth")
    UNSLOTH_AVAILABLE = False

@dataclass
class TrainingMetrics:
    """Data class to store training metrics and statistics."""
    train_runtime: float
    train_samples_per_second: float
    train_steps_per_second: float
    train_loss: float
    eval_loss: Optional[float] = None
    peak_memory_gb: float = 0.0
    memory_percentage: float = 0.0

class FineTuner:
    """
    Fine-tuning implementation using Unsloth for memory-efficient training.
    Supports LoRA and QLoRA for parameter-efficient fine-tuning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the fine-tuner with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.base_model_name = config.get('base_model', 'unsloth/Llama-3.2-3B-Instruct')
        self.max_seq_length = config.get('max_seq_length', 2048)
        self.load_in_4bit = config.get('load_in_4bit', True)
        self.load_in_8bit = config.get('load_in_8bit', False)
        self.full_finetuning = config.get('full_finetuning', False)
        
        # LoRA configuration
        self.lora_r = config.get('lora_r', 16)
        self.lora_alpha = config.get('lora_alpha', 16)
        self.lora_dropout = config.get('lora_dropout', 0.0)
        self.target_modules = config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        self.bias = config.get('bias', 'none')
        self.use_rslora = config.get('use_rslora', False)
        
        # Training parameters
        self.training_params = config.get('training_params', {})
        self.output_path = Path(config.get('output_path', 'models/fine_tuned'))
        self.save_method = config.get('save_method', 'lora')
        
        # Initialize model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        self.logger.info(f"FineTuner initialized with base model: {self.base_model_name}")
    
    def load_base_model(self, model_name: Optional[str] = None, 
                       max_seq_length: Optional[int] = None,
                       load_in_4bit: Optional[bool] = None) -> Tuple[Any, Any]:
        """
        Load the base model using Unsloth FastLanguageModel.
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available. Please install unsloth package.")
        
        model_name = model_name or self.base_model_name
        max_seq_length = max_seq_length or self.max_seq_length
        load_in_4bit = load_in_4bit if load_in_4bit is not None else self.load_in_4bit
        
        self.logger.info(f"Loading base model: {model_name}")
        
        try:
            # Get memory stats before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                start_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
                gpu_stats = torch.cuda.get_device_properties(0)
                max_memory = gpu_stats.total_memory / 1024 / 1024 / 1024
                
                self.logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory:.2f} GB")
                self.logger.info(f"Memory before loading: {start_memory:.2f} GB")
            
            # Load model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                full_finetuning=self.full_finetuning,
                dtype=None,  # Auto-detect
                # token="hf_...",  # Use if using gated models
            )
            
            # Log memory usage after loading
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
                memory_used = current_memory - start_memory
                self.logger.info(f"Model loaded. Memory used: {memory_used:.2f} GB")
            
            self.model = model
            self.tokenizer = tokenizer
            
            self.logger.info("Base model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def save_model(self, model: Any, tokenizer: Any, save_path: Optional[str] = None,
                   save_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the fine-tuned model in various formats.
        """
        save_path = Path(save_path) if save_path else self.output_path
        save_method = save_method or self.save_method
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving model using method: {save_method}")
        
        try:
            model_info = {
                'save_method': save_method,
                'path': str(save_path),
                'base_model': self.base_model_name,
                'lora_config': {
                    'r': self.lora_r,
                    'alpha': self.lora_alpha,
                    'target_modules': self.target_modules
                }
            }
            
            if save_method == 'lora':
                # Save only LoRA adapters
                lora_path = save_path / "lora_adapters"
                model.save_pretrained(str(lora_path))
                tokenizer.save_pretrained(str(lora_path))
                model_info['lora_path'] = str(lora_path)
                
            elif save_method == 'merged_16bit':
                # Save merged model in 16-bit
                merged_path = save_path / "merged_16bit"
                model.save_pretrained_merged(
                    str(merged_path), 
                    tokenizer, 
                    save_method="merged_16bit"
                )
                model_info['merged_path'] = str(merged_path)
                
            elif save_method == 'merged_4bit':
                # Save merged model in 4-bit
                merged_path = save_path / "merged_4bit"
                model.save_pretrained_merged(
                    str(merged_path), 
                    tokenizer, 
                    save_method="merged_4bit"
                )
                model_info['merged_path'] = str(merged_path)
                
            elif save_method == 'gguf':
                # Save in GGUF format for llama.cpp compatibility
                gguf_path = save_path / "gguf"
                model.save_pretrained_gguf(
                    str(gguf_path), 
                    tokenizer,
                    quantization_method="q8_0"
                )
                model_info['gguf_path'] = str(gguf_path)
                
            else:
                raise ValueError(f"Unsupported save method: {save_method}")
            
            # Save model metadata
            metadata_path = save_path / "model_info.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Model saved successfully to {save_path}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_fine_tuned_model(self, model_path: str, 
                             model_info: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
        """
        Load a previously fine-tuned model.
        """
        model_path = Path(model_path)
        
        # Load model info if not provided
        if model_info is None:
            info_path = model_path / "model_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                raise FileNotFoundError(f"Model info not found at {info_path}")
        
        save_method = model_info.get('save_method', 'lora')
        
        self.logger.info(f"Loading fine-tuned model from {model_path} (method: {save_method})")
        
        try:
            if save_method == 'lora':
                # Load LoRA adapters
                lora_path = model_path / "lora_adapters"
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(lora_path),
                    max_seq_length=self.max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                )
                
            elif save_method in ['merged_16bit', 'merged_4bit']:
                # Load merged model
                merged_path = model_path / save_method
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(merged_path),
                    max_seq_length=self.max_seq_length,
                    dtype=None,
                    load_in_4bit=(save_method == 'merged_4bit'),
                )
                
            else:
                raise ValueError(f"Loading not supported for save method: {save_method}")
            
            # Enable inference mode
            FastLanguageModel.for_inference(model)
            
            self.logger.info("Fine-tuned model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load fine-tuned model: {str(e)}")
            raise
    
    def inference(self, model: Any, tokenizer: Any, prompt: str,
                  max_new_tokens: int = 256, temperature: float = 0.1,
                  stream: bool = False) -> str:
        """
        Run inference on the fine-tuned model.
        """
        # Format prompt as conversation
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if stream:
                # Streaming generation
                text_streamer = TextStreamer(tokenizer, skip_prompt=True)
                _ = model.generate(
                    input_ids=inputs, 
                    streamer=text_streamer,
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
                return ""  # Text is streamed, not returned
            else:
                # Regular generation
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode only the generated part
                generated_ids = outputs[0][len(inputs[0]):]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                return response.strip()
                
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            raise
    
    def get_model_size(self, model: Any) -> Dict[str, Any]:
        """Get information about model size and parameters."""
        try:
            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Calculate model size in memory
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            size_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'trainable_percentage': (trainable_params / total_params) * 100,
                'model_size_mb': model_size_mb,
                'model_size_gb': model_size_mb / 1024
            }
            
            return size_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model size info: {str(e)}")
            return {}
    
    def _simple_format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Simple fallback conversation formatting."""
        formatted = ""
        for message in conversation:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted += f"System: {content}\n"
            elif role == 'user':
                formatted += f"User: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
        
        return formatted
    
    def validate_training_config(self) -> bool:
        """Validate training configuration parameters."""
        required_params = ['per_device_train_batch_size', 'learning_rate']
        
        for param in required_params:
            if param not in self.training_params:
                self.logger.error(f"Missing required training parameter: {param}")
                return False
        
        # Validate batch size
        batch_size = self.training_params.get('per_device_train_batch_size', 2)
        if batch_size < 1:
            self.logger.error("Batch size must be at least 1")
            return False
        
        # Validate learning rate
        lr = self.training_params.get('learning_rate', 2e-4)
        if lr <= 0 or lr > 1:
            self.logger.error("Learning rate must be between 0 and 1")
            return False
        
        return True
    
    def estimate_memory_requirements(self, dataset_size: int) -> Dict[str, float]:
        """Estimate memory requirements for training."""
        # Base memory estimates (in GB)
        base_model_memory = {
            '3B': 6.0,
            '7B': 14.0,
            '13B': 26.0
        }
        
        # Extract model size from name
        model_size = '3B'  # Default
        if '7B' in self.base_model_name:
            model_size = '7B'
        elif '13B' in self.base_model_name:
            model_size = '13B'
        
        base_memory = base_model_memory.get(model_size, 6.0)
        
        # Estimate additional memory for training
        batch_size = self.training_params.get('per_device_train_batch_size', 2)
        gradient_accumulation = self.training_params.get('gradient_accumulation_steps', 4)
        effective_batch_size = batch_size * gradient_accumulation
        
        # Additional memory for gradients, optimizer states, etc.
        training_overhead = base_memory * 2.0  # Rough estimate
        batch_memory = effective_batch_size * 0.1  # Rough estimate per batch
        
        total_estimated = base_memory + training_overhead + batch_memory
        
        # Apply reduction for quantization
        if self.load_in_4bit:
            total_estimated *= 0.5
        elif self.load_in_8bit:
            total_estimated *= 0.75
        
        return {
            'base_model_memory_gb': base_memory,
            'training_overhead_gb': training_overhead,
            'batch_memory_gb': batch_memory,
            'total_estimated_gb': total_estimated,
            'quantization_applied': self.load_in_4bit or self.load_in_8bit
        }
    
    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("FineTuner cleanup completed"):
            self.logger.error(f"Failed to load base model: {str(e)}")
            raise
    
    def apply_lora(self, model: Any, r: Optional[int] = None,
                   target_modules: Optional[List[str]] = None,
                   lora_alpha: Optional[int] = None) -> Any:
        """
        Apply LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available for LoRA application.")
        
        r = r or self.lora_r
        target_modules = target_modules or self.target_modules
        lora_alpha = lora_alpha or self.lora_alpha
        
        self.logger.info(f"Applying LoRA with r={r}, alpha={lora_alpha}")
        self.logger.info(f"Target modules: {target_modules}")
        
        try:
            # Apply LoRA using Unsloth
            model = FastLanguageModel.get_peft_model(
                model,
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
                random_state=3407,
                use_rslora=self.use_rslora,
                loftq_config=None,
            )
            
            # Log LoRA configuration
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            
            self.logger.info("LoRA applied successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to apply LoRA: {str(e)}")
            raise
    
    def format_dataset(self, dataset: Dataset, tokenizer: Any) -> Dataset:
        """
        Format the dataset for fine-tuning using Llama 3.2 chat template.
        Based on the formatting approach in the reference notebook.
        """
        self.logger.info(f"Formatting dataset with {len(dataset)} examples")
        
        def formatting_prompts_func(examples):
            """Format conversations using the chat template."""
            convos = examples["messages"]
            texts = []
            
            for convo in convos:
                try:
                    # Apply chat template without tokenization for training
                    text = tokenizer.apply_chat_template(
                        convo, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    self.logger.warning(f"Failed to format conversation: {str(e)}")
                    # Fallback to simple format
                    texts.append(self._simple_format_conversation(convo))
            
            return {"text": texts}
        
        # Apply formatting
        formatted_dataset = dataset.map(
            formatting_prompts_func, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        self.logger.info(f"Dataset formatted successfully. Sample length: {len(formatted_dataset[0]['text'])}")
        return formatted_dataset
    
    def train(self, model: Any, tokenizer: Any, dataset: Dataset,
              training_config: Optional[Dict[str, Any]] = None) -> TrainingMetrics:
        """
        Train the model using Hugging Face TRL's SFTTrainer.
        """
        training_config = training_config or self.training_params
        
        self.logger.info("Starting model training...")
        self.logger.info(f"Training dataset size: {len(dataset)}")
        
        # Get memory stats before training
        start_gpu_memory = 0
        max_memory = 0
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
            max_memory = gpu_stats.total_memory / 1024 / 1024 / 1024
            self.logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory:.2f} GB")
            self.logger.info(f"Memory before training: {start_gpu_memory:.2f} GB")
        
        try:
            # Configure training arguments using SFTConfig
            sft_config = SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
                gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
                warmup_steps=training_config.get('warmup_steps', 5),
                max_steps=training_config.get('max_steps', 60),
                learning_rate=training_config.get('learning_rate', 2e-4),
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=training_config.get('logging_steps', 1),
                optim=training_config.get('optim', 'adamw_8bit'),
                weight_decay=training_config.get('weight_decay', 0.01),
                lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
                seed=training_config.get('seed', 3407),
                output_dir=str(self.output_path / "checkpoints"),
                report_to=training_config.get('report_to', 'none'),
                save_strategy="steps",
                save_steps=training_config.get('save_steps', 20),
                dataloader_num_workers=0,  # Avoid multiprocessing issues
            )
            
            # Initialize trainer
            self.trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                eval_dataset=None,  # Can be added for validation
                args=sft_config,
            )
            
            # Start training
            self.logger.info("Starting training process...")
            trainer_stats = self.trainer.train()
            
            # Calculate memory usage
            used_memory = 0
            memory_percentage = 0
            if torch.cuda.is_available():
                used_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
                used_memory_for_training = used_memory - start_gpu_memory
                memory_percentage = (used_memory / max_memory) * 100
                
                self.logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
                self.logger.info(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
                self.logger.info(f"Peak memory usage: {used_memory:.2f} GB")
                self.logger.info(f"Memory used for training: {used_memory_for_training:.2f} GB")
                self.logger.info(f"Memory usage percentage: {memory_percentage:.1f}%")
            
            # Create training metrics
            metrics = TrainingMetrics(
                train_runtime=trainer_stats.metrics['train_runtime'],
                train_samples_per_second=trainer_stats.metrics.get('train_samples_per_second', 0),
                train_steps_per_second=trainer_stats.metrics.get('train_steps_per_second', 0),
                train_loss=trainer_stats.metrics.get('train_loss', 0),
                peak_memory_gb=used_memory,
                memory_percentage=memory_percentage
            )
            
            self.logger.info("Training completed successfully")
            return metrics
            
        except Exception as e