"""
Synthetic Data Generator using Meta's Synthetic Data Kit and Unsloth
Based on the Meta Llama 3.2 implementation for QA pair generation
"""

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

try:
    from unsloth.dataprep import SyntheticDataKit
except ImportError:
    print("Warning: Unsloth not installed. Please install with: pip install unsloth")
    SyntheticDataKit = None

class SyntheticDataGenerator:
    """
    Generates synthetic QA pairs using Meta's Synthetic Data Kit with Unsloth optimizations.
    Based on the approach from Meta_Synthetic_Data_Llama3_2_(3B).ipynb
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the synthetic data generator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.generator = None
        self.base_model_name = config.get('base_model', 'unsloth/Llama-3.2-3B-Instruct')
        self.max_seq_length = config.get('max_seq_length', 2048)
        self.output_dir = Path(config.get('output_dir', 'data/synthetic'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation parameters
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.95)
        self.overlap = config.get('overlap', 64)
        self.max_generation_tokens = config.get('max_generation_tokens', 512)
        self.pairs_per_chunk = config.get('pairs_per_chunk', 25)
        
        self.logger.info(f"SyntheticDataGenerator initialized with model: {self.base_model_name}")
    
    def initialize_model(self, model_name: Optional[str] = None, max_seq_length: Optional[int] = None) -> None:
        """Initialize the synthetic data kit with the specified model."""
        if SyntheticDataKit is None:
            raise ImportError("Unsloth not available. Please install unsloth package.")
        
        model_name = model_name or self.base_model_name
        max_seq_length = max_seq_length or self.max_seq_length
        
        self.logger.info(f"Loading model {model_name} for synthetic data generation...")
        
        try:
            self.generator = SyntheticDataKit.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length
            )
            
            # Configure QA generation parameters
            self.generator.prepare_qa_generation(
                output_folder=str(self.output_dir),
                temperature=self.temperature,
                top_p=self.top_p,
                overlap=self.overlap,
                max_generation_tokens=self.max_generation_tokens
            )
            
            self.logger.info("Synthetic data generator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize synthetic data generator: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Chunk documents into smaller pieces for QA generation.
        Based on the chunking approach in the reference notebook.
        """
        self.logger.info(f"Chunking {len(documents)} documents...")
        
        # Create temporary directory for document processing
        temp_dir = self.output_dir / "temp_documents"
        temp_dir.mkdir(exist_ok=True)
        
        # Save documents as text files for processing
        document_files = []
        for i, doc in enumerate(documents):
            doc_path = temp_dir / f"document_{i}.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                # Extract text content based on document structure
                if isinstance(doc, dict):
                    content = doc.get('content', doc.get('text', str(doc)))
                else:
                    content = str(doc)
                f.write(content)
            document_files.append(str(doc_path))
        
        # Use synthetic data kit to chunk the documents
        try:
            all_chunks = []
            for doc_file in document_files:
                chunks = self.generator.chunk_data(doc_file)
                all_chunks.extend(chunks)
            
            self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Error during document chunking: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def generate_qa_pairs_from_chunks(self, chunks: List[str], domain_topic: str = "") -> List[Dict[str, Any]]:
        """
        Generate QA pairs from document chunks using synthetic-data-kit CLI.
        Based on the approach in the reference notebook.
        """
        self.logger.info(f"Generating QA pairs from {len(chunks)} chunks...")
        
        all_qa_pairs = []
        config_path = self._create_sdk_config()
        
        try:
            # Process chunks in batches to avoid memory issues
            for i, chunk_file in enumerate(chunks[:3]):  # Limit to first 3 chunks as in reference
                self.logger.info(f"Processing chunk {i+1}/{min(len(chunks), 3)}: {chunk_file}")
                
                # Generate QA pairs using CLI
                cmd = [
                    "synthetic-data-kit",
                    "-c", str(config_path),
                    "create", chunk_file,
                    "--num-pairs", str(self.pairs_per_chunk),
                    "--type", "qa"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        self.logger.warning(f"Failed to generate QA pairs for chunk {i}: {result.stderr}")
                        continue
                    
                    # Load generated QA pairs
                    qa_file = self.output_dir / "generated" / f"{Path(chunk_file).stem}_qa_pairs.json"
                    if qa_file.exists():
                        with open(qa_file, 'r', encoding='utf-8') as f:
                            chunk_qa_pairs = json.load(f)
                            all_qa_pairs.extend(chunk_qa_pairs)
                    
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Timeout generating QA pairs for chunk {i}")
                except Exception as e:
                    self.logger.warning(f"Error processing chunk {i}: {str(e)}")
                
                # Sleep between chunks to avoid overwhelming the system
                time.sleep(2)
            
            self.logger.info(f"Generated {len(all_qa_pairs)} QA pairs from chunks")
            return all_qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error during QA pair generation: {str(e)}")
            raise
    
    def generate_qa_pairs(self, documents: List[Dict[str, Any]], 
                         num_pairs_per_chunk: Optional[int] = None,
                         temperature: Optional[float] = None,
                         domain_topic: str = "") -> List[Dict[str, Any]]:
        """
        Main method to generate QA pairs from documents.
        """
        num_pairs_per_chunk = num_pairs_per_chunk or self.pairs_per_chunk
        temperature = temperature or self.temperature
        
        self.logger.info(f"Starting QA pair generation for domain: {domain_topic}")
        
        # Step 1: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 2: Generate QA pairs from chunks
        qa_pairs = self.generate_qa_pairs_from_chunks(chunks, domain_topic)
        
        # Step 3: Post-process and enhance QA pairs
        enhanced_qa_pairs = self._enhance_qa_pairs(qa_pairs, domain_topic)
        
        return enhanced_qa_pairs
    
    def prepare_training_dataset(self, qa_pairs: List[Dict[str, Any]]) -> Dataset:
        """
        Convert QA pairs to training dataset format compatible with Llama 3.2.
        Based on the format used in the reference notebook.
        """
        self.logger.info(f"Preparing training dataset from {len(qa_pairs)} QA pairs...")
        
        # Convert QA pairs to conversation format
        conversations = []
        for qa_pair in qa_pairs:
            # Handle different QA pair formats
            if isinstance(qa_pair, dict):
                if 'question' in qa_pair and 'answer' in qa_pair:
                    question = qa_pair['question']
                    answer = qa_pair['answer']
                elif 'input' in qa_pair and 'output' in qa_pair:
                    question = qa_pair['input']
                    answer = qa_pair['output']
                else:
                    # Try to extract from messages format
                    messages = qa_pair.get('messages', [])
                    if len(messages) >= 2:
                        question = messages[-2].get('content', '')
                        answer = messages[-1].get('content', '')
                    else:
                        continue
            else:
                continue
            
            # Create conversation in Llama 3.2 format
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant specialized in {self.config.get('domain_topic', 'general knowledge')}."
                    },
                    {
                        "role": "user", 
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            }
            conversations.append(conversation)
        
        # Convert to pandas DataFrame then to Dataset
        df = pd.DataFrame(conversations)
        dataset = Dataset.from_pandas(df)
        
        self.logger.info(f"Created training dataset with {len(dataset)} examples")
        return dataset
    
    def save_dataset(self, dataset: Dataset, filename: Optional[str] = None) -> str:
        """Save the training dataset to disk."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"training_dataset_{timestamp}.json"
        
        save_path = self.output_dir / filename
        
        # Save as JSON for compatibility
        dataset.to_json(str(save_path))
        
        self.logger.info(f"Training dataset saved to {save_path}")
        return str(save_path)
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.generator is not None:
            try:
                self.generator.cleanup()
                self.logger.info("Synthetic data generator cleaned up successfully")
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {str(e)}")
        
        self.generator = None
    
    def _create_sdk_config(self) -> Path:
        """Create configuration file for synthetic-data-kit CLI."""
        config_data = {
            'generator': {
                'model_name': self.base_model_name,
                'max_seq_length': self.max_seq_length,
                'dtype': 'auto',
                'load_in_4bit': True
            },
            'generation': {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'max_generation_tokens': self.max_generation_tokens,
                'overlap': self.overlap
            },
            'qa_generation': {
                'num_pairs': self.pairs_per_chunk,
                'question_types': ['factual', 'analytical', 'explanatory'],
                'answer_length': {'min': 20, 'max': 200}
            },
            'output': {
                'folder': str(self.output_dir),
                'format': 'json',
                'save_intermediate': True
            }
        }
        
        config_path = self.output_dir / "synthetic_data_kit_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        return config_path
    
    def _enhance_qa_pairs(self, qa_pairs: List[Dict[str, Any]], domain_topic: str) -> List[Dict[str, Any]]:
        """Post-process and enhance generated QA pairs."""
        enhanced_pairs = []
        
        for qa_pair in qa_pairs:
            # Add domain context if missing
            if domain_topic and 'context' not in qa_pair:
                qa_pair['context'] = domain_topic
            
            # Ensure required fields exist
            if self._validate_qa_pair(qa_pair):
                enhanced_pairs.append(qa_pair)
        
        self.logger.info(f"Enhanced {len(enhanced_pairs)} valid QA pairs")
        return enhanced_pairs
    
    def _validate_qa_pair(self, qa_pair: Dict[str, Any]) -> bool:
        """Validate that a QA pair has required fields and quality."""
        required_fields = ['question', 'answer']
        
        # Check for different field name variations
        if 'input' in qa_pair and 'output' in qa_pair:
            qa_pair['question'] = qa_pair['input']
            qa_pair['answer'] = qa_pair['output']
        
        # Validate required fields exist and are non-empty
        for field in required_fields:
            if field not in qa_pair or not qa_pair[field] or len(qa_pair[field].strip()) < 10:
                return False
        
        return True
    
    def convert_to_ft_format(self, qa_pairs_files: List[str]) -> List[str]:
        """
        Convert QA pairs to fine-tuning format using synthetic-data-kit.
        Based on the save-as command in the reference notebook.
        """
        ft_files = []
        config_path = self._create_sdk_config()
        
        for qa_file in qa_pairs_files:
            try:
                # Convert to fine-tuning format
                cmd = [
                    "synthetic-data-kit",
                    "-c", str(config_path),
                    "save-as", qa_file,
                    "-f", "ft"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    # Determine output file path
                    base_name = Path(qa_file).stem
                    ft_file = self.output_dir / "final" / f"{base_name}_ft.json"
                    if ft_file.exists():
                        ft_files.append(str(ft_file))
                else:
                    self.logger.warning(f"Failed to convert {qa_file} to FT format: {result.stderr}")
                    
            except Exception as e:
                self.logger.warning(f"Error converting {qa_file}: {str(e)}")
        
        return ft_files
    
    def load_and_combine_datasets(self, ft_files: List[str]) -> Dataset:
        """Load and combine multiple fine-tuning datasets."""
        all_conversations = []
        
        for ft_file in ft_files:
            try:
                df = pd.read_json(ft_file)
                all_conversations.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to load {ft_file}: {str(e)}")
        
        if all_conversations:
            combined_df = pd.concat(all_conversations, ignore_index=True)
            dataset = Dataset.from_pandas(combined_df)
            self.logger.info(f"Combined dataset with {len(dataset)} examples")
            return dataset
        else:
            raise ValueError("No valid datasets found to combine")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the data generation process."""
        output_dir = Path(self.output_dir)
        
        stats = {
            'total_chunks': 0,
            'total_qa_pairs': 0,
            'generated_files': [],
            'final_datasets': []
        }
        
        # Count generated files
        generated_dir = output_dir / "generated"
        if generated_dir.exists():
            qa_files = list(generated_dir.glob("*_qa_pairs.json"))
            stats['generated_files'] = [str(f) for f in qa_files]
            
            # Count total QA pairs
            for qa_file in qa_files:
                try:
                    with open(qa_file, 'r') as f:
                        qa_data = json.load(f)
                        stats['total_qa_pairs'] += len(qa_data)
                except:
                    pass
        
        # Count final datasets
        final_dir = output_dir / "final"
        if final_dir.exists():
            ft_files = list(final_dir.glob("*_ft.json"))
            stats['final_datasets'] = [str(f) for f in ft_files]
        
        return stats