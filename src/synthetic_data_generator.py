"""
Usage:
    python synthetic_data_generator.py --input path/to/file.pdf
    python synthetic_data_generator.py --input path/to/pdf_folder/
    python synthetic_data_generator.py --input https://arxiv.org/html/2412.09871v1
    python synthetic_data_generator.py --config custom_config.yaml --input data/pdfs/ --chunks 5
"""

import os
import sys
import json
import time
import yaml
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
import pandas as pd
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


class SyntheticDataGenerator:
    """
    Complete synthetic data generation pipeline using Meta's Synthetic Data Kit.
    Handles PDF files, folders, and URLs to generate QA pairs for fine-tuning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the synthetic data generator."""
        self.config_path = config_path or "synthetic_data_kit_config.yaml"
        self.generator = None
        self.output_folder = "data"
        self.generated_files = []
        
    def setup_generator(self, 
                       model_name: str = "unsloth/Llama-3.2-3B-Instruct",
                       max_seq_length: int = 2048,
                       temperature: float = 0.7,
                       top_p: float = 0.95,
                       overlap: int = 64,
                       max_generation_tokens: int = 512,
                       output_folder: str = "data"):
        """Setup the synthetic data kit generator."""
        
        logger.info(f"Setting up SyntheticDataKit with model: {model_name}")
        
        try:
            from unsloth.dataprep import SyntheticDataKit
            
            self.generator = SyntheticDataKit.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length
            )
            
            self.output_folder = output_folder
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            
            self.generator.prepare_qa_generation(
                output_folder=output_folder,
                temperature=temperature,
                top_p=top_p,
                overlap=overlap,
                max_generation_tokens=max_generation_tokens
            )
            
            logger.info("SyntheticDataKit setup completed successfully")
            
        except ImportError as e:
            logger.error("Failed to import SyntheticDataKit. Please install unsloth package.")
            raise e
        except Exception as e:
            logger.error(f"Failed to setup SyntheticDataKit: {str(e)}")
            raise e
    
    def create_config_file(self, config_path: str = None):
        """Create a synthetic-data-kit configuration file."""
        config_path = config_path or self.config_path
        
        config_data = {
            'generator': {
                'model_name': 'unsloth/Llama-3.2-3B-Instruct',
                'max_seq_length': 2048,
                'dtype': 'auto',
                'load_in_4bit': True
            },
            'generation': {
                'temperature': 0.7,
                'top_p': 0.95,
                'max_generation_tokens': 512,
                'overlap': 64
            },
            'qa_generation': {
                'num_pairs': 25,
                'question_types': ['factual', 'analytical', 'explanatory'],
                'answer_length': {'min': 20, 'max': 200}
            },
            'output': {
                'folder': self.output_folder,
                'format': 'json',
                'save_intermediate': True
            },
            'curation': {
                'threshold': 0.7,
                'filters': ['relevance', 'quality', 'coherence'],
                'max_examples': 1000
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration file created: {config_path}")
        return config_path
    
    def system_check(self):
        """Run synthetic-data-kit system check."""
        logger.info("Running system check...")
        
        try:
            result = subprocess.run(['synthetic-data-kit', 'system-check'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ System check passed")
                logger.info(result.stdout)
            else:
                logger.warning("⚠️ System check failed")
                logger.warning(result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error("System check timed out")
        except FileNotFoundError:
            logger.error("synthetic-data-kit command not found. Please install synthetic-data-kit package.")
        except Exception as e:
            logger.error(f"System check failed: {str(e)}")
    
    def process_pdf_file(self, pdf_path: str) -> str:
        """Process a single PDF file and extract text content."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF using synthetic-data-kit ingest
        try:
            output_name = pdf_path.stem.replace(' ', '_').replace('-', '_')
            
            # Use synthetic-data-kit to ingest the PDF
            cmd = [
                'synthetic-data-kit',
                '-c', self.config_path,
                'ingest', str(pdf_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"Failed to ingest PDF {pdf_path}: {result.stderr}")
                return None
            
            # Find the generated text file
            expected_output = Path(self.output_folder) / 'output' / f"{output_name}.txt"
            
            # Look for any .txt file in output folder if expected name not found
            if not expected_output.exists():
                output_dir = Path(self.output_folder) / 'output'
                if output_dir.exists():
                    txt_files = list(output_dir.glob('*.txt'))
                    if txt_files:
                        expected_output = txt_files[-1]  # Use the most recent
            
            if expected_output.exists():
                logger.info(f"✅ PDF processed successfully: {expected_output}")
                return str(expected_output)
            else:
                logger.error(f"❌ Expected output file not found: {expected_output}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"PDF processing timed out: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
    
    def process_url(self, url: str) -> str:
        """Process a URL and extract content."""
        logger.info(f"Processing URL: {url}")
        
        try:
            # Use synthetic-data-kit to ingest the URL
            cmd = [
                'synthetic-data-kit',
                '-c', self.config_path,
                'ingest', url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"Failed to ingest URL {url}: {result.stderr}")
                return None
            
            # Find the generated text file
            output_dir = Path(self.output_folder) / 'output'
            if output_dir.exists():
                txt_files = list(output_dir.glob('*.txt'))
                if txt_files:
                    # Sort by modification time and get the most recent
                    latest_file = max(txt_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"✅ URL processed successfully: {latest_file}")
                    return str(latest_file)
            
            logger.error(f"❌ No output file found for URL: {url}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"URL processing timed out: {url}")
            return None
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return None
    
    def chunk_document(self, input_file: str) -> List[str]:
        """Chunk a document into smaller pieces for processing."""
        logger.info(f"Chunking document: {input_file}")
        
        try:
            filenames = self.generator.chunk_data(input_file)
            logger.info(f"Document chunked into {len(filenames)} pieces")
            logger.info(f"First 3 chunks: {filenames[:3]}")
            return filenames
            
        except Exception as e:
            logger.error(f"Error chunking document {input_file}: {str(e)}")
            return []
    
    def generate_qa_pairs(self, chunk_files: List[str], num_pairs: int = 25, max_chunks: Optional[int] = None) -> List[str]:
        """Generate QA pairs from chunked documents."""
        if max_chunks:
            chunk_files = chunk_files[:max_chunks]
        
        logger.info(f"Generating QA pairs from {len(chunk_files)} chunks ({num_pairs} pairs per chunk)")
        
        qa_files = []
        
        for i, filename in enumerate(chunk_files):
            logger.info(f"Processing chunk {i+1}/{len(chunk_files)}: {Path(filename).name}")
            
            try:
                # Generate QA pairs using synthetic-data-kit
                cmd = [
                    'synthetic-data-kit',
                    '-c', self.config_path,
                    'create', filename,
                    '--num-pairs', str(num_pairs),
                    '--type', 'qa'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Find the generated QA file
                    base_name = Path(filename).stem
                    qa_file = Path(self.output_folder) / 'generated' / f"{base_name}_qa_pairs.json"
                    
                    if qa_file.exists():
                        qa_files.append(str(qa_file))
                        logger.info(f"✅ QA pairs generated: {qa_file}")
                    else:
                        logger.warning(f"⚠️ Expected QA file not found: {qa_file}")
                else:
                    logger.error(f"❌ Failed to generate QA pairs for {filename}: {result.stderr}")
                
                # Small delay between chunks
                time.sleep(2)
                
            except subprocess.TimeoutExpired:
                logger.error(f"QA generation timed out for chunk: {filename}")
            except Exception as e:
                logger.error(f"Error generating QA pairs for {filename}: {str(e)}")
        
        logger.info(f"Generated QA pairs for {len(qa_files)} chunks")
        return qa_files
    
    def convert_to_training_format(self, qa_files: List[str]) -> List[str]:
        """Convert QA pairs to fine-tuning format."""
        logger.info(f"Converting {len(qa_files)} QA files to training format")
        
        ft_files = []
        
        for qa_file in qa_files:
            try:
                logger.info(f"Converting: {Path(qa_file).name}")
                
                # Convert to fine-tuning format
                cmd = [
                    'synthetic-data-kit',
                    '-c', self.config_path,
                    'save-as', qa_file,
                    '-f', 'ft'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Find the generated FT file
                    base_name = Path(qa_file).stem
                    ft_file = Path(self.output_folder) / 'final' / f"{base_name}_ft.json"
                    
                    if ft_file.exists():
                        ft_files.append(str(ft_file))
                        logger.info(f"✅ Converted to training format: {ft_file}")
                    else:
                        logger.warning(f"⚠️ Expected FT file not found: {ft_file}")
                else:
                    logger.error(f"❌ Failed to convert {qa_file}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error converting {qa_file}: {str(e)}")
        
        logger.info(f"Converted {len(ft_files)} files to training format")
        return ft_files
    
    def create_final_dataset(self, ft_files: List[str]) -> Dataset:
        """Create final dataset from fine-tuning files."""
        logger.info(f"Creating final dataset from {len(ft_files)} files")
        
        try:
            conversations = []
            
            for ft_file in ft_files:
                try:
                    df = pd.read_json(ft_file)
                    conversations.append(df)
                    logger.info(f"Loaded {len(df)} conversations from {Path(ft_file).name}")
                except Exception as e:
                    logger.error(f"Failed to load {ft_file}: {str(e)}")
            
            if conversations:
                combined_df = pd.concat(conversations, ignore_index=True)
                dataset = Dataset.from_pandas(combined_df)
                
                logger.info(f"✅ Final dataset created with {len(dataset)} examples")
                
                # Save combined dataset
                output_path = Path(self.output_folder) / 'final_dataset.json'
                dataset.to_json(str(output_path))
                logger.info(f"Dataset saved to: {output_path}")
                
                return dataset
            else:
                logger.error("No valid conversations found")
                return None
                
        except Exception as e:
            logger.error(f"Error creating final dataset: {str(e)}")
            return None
    
    def process_input(self, input_path: str, num_pairs: int = 25, max_chunks: Optional[int] = None) -> Dataset:
        """Process input (PDF file, folder, or URL) and generate QA dataset."""
        logger.info(f"Processing input: {input_path}")
        
        # Determine input type and process accordingly
        if input_path.startswith(('http://', 'https://')):
            # URL input
            text_file = self.process_url(input_path)
        elif Path(input_path).is_file():
            # Single file input
            if input_path.lower().endswith('.pdf'):
                text_file = self.process_pdf_file(input_path)
            else:
                raise ValueError(f"Unsupported file type: {input_path}")
        elif Path(input_path).is_dir():
            # Directory input - process all PDFs
            pdf_files = list(Path(input_path).glob('*.pdf'))
            if not pdf_files:
                raise ValueError(f"No PDF files found in directory: {input_path}")
            
            logger.info(f"Found {len(pdf_files)} PDF files in directory")
            
            # Process all PDFs and combine
            all_text_files = []
            for pdf_file in pdf_files:
                text_file = self.process_pdf_file(str(pdf_file))
                if text_file:
                    all_text_files.append(text_file)
            
            if not all_text_files:
                raise ValueError("Failed to process any PDF files")
            
            # For multiple files, process each one separately then combine
            all_qa_files = []
            for text_file in all_text_files:
                chunks = self.chunk_document(text_file)
                if chunks:
                    qa_files = self.generate_qa_pairs(chunks, num_pairs, max_chunks)
                    all_qa_files.extend(qa_files)
            
            if not all_qa_files:
                raise ValueError("Failed to generate any QA pairs")
            
            # Convert to training format
            ft_files = self.convert_to_training_format(all_qa_files)
            
            # Create final dataset
            return self.create_final_dataset(ft_files)
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        if not text_file:
            raise ValueError(f"Failed to process input: {input_path}")
        
        # Chunk the document
        chunks = self.chunk_document(text_file)
        if not chunks:
            raise ValueError(f"Failed to chunk document: {text_file}")
        
        # Generate QA pairs
        qa_files = self.generate_qa_pairs(chunks, num_pairs, max_chunks)
        if not qa_files:
            raise ValueError("Failed to generate QA pairs")
        
        # Convert to training format
        ft_files = self.convert_to_training_format(qa_files)
        if not ft_files:
            raise ValueError("Failed to convert to training format")
        
        # Create final dataset
        dataset = self.create_final_dataset(ft_files)
        if dataset is None:
            raise ValueError("Failed to create final dataset")
        
        return dataset
    
    def cleanup(self):
        """Clean up resources."""
        if self.generator:
            try:
                self.generator.cleanup()
                logger.info("Generator cleanup completed")
            except Exception as e:
                logger.warning(f"Error during cleanup: {str(e)}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA pairs from PDFs or URLs using Meta's Synthetic Data Kit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF file
  python %(prog)s --input document.pdf
  
  # Process a folder of PDFs
  python %(prog)s --input /path/to/pdf_folder/
  
  # Process a URL
  python %(prog)s --input https://arxiv.org/html/2412.09871v1
  
  # Custom configuration
  python %(prog)s --input data.pdf --config custom_config.yaml --chunks 5 --pairs 30
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input path: PDF file, folder containing PDFs, or URL')
    
    # Optional arguments
    parser.add_argument('--config', '-c', 
                       help='Path to synthetic-data-kit config file (will be created if not exists)')
    
    parser.add_argument('--model', '-m', default='unsloth/Llama-3.2-3B-Instruct',
                       help='Model name for synthetic data generation (default: unsloth/Llama-3.2-3B-Instruct)')
    
    parser.add_argument('--output', '-o', default='data',
                       help='Output folder for generated data (default: data)')
    
    parser.add_argument('--chunks', type=int,
                       help='Maximum number of chunks to process (default: all chunks)')
    
    parser.add_argument('--pairs', '-p', type=int, default=25,
                       help='Number of QA pairs to generate per chunk (default: 25)')
    
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length for the model (default: 2048)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation (default: 0.7)')
    
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Top-p for generation (default: 0.95)')
    
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum generation tokens (default: 512)')
    
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap for chunking (default: 64)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--no-system-check', action='store_true',
                       help='Skip system check')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(args.config)
        
        # Create config file if not exists
        if not args.config or not Path(args.config).exists():
            config_path = generator.create_config_file(args.config)
            generator.config_path = config_path
        
        # Run system check
        if not args.no_system_check:
            generator.system_check()
        
        # Setup the generator
        generator.setup_generator(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            temperature=args.temperature,
            top_p=args.top_p,
            overlap=args.overlap,
            max_generation_tokens=args.max_tokens,
            output_folder=args.output
        )
        
        # Process input and generate dataset
        logger.info("🚀 Starting synthetic data generation pipeline...")
        
        dataset = generator.process_input(
            input_path=args.input,
            num_pairs=args.pairs,
            max_chunks=args.chunks
        )
        
        if dataset:
            logger.info("🎉 Synthetic data generation completed successfully!")
            logger.info(f"📊 Final dataset contains {len(dataset)} examples")
            
            # Display sample data
            logger.info("📝 Sample from generated dataset:")
            if len(dataset) > 0:
                sample = dataset[0]
                if 'messages' in sample:
                    for msg in sample['messages'][:2]:  # Show first 2 messages
                        logger.info(f"  {msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}...")
            
            # Save final statistics
            stats = {
                'total_examples': len(dataset),
                'model_used': args.model,
                'chunks_processed': args.chunks or 'all',
                'pairs_per_chunk': args.pairs,
                'input_source': args.input,
                'output_folder': args.output,
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            stats_path = Path(args.output) / 'generation_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"📈 Generation statistics saved to: {stats_path}")
            
        else:
            logger.error("❌ Failed to generate dataset")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            generator.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()


    """
    # Basic usage
python synthetic_data_generator.py --input document.pdf

# Advanced usage with all options
python synthetic_data_generator.py \
    --input /path/to/pdfs/ \
    --config custom_config.yaml \
    --model unsloth/Llama-3.2-3B-Instruct \
    --output my_data \
    --chunks 5 \
    --pairs 30 \
    --temperature 0.8 \
    --verbose
    """


    """
    # Process single PDF
python synthetic_data_generator.py --input research_paper.pdf

# Process folder of PDFs with custom settings
python synthetic_data_generator.py \
    --input ./research_papers/ \
    --pairs 40 \
    --chunks 10 \
    --output synthetic_data \
    --verbose

# Process URL with custom model
python synthetic_data_generator.py \
    --input https://arxiv.org/html/2412.09871v1 \
    --model unsloth/Llama-3.2-7B-Instruct \
    --temperature 0.8 \
    --max-tokens 768

# Resume with existing config
python synthetic_data_generator.py \
    --input data.pdf \
    --config existing_config.yaml \
    --no-system-check

data/
├── output/           # Extracted text files
├── generated/        # Generated QA pairs
├── final/           # Training-ready format
├── final_dataset.json    # Combined dataset
├── generation_stats.json # Statistics
└── synthetic_data_kit_config.yaml


    """