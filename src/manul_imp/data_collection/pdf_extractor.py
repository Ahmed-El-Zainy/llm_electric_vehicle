# src/data_collection/pdf_extractor.py
import fitz 
import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import logging
from PIL import Image
import io
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


from src.config.settings import settings
from src.utils.storage import DataStorage


@dataclass
class ExtractedPDFContent:
    file_path: str
    title: str
    content: str
    pages: List[Dict]
    metadata: Dict
    images: List[Dict]
    tables: List[Dict]
    content_hash: str

@dataclass 
class PageLayout:
    page_num: int
    text_blocks: List[Dict]
    images: List[Dict]
    tables: List[Dict]
    headers: List[str]
    footers: List[str]

class PDFExtractor:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.storage = DataStorage()
        
        # Text extraction parameters
        self.min_font_size = 8
        self.header_font_threshold = 14
        self.table_detection_threshold = 0.7
    
    def extract_pdf_content(self, pdf_path: str) -> Optional[ExtractedPDFContent]:
        """Extract content from a PDF file with layout preservation"""
        try:
            self.logger.info(f"Extracting content from PDF: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = self._extract_metadata(doc, pdf_path)
            
            # Extract content from each page
            pages_content = []
            all_text = []
            all_images = []
            all_tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract page layout
                page_layout = self._extract_page_layout(page, page_num)
                pages_content.append(page_layout.__dict__)
                
                # Collect text
                page_text = self._extract_text_with_formatting(page)
                if page_text.strip():
                    all_text.append(page_text)
                
                # Extract images
                page_images = self._extract_images(page, page_num, pdf_path)
                all_images.extend(page_images)
                
                # Extract tables
                page_tables = self._extract_tables(page, page_num)
                all_tables.extend(page_tables)
            
            doc.close()
            
            # Combine all text
            full_content = '\n\n'.join(all_text)
            content_hash = hashlib.md5(full_content.encode()).hexdigest()
            
            # Get document title
            title = metadata.get('title', '') or self._extract_title_from_content(full_content) or Path(pdf_path).stem
            
            return ExtractedPDFContent(
                file_path=pdf_path,
                title=title,
                content=full_content,
                pages=pages_content,
                metadata=metadata,
                images=all_images,
                tables=all_tables,
                content_hash=content_hash
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF content from {pdf_path}: {e}")
            return None
    
    def _extract_metadata(self, doc: fitz.Document, file_path: str) -> Dict:
        """Extract PDF metadata"""
        metadata = doc.metadata
        file_stats = os.stat(file_path)
        
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
            'page_count': len(doc),
            'file_size': file_stats.st_size,
            'file_path': file_path,
            'has_ev_keywords': False,  # Will be updated after content analysis
            'language': 'en',
            'source_type': 'pdf'
        }
    
    def _extract_page_layout(self, page: fitz.Page, page_num: int) -> PageLayout:
        """Extract structured layout information from a page"""
        # Get text blocks with formatting info
        blocks = page.get_text("dict")
        
        text_blocks = []
        headers = []
        footers = []
        
        page_height = page.rect.height
        header_threshold = page_height * 0.1  # Top 10%
        footer_threshold = page_height * 0.9   # Bottom 10%
        
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        bbox = span["bbox"]
                        font_size = span["size"]
                        font_flags = span["flags"]
                        
                        block_info = {
                            'text': text,
                            'bbox': bbox,
                            'font_size': font_size,
                            'font_flags': font_flags,
                            'is_bold': bool(font_flags & 2**4),
                            'is_italic': bool(font_flags & 2**1),
                            'position_y': bbox[1]
                        }
                        
                        text_blocks.append(block_info)
                        
                        # Classify as header, footer, or body text
                        if bbox[1] < header_threshold and font_size >= self.header_font_threshold:
                            headers.append(text)
                        elif bbox[1] > footer_threshold:
                            footers.append(text)
        
        return PageLayout(
            page_num=page_num,
            text_blocks=text_blocks,
            images=[],  # Will be filled by _extract_images
            tables=[],  # Will be filled by _extract_tables
            headers=headers,
            footers=footers
        )
    
    def _extract_text_with_formatting(self, page: fitz.Page) -> str:
        """Extract text while preserving some formatting structure"""
        blocks = page.get_text("dict")
        formatted_text = []
        
        for block in blocks.get("blocks", []):
            if "lines" in block:
                block_text = []
                
                for line in block["lines"]:
                    line_text = []
                    
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        font_flags = span["flags"]
                        
                        # Add formatting markers for headers
                        if font_size >= self.header_font_threshold:
                            text = f"## {text}"
                        elif font_flags & 2**4:  # Bold
                            text = f"**{text}**"
                        
                        line_text.append(text)
                    
                    if line_text:
                        block_text.append(" ".join(line_text))
                
                if block_text:
                    formatted_text.append("\n".join(block_text))
        
        return "\n\n".join(formatted_text)
    
    def _extract_images(self, page: fitz.Page, page_num: int, pdf_path: str) -> List[Dict]:
        """Extract images from a PDF page"""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # Skip if not RGB/RGBA
                    # Convert to RGB if needed
                    if pix.n != 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    # Save image
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = Path(settings.get_data_path('raw')) / 'images' / img_filename
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    pix.save(str(img_path))
                    
                    # Extract image metadata
                    img_info = {
                        'page_num': page_num,
                        'img_index': img_index,
                        'filename': img_filename,
                        'path': str(img_path),
                        'width': pix.width,
                        'height': pix.height,
                        'size_bytes': len(pix.tobytes()),
                        'source_pdf': pdf_path
                    }
                    
                    images.append(img_info)
                
                pix = None  # Free memory
                
            except Exception as e:
                self.logger.warning(f"Could not extract image {img_index} from page {page_num}: {e}")
                continue
        
        return images
    
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract tables from a PDF page using text positioning"""
        tables = []
        
        try:
            # Get text with detailed positioning
            text_dict = page.get_text("dict")
            
            # Simple table detection based on text alignment
            potential_table_blocks = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    lines = []
                    for line in block["lines"]:
                        line_text = ""
                        x_positions = []
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            x_positions.append(span["bbox"][0])
                        
                        if line_text.strip():
                            lines.append({
                                'text': line_text.strip(),
                                'x_positions': x_positions,
                                'y_position': line["bbox"][1]
                            })
                    
                    if len(lines) > 2:  # Potential table
                        potential_table_blocks.append(lines)
            
            # Analyze blocks for table-like structure
            for block_idx, block_lines in enumerate(potential_table_blocks):
                if self._is_table_like(block_lines):
                    table_data = self._parse_table_structure(block_lines)
                    
                    table_info = {
                        'page_num': page_num,
                        'table_index': block_idx,
                        'rows': len(table_data),
                        'cols': len(table_data[0]) if table_data else 0,
                        'data': table_data,
                        'raw_text': '\n'.join([line['text'] for line in block_lines])
                    }
                    
                    tables.append(table_info)
        
        except Exception as e:
            self.logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def _is_table_like(self, lines: List[Dict]) -> bool:
        """Determine if a block of lines represents a table"""
        if len(lines) < 3:
            return False
        
        # Check for consistent column separators
        separator_patterns = [r'\t', r'\s{3,}', r'\|']
        
        for pattern in separator_patterns:
            consistent_columns = 0
            expected_cols = None
            
            for line in lines:
                text = line['text']
                columns = re.split(pattern, text)
                
                if len(columns) > 1:
                    if expected_cols is None:
                        expected_cols = len(columns)
                    elif len(columns) == expected_cols:
                        consistent_columns += 1
            
            # If more than 70% of lines have consistent columns
            if consistent_columns / len(lines) > self.table_detection_threshold:
                return True
        
        return False
    
    def _parse_table_structure(self, lines: List[Dict]) -> List[List[str]]:
        """Parse table structure from text lines"""
        table_data = []
        
        # Try different separator patterns
        separator_patterns = [r'\t', r'\s{3,}', r'\|']
        
        best_pattern = None
        max_consistency = 0
        
        for pattern in separator_patterns:
            consistency = self._calculate_pattern_consistency(lines, pattern)
            if consistency > max_consistency:
                max_consistency = consistency
                best_pattern = pattern
        
        if best_pattern:
            for line in lines:
                columns = [col.strip() for col in re.split(best_pattern, line['text'])]
                if len(columns) > 1:
                    table_data.append(columns)
        
        return table_data
    
    def _calculate_pattern_consistency(self, lines: List[Dict], pattern: str) -> float:
        """Calculate how consistently a pattern separates columns"""
        column_counts = []
        
        for line in lines:
            columns = re.split(pattern, line['text'])
            if len(columns) > 1:
                column_counts.append(len(columns))
        
        if not column_counts:
            return 0.0
        
        # Calculate consistency as inverse of standard deviation
        import statistics
        if len(set(column_counts)) == 1:
            return 1.0
        
        try:
            std_dev = statistics.stdev(column_counts)
            mean_cols = statistics.mean(column_counts)
            return max(0, 1 - (std_dev / mean_cols))
        except:
            return 0.0
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract potential title from document content"""
        lines = content.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and not line.startswith('#') and len(line) < 100:
                # Check if it looks like a title (capitalized, not too long)
                if line[0].isupper() and len(line.split()) <= 10:
                    return line
        
        return None
    
    def _has_ev_keywords(self, content: str) -> bool:
        """Check if content contains EV-related keywords"""
        ev_keywords = [
            'electric vehicle', 'ev charging', 'charging station', 'chargepoint',
            'tesla supercharger', 'fast charging', 'dc fast charging', 'level 2 charging',
            'electric car', 'battery', 'plug-in', 'ev infrastructure', 'charging network',
            'kilowatt', 'kwh', 'amperage', 'voltage', 'connector type', 'chademo', 'ccs'
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in ev_keywords)
    
    def extract_from_directory(self, directory_path: str) -> List[ExtractedPDFContent]:
        """Extract content from all PDFs in a directory"""
        if not settings.data_sources.pdf_sources['enabled']:
            self.logger.info("PDF extraction is disabled")
            return []
        
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.warning(f"PDF directory does not exist: {directory_path}")
            return []
        
        pdf_files = list(directory.rglob("*.pdf"))
        max_files = settings.data_sources.pdf_sources.get('max_files', 50)
        
        if len(pdf_files) > max_files:
            pdf_files = pdf_files[:max_files]
            self.logger.info(f"Processing first {max_files} PDF files")
        
        extracted_content = []
        
        for pdf_file in pdf_files:
            try:
                content = self.extract_pdf_content(str(pdf_file))
                if content:
                    # Update metadata with EV keyword check
                    content.metadata['has_ev_keywords'] = self._has_ev_keywords(content.content)
                    
                    # Only keep PDFs with relevant content
                    if content.metadata['has_ev_keywords'] or len(content.content) > 500:
                        extracted_content.append(content)
                        self.logger.info(f"Successfully extracted: {pdf_file}")
                    else:
                        self.logger.info(f"Skipped (no relevant content): {pdf_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        # Save extracted content
        self._save_pdf_content(extracted_content)
        
        self.logger.info(f"Extracted content from {len(extracted_content)} PDF files")
        return extracted_content
    
    def extract_from_sources(self) -> List[ExtractedPDFContent]:
        """Extract content from all configured PDF sources"""
        all_content = []
        
        sources = settings.data_sources.pdf_sources.get('sources', [])
        
        for source in sources:
            source_path = Path(source)
            if source_path.exists():
                content = self.extract_from_directory(str(source_path))
                all_content.extend(content)
            else:
                self.logger.warning(f"PDF source directory not found: {source}")
        
        return all_content
    
    def _save_pdf_content(self, content: List[ExtractedPDFContent]):
        """Save extracted PDF content to storage"""
        data_to_save = []
        
        for item in content:
            data_to_save.append({
                'file_path': item.file_path,
                'title': item.title,
                'content': item.content,
                'pages': item.pages,
                'metadata': item.metadata,
                'images': item.images,
                'tables': item.tables,
                'content_hash': item.content_hash,
                'source_type': 'pdf'
            })
        
        # Save to JSON file
        output_path = Path(settings.get_data_path('raw')) / 'pdf_content.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved PDF content to {output_path}")

# Usage example
def main():
    extractor = PDFExtractor()
    content = extractor.extract_from_sources()
    print(f"Extracted content from {len(content)} PDF files")

if __name__ == "__main__":
    main()