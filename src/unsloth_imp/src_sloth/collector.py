"""
Data Collection Module for web scraping, PDF extraction, and metadata processing.
Implements data collection requirements from the pipeline specification.
"""

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

# Web scraping
try:
    from bs4 import BeautifulSoup
    import requests
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    print("Warning: Web scraping dependencies not available. Install with: pip install beautifulsoup4 requests")
    WEB_SCRAPING_AVAILABLE = False

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    print("Warning: PDF processing dependencies not available. Install with: pip install PyPDF2 pdfplumber")
    PDF_PROCESSING_AVAILABLE = False

# Document processing
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

class DataCollector:
    """
    Handles collection of domain-specific data from various sources including
    web scraping, PDF extraction, and document processing with metadata attribution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data collector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.output_dir = Path(config.get('output_dir', 'data/raw'))
        self.max_pages_per_source = config.get('max_pages_per_source', 100)
        self.scraping_delay = config.get('scraping_delay', 1.0)
        self.web_sources = config.get('web_sources', [])
        self.pdf_sources = config.get('pdf_sources', [])
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session for web requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.logger.info(f"DataCollector initialized with output dir: {self.output_dir}")
    
    def scrape_web_sources(self, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Scrape web sources for domain-specific content with metadata extraction.
        """
        if not WEB_SCRAPING_AVAILABLE:
            self.logger.error("Web scraping dependencies not available")
            return []
        
        sources = sources or self.web_sources
        if not sources:
            self.logger.warning("No web sources provided for scraping")
            return []
        
        self.logger.info(f"Starting web scraping for {len(sources)} sources")
        
        scraped_data = []
        
        for i, source_url in enumerate(sources):
            self.logger.info(f"Scraping source {i+1}/{len(sources)}: {source_url}")
            
            try:
                # Scrape single source
                source_data = self._scrape_single_source(source_url)
                if source_data:
                    scraped_data.extend(source_data)
                
                # Respect rate limiting
                if i < len(sources) - 1:
                    time.sleep(self.scraping_delay)
                    
            except Exception as e:
                self.logger.error(f"Failed to scrape {source_url}: {str(e)}")
                continue
        
        self.logger.info(f"Web scraping completed. Collected {len(scraped_data)} pages")
        return scraped_data
    
    def _scrape_single_source(self, url: str) -> List[Dict[str, Any]]:
        """Scrape a single web source and extract content with metadata."""
        pages_data = []
        visited_urls = set()
        urls_to_visit = [url]
        pages_scraped = 0
        
        while urls_to_visit and pages_scraped < self.max_pages_per_source:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            try:
                # Fetch page content
                response = self.session.get(current_url, timeout=30)
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract page data
                page_data = self._extract_page_content(soup, current_url, response)
                if page_data:
                    pages_data.append(page_data)
                    pages_scraped += 1
                
                # Find additional URLs to scrape (same domain)
                if pages_scraped < self.max_pages_per_source:
                    new_urls = self._extract_internal_links(soup, current_url)
                    for new_url in new_urls[:5]:  # Limit new URLs per page
                        if new_url not in visited_urls and new_url not in urls_to_visit:
                            urls_to_visit.append(new_url)
                
            except Exception as e:
                self.logger.warning(f"Failed to scrape page {current_url}: {str(e)}")
                continue
        
        return pages_data
    
    def _extract_page_content(self, soup: BeautifulSoup, url: str, response) -> Optional[Dict[str, Any]]:
        """Extract content and metadata from a single page."""
        try:
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ''
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Skip if content is too short
            if len(content.strip()) < 100:
                return None
            
            # Extract metadata
            metadata = self._extract_page_metadata(soup, url, response)
            
            # Create content hash for deduplication
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            page_data = {
                'url': url,
                'title': title,
                'content': content,
                'content_hash': content_hash,
                'metadata': metadata,
                'source_type': 'web',
                'collected_at': datetime.now().isoformat()
            }
            
            return page_data
            
        except Exception as e:
            self.logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML, removing navigation and ads."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            element.decompose()
        
        # Try to find main content areas
        main_content = ''
        
        # Look for main content tags
        main_tags = soup.find_all(['main', 'article', 'div'], 
                                 class_=re.compile(r'(content|main|article|post|body)', re.I))
        
        if main_tags:
            for tag in main_tags:
                main_content += tag.get_text(separator=' ', strip=True) + '\n'
        else:
            # Fallback: extract all paragraph and heading text
            text_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            for tag in text_tags:
                text = tag.get_text(strip=True)
                if len(text) > 10:  # Filter out very short text
                    main_content += text + '\n'
        
        # Clean up the content
        main_content = re.sub(r'\n\s*\n', '\n\n', main_content)  # Remove excessive newlines
        main_content = re.sub(r'[ \t]+', ' ', main_content)  # Normalize whitespace
        
        return main_content.strip()
    
    def _extract_page_metadata(self, soup: BeautifulSoup, url: str, response) -> Dict[str, Any]:
        """Extract metadata from page headers and meta tags."""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'response_status': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'last_modified': response.headers.get('last-modified', ''),
        }
        
        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                meta_tags[name.lower()] = content
        
        metadata['meta_tags'] = meta_tags
        
        # Extract structured data (JSON-LD)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        structured_data = []
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except:
                pass
        
        if structured_data:
            metadata['structured_data'] = structured_data
        
        return metadata
    
    def _extract_internal_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract internal links from the page."""
        base_domain = urlparse(base_url).netloc
        internal_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            # Check if it's an internal link
            if urlparse(absolute_url).netloc == base_domain:
                # Skip non-content links
                if not any(skip in href.lower() for skip in ['#', 'javascript:', 'mailto:', 'tel:', '.pdf', '.doc']):
                    internal_links.append(absolute_url)
        
        return list(set(internal_links))  # Remove duplicates
    
    def extract_pdf_content(self, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract content from PDF files with layout preservation and metadata.
        """
        if not PDF_PROCESSING_AVAILABLE:
            self.logger.error("PDF processing dependencies not available")
            return []
        
        sources = sources or self.pdf_sources
        if not sources:
            self.logger.warning("No PDF sources provided")
            return []
        
        self.logger.info(f"Processing {len(sources)} PDF sources")
        
        pdf_data = []
        
        for pdf_source in sources:
            self.logger.info(f"Processing PDF: {pdf_source}")
            
            try:
                if pdf_source.startswith(('http://', 'https://')):
                    # Download PDF from URL
                    pdf_content = self._download_pdf(pdf_source)
                    source_path = pdf_source
                else:
                    # Local PDF file
                    pdf_path = Path(pdf_source)
                    if not pdf_path.exists():
                        self.logger.error(f"PDF file not found: {pdf_source}")
                        continue
                    
                    with open(pdf_path, 'rb') as f:
                        pdf_content = f.read()
                    source_path = str(pdf_path)
                
                # Extract content and metadata
                extracted_data = self._extract_pdf_content(pdf_content, source_path)
                if extracted_data:
                    pdf_data.append(extracted_data)
                    
            except Exception as e:
                self.logger.error(f"Failed to process PDF {pdf_source}: {str(e)}")
                continue
        
        self.logger.info(f"PDF processing completed. Extracted {len(pdf_data)} documents")
        return pdf_data
    
    def _download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            # Check if content starts with PDF signature
            if not response.content.startswith(b'%PDF-'):
                raise ValueError(f"URL does not appear to contain a PDF: {url}")
        
        return response.content
    
    def _extract_pdf_content(self, pdf_content: bytes, source_path: str) -> Optional[Dict[str, Any]]:
        """Extract text content and metadata from PDF bytes."""
        try:
            import io
            
            # Try pdfplumber first (better layout preservation)
            extracted_data = None
            
            try:
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    extracted_data = self._extract_with_pdfplumber(pdf, source_path)
            except Exception as e:
                self.logger.warning(f"pdfplumber failed for {source_path}: {str(e)}")
                
                # Fallback to PyPDF2
                try:
                    extracted_data = self._extract_with_pypdf2(pdf_content, source_path)
                except Exception as e2:
                    self.logger.error(f"PyPDF2 also failed for {source_path}: {str(e2)}")
                    return None
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract PDF content from {source_path}: {str(e)}")
            return None
    
    def _extract_with_pdfplumber(self, pdf, source_path: str) -> Dict[str, Any]:
        """Extract content using pdfplumber (preserves layout better)."""
        pages_content = []
        total_text = ""
        
        for page_num, page in enumerate(pdf.pages):
            try:
                # Extract text with layout
                page_text = page.extract_text()
                if page_text:
                    pages_content.append({
                        'page_number': page_num + 1,
                        'content': page_text.strip(),
                        'char_count': len(page_text)
                    })
                    total_text += page_text + "\n\n"
                
            except Exception as e:
                self.logger.warning(f"Failed to extract page {page_num + 1} from {source_path}: {str(e)}")
                continue
        
        # Extract metadata
        metadata = {
            'total_pages': len(pdf.pages),
            'pages_extracted': len(pages_content),
            'extraction_method': 'pdfplumber',
            'source_path': source_path,
            'total_chars': len(total_text),
            'extracted_at': datetime.now().isoformat()
        }
        
        # Try to get PDF metadata
        try:
            if hasattr(pdf, 'metadata') and pdf.metadata:
                metadata['pdf_metadata'] = dict(pdf.metadata)
        except:
            pass
        
        content_hash = hashlib.md5(total_text.encode('utf-8')).hexdigest()
        
        return {
            'source_path': source_path,
            'content': total_text.strip(),
            'content_hash': content_hash,
            'pages': pages_content,
            'metadata': metadata,
            'source_type': 'pdf',
        return {
            'source_path': source_path,
            'content': total_text.strip(),
            'content_hash': content_hash,
            'pages': pages_content,
            'metadata': metadata,
            'source_type': 'pdf',
            'collected_at': datetime.now().isoformat()
        }
    
    def extract_document_content(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract content from various document formats (DOCX, TXT, etc.).
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None
        
        self.logger.info(f"Extracting content from: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.docx':
                return self._extract_docx_content(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return self._extract_text_content(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract content from {file_path}: {str(e)}")
            return None
    
    def _extract_docx_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from DOCX files."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available for DOCX processing")
        
        doc = DocxDocument(str(file_path))
        
        # Extract paragraphs
        paragraphs = []
        full_text = ""
        
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
                full_text += para.text + "\n"
        
        # Extract metadata
        core_props = doc.core_properties
        metadata = {
            'title': core_props.title or '',
            'author': core_props.author or '',
            'subject': core_props.subject or '',
            'created': core_props.created.isoformat() if core_props.created else '',
            'modified': core_props.modified.isoformat() if core_props.modified else '',
            'paragraphs_count': len(paragraphs),
            'extraction_method': 'python-docx',
            'source_path': str(file_path),
            'total_chars': len(full_text),
            'extracted_at': datetime.now().isoformat()
        }
        
        content_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
        
        return {
            'source_path': str(file_path),
            'content': full_text.strip(),
            'content_hash': content_hash,
            'paragraphs': paragraphs,
            'metadata': metadata,
            'source_type': 'docx',
            'collected_at': datetime.now().isoformat()
        }
    
    def _extract_text_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from plain text files."""
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        content = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode file with any supported encoding: {file_path}")
        
        # Basic file stats
        file_stats = file_path.stat()
        
        metadata = {
            'encoding': encoding_used,
            'file_size': file_stats.st_size,
            'lines_count': len(content.splitlines()),
            'extraction_method': 'text_reader',
            'source_path': str(file_path),
            'total_chars': len(content),
            'extracted_at': datetime.now().isoformat(),
            'file_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }
        
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        return {
            'source_path': str(file_path),
            'content': content.strip(),
            'content_hash': content_hash,
            'metadata': metadata,
            'source_type': 'text',
            'collected_at': datetime.now().isoformat()
        }
    
    def save_raw_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save collected raw data to disk with metadata."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_data_{timestamp}.json"
        
        save_path = self.output_dir / filename
        
        # Add collection summary
        collection_summary = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_sources': len(data.get('web_content', [])) + len(data.get('pdf_content', [])),
            'web_sources_count': len(data.get('web_content', [])),
            'pdf_sources_count': len(data.get('pdf_content', [])),
            'total_content_chars': sum(
                len(item.get('content', '')) 
                for items in [data.get('web_content', []), data.get('pdf_content', [])]
                for item in items
            )
        }
        
        # Combine data with summary
        save_data = {
            **data,
            'collection_summary': collection_summary
        }
        
        # Save to JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Raw data saved to: {save_path}")
        return str(save_path)
    
    def load_raw_data(self, filename: str) -> Dict[str, Any]:
        """Load previously collected raw data."""
        load_path = self.output_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Raw data loaded from: {load_path}")
        return data
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        stats = {
            'output_directory': str(self.output_dir),
            'raw_data_files': [],
            'total_files': 0,
            'total_size_mb': 0.0
        }
        
        # Scan output directory
        if self.output_dir.exists():
            for file_path in self.output_dir.glob('*.json'):
                file_stats = file_path.stat()
                stats['raw_data_files'].append({
                    'filename': file_path.name,
                    'size_mb': file_stats.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
                stats['total_size_mb'] += file_stats.st_size / (1024 * 1024)
            
            stats['total_files'] = len(stats['raw_data_files'])
        
        return stats
    
    def validate_collected_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate collected data quality and completeness."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check web content
        web_content = data.get('web_content', [])
        valid_web_items = 0
        total_web_chars = 0
        
        for item in web_content:
            if not item.get('content') or len(item['content'].strip()) < 50:
                validation_results['warnings'].append(f"Short web content from {item.get('url', 'unknown')}")
            else:
                valid_web_items += 1
                total_web_chars += len(item['content'])
        
        # Check PDF content
        pdf_content = data.get('pdf_content', [])
        valid_pdf_items = 0
        total_pdf_chars = 0
        
        for item in pdf_content:
            if not item.get('content') or len(item['content'].strip()) < 100:
                validation_results['warnings'].append(f"Short PDF content from {item.get('source_path', 'unknown')}")
            else:
                valid_pdf_items += 1
                total_pdf_chars += len(item['content'])
        
        # Overall validation
        total_valid_items = valid_web_items + valid_pdf_items
        if total_valid_items == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No valid content items found")
        
        if total_web_chars + total_pdf_chars < 1000:
            validation_results['warnings'].append("Very low total content volume")
        
        # Statistics
        validation_results['statistics'] = {
            'total_items': len(web_content) + len(pdf_content),
            'valid_items': total_valid_items,
            'web_items': {
                'total': len(web_content),
                'valid': valid_web_items,
                'total_chars': total_web_chars
            },
            'pdf_items': {
                'total': len(pdf_content),
                'valid': valid_pdf_items,
                'total_chars': total_pdf_chars
            },
            'total_content_chars': total_web_chars + total_pdf_chars
        }
        
        return validation_results
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during collection."""
        temp_dir = self.output_dir / 'temp'
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info("Temporary files cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def _extract_with_pypdf2(self, pdf_content: bytes, source_path: str) -> Dict[str, Any]:
        """Fallback extraction using PyPDF2."""
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        pages_content = []
        total_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    pages_content.append({
                        'page_number': page_num + 1,
                        'content': page_text.strip(),
                        'char_count': len(page_text)
                    })
                    total_text += page_text + "\n\n"
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract page {page_num + 1} from {source_path}: {str(e)}")
                continue
        
        # Extract metadata
        metadata = {
            'total_pages': len(pdf_reader.pages),
            'pages_extracted': len(pages_content),
            'extraction_method': 'PyPDF2',
            'source_path': source_path,
            'total_chars': len(total_text),
            'extracted_at': datetime.now().isoformat()
        }
        
        # Try to get PDF metadata
        try:
            if pdf_reader.metadata:
                metadata['pdf_metadata'] = {
                    str(key): str(value) for key, value in pdf_reader.metadata.items()
                }
        except:
            pass
        
        content_hash = hashlib.md5(total_text.encode('utf-8')).hexdigest()
        
        return {
            'source_path': source_path,
            'content': total_text.strip(),
            'content_hash': content_hash,
            'pages': pages_content,
            'metadata': metadata,
            'source_type': 'pdf',
            'collecte