# src/data_collection/web_scraper.py
import asyncio
import aiohttp
import time
import hashlib
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import re
import os 
import sys 


# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from src.config.settings import settings
from src.utils.logging import get_logger
from src.utils.storage import DataStorage


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


@dataclass
class ScrapedContent:
    url: str
    title: str
    content: str
    metadata: Dict
    timestamp: float
    content_hash: str

class WebScraper:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.storage = DataStorage()
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.scraped_content: List[ScrapedContent] = []
        
        # Configure headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _is_valid_url(self, url: str, allowed_domains: List[str]) -> bool:
        """Check if URL is valid and from allowed domains"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check if domain is in allowed list
            return any(allowed_domain in domain for allowed_domain in allowed_domains)
        except Exception:
            return False
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, allowed_domains: List[str]) -> List[str]:
        """Extract valid links from HTML content"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if (self._is_valid_url(full_url, allowed_domains) and 
                full_url not in self.visited_urls and
                not self._is_file_url(full_url)):
                links.append(full_url)
        
        return links
    
    def _is_file_url(self, url: str) -> bool:
        """Check if URL points to a file rather than a page"""
        file_extensions = {'.pdf', '.doc', '.docx', '.zip', '.exe', '.dmg', '.jpg', '.png', '.gif'}
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in file_extensions)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        # Remove common navigation and footer text
        patterns_to_remove = [
            r'cookie policy.*?accept',
            r'privacy policy',
            r'terms of service',
            r'follow us on.*?social media',
            r'newsletter.*?subscribe',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Optional[ScrapedContent]:
        """Extract meaningful content from HTML"""
        try:
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.article-content', '.post-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body or soup
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract text content
            text_content = main_content.get_text()
            cleaned_content = self._clean_text(text_content)
            
            # Skip if content is too short
            if len(cleaned_content) < settings.data_processing.min_text_length:
                return None
            
            # Create content hash for deduplication
            content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()
            
            # Extract metadata
            metadata = {
                'domain': urlparse(url).netloc,
                'length': len(cleaned_content),
                'word_count': len(cleaned_content.split()),
                'has_ev_keywords': self._has_ev_keywords(cleaned_content),
                'language': 'en',  # Could implement language detection
            }
            
            return ScrapedContent(
                url=url,
                title=title,
                content=cleaned_content,
                metadata=metadata,
                timestamp=time.time(),
                content_hash=content_hash
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _has_ev_keywords(self, text: str) -> bool:
        """Check if text contains EV-related keywords"""
        ev_keywords = [
            'electric vehicle', 'ev charging', 'charging station', 'chargepoint',
            'tesla supercharger', 'fast charging', 'dc fast charging', 'level 2 charging',
            'electric car', 'battery', 'plug-in', 'ev infrastructure', 'charging network'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ev_keywords)
    
    async def _scrape_url(self, url: str, session: aiohttp.ClientSession) -> Optional[ScrapedContent]:
        """Scrape a single URL"""
        try:
            self.logger.info(f"Scraping URL: {url}")
            
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                content = self._extract_content(soup, url)
                if content and content.metadata['has_ev_keywords']:
                    return content
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
    
    async def scrape_domain(self, domain: str, max_pages: int = 100) -> List[ScrapedContent]:
        """Scrape a domain with BFS approach"""
        if not self.session:
            raise RuntimeError("WebScraper must be used as async context manager")
        
        # Start with domain homepage
        start_url = f"https://{domain}" if not domain.startswith('http') else domain
        urls_to_visit = [start_url]
        scraped_contents = []
        
        allowed_domains = [domain.replace('https://', '').replace('http://', '')]
        
        while urls_to_visit and len(self.visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            self.visited_urls.add(current_url)
            
            # Rate limiting
            await asyncio.sleep(settings.data_sources.web_scraping.get('delay_seconds', 1))
            
            try:
                # Scrape current URL
                content = await self._scrape_url(current_url, self.session)
                if content:
                    scraped_contents.append(content)
                    self.logger.info(f"Successfully scraped: {current_url}")
                
                # Get HTML for link extraction
                async with self.session.get(current_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract new links
                        new_links = self._extract_links(soup, current_url, allowed_domains)
                        urls_to_visit.extend(new_links[:5])  # Limit new links per page
                
            except Exception as e:
                self.logger.error(f"Error processing {current_url}: {e}")
                continue
        
        self.logger.info(f"Scraped {len(scraped_contents)} pages from {domain}")
        return scraped_contents
    
    async def scrape_all_domains(self) -> List[ScrapedContent]:
        """Scrape all configured domains"""
        if not settings.data_sources.web_scraping['enabled']:
            self.logger.info("Web scraping is disabled")
            return []
        
        domains = settings.data_sources.web_scraping['domains']
        max_pages_per_domain = settings.data_sources.web_scraping.get('max_pages', 1000) // len(domains)
        
        all_content = []
        
        for domain in domains:
            try:
                self.logger.info(f"Starting scraping for domain: {domain}")
                content = await self.scrape_domain(domain, max_pages_per_domain)
                all_content.extend(content)
                
            except Exception as e:
                self.logger.error(f"Error scraping domain {domain}: {e}")
        
        # Save scraped content
        await self._save_scraped_content(all_content)
        
        self.logger.info(f"Total scraped content: {len(all_content)} items")
        return all_content
    
    async def _save_scraped_content(self, content: List[ScrapedContent]):
        """Save scraped content to storage"""
        data_to_save = []
        
        for item in content:
            data_to_save.append({
                'url': item.url,
                'title': item.title,
                'content': item.content,
                'metadata': item.metadata,
                'timestamp': item.timestamp,
                'content_hash': item.content_hash,
                'source_type': 'web_scraping'
            })
        
        # Save to JSON file
        output_path = Path(settings.get_data_path('raw')) / 'scraped_content.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved scraped content to {output_path}")

# Usage example
async def main():
    async with WebScraper() as scraper:
        content = await scraper.scrape_all_domains()
        print(f"Scraped {len(content)} items")

if __name__ == "__main__":
    asyncio.run(main())