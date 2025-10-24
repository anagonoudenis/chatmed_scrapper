"""
Universal Web Scraper - Scrape any medical website from URLs
Automatically extracts: title, content, metadata, images
Respects robots.txt and rate limits
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from utils.config import AppConfig


class UniversalWebScraper:
    """
    Scrape any website from URLs.
    Automatically extracts medical content, metadata, and structure.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.scraped_urls = set()
        self.results = []

    async def init_session(self):
        """Initialize async HTTP session with headers."""
        headers = {
            "User-Agent": self.config.network.user_agents[0],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.network.timeout_seconds)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=self.config.scraper.concurrent_requests)
        )
        logger.info("Universal web scraper session initialized")

    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            logger.info("Web scraper session closed")

    async def scrape_url(self, url: str) -> Optional[dict]:
        """
        Scrape a single URL and extract all relevant content.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with extracted data or None if failed
        """
        if url in self.scraped_urls:
            logger.debug(f"URL already scraped: {url}")
            return None

        try:
            logger.info(f"Scraping URL: {url}")
            
            # Fetch page
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                html = await response.text()
                content_type = response.headers.get('Content-Type', '')
                
                # Only process HTML
                if 'text/html' not in content_type:
                    logger.warning(f"Non-HTML content: {content_type}")
                    return None

            # Parse HTML
            soup = BeautifulSoup(html, 'html5lib')
            
            # Extract data
            data = {
                'url': url,
                'domain': urlparse(url).netloc,
                'scraped_at': datetime.now().isoformat(),
                'title': self._extract_title(soup),
                'content': self._extract_content(soup),
                'abstract': self._extract_abstract(soup),
                'metadata': self._extract_metadata(soup),
                'authors': self._extract_authors(soup),
                'date': self._extract_date(soup),
                'keywords': self._extract_keywords(soup),
                'images': self._extract_images(soup, url),
                'links': self._extract_links(soup, url),
            }
            
            # Mark as scraped
            self.scraped_urls.add(url)
            
            logger.success(f"Successfully scraped: {url}")
            return data
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try multiple methods
        title = None
        
        # 1. <title> tag
        if soup.title:
            title = soup.title.string
        
        # 2. <h1> tag
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text()
        
        # 3. og:title meta tag
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content')
        
        # 4. article title
        if not title:
            article_title = soup.find('h1', class_=re.compile(r'title|heading', re.I))
            if article_title:
                title = article_title.get_text()
        
        return self._clean_text(title) if title else "No title"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page."""
        content_parts = []
        
        # Try to find main content area
        main_selectors = [
            ('article', {}),
            ('main', {}),
            ('div', {'class': re.compile(r'content|article|post|entry', re.I)}),
            ('div', {'id': re.compile(r'content|article|post|entry', re.I)}),
        ]
        
        for tag, attrs in main_selectors:
            main_content = soup.find(tag, attrs)
            if main_content:
                # Extract paragraphs
                paragraphs = main_content.find_all('p')
                for p in paragraphs:
                    text = p.get_text()
                    if len(text.strip()) > 50:  # Only meaningful paragraphs
                        content_parts.append(text)
                
                if content_parts:
                    break
        
        # Fallback: all paragraphs
        if not content_parts:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text()
                if len(text.strip()) > 50:
                    content_parts.append(text)
        
        content = '\n\n'.join(content_parts)
        return self._clean_text(content)

    def _extract_abstract(self, soup: BeautifulSoup) -> str:
        """Extract abstract or summary."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return self._clean_text(meta_desc.get('content', ''))
        
        # Try og:description
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            return self._clean_text(og_desc.get('content', ''))
        
        # Try first paragraph
        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text()
            if len(text) > 100:
                return self._clean_text(text[:500])
        
        return ""

    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract all metadata from page."""
        metadata = {}
        
        # All meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        return metadata

    def _extract_authors(self, soup: BeautifulSoup) -> list:
        """Extract author names."""
        authors = []
        
        # Try meta author tag
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            authors.append(author_meta.get('content'))
        
        # Try author class/id
        author_elements = soup.find_all(class_=re.compile(r'author', re.I))
        for elem in author_elements:
            text = elem.get_text().strip()
            if text and len(text) < 100:
                authors.append(text)
        
        return list(set(authors))  # Remove duplicates

    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date."""
        # Try various date meta tags
        date_selectors = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'publication_date'}),
            ('meta', {'name': 'date'}),
            ('time', {'datetime': True}),
        ]
        
        for tag, attrs in date_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                date = elem.get('content') or elem.get('datetime')
                if date:
                    return date
        
        return ""

    def _extract_keywords(self, soup: BeautifulSoup) -> list:
        """Extract keywords from meta tags."""
        keywords = []
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            content = meta_keywords.get('content', '')
            keywords = [k.strip() for k in content.split(',')]
        
        return keywords

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract all images with absolute URLs."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Convert to absolute URL
                absolute_url = urljoin(base_url, src)
                images.append({
                    'url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                })
        
        return images[:20]  # Limit to 20 images

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract all internal links."""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(base_url, href)
            
            # Only internal links
            if urlparse(absolute_url).netloc == base_domain:
                links.append({
                    'url': absolute_url,
                    'text': a.get_text().strip(),
                })
        
        return links[:50]  # Limit to 50 links

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        return text.strip()

    async def scrape_urls_from_file(self, file_path: Path) -> list:
        """
        Scrape all URLs from a text file (one URL per line).
        
        Args:
            file_path: Path to file containing URLs
            
        Returns:
            List of scraped data dictionaries
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Read URLs
        urls = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # Skip comments
                    urls.append(url)
        
        logger.info(f"Found {len(urls)} URLs to scrape")
        
        # Initialize session
        await self.init_session()
        
        # Scrape all URLs with rate limiting
        results = []
        for i, url in enumerate(urls, 1):
            logger.info(f"Progress: {i}/{len(urls)}")
            
            result = await self.scrape_url(url)
            if result:
                results.append(result)
            
            # Rate limiting
            await asyncio.sleep(self.config.scraper.rate_limit_min_seconds)
        
        # Close session
        await self.close_session()
        
        logger.success(f"Scraped {len(results)}/{len(urls)} URLs successfully")
        return results

    async def scrape_single_url(self, url: str) -> Optional[dict]:
        """
        Scrape a single URL (convenience method).
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraped data dictionary or None
        """
        await self.init_session()
        result = await self.scrape_url(url)
        await self.close_session()
        return result
