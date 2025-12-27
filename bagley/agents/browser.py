"""
🌐 Browser Agent - Web Research & Search
Custom Chromium-based search engine integration
"""

import urllib.parse
from typing import Optional, List, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class BrowserAgent:
    """
    🌐 Browser Agent for Web Research
    
    Features:
    - Web search via multiple engines
    - Page content extraction
    - Screenshot capture
    - Link following
    - Custom Chromium automation
    """
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.search_engines = {
            "google": "https://www.google.com/search?q=",
            "duckduckgo": "https://duckduckgo.com/?q=",
            "bing": "https://www.bing.com/search?q=",
        }
        self.default_engine = "duckduckgo"
        self.actions_taken: List[str] = []
        
        # Browser instance (lazy loaded)
        self._browser = None
        self._page = None
        
        logger.info("Initialized BrowserAgent")
    
    async def search(self, query: str, engine: Optional[str] = None) -> str:
        """
        Perform web search.
        
        Args:
            query: Search query
            engine: Search engine (google, duckduckgo, bing)
            
        Returns:
            Search results summary
        """
        engine = engine or self.default_engine
        base_url = self.search_engines.get(engine, self.search_engines["duckduckgo"])
        
        search_url = base_url + urllib.parse.quote(query)
        
        self.actions_taken.append(f"Searched: {query}")
        
        # Try to fetch and parse results
        try:
            content = await self._fetch_url(search_url)
            results = self._extract_search_results(content)
            
            return f"Search results for '{query}':\n\n" + "\n".join(results[:10])
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Searched for '{query}' on {engine}. URL: {search_url}"
    
    async def fetch_page(self, url: str) -> str:
        """
        Fetch and extract content from a webpage.
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted text content
        """
        self.actions_taken.append(f"Fetched: {url}")
        
        try:
            content = await self._fetch_url(url)
            text = self._extract_text(content)
            
            # Truncate if too long
            if len(text) > 10000:
                text = text[:10000] + "\n\n[Content truncated...]"
            
            return text
            
        except Exception as e:
            return f"Error fetching {url}: {e}"
    
    async def _fetch_url(self, url: str) -> str:
        """Fetch URL content"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                async with session.get(url, headers=headers, timeout=30) as response:
                    return await response.text()
                    
        except ImportError:
            # Fallback to urllib
            import urllib.request
            
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode('utf-8', errors='ignore')
    
    def _extract_search_results(self, html: str) -> List[str]:
        """Extract search results from HTML"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # DuckDuckGo results
            for result in soup.select('.result__title a'):
                title = result.get_text(strip=True)
                href = result.get('href', '')
                if title:
                    results.append(f"• {title}")
            
            # Google results
            for result in soup.select('h3'):
                title = result.get_text(strip=True)
                if title:
                    results.append(f"• {title}")
            
            return results if results else ["No results extracted"]
            
        except ImportError:
            return ["Install beautifulsoup4 for better results extraction"]
    
    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts and styles
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)
            
        except ImportError:
            # Basic extraction without BeautifulSoup
            import re
            
            # Remove tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Clean up
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    async def screenshot(self, url: str, output_path: str) -> str:
        """Take screenshot of webpage"""
        self.actions_taken.append(f"Screenshot: {url}")
        
        try:
            # Would use Playwright/Puppeteer
            return f"Screenshot functionality requires Playwright. URL: {url}"
        except Exception as e:
            return f"Screenshot error: {e}"
    
    async def get_links(self, url: str) -> List[str]:
        """Extract all links from a page"""
        try:
            content = await self._fetch_url(url)
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('http'):
                    links.append(href)
            
            return links[:50]  # Limit
            
        except Exception as e:
            return [f"Error: {e}"]
    
    def get_actions(self) -> List[str]:
        return self.actions_taken.copy()
