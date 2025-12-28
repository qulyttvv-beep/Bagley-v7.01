"""
🌐 Bagley Web Intelligence System
=================================
Daily scraping of news, Twitter/X, Reddit
With anti-lockout protection and error recovery
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import re
from urllib.parse import urlparse, quote_plus

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class ScraperConfig:
    """Configuration for web scraping"""
    
    # Rate limiting (be nice to servers)
    requests_per_minute: int = 30
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 5.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 10.0
    
    # Anti-detection
    rotate_user_agents: bool = True
    use_random_delays: bool = True
    
    # Storage
    data_dir: str = "web_intelligence"
    cache_hours: int = 1  # Don't re-fetch same content within this time
    
    # Content limits
    max_articles_per_source: int = 100
    max_posts_per_subreddit: int = 100
    max_tweets_per_search: int = 100


# User agents rotation pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]


# ==================== Data Models ====================

class ContentType(Enum):
    NEWS = "news"
    TWEET = "tweet"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"


@dataclass
class WebContent:
    """Represents scraped web content"""
    content_type: ContentType
    source: str
    title: str
    content: str
    url: str
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scraped_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content_type': self.content_type.value,
            'source': self.source,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'author': self.author,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
            'scraped_at': self.scraped_at.isoformat()
        }
    
    def content_hash(self) -> str:
        """Generate hash for deduplication"""
        return hashlib.md5(f"{self.url}{self.content[:100]}".encode()).hexdigest()


# ==================== Rate Limiter ====================

class RateLimiter:
    """Smart rate limiting to avoid bans"""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests: List[float] = []
        self._lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're going too fast"""
        with self._lock:
            now = time.time()
            minute_ago = now - 60
            
            # Remove old requests
            self.requests = [t for t in self.requests if t > minute_ago]
            
            # Check if we need to wait
            if len(self.requests) >= self.requests_per_minute:
                oldest = self.requests[0]
                wait_time = 60 - (now - oldest) + random.uniform(1, 3)
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
            
            # Record this request
            self.requests.append(time.time())
    
    def add_random_delay(self, min_s: float = 1.0, max_s: float = 3.0):
        """Add random delay to seem more human"""
        delay = random.uniform(min_s, max_s)
        time.sleep(delay)


# ==================== Base Scraper ====================

class BaseScraper(ABC):
    """Base class for all scrapers"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup HTTP session with retries"""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            self.session = requests.Session()
            
            # Retry strategy
            retry = Retry(
                total=self.config.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
            
        except ImportError:
            logger.warning("requests not installed")
            self.session = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with rotating user agent"""
        ua = random.choice(USER_AGENTS) if self.config.rotate_user_agents else USER_AGENTS[0]
        return {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _make_request(self, url: str, **kwargs) -> Optional[str]:
        """Make HTTP request with rate limiting and error handling"""
        if not self.session:
            logger.error("No HTTP session available")
            return None
        
        self.rate_limiter.wait_if_needed()
        
        if self.config.use_random_delays:
            self.rate_limiter.add_random_delay(
                self.config.min_delay_seconds,
                self.config.max_delay_seconds
            )
        
        try:
            headers = self._get_headers()
            headers.update(kwargs.pop('headers', {}))
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=30,
                **kwargs
            )
            
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                # Rate limited - back off
                logger.warning(f"Rate limited by {urlparse(url).netloc}, backing off...")
                time.sleep(self.config.retry_delay_seconds * 2)
                return None
            elif response.status_code == 403:
                logger.warning(f"Forbidden (403) from {urlparse(url).netloc}")
                return None
            else:
                logger.warning(f"HTTP {response.status_code} from {url}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    @abstractmethod
    def scrape(self) -> List[WebContent]:
        """Scrape content from source"""
        pass


# ==================== News Scraper ====================

class NewsScraper(BaseScraper):
    """
    📰 News scraper using RSS feeds and APIs
    Much safer than direct scraping
    """
    
    # RSS feeds (no auth needed, publicly available)
    RSS_FEEDS = {
        # Tech
        'hackernews': 'https://hnrss.org/frontpage',
        'techcrunch': 'https://techcrunch.com/feed/',
        'arstechnica': 'https://feeds.arstechnica.com/arstechnica/index',
        'theverge': 'https://www.theverge.com/rss/index.xml',
        'wired': 'https://www.wired.com/feed/rss',
        
        # General
        'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
        'reuters': 'https://www.reutersagency.com/feed/',
        'ap': 'https://rsshub.app/apnews/topics/apf-topnews',
        'npr': 'https://feeds.npr.org/1001/rss.xml',
        
        # Science
        'nature': 'https://www.nature.com/nature.rss',
        'science': 'https://www.science.org/rss/news_current.xml',
        'arxiv_cs': 'http://arxiv.org/rss/cs',
        'arxiv_ai': 'http://arxiv.org/rss/cs.AI',
        
        # Business
        'bloomberg': 'https://www.bloomberg.com/feed/podcast/etf-iq.xml',
        'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        
        # AI specific
        'openai_blog': 'https://openai.com/blog/rss/',
        'deepmind': 'https://www.deepmind.com/blog/rss.xml',
        'huggingface': 'https://huggingface.co/blog/feed.xml',
    }
    
    def scrape(self) -> List[WebContent]:
        """Scrape news from RSS feeds"""
        all_content = []
        
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed: pip install feedparser")
            return []
        
        for source_name, feed_url in self.RSS_FEEDS.items():
            try:
                logger.info(f"Fetching news from {source_name}...")
                
                self.rate_limiter.wait_if_needed()
                feed = feedparser.parse(feed_url)
                
                if feed.bozo and feed.bozo_exception:
                    logger.warning(f"Feed error for {source_name}: {feed.bozo_exception}")
                    continue
                
                for entry in feed.entries[:self.config.max_articles_per_source]:
                    # Parse timestamp
                    timestamp = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        timestamp = datetime(*entry.published_parsed[:6])
                    
                    # Get content
                    content = ""
                    if hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    
                    # Clean HTML
                    content = self._clean_html(content)
                    
                    web_content = WebContent(
                        content_type=ContentType.NEWS,
                        source=source_name,
                        title=entry.get('title', 'No title'),
                        content=content,
                        url=entry.get('link', ''),
                        author=entry.get('author'),
                        timestamp=timestamp,
                        metadata={
                            'tags': [t.term for t in entry.get('tags', [])] if hasattr(entry, 'tags') else []
                        }
                    )
                    
                    all_content.append(web_content)
                
                logger.info(f"  Got {len(feed.entries)} articles from {source_name}")
                
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
                continue
        
        return all_content
    
    def _clean_html(self, html: str) -> str:
        """Remove HTML tags"""
        import re
        clean = re.sub(r'<[^>]+>', '', html)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


# ==================== Reddit Scraper ====================

class RedditScraper(BaseScraper):
    """
    🔴 Reddit scraper using public JSON API
    No authentication needed for public subreddits
    """
    
    SUBREDDITS = [
        # AI & Tech
        'artificial', 'MachineLearning', 'LocalLLaMA', 'singularity',
        'ChatGPT', 'OpenAI', 'StableDiffusion', 'technology',
        'programming', 'learnprogramming', 'Python', 'javascript',
        
        # News & Discussion
        'worldnews', 'news', 'science', 'Futurology',
        'explainlikeimfive', 'askscience', 'AskReddit',
        
        # Specific interests
        'gaming', 'movies', 'books', 'music',
    ]
    
    def scrape(self) -> List[WebContent]:
        """Scrape posts from Reddit using public JSON API"""
        all_content = []
        
        for subreddit in self.SUBREDDITS:
            try:
                logger.info(f"Fetching r/{subreddit}...")
                
                # Reddit's public JSON API
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={self.config.max_posts_per_subreddit}"
                
                # Reddit requires a proper User-Agent
                headers = self._get_headers()
                headers['User-Agent'] = f'Bagley/7.01 (Educational AI Project)'
                
                response_text = self._make_request(url, headers=headers)
                
                if not response_text:
                    continue
                
                data = json.loads(response_text)
                posts = data.get('data', {}).get('children', [])
                
                for post in posts:
                    post_data = post.get('data', {})
                    
                    # Skip removed/deleted
                    if post_data.get('removed_by_category') or post_data.get('selftext') == '[removed]':
                        continue
                    
                    # Parse timestamp
                    timestamp = None
                    if post_data.get('created_utc'):
                        timestamp = datetime.fromtimestamp(post_data['created_utc'])
                    
                    content = post_data.get('selftext', '') or post_data.get('title', '')
                    
                    web_content = WebContent(
                        content_type=ContentType.REDDIT_POST,
                        source=f"r/{subreddit}",
                        title=post_data.get('title', ''),
                        content=content,
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        author=post_data.get('author'),
                        timestamp=timestamp,
                        metadata={
                            'score': post_data.get('score', 0),
                            'num_comments': post_data.get('num_comments', 0),
                            'upvote_ratio': post_data.get('upvote_ratio', 0),
                            'is_self': post_data.get('is_self', False),
                            'link_flair': post_data.get('link_flair_text'),
                        }
                    )
                    
                    all_content.append(web_content)
                
                logger.info(f"  Got {len(posts)} posts from r/{subreddit}")
                
                # Extra delay for Reddit
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit}: {e}")
                continue
        
        return all_content


# ==================== Twitter/X Scraper ====================

class TwitterScraper(BaseScraper):
    """
    🐦 Twitter/X content via alternative methods
    Uses Nitter instances or RSS bridges
    """
    
    # Nitter instances (open source Twitter frontend)
    NITTER_INSTANCES = [
        'https://nitter.net',
        'https://nitter.privacydev.net',
        'https://nitter.poast.org',
        'https://nitter.woodland.cafe',
    ]
    
    # Accounts to follow
    ACCOUNTS = [
        # AI Researchers
        'ylecun', 'kaborir', 'iloannides', 'sama',
        'AndrewYNg', 'jeffdean', 'goodloopsteve',
        
        # AI Companies
        'OpenAI', 'AnthropicAI', 'GoogleDeepMind', 'xaboratory',
        'huggingface', 'MistralAI', 'AIatMeta',
        
        # Tech News
        'veraboringbrain', 'TechCrunch', 'WIRED', 'engadget',
        
        # Influential
        'elonmusk', 'BillGates', 'satlovernadeena',
    ]
    
    # Search terms
    SEARCH_TERMS = [
        'AI breakthrough',
        'GPT-5',
        'artificial intelligence',
        'machine learning',
        'large language model',
        'neural network',
    ]
    
    def scrape(self) -> List[WebContent]:
        """Scrape Twitter via Nitter or RSS bridges"""
        all_content = []
        
        # Try RSS bridge approach (more reliable)
        all_content.extend(self._scrape_via_rss())
        
        # Try Nitter instances
        all_content.extend(self._scrape_via_nitter())
        
        return all_content
    
    def _scrape_via_rss(self) -> List[WebContent]:
        """Use RSS bridges for Twitter content"""
        content = []
        
        try:
            import feedparser
        except ImportError:
            return content
        
        # RSSHub Twitter feeds
        rsshub_base = "https://rsshub.app/twitter/user"
        
        for account in self.ACCOUNTS[:10]:  # Limit to avoid rate limits
            try:
                url = f"{rsshub_base}/{account}"
                
                self.rate_limiter.wait_if_needed()
                feed = feedparser.parse(url)
                
                if feed.bozo:
                    continue
                
                for entry in feed.entries[:20]:
                    timestamp = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        timestamp = datetime(*entry.published_parsed[:6])
                    
                    web_content = WebContent(
                        content_type=ContentType.TWEET,
                        source=f"@{account}",
                        title=entry.get('title', '')[:100],
                        content=self._clean_html(entry.get('summary', '')),
                        url=entry.get('link', ''),
                        author=account,
                        timestamp=timestamp,
                    )
                    content.append(web_content)
                
            except Exception as e:
                logger.debug(f"RSS bridge failed for @{account}: {e}")
                continue
        
        return content
    
    def _scrape_via_nitter(self) -> List[WebContent]:
        """Scrape via Nitter instances"""
        content = []
        
        # Try each Nitter instance
        for instance in self.NITTER_INSTANCES:
            try:
                # Test if instance is up
                test_response = self._make_request(f"{instance}/")
                if not test_response:
                    continue
                
                logger.info(f"Using Nitter instance: {instance}")
                
                for account in self.ACCOUNTS[:5]:
                    try:
                        url = f"{instance}/{account}/rss"
                        
                        import feedparser
                        self.rate_limiter.wait_if_needed()
                        feed = feedparser.parse(url)
                        
                        for entry in feed.entries[:10]:
                            timestamp = None
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                timestamp = datetime(*entry.published_parsed[:6])
                            
                            web_content = WebContent(
                                content_type=ContentType.TWEET,
                                source=f"@{account}",
                                title=entry.get('title', '')[:100],
                                content=self._clean_html(entry.get('description', '')),
                                url=entry.get('link', '').replace(instance, 'https://twitter.com'),
                                author=account,
                                timestamp=timestamp,
                            )
                            content.append(web_content)
                        
                    except Exception as e:
                        logger.debug(f"Nitter failed for @{account}: {e}")
                        continue
                
                # If we got content, don't try other instances
                if content:
                    break
                    
            except Exception as e:
                logger.debug(f"Nitter instance {instance} failed: {e}")
                continue
        
        return content
    
    def _clean_html(self, html: str) -> str:
        """Remove HTML tags"""
        import re
        clean = re.sub(r'<[^>]+>', '', html)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


# ==================== Content Storage ====================

class WebIntelligenceStorage:
    """
    💾 Store and deduplicate web content
    """
    
    def __init__(self, data_dir: str = "web_intelligence"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.seen_hashes: set = set()
        self._load_seen_hashes()
    
    def _load_seen_hashes(self):
        """Load previously seen content hashes"""
        hash_file = self.data_dir / "seen_hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    self.seen_hashes = set(json.load(f))
            except:
                self.seen_hashes = set()
    
    def _save_seen_hashes(self):
        """Save seen hashes"""
        hash_file = self.data_dir / "seen_hashes.json"
        with open(hash_file, 'w') as f:
            json.dump(list(self.seen_hashes), f)
    
    def store(self, content: List[WebContent]) -> int:
        """Store content, returns number of new items stored"""
        new_count = 0
        
        # Organize by date
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.data_dir / today
        day_dir.mkdir(exist_ok=True)
        
        for item in content:
            content_hash = item.content_hash()
            
            if content_hash in self.seen_hashes:
                continue
            
            self.seen_hashes.add(content_hash)
            new_count += 1
            
            # Save to file
            filename = f"{item.content_type.value}_{content_hash[:8]}.json"
            filepath = day_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(item.to_dict(), f, indent=2, ensure_ascii=False)
        
        self._save_seen_hashes()
        return new_count
    
    def get_recent(self, days: int = 7, content_type: Optional[ContentType] = None) -> List[Dict]:
        """Get recent content"""
        results = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            day_dir = self.data_dir / date
            
            if not day_dir.exists():
                continue
            
            for file in day_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if content_type and data.get('content_type') != content_type.value:
                        continue
                    
                    results.append(data)
                except:
                    continue
        
        return results
    
    def export_for_training(self, output_file: str) -> int:
        """Export all content as JSONL for training"""
        count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for day_dir in sorted(self.data_dir.glob("20*")):
                for file in day_dir.glob("*.json"):
                    try:
                        with open(file, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                        
                        # Convert to training format
                        training_item = {
                            "messages": [
                                {"role": "system", "content": f"Recent {data['content_type']} from {data['source']}"},
                                {"role": "user", "content": f"What's happening with: {data['title']}?"},
                                {"role": "assistant", "content": data['content']}
                            ]
                        }
                        
                        f.write(json.dumps(training_item, ensure_ascii=False) + '\n')
                        count += 1
                    except:
                        continue
        
        return count


# ==================== Main Intelligence System ====================

class WebIntelligenceSystem:
    """
    🧠 Bagley's Web Intelligence System
    Coordinates all scrapers and runs on schedule
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.storage = WebIntelligenceStorage(self.config.data_dir)
        
        # Initialize scrapers
        self.scrapers = {
            'news': NewsScraper(self.config),
            'reddit': RedditScraper(self.config),
            'twitter': TwitterScraper(self.config),
        }
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_run: Dict[str, datetime] = {}
    
    def scrape_all(self) -> Dict[str, int]:
        """Run all scrapers and store results"""
        results = {}
        
        for name, scraper in self.scrapers.items():
            try:
                logger.info(f"{'='*50}")
                logger.info(f"Running {name} scraper...")
                logger.info(f"{'='*50}")
                
                content = scraper.scrape()
                new_count = self.storage.store(content)
                
                results[name] = {
                    'total': len(content),
                    'new': new_count
                }
                
                self._last_run[name] = datetime.now()
                
                logger.info(f"  {name}: {len(content)} items, {new_count} new")
                
            except Exception as e:
                logger.error(f"Scraper {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def scrape_news(self) -> int:
        """Scrape only news"""
        content = self.scrapers['news'].scrape()
        return self.storage.store(content)
    
    def scrape_reddit(self) -> int:
        """Scrape only Reddit"""
        content = self.scrapers['reddit'].scrape()
        return self.storage.store(content)
    
    def scrape_twitter(self) -> int:
        """Scrape only Twitter"""
        content = self.scrapers['twitter'].scrape()
        return self.storage.store(content)
    
    def start_scheduled(self, interval_hours: float = 6.0):
        """Start scheduled scraping in background"""
        if self._running:
            return
        
        self._running = True
        
        def run_loop():
            while self._running:
                try:
                    logger.info("Starting scheduled scrape...")
                    self.scrape_all()
                    logger.info("Scheduled scrape complete")
                except Exception as e:
                    logger.error(f"Scheduled scrape failed: {e}")
                
                # Sleep until next run
                sleep_seconds = interval_hours * 3600
                for _ in range(int(sleep_seconds / 10)):
                    if not self._running:
                        break
                    time.sleep(10)
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Scheduled scraping started (every {interval_hours} hours)")
    
    def stop_scheduled(self):
        """Stop scheduled scraping"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduled scraping stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'running': self._running,
            'last_run': {k: v.isoformat() for k, v in self._last_run.items()},
            'storage_dir': str(self.storage.data_dir),
            'seen_count': len(self.storage.seen_hashes),
        }
    
    def get_recent_content(self, days: int = 1) -> List[Dict]:
        """Get recent scraped content"""
        return self.storage.get_recent(days)
    
    def export_training_data(self, output_file: str = "web_training.jsonl") -> int:
        """Export all web content for training"""
        return self.storage.export_for_training(output_file)


# ==================== Factory ====================

def create_web_intelligence() -> WebIntelligenceSystem:
    """Create web intelligence system with default config"""
    return WebIntelligenceSystem()


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Bagley Web Intelligence")
    parser.add_argument('--scrape', choices=['all', 'news', 'reddit', 'twitter'], default='all')
    parser.add_argument('--export', type=str, help='Export to training file')
    parser.add_argument('--schedule', type=float, help='Run every N hours')
    
    args = parser.parse_args()
    
    system = create_web_intelligence()
    
    if args.export:
        count = system.export_training_data(args.export)
        print(f"Exported {count} items to {args.export}")
    elif args.schedule:
        system.start_scheduled(args.schedule)
        print(f"Running every {args.schedule} hours. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop_scheduled()
    else:
        if args.scrape == 'all':
            results = system.scrape_all()
        elif args.scrape == 'news':
            results = {'news': system.scrape_news()}
        elif args.scrape == 'reddit':
            results = {'reddit': system.scrape_reddit()}
        elif args.scrape == 'twitter':
            results = {'twitter': system.scrape_twitter()}
        
        print(f"\nResults: {json.dumps(results, indent=2)}")


# ==================== Exports ====================

__all__ = [
    'ScraperConfig',
    'ContentType',
    'WebContent',
    'NewsScraper',
    'RedditScraper',
    'TwitterScraper',
    'WebIntelligenceStorage',
    'WebIntelligenceSystem',
    'create_web_intelligence',
]
