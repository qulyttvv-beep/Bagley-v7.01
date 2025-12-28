"""
📅 Bagley Daily Intelligence Scheduler
======================================
Automated daily scraping with smart scheduling
and error recovery
"""

import os
import sys
import json
import time
import logging
import threading
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for scheduled intelligence gathering"""
    
    # Scrape times (24h format)
    morning_scrape: str = "06:00"      # 6 AM - morning news
    afternoon_scrape: str = "12:00"    # 12 PM - midday update
    evening_scrape: str = "18:00"      # 6 PM - evening roundup
    night_scrape: str = "23:00"        # 11 PM - end of day
    
    # What to scrape at each time
    scrape_news: bool = True
    scrape_reddit: bool = True
    scrape_twitter: bool = True
    
    # Recovery
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay_minutes: int = 30
    
    # Notifications
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None


class DailyIntelligenceScheduler:
    """
    🤖 Automated daily intelligence gathering for Bagley
    
    Features:
    - Multiple daily scrape times
    - Automatic error recovery
    - Smart retry logic
    - Status tracking
    """
    
    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self._web_intel = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_results: Dict[str, Any] = {}
        self._failure_count: Dict[str, int] = {}
        self._stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_items': 0,
            'new_items': 0,
            'started_at': None,
        }
    
    def _get_web_intel(self):
        """Lazy load web intelligence system"""
        if self._web_intel is None:
            from bagley.core.web_intelligence import WebIntelligenceSystem
            self._web_intel = WebIntelligenceSystem()
        return self._web_intel
    
    def _run_scrape(self, scrape_type: str = "all"):
        """Run a scrape with error handling"""
        logger.info(f"{'='*60}")
        logger.info(f"🔄 Starting {scrape_type} scrape at {datetime.now()}")
        logger.info(f"{'='*60}")
        
        self._stats['total_runs'] += 1
        
        try:
            web_intel = self._get_web_intel()
            results = {}
            
            if scrape_type == "all" or scrape_type == "full":
                results = web_intel.scrape_all()
            elif scrape_type == "news":
                if self.config.scrape_news:
                    new = web_intel.scrape_news()
                    results['news'] = {'new': new}
            elif scrape_type == "reddit":
                if self.config.scrape_reddit:
                    new = web_intel.scrape_reddit()
                    results['reddit'] = {'new': new}
            elif scrape_type == "twitter":
                if self.config.scrape_twitter:
                    new = web_intel.scrape_twitter()
                    results['twitter'] = {'new': new}
            elif scrape_type == "quick":
                # Quick scrape - just news (fastest)
                if self.config.scrape_news:
                    new = web_intel.scrape_news()
                    results['news'] = {'new': new}
            
            # Track results
            self._last_results = {
                'timestamp': datetime.now().isoformat(),
                'type': scrape_type,
                'results': results,
                'success': True
            }
            
            # Update stats
            self._stats['successful_runs'] += 1
            for source, data in results.items():
                if isinstance(data, dict):
                    self._stats['total_items'] += data.get('total', 0)
                    self._stats['new_items'] += data.get('new', 0)
            
            # Reset failure count
            self._failure_count = {}
            
            logger.info(f"✅ Scrape complete: {results}")
            
            if self.config.on_success:
                self.config.on_success(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Scrape failed: {e}")
            
            self._stats['failed_runs'] += 1
            self._last_results = {
                'timestamp': datetime.now().isoformat(),
                'type': scrape_type,
                'error': str(e),
                'success': False
            }
            
            # Track failures for retry
            self._failure_count[scrape_type] = self._failure_count.get(scrape_type, 0) + 1
            
            if self.config.on_failure:
                self.config.on_failure(e)
            
            # Schedule retry if needed
            if self.config.retry_on_failure:
                if self._failure_count[scrape_type] < self.config.max_retries:
                    logger.info(f"🔄 Scheduling retry in {self.config.retry_delay_minutes} minutes")
                    schedule.every(self.config.retry_delay_minutes).minutes.do(
                        lambda: self._run_scrape(scrape_type)
                    ).tag('retry')
            
            return {'error': str(e)}
    
    def _schedule_jobs(self):
        """Set up the daily schedule"""
        schedule.clear()
        
        # Morning - full scrape
        schedule.every().day.at(self.config.morning_scrape).do(
            lambda: self._run_scrape("all")
        )
        
        # Afternoon - quick update
        schedule.every().day.at(self.config.afternoon_scrape).do(
            lambda: self._run_scrape("quick")
        )
        
        # Evening - full scrape
        schedule.every().day.at(self.config.evening_scrape).do(
            lambda: self._run_scrape("all")
        )
        
        # Night - full scrape
        schedule.every().day.at(self.config.night_scrape).do(
            lambda: self._run_scrape("all")
        )
        
        logger.info(f"📅 Scheduled scrapes:")
        logger.info(f"   Morning:   {self.config.morning_scrape} (full)")
        logger.info(f"   Afternoon: {self.config.afternoon_scrape} (quick)")
        logger.info(f"   Evening:   {self.config.evening_scrape} (full)")
        logger.info(f"   Night:     {self.config.night_scrape} (full)")
    
    def start(self, run_immediately: bool = True):
        """Start the scheduler"""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._stats['started_at'] = datetime.now().isoformat()
        
        # Set up schedule
        self._schedule_jobs()
        
        # Run immediately if requested
        if run_immediately:
            self._run_scrape("all")
        
        # Start background thread
        def run_scheduler():
            while self._running:
                try:
                    schedule.run_pending()
                    
                    # Clear completed retry jobs
                    schedule.clear('retry')
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                
                time.sleep(60)  # Check every minute
        
        self._thread = threading.Thread(target=run_scheduler, daemon=True)
        self._thread.start()
        
        logger.info("🚀 Daily intelligence scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        schedule.clear()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("🛑 Daily intelligence scheduler stopped")
    
    def run_now(self, scrape_type: str = "all"):
        """Run a scrape immediately"""
        return self._run_scrape(scrape_type)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            'running': self._running,
            'stats': self._stats,
            'last_results': self._last_results,
            'next_runs': [
                {
                    'time': job.next_run.isoformat() if job.next_run else None,
                    'job': str(job)
                }
                for job in schedule.get_jobs()
            ],
            'config': {
                'morning': self.config.morning_scrape,
                'afternoon': self.config.afternoon_scrape,
                'evening': self.config.evening_scrape,
                'night': self.config.night_scrape,
            }
        }
    
    def get_recent_content(self, hours: int = 24) -> list:
        """Get content from the last N hours"""
        web_intel = self._get_web_intel()
        days = max(1, hours // 24 + 1)
        return web_intel.get_recent_content(days)


class BagleyKnowledgeUpdater:
    """
    🧠 Updates Bagley's knowledge base with web intelligence
    
    This class converts scraped web content into
    formats Bagley can use for context and training
    """
    
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        self._daily_summaries: Dict[str, str] = {}
    
    def process_content(self, content: list) -> Dict[str, Any]:
        """Process web content into knowledge format"""
        
        # Group by type
        by_type = {}
        for item in content:
            ct = item.get('content_type', 'unknown')
            if ct not in by_type:
                by_type[ct] = []
            by_type[ct].append(item)
        
        # Create daily summary
        summary = self._create_summary(by_type)
        
        # Save knowledge files
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Save raw content
        raw_file = self.knowledge_dir / f"{today}_raw.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = self.knowledge_dir / f"{today}_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self._daily_summaries[today] = summary
        
        return {
            'items_processed': len(content),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'summary_file': str(summary_file),
            'raw_file': str(raw_file),
        }
    
    def _create_summary(self, by_type: Dict[str, list]) -> str:
        """Create a human-readable summary"""
        lines = [
            f"# Bagley Daily Intelligence Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        
        # News summary
        if 'news' in by_type:
            lines.append("## 📰 News Headlines")
            for item in by_type['news'][:20]:
                lines.append(f"- **{item.get('source', 'Unknown')}**: {item.get('title', 'No title')}")
            lines.append("")
        
        # Reddit summary
        if 'reddit_post' in by_type:
            lines.append("## 🔴 Reddit Trending")
            for item in by_type['reddit_post'][:20]:
                score = item.get('metadata', {}).get('score', 0)
                lines.append(f"- **{item.get('source', '')}** ({score}↑): {item.get('title', '')}")
            lines.append("")
        
        # Twitter summary
        if 'tweet' in by_type:
            lines.append("## 🐦 Twitter/X Updates")
            for item in by_type['tweet'][:20]:
                lines.append(f"- **{item.get('author', 'Unknown')}**: {item.get('content', '')[:100]}...")
            lines.append("")
        
        # Stats
        total = sum(len(v) for v in by_type.values())
        lines.extend([
            "---",
            f"**Total Items:** {total}",
            f"**Sources:** News ({len(by_type.get('news', []))}), "
            f"Reddit ({len(by_type.get('reddit_post', []))}), "
            f"Twitter ({len(by_type.get('tweet', []))})",
        ])
        
        return '\n'.join(lines)
    
    def get_context_for_chat(self, query: str = None) -> str:
        """Get relevant context for chat responses"""
        # Get most recent summary
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in self._daily_summaries:
            return self._daily_summaries[today]
        
        # Try to load from file
        summary_file = self.knowledge_dir / f"{today}_summary.md"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        return "No recent intelligence data available."
    
    def export_for_finetuning(self, output_file: str = "web_finetune.jsonl") -> int:
        """Export knowledge for fine-tuning"""
        count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for raw_file in sorted(self.knowledge_dir.glob("*_raw.json")):
                try:
                    with open(raw_file, 'r', encoding='utf-8') as rf:
                        content = json.load(rf)
                    
                    for item in content:
                        # Create training example
                        example = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": f"You are Bagley, an AI assistant with access to current news and social media."
                                },
                                {
                                    "role": "user",
                                    "content": f"What do you know about: {item.get('title', 'this topic')}?"
                                },
                                {
                                    "role": "assistant",
                                    "content": f"Based on recent {item.get('content_type', 'information')} from {item.get('source', 'the web')}: {item.get('content', '')}"
                                }
                            ]
                        }
                        
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                        count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {raw_file}: {e}")
                    continue
        
        return count


# ==================== Integration with Bagley ====================

class BagleyWebIntegration:
    """
    🔗 Integration layer between web intelligence and Bagley's core
    """
    
    def __init__(self):
        self.scheduler = DailyIntelligenceScheduler()
        self.knowledge = BagleyKnowledgeUpdater()
        self._initialized = False
    
    def initialize(self):
        """Initialize the web intelligence system"""
        if self._initialized:
            return
        
        logger.info("🌐 Initializing Bagley Web Intelligence...")
        
        # Set up success callback to process content
        def on_scrape_success(results):
            content = self.scheduler.get_recent_content(hours=1)
            self.knowledge.process_content(content)
        
        self.scheduler.config.on_success = on_scrape_success
        
        self._initialized = True
        logger.info("✅ Web Intelligence initialized")
    
    def start(self, run_immediately: bool = True):
        """Start web intelligence gathering"""
        self.initialize()
        self.scheduler.start(run_immediately)
    
    def stop(self):
        """Stop web intelligence gathering"""
        self.scheduler.stop()
    
    def get_current_context(self) -> str:
        """Get current web context for chat"""
        return self.knowledge.get_context_for_chat()
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status"""
        return {
            'scheduler': self.scheduler.get_status(),
            'knowledge_dir': str(self.knowledge.knowledge_dir),
            'initialized': self._initialized,
        }
    
    def scrape_now(self, source: str = "all"):
        """Run scrape immediately"""
        return self.scheduler.run_now(source)


# ==================== Factory ====================

_global_integration: Optional[BagleyWebIntegration] = None

def get_web_integration() -> BagleyWebIntegration:
    """Get or create the global web integration instance"""
    global _global_integration
    if _global_integration is None:
        _global_integration = BagleyWebIntegration()
    return _global_integration


def start_daily_intelligence():
    """Start daily intelligence gathering"""
    integration = get_web_integration()
    integration.start()
    return integration


def stop_daily_intelligence():
    """Stop daily intelligence gathering"""
    global _global_integration
    if _global_integration:
        _global_integration.stop()


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Bagley Daily Intelligence")
    parser.add_argument('--start', action='store_true', help='Start scheduler')
    parser.add_argument('--scrape', choices=['all', 'news', 'reddit', 'twitter', 'quick'])
    parser.add_argument('--export', type=str, help='Export for fine-tuning')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    integration = get_web_integration()
    
    if args.status:
        print(json.dumps(integration.get_status(), indent=2, default=str))
    elif args.export:
        count = integration.knowledge.export_for_finetuning(args.export)
        print(f"Exported {count} examples to {args.export}")
    elif args.scrape:
        integration.initialize()
        results = integration.scrape_now(args.scrape)
        print(json.dumps(results, indent=2, default=str))
    elif args.start:
        print("Starting daily intelligence scheduler...")
        print("Press Ctrl+C to stop")
        integration.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            integration.stop()
    else:
        parser.print_help()


# ==================== Exports ====================

__all__ = [
    'ScheduleConfig',
    'DailyIntelligenceScheduler',
    'BagleyKnowledgeUpdater',
    'BagleyWebIntegration',
    'get_web_integration',
    'start_daily_intelligence',
    'stop_daily_intelligence',
]
