"""
Autonomous Medical Scraping Agent
Generates topics, finds URLs, scrapes, and enriches data automatically.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from utils.deepseek_client import DeepSeekClient, create_deepseek_client
from core.web_scraper import UniversalWebScraper


console = Console()


@dataclass
class TopicResult:
    """Result of scraping a single topic"""
    topic: str
    urls_found: int
    urls_scraped: int
    pages_kept: int
    pages_rejected: int
    avg_quality_score: float
    output_file: str
    started_at: str
    completed_at: str
    duration_seconds: float
    errors: List[str]


@dataclass
class AgentStats:
    """Overall agent statistics"""
    total_topics: int
    topics_completed: int
    topics_failed: int
    total_pages_scraped: int
    total_pages_kept: int
    total_pages_rejected: int
    avg_quality_score: float
    started_at: str
    completed_at: Optional[str]
    total_duration_seconds: float


class AutonomousAgent:
    """
    Autonomous agent that generates medical topics and scrapes them automatically.
    Zero human intervention required.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.deepseek_client: Optional[DeepSeekClient] = None
        self.web_scraper: Optional[UniversalWebScraper] = None
        self.results: List[TopicResult] = []
        self.stats: Optional[AgentStats] = None
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        
    async def initialize(self):
        """Initialize agent components"""
        logger.info("Initializing Autonomous Agent...")
        
        # Check if DeepSeek API key is configured
        if not self.config.deepseek.api_key:
            raise ValueError(
                "DeepSeek API key not configured. "
                "Please add your API key in config.toml under [deepseek] section."
            )
        
        # Initialize DeepSeek client
        self.deepseek_client = create_deepseek_client(self.config)
        await self.deepseek_client.init_session()
        
        # Initialize web scraper
        self.web_scraper = UniversalWebScraper(self.config)
        
        logger.success("Autonomous Agent initialized")
        
    async def cleanup(self):
        """Cleanup agent resources"""
        if self.deepseek_client:
            await self.deepseek_client.close_session()
        logger.info("Autonomous Agent cleaned up")
    
    async def run_autonomous(
        self,
        topics_count: Optional[int] = None,
        continuous: bool = False
    ) -> AgentStats:
        """
        Run autonomous scraping
        
        Args:
            topics_count: Number of topics to generate (None = use config)
            continuous: Run continuously until stopped
            
        Returns:
            Agent statistics
        """
        topics_count = topics_count or self.config.autonomous.topics_per_run
        
        console.print("\n[bold cyan]🤖 Autonomous Medical Scraping Agent[/bold cyan]")
        console.print(f"[yellow]Mode:[/yellow] {'Continuous' if continuous else f'{topics_count} topics'}")
        console.print(f"[yellow]Quality threshold:[/yellow] {self.config.autonomous.quality_threshold}")
        console.print()
        
        start_time = datetime.now()
        
        # Initialize stats
        self.stats = AgentStats(
            total_topics=topics_count,
            topics_completed=0,
            topics_failed=0,
            total_pages_scraped=0,
            total_pages_kept=0,
            total_pages_rejected=0,
            avg_quality_score=0.0,
            started_at=start_time.isoformat(),
            completed_at=None,
            total_duration_seconds=0.0
        )
        
        try:
            while True:
                # Step 1: Generate topics
                console.print("[bold magenta]📋 Step 1: Generating medical topics...[/bold magenta]")
                topics = await self.deepseek_client.generate_medical_topics(
                    count=topics_count,
                    language="fr"
                )
                
                if not topics:
                    logger.error("Failed to generate topics")
                    break
                
                console.print(f"[green]✓[/green] Generated {len(topics)} topics\n")
                
                # Display topics
                self._display_topics(topics)
                
                # Step 2: Process each topic
                for i, topic in enumerate(topics, 1):
                    console.print(f"\n[bold yellow]{'═' * 60}[/bold yellow]")
                    console.print(f"[bold cyan]Processing {i}/{len(topics)}: {topic}[/bold cyan]")
                    console.print(f"[bold yellow]{'═' * 60}[/bold yellow]\n")
                    
                    try:
                        result = await self._process_topic(topic)
                        self.results.append(result)
                        self.stats.topics_completed += 1
                        
                        # Update stats
                        self.stats.total_pages_scraped += result.urls_scraped
                        self.stats.total_pages_kept += result.pages_kept
                        self.stats.total_pages_rejected += result.pages_rejected
                        
                    except Exception as e:
                        logger.error(f"Failed to process topic '{topic}': {e}")
                        self.stats.topics_failed += 1
                    
                    # Sleep between topics
                    if i < len(topics):
                        await asyncio.sleep(self.config.autonomous.sleep_between_topics)
                
                # If not continuous, break after one run
                if not continuous:
                    break
                
                console.print("\n[bold green]Continuous mode: Starting next batch...[/bold green]\n")
                await asyncio.sleep(60)  # Wait 1 minute before next batch
                
        except KeyboardInterrupt:
            console.print("\n[bold red]Agent stopped by user[/bold red]")
        finally:
            # Finalize stats
            end_time = datetime.now()
            self.stats.completed_at = end_time.isoformat()
            self.stats.total_duration_seconds = (end_time - start_time).total_seconds()
            
            # Calculate average quality
            if self.stats.total_pages_kept > 0:
                total_quality = sum(r.avg_quality_score * r.pages_kept for r in self.results)
                self.stats.avg_quality_score = total_quality / self.stats.total_pages_kept
            
            # Display final stats
            self._display_final_stats()
            
            # Save report
            await self._save_report()
        
        return self.stats
    
    async def _process_topic(self, topic: str) -> TopicResult:
        """Process a single medical topic"""
        start_time = datetime.now()
        errors = []
        
        # Step 1: Find URLs
        console.print(f"[cyan]🔍 Finding sources for '{topic}'...[/cyan]")
        urls = await self.deepseek_client.suggest_urls_for_topic(
            topic,
            max_urls=self.config.autonomous.max_urls_per_topic
        )
        
        if not urls:
            logger.warning(f"No URLs found for topic: {topic}")
            urls = []
        
        console.print(f"[green]✓[/green] Found {len(urls)} URLs\n")
        
        # Step 2: Scrape URLs
        console.print(f"[cyan]🌐 Scraping {len(urls)} URLs...[/cyan]")
        
        scraped_data = []
        if urls:
            # Create temporary file with URLs
            temp_file = Path(f"temp_{topic.replace(' ', '_')}.txt")
            temp_file.write_text('\n'.join(urls), encoding='utf-8')
            
            try:
                scraped_data = await self.web_scraper.scrape_urls_from_file(temp_file)
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                errors.append(f"Scraping error: {str(e)}")
            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
        
        console.print(f"[green]✓[/green] Scraped {len(scraped_data)} pages\n")
        
        # Step 3: Validate and enrich with DeepSeek
        console.print(f"[cyan]🤖 Validating and enriching content...[/cyan]")
        
        enriched_data = []
        pages_kept = 0
        pages_rejected = 0
        quality_scores = []
        
        for data in scraped_data:
            try:
                # Validate content
                validation = await self.deepseek_client.validate_medical_content(
                    data.get('content', '')[:5000],  # Limit content size
                    data.get('url', '')
                )
                
                quality_score = validation.get('quality_score', 0.0)
                quality_scores.append(quality_score)
                
                # Keep only high-quality content
                if quality_score >= self.config.autonomous.quality_threshold:
                    # Enrich with Q&A pairs
                    if self.config.autonomous.multilingual_enabled:
                        # Generate multilingual Q&A
                        multilingual_qa = await self.deepseek_client.generate_multilingual_qa_pairs(
                            data.get('content', '')[:15000],
                            count=5,  # 5 Q&A par langue
                            languages=self.config.autonomous.target_languages
                        )
                        data['qa_pairs_multilingual'] = multilingual_qa
                        # Count total Q&A
                        total_qa = sum(len(qa_list) for qa_list in multilingual_qa.values())
                        logger.info(f"Generated {total_qa} Q&A in {len(multilingual_qa)} languages")
                    else:
                        # Single language Q&A (legacy)
                        qa_pairs = await self.deepseek_client.generate_qa_pairs(
                            data.get('content', '')[:15000],
                            count=10
                        )
                        data['qa_pairs'] = qa_pairs
                    
                    # Add enrichment
                    data['deepseek_validation'] = validation
                    data['quality_score'] = quality_score
                    
                    enriched_data.append(data)
                    pages_kept += 1
                else:
                    pages_rejected += 1
                    logger.debug(f"Rejected low-quality page: {data.get('url', 'unknown')}")
                    
            except Exception as e:
                logger.error(f"Enrichment failed for {data.get('url', 'unknown')}: {e}")
                errors.append(f"Enrichment error: {str(e)}")
                pages_rejected += 1
        
        console.print(f"[green]✓[/green] Kept {pages_kept} pages, rejected {pages_rejected}\n")
        
        # Step 4: Save results
        output_dir = Path(f"data/output/{topic.replace(' ', '_').lower()}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"autonomous_scrape_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in enriched_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        console.print(f"[green]✓[/green] Saved to: {output_file}\n")
        
        # Create result
        end_time = datetime.now()
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return TopicResult(
            topic=topic,
            urls_found=len(urls),
            urls_scraped=len(scraped_data),
            pages_kept=pages_kept,
            pages_rejected=pages_rejected,
            avg_quality_score=avg_quality,
            output_file=str(output_file),
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            errors=errors
        )
    
    def _display_topics(self, topics: List[str]):
        """Display generated topics in a table"""
        table = Table(title="Generated Medical Topics", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Topic", style="cyan")
        
        for i, topic in enumerate(topics[:20], 1):  # Show first 20
            table.add_row(str(i), topic)
        
        if len(topics) > 20:
            table.add_row("...", f"... and {len(topics) - 20} more")
        
        console.print(table)
        console.print()
    
    def _display_final_stats(self):
        """Display final statistics"""
        console.print(f"\n[bold green]{'═' * 60}[/bold green]")
        console.print(f"[bold green]🎉 AUTONOMOUS SCRAPING COMPLETED[/bold green]")
        console.print(f"[bold green]{'═' * 60}[/bold green]\n")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Topics Processed", f"{self.stats.topics_completed}/{self.stats.total_topics}")
        table.add_row("Topics Failed", str(self.stats.topics_failed))
        table.add_row("Total Pages Scraped", str(self.stats.total_pages_scraped))
        table.add_row("Pages Kept", str(self.stats.total_pages_kept))
        table.add_row("Pages Rejected", str(self.stats.total_pages_rejected))
        table.add_row("Avg Quality Score", f"{self.stats.avg_quality_score:.2f}")
        table.add_row("Total Duration", f"{self.stats.total_duration_seconds:.1f}s")
        
        console.print(table)
        console.print()
    
    async def _save_report(self):
        """Save detailed report"""
        report_dir = Path("data/output/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"autonomous_report_{timestamp}.json"
        
        report = {
            "stats": asdict(self.stats),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓[/green] Report saved to: {report_file}\n")
