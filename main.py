"""
ChatMed Medical Data Scraper - Main Entry Point
Production-ready scraper with zero-bug guarantees (2025).

Usage:
    python main.py scrape --query "diabetes symptoms" --pages 10
    python main.py scrape --query "diabète Afrique" --source pubmed --dry-run
    python main.py api --host 127.0.0.1 --port 8000
    python main.py validate --file data.jsonl
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
import uvicorn
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Fallback pour pendulum
try:
    import pendulum
    PENDULUM_AVAILABLE = True
except ImportError:
    PENDULUM_AVAILABLE = False
    from dateutil import parser as dateparser

from core.cleaner import DataPipeline
from core.scraper import AsyncScraper
from core.storage import DataStorage
from core.web_scraper import UniversalWebScraper
from core.autonomous_agent import AutonomousAgent
from utils.config import get_config

# Initialize Typer app
app = typer.Typer(
    name="chatmed-scraper",
    help="Production-ready medical data scraper for ChatMed (2025)",
    add_completion=False,
)

console = Console()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with Loguru."""
    logger.remove()  # Remove default handler

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler (from config)
    config = get_config()
    log_file = Path(config.monitoring.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=log_level,
        rotation=config.monitoring.log_rotation,
        compression="zip",
        serialize=config.monitoring.log_format == "json",
    )


@app.command()
def scrape(
    query: str = typer.Option(..., "--query", "-q", help="Search query for medical data"),
    pages: int = typer.Option(15, "--pages", "-p", help="Maximum pages to scrape"),
    source: str = typer.Option("pubmed", "--source", "-s", help="Data source (pubmed, who)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Test mode without saving data"),
    validate_only: bool = typer.Option(
        False, "--validate-only", help="Only validate without enrichment"
    ),
    config_file: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """
    Scrape medical data from specified source.
    Processes, validates, and exports data locally.
    """
    setup_logging(log_level)

    try:
        # Load configuration
        config = get_config(config_file)

        console.print(f"\n[bold cyan]ChatMed Medical Scraper 2025[/bold cyan]")
        console.print(f"Query: [yellow]{query}[/yellow]")
        console.print(f"Source: [yellow]{source}[/yellow]")
        console.print(f"Max pages: [yellow]{pages}[/yellow]")
        console.print(f"Dry run: [yellow]{dry_run}[/yellow]\n")

        # Run scraping
        asyncio.run(
            _run_scrape(
                query=query,
                max_results=pages * 10,
                source=source,
                dry_run=dry_run,
                validate_only=validate_only,
                config=config,
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Scraping failed")
        raise typer.Exit(code=1)


async def _run_scrape(
    query: str,
    max_results: int,
    source: str,
    dry_run: bool,
    validate_only: bool,
    config: any,
) -> None:
    """Internal async function for scraping."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize components
        task1 = progress.add_task("[cyan]Initializing scraper...", total=None)

        async with AsyncScraper(config) as scraper:
            pipeline = DataPipeline(config)
            storage = DataStorage(config.data)

            progress.update(task1, completed=True)

            # Scrape data
            task2 = progress.add_task(f"[cyan]Scraping {source}...", total=None)

            if source == "pubmed":
                raw_entries = await scraper.fetch_pubmed_api(query, max_results)
            else:
                console.print(f"[yellow]Warning: Source '{source}' not fully implemented[/yellow]")
                raw_entries = []

            progress.update(task2, completed=True)

            if not raw_entries:
                console.print("[yellow]No results found[/yellow]")
                return

            console.print(f"\n[green]✓[/green] Scraped {len(raw_entries)} entries")

            # Process data
            task3 = progress.add_task("[cyan]Processing data...", total=len(raw_entries))

            processed_entries = []
            for i, raw_entry in enumerate(raw_entries):
                result = pipeline.process(raw_entry)
                if result is not None:
                    processed_entries.append(result)
                progress.update(task3, advance=1)

            console.print(f"[green]✓[/green] Processed {len(processed_entries)} entries")

            # Export data
            if not dry_run and processed_entries:
                task4 = progress.add_task("[cyan]Exporting data...", total=None)

                export_results = storage.export_all(processed_entries)

                progress.update(task4, completed=True)

                # Display export results
                console.print("\n[bold green]Export Complete[/bold green]")
                for format_name, path in export_results.items():
                    if path:
                        console.print(f"  • {format_name.upper()}: [cyan]{path}[/cyan]")

            # Display statistics
            _display_stats(scraper.get_stats(), pipeline.get_stats(), storage.get_stats())

            # Display sample entries
            if processed_entries:
                _display_sample_entries(processed_entries[:3])


def _display_stats(
    scraper_stats: dict, pipeline_stats: dict, storage_stats: dict
) -> None:
    """Display statistics in a formatted table."""
    table = Table(title="Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Value", justify="right", style="green")

    # Scraper stats
    for key, value in scraper_stats.items():
        table.add_row("Scraper", key.replace("_", " ").title(), str(value))

    # Pipeline stats
    for key, value in pipeline_stats.items():
        table.add_row("Pipeline", key.replace("_", " ").title(), str(value))

    # Storage stats
    for key, value in storage_stats.items():
        if isinstance(value, float):
            table.add_row("Storage", key.replace("_", " ").title(), f"{value:.2f}")
        else:
            table.add_row("Storage", key.replace("_", " ").title(), str(value))

    console.print("\n")
    console.print(table)


def _display_sample_entries(entries: list) -> None:
    """Display sample entries."""
    console.print("\n[bold cyan]Sample Entries[/bold cyan]\n")

    for i, entry in enumerate(entries, 1):
        console.print(f"[bold yellow]Entry {i}:[/bold yellow]")
        console.print(f"  Title: {entry.title[:100]}...")
        console.print(f"  Abstract: {entry.abstract[:200]}...")
        console.print(f"  Quality: {entry.quality_score:.2f}")
        console.print(f"  Relevance: {entry.relevance_score:.2f}")
        if entry.qa_pairs:
            console.print(f"  Q&A Pairs: {len(entry.qa_pairs)}")
        console.print()


@app.command()
def scrape_web(
    urls_file: Path = typer.Option(..., "--file", "-f", help="File containing URLs (one per line)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (auto-generated if not specified)"),
    config_file: Path = typer.Option(Path("config.toml"), "--config", "-c", help="Path to config file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """
    Scrape any websites from a list of URLs.
    Universal web scraper - works with any medical website.
    Each URL file gets its own output folder.
    
    Example:
        python main.py scrape-web --file mes_sites.txt
        Output: data/output/mes_sites/web_scrape_YYYYMMDD_HHMMSS.jsonl
    """
    setup_logging(log_level)
    
    try:
        config = get_config(config_file)
        
        # Auto-generate output path based on input filename
        if output is None:
            # Get filename without extension
            file_stem = urls_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"data/output/{file_stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"web_scrape_{timestamp}.jsonl"
        
        console.print(f"\n[bold cyan]Universal Web Scraper[/bold cyan]")
        console.print(f"URLs file: [yellow]{urls_file}[/yellow]")
        console.print(f"Output folder: [yellow]{output.parent}[/yellow]")
        console.print(f"Output file: [yellow]{output.name}[/yellow]\n")
        
        # Run scraping
        asyncio.run(_run_web_scrape(urls_file, output, config))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Web scraping failed")
        raise typer.Exit(code=1)


async def _run_web_scrape(urls_file: Path, output: Path, config: any) -> None:
    """Run web scraping from URLs file."""
    import json
    
    scraper = UniversalWebScraper(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scraping websites...", total=None)
        
        # Scrape all URLs
        results = await scraper.scrape_urls_from_file(urls_file)
        
        progress.update(task, description=f"Scraped {len(results)} URLs")
    
    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    console.print(f"\n[bold green]✓[/bold green] Saved {len(results)} results to {output}")
    
    # Display sample
    if results:
        console.print("\n[bold cyan]Sample Results:[/bold cyan]\n")
        for i, result in enumerate(results[:3], 1):
            console.print(f"[bold yellow]{i}. {result['title']}[/bold yellow]")
            console.print(f"   URL: {result['url']}")
            console.print(f"   Content length: {len(result['content'])} chars")
            console.print(f"   Images: {len(result['images'])}")
            console.print(f"   Links: {len(result['links'])}\n")


@app.command()
def scrape_all(
    folder: Path = typer.Option(Path("."), "--folder", "-d", help="Folder containing URL files"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern to match"),
    config_file: Path = typer.Option(Path("config.toml"), "--config", "-c", help="Path to config file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """
    Scrape ALL URL files in a folder at once.
    Each file gets its own output folder automatically.
    
    Example:
        python main.py scrape-all
        python main.py scrape-all --folder . --pattern "*.txt"
    
    This will scrape:
        - diabete.txt → data/output/diabete/
        - paludisme.txt → data/output/paludisme/
        - mes_sites.txt → data/output/mes_sites/
    """
    setup_logging(log_level)
    
    try:
        config = get_config(config_file)
        
        # Find all URL files
        url_files = list(folder.glob(pattern))
        
        # Filter out non-URL files (exclude config, requirements, etc.)
        url_files = [f for f in url_files if f.name not in ['config.toml', 'requirements.txt', 'requirements-minimal.txt']]
        
        if not url_files:
            console.print(f"[bold red]No URL files found matching pattern '{pattern}' in {folder}[/bold red]")
            raise typer.Exit(code=1)
        
        console.print(f"\n[bold cyan]Scraping Multiple URL Files[/bold cyan]")
        console.print(f"Found {len(url_files)} file(s) to process:\n")
        
        for f in url_files:
            console.print(f"  • [yellow]{f.name}[/yellow]")
        
        console.print()
        
        # Process each file
        total_results = 0
        for i, urls_file in enumerate(url_files, 1):
            console.print(f"\n[bold magenta]═══ Processing {i}/{len(url_files)}: {urls_file.name} ═══[/bold magenta]\n")
            
            # Auto-generate output path
            file_stem = urls_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"data/output/{file_stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"web_scrape_{timestamp}.jsonl"
            
            console.print(f"Output: [yellow]{output}[/yellow]\n")
            
            # Run scraping
            asyncio.run(_run_web_scrape(urls_file, output, config))
            
            # Count results
            if output.exists():
                with open(output, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                    total_results += count
        
        # Summary
        console.print(f"\n[bold green]{'═' * 60}[/bold green]")
        console.print(f"[bold green]✓ COMPLETED[/bold green]")
        console.print(f"[bold green]{'═' * 60}[/bold green]")
        console.print(f"Processed: [yellow]{len(url_files)}[/yellow] files")
        console.print(f"Total results: [yellow]{total_results}[/yellow] websites scraped")
        console.print(f"Output location: [yellow]data/output/[/yellow]\n")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Batch scraping failed")
        raise typer.Exit(code=1)


@app.command()
def auto_scrape(
    topics: Optional[int] = typer.Option(None, "--topics", "-t", help="Number of topics to generate"),
    continuous: bool = typer.Option(False, "--continuous", help="Run continuously"),
    config_file: Path = typer.Option(Path("config.toml"), "--config", "-c", help="Path to config file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """
    🤖 AUTONOMOUS SCRAPING MODE
    
    The agent will:
    1. Generate medical topics automatically using DeepSeek AI
    2. Find the best URLs for each topic
    3. Scrape and validate content
    4. Enrich data with Q&A pairs
    5. Save organized datasets
    
    Zero human intervention required!
    
    Examples:
        # Scrape 50 topics (default)
        python main.py auto-scrape
        
        # Scrape 100 topics
        python main.py auto-scrape --topics 100
        
        # Run continuously (infinite loop)
        python main.py auto-scrape --continuous
    """
    setup_logging(log_level)
    
    try:
        config = get_config(config_file)
        
        # Check if autonomous mode is enabled
        if not config.autonomous.enabled:
            console.print("[bold red]Error:[/bold red] Autonomous mode is disabled in config.toml")
            console.print("Enable it by setting [autonomous] enabled = true")
            raise typer.Exit(code=1)
        
        # Check DeepSeek API key
        if not config.deepseek.api_key:
            console.print("[bold red]Error:[/bold red] DeepSeek API key not configured")
            console.print("Add your API key in config.toml under [deepseek] section:")
            console.print("  api_key = \"your-api-key-here\"")
            console.print("\nGet your API key at: https://platform.deepseek.com/")
            raise typer.Exit(code=1)
        
        # Run autonomous agent
        asyncio.run(_run_autonomous_agent(config, topics, continuous))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Autonomous scraping failed")
        raise typer.Exit(code=1)


async def _run_autonomous_agent(config: any, topics: Optional[int], continuous: bool) -> None:
    """Run the autonomous agent"""
    async with AutonomousAgent(config) as agent:
        stats = await agent.run_autonomous(topics_count=topics, continuous=continuous)
        
        # Display success message
        if stats.topics_completed > 0:
            console.print(f"[bold green]✓ Successfully processed {stats.topics_completed} topics![/bold green]")
            console.print(f"[yellow]Check data/output/ for results[/yellow]")


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", "--host", help="API host"),
    port: int = typer.Option(8000, "--port", help="API port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    config_file: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config file"
    ),
) -> None:
    """
    Start FastAPI server for scraper API.
    Provides REST endpoints for scraping and validation.
    """
    setup_logging("INFO")

    try:
        config = get_config(config_file)

        if not config.api.enabled:
            console.print(
                "[yellow]Warning: API is disabled in config. Enable it in config.toml[/yellow]"
            )

        console.print(f"\n[bold cyan]Starting ChatMed Scraper API[/bold cyan]")
        console.print(f"Host: [yellow]{host}[/yellow]")
        console.print(f"Port: [yellow]{port}[/yellow]")
        console.print(f"Docs: [cyan]http://{host}:{port}/docs[/cyan]\n")

        # Import and create FastAPI app
        from fastapi import FastAPI
        from api.routes import router

        api_app = FastAPI(
            title="ChatMed Medical Scraper API",
            description="Production-ready medical data scraping API (2025)",
            version="1.0.0",
        )
        api_app.include_router(router, prefix="/api/v1")

        # Run server
        uvicorn.run(api_app, host=host, port=port, reload=reload, log_level="info")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("API startup failed")
        raise typer.Exit(code=1)


@app.command()
def validate(
    file: Path = typer.Option(..., "--file", "-f", help="JSONL file to validate"),
    config_file: Path = typer.Option(
        Path("config.toml"), "--config", "-c", help="Path to config file"
    ),
) -> None:
    """
    Validate data from JSONL file.
    Checks data quality and compliance.
    """
    setup_logging("INFO")

    try:
        config = get_config(config_file)
        storage = DataStorage(config.data)
        pipeline = DataPipeline(config)

        console.print(f"\n[bold cyan]Validating data from:[/bold cyan] {file}\n")

        # Load data
        entries = storage.load_jsonl(file.name)

        if not entries:
            console.print("[yellow]No entries found in file[/yellow]")
            return

        # Validate each entry
        valid_count = 0
        invalid_count = 0

        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Validating...", total=len(entries))

            for entry in entries:
                validated = pipeline.validate_step(entry)
                if validated is not None:
                    valid_count += 1
                else:
                    invalid_count += 1
                progress.update(task, advance=1)

        # Display results
        console.print(f"\n[green]✓[/green] Valid entries: {valid_count}")
        console.print(f"[red]✗[/red] Invalid entries: {invalid_count}")
        console.print(
            f"[cyan]Validation rate:[/cyan] {valid_count / len(entries) * 100:.1f}%\n"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Validation failed")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Display system information and configuration."""
    try:
        config = get_config()

        console.print("\n[bold cyan]ChatMed Medical Scraper - System Info[/bold cyan]\n")

        # Configuration info
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Default Query", config.scraper.default_query)
        table.add_row("Max Pages", str(config.scraper.max_pages))
        table.add_row("Concurrent Requests", str(config.scraper.concurrent_requests))
        table.add_row("Output Directory", config.data.output_dir)
        table.add_row("Export JSONL", "✓" if config.data.export_jsonl else "✗")
        table.add_row("Export Parquet", "✓" if config.data.export_parquet else "✗")
        table.add_row("Export SQLite", "✓" if config.data.export_sqlite else "✗")
        table.add_row("ML Enrichment", "✓" if config.ml.extract_entities else "✗")
        table.add_row("PII Anonymization", "✓" if config.ethics.anonymize_pii else "✗")
        table.add_row("API Enabled", "✓" if config.api.enabled else "✗")

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
