"""
FastAPI routes for medical data scraper API.
Provides endpoints for scraping, validation, and data retrieval.
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.cleaner import DataPipeline, MedicalEntry
from core.scraper import AsyncScraper
from core.storage import DataStorage
from utils.config import AppConfig, get_config

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# API key security (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

router = APIRouter()


# Request/Response models
class ScrapeRequest(BaseModel):
    """Request model for scraping endpoint."""

    query: str = Field(min_length=3, max_length=500)
    max_results: int = Field(ge=1, le=1000, default=100)
    source: str = Field(default="pubmed", pattern="^(pubmed|who|generic)$")
    dry_run: bool = False


class ScrapeResponse(BaseModel):
    """Response model for scraping endpoint."""

    status: str
    entries_scraped: int
    entries_processed: int
    entries_exported: int
    export_paths: dict[str, Optional[str]]
    stats: dict[str, Any]


class ValidationRequest(BaseModel):
    """Request model for data validation endpoint."""

    data: dict[str, Any]


class ValidationResponse(BaseModel):
    """Response model for validation endpoint."""

    valid: bool
    entry: Optional[dict[str, Any]] = None
    errors: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""

    scraper_stats: dict[str, int]
    pipeline_stats: dict[str, int]
    storage_stats: dict[str, Any]


# Dependency for API key authentication
async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header), config: AppConfig = Depends(get_config)
) -> None:
    """Verify API key if authentication is enabled."""
    if not config.api.require_auth:
        return

    if not api_key or api_key != config.api.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


@router.post("/scrape", response_model=ScrapeResponse)
@limiter.limit("10/minute")
async def scrape_endpoint(
    request: ScrapeRequest,
    config: AppConfig = Depends(get_config),
    _: None = Depends(verify_api_key),
) -> ScrapeResponse:
    """
    Scrape medical data from specified source.
    Processes and exports data to all enabled formats.
    """
    try:
        logger.info(f"Scraping request: query={request.query}, source={request.source}")

        # Initialize components
        async with AsyncScraper(config) as scraper:
            pipeline = DataPipeline(config)
            storage = DataStorage(config.data)

            # Scrape data
            if request.source == "pubmed":
                raw_entries = await scraper.fetch_pubmed_api(request.query, request.max_results)
            else:
                # Generic scraping (would need URLs)
                raw_entries = []

            if not raw_entries:
                return ScrapeResponse(
                    status="no_results",
                    entries_scraped=0,
                    entries_processed=0,
                    entries_exported=0,
                    export_paths={},
                    stats={},
                )

            # Process entries
            processed_entries = pipeline.process_batch(raw_entries)

            # Export if not dry run
            export_paths: dict[str, Optional[str]] = {}
            if not request.dry_run and processed_entries:
                export_results = storage.export_all(processed_entries)
                export_paths = {k: str(v) if v else None for k, v in export_results.items()}

            # Get statistics
            stats = {
                "scraper": scraper.get_stats(),
                "pipeline": pipeline.get_stats(),
                "storage": storage.get_stats(),
            }

            return ScrapeResponse(
                status="success",
                entries_scraped=len(raw_entries),
                entries_processed=len(processed_entries),
                entries_exported=len(processed_entries) if not request.dry_run else 0,
                export_paths=export_paths,
                stats=stats,
            )

    except Exception as e:
        logger.error(f"Error in scrape endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
@limiter.limit("30/minute")
async def validate_endpoint(
    request: ValidationRequest,
    config: AppConfig = Depends(get_config),
    _: None = Depends(verify_api_key),
) -> ValidationResponse:
    """
    Validate medical data entry.
    Checks data quality and compliance with schema.
    """
    try:
        pipeline = DataPipeline(config)

        # Clean and validate
        cleaned = pipeline.clean_step(request.data)
        if cleaned is None:
            return ValidationResponse(
                valid=False, errors=["Data cleaning failed or entry is duplicate"]
            )

        validated = pipeline.validate_step(cleaned)
        if validated is None:
            return ValidationResponse(
                valid=False, errors=["Data validation failed or quality too low"]
            )

        return ValidationResponse(valid=True, entry=validated.model_dump(), errors=[])

    except Exception as e:
        logger.error(f"Error in validate endpoint: {e}")
        return ValidationResponse(valid=False, errors=[str(e)])


@router.get("/stats", response_model=StatsResponse)
@limiter.limit("60/minute")
async def stats_endpoint(
    config: AppConfig = Depends(get_config), _: None = Depends(verify_api_key)
) -> StatsResponse:
    """
    Get scraper statistics.
    Returns current stats for scraper, pipeline, and storage.
    """
    try:
        # Note: In production, these would be persistent
        return StatsResponse(
            scraper_stats={},
            pipeline_stats={},
            storage_stats=DataStorage(config.data).get_stats(),
        )

    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "chatmed-scraper"}
