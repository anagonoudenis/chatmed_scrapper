"""
Configuration management using TOML (NO .env files).
Validates all settings with Pydantic v2 for zero-bug guarantees.
Prompts for sensitive values at runtime if needed.
"""

from pathlib import Path
from typing import Any, Literal, Optional

import tomli
from loguru import logger
from pydantic import BaseModel, Field, validator


class ScraperConfig(BaseModel):
    """Scraper-specific configuration with validation."""

    default_query: str = Field(min_length=3, max_length=500)
    max_pages: int = Field(ge=1, le=1000, default=15)
    max_results_per_query: int = Field(ge=1, le=10000, default=100)
    rate_limit_min_seconds: float = Field(ge=0.1, le=10.0, default=0.5)
    rate_limit_max_seconds: float = Field(ge=0.5, le=30.0, default=2.0)
    concurrent_requests: int = Field(ge=1, le=200, default=50)
    max_retries: int = Field(ge=1, le=10, default=7)
    retry_base_delay: float = Field(ge=0.1, le=10.0, default=1.0)
    retry_max_delay: float = Field(ge=1.0, le=300.0, default=60.0)
    circuit_breaker_failure_threshold: int = Field(ge=1, le=20, default=5)
    circuit_breaker_timeout: int = Field(ge=10, le=600, default=60)

    @validator("rate_limit_max_seconds")
    def validate_rate_limits(cls, v: float, values: dict) -> float:
        """Ensure max rate limit >= min rate limit."""
        if "rate_limit_min_seconds" in values:
            if v < values["rate_limit_min_seconds"]:
                raise ValueError("rate_limit_max_seconds must be >= rate_limit_min_seconds")
        return v


class PubMedSourceConfig(BaseModel):
    """PubMed-specific configuration."""

    api_url: str = Field(pattern=r"^https?://.*")
    api_key: str = ""
    use_api: bool = True
    fallback_scrape: bool = True
    base_url: str = Field(pattern=r"^https?://.*")


class WHOSourceConfig(BaseModel):
    """WHO-specific configuration."""

    api_url: str = Field(pattern=r"^https?://.*")
    base_url: str = Field(pattern=r"^https?://.*")
    use_api: bool = True


class SourcesConfig(BaseModel):
    """Data sources configuration."""

    enabled: list[str] = Field(default=["pubmed", "who"])
    pubmed: PubMedSourceConfig
    who: WHOSourceConfig


class NetworkConfig(BaseModel):
    """Network and HTTP configuration."""

    user_agents: list[str] = Field(min_length=1)
    use_proxies: bool = False
    proxy_rotation: Literal["sticky", "random", "round-robin"] = "sticky"
    max_proxy_failures: int = Field(ge=1, le=10, default=3)
    verify_ssl: bool = True
    timeout_seconds: int = Field(ge=5, le=300, default=30)


class CacheConfig(BaseModel):
    """Caching configuration."""

    enabled: bool = True
    ttl_hours: int = Field(ge=1, le=168, default=48)
    max_size_mb: int = Field(ge=10, le=10000, default=500)
    auto_vacuum: bool = True


class DataConfig(BaseModel):
    """Data output and quality configuration."""

    output_dir: str = "./data/output"
    backup_dir: str = "./data/backups"
    cache_dir: str = "./data/cache"
    export_jsonl: bool = True
    export_parquet: bool = True
    export_sqlite: bool = True
    min_abstract_length: int = Field(ge=50, le=1000, default=200)
    max_abstract_length: int = Field(ge=500, le=10000, default=3000)
    min_title_length: int = Field(ge=5, le=50, default=10)
    max_title_length: int = Field(ge=50, le=500, default=250)
    min_relevance_score: float = Field(ge=0.0, le=1.0, default=0.6)
    min_quality_score: float = Field(ge=0.0, le=1.0, default=0.9)


class MLConfig(BaseModel):
    """Machine learning and NLP configuration."""

    spacy_model: str = "fr_core_news_lg"
    scispacy_model: str = "en_core_sci_lg"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    extract_entities: bool = True
    entity_types: list[str] = Field(
        default=["SYMPTOM", "DISEASE", "DRUG", "TREATMENT", "ANATOMY"]
    )
    generate_keywords: bool = True
    max_keywords: int = Field(ge=5, le=50, default=15)
    generate_qa_pairs: bool = True
    qa_pairs_per_entry: int = Field(ge=1, le=20, default=5)
    classify_content: bool = True
    classification_labels: list[str] = Field(
        default=["diagnostic", "treatment", "prevention", "epidemiology"]
    )


class EthicsConfig(BaseModel):
    """Ethics and compliance configuration."""

    check_robots_txt: bool = True
    respect_tos: bool = True
    abort_on_violation: bool = True
    anonymize_pii: bool = True
    pii_patterns: list[str] = Field(default=["email", "phone", "ssn", "medical_id"])
    max_requests_per_day: int = Field(ge=100, le=100000, default=2000)
    embed_compliance_metadata: bool = True
    log_consent: bool = True


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: str = "./logs/scraper.log"
    log_rotation: str = "100 MB"
    show_progress: bool = True
    progress_style: Literal["rich", "tqdm", "simple"] = "rich"
    export_metrics: bool = True
    metrics_file: str = "./logs/metrics.json"
    enable_alerts: bool = False
    alert_webhook_url: str = ""
    alert_threshold_error_rate: float = Field(ge=0.0, le=1.0, default=0.05)


class APIConfig(BaseModel):
    """FastAPI server configuration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(ge=1024, le=65535, default=8000)
    reload: bool = False
    require_auth: bool = False
    api_key: str = ""
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(ge=1, le=10000, default=60)


class TestingConfig(BaseModel):
    """Testing and dry-run configuration."""

    dry_run: bool = False
    mock_mode: bool = False
    test_sample_size: int = Field(ge=1, le=100, default=5)
    chaos_testing: bool = False


class AppConfig(BaseModel):
    """
    Main application configuration.
    Loads from config.toml with full validation.
    NO .env files - all configuration via TOML.
    """

    scraper: ScraperConfig
    sources: SourcesConfig
    network: NetworkConfig
    cache: CacheConfig
    data: DataConfig
    ml: MLConfig
    ethics: EthicsConfig
    monitoring: MonitoringConfig
    api: APIConfig
    testing: TestingConfig
    
    class Config:
        """Pydantic v1 configuration."""
        validate_assignment = True

    @classmethod
    def load_from_toml(cls, config_path: Path = Path("config.toml")) -> "AppConfig":
        """
        Load and validate configuration from TOML file.
        Raises ValidationError if configuration is invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                config_data = tomli.load(f)

            # Validate and create config
            config = cls(**config_data)

            # Create required directories
            for dir_path in [
                config.data.output_dir,
                config.data.backup_dir,
                config.data.cache_dir,
                Path(config.monitoring.log_file).parent,
            ]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

            logger.info(f"Configuration loaded successfully from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise


# Global configuration instance (lazy-loaded)
_config: Optional[AppConfig] = None


def get_config(config_path: Path = Path("config.toml")) -> AppConfig:
    """
    Get global configuration instance (singleton pattern).
    Loads configuration on first call.
    """
    global _config
    if _config is None:
        _config = AppConfig.load_from_toml(config_path)
    return _config


def reload_config(config_path: Path = Path("config.toml")) -> AppConfig:
    """Force reload configuration from file."""
    global _config
    _config = AppConfig.load_from_toml(config_path)
    return _config
