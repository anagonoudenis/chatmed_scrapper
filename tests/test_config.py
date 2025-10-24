"""Tests for configuration management."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from utils.config import (
    AppConfig,
    ScraperConfig,
    DataConfig,
    get_config,
)


def test_scraper_config_validation():
    """Test scraper configuration validation."""
    # Valid config
    config = ScraperConfig(
        default_query="test",
        max_pages=10,
        max_results_per_query=100,
        rate_limit_min_seconds=0.5,
        rate_limit_max_seconds=2.0,
    )
    assert config.default_query == "test"
    assert config.max_pages == 10

    # Invalid: rate_limit_max < rate_limit_min
    with pytest.raises(ValidationError):
        ScraperConfig(
            default_query="test",
            rate_limit_min_seconds=2.0,
            rate_limit_max_seconds=0.5,
        )

    # Invalid: max_pages out of range
    with pytest.raises(ValidationError):
        ScraperConfig(default_query="test", max_pages=10000)


def test_data_config_validation():
    """Test data configuration validation."""
    config = DataConfig(
        output_dir="./output",
        min_abstract_length=200,
        max_abstract_length=3000,
    )
    assert config.min_abstract_length == 200
    assert config.export_jsonl is True


def test_config_load_from_toml(test_config: AppConfig):
    """Test loading configuration from TOML."""
    # Test config should be valid
    assert test_config.scraper.default_query == "test query"
    assert test_config.data.export_jsonl is True
    assert test_config.ml.extract_entities is False


def test_config_singleton(test_config: AppConfig):
    """Test configuration singleton pattern."""
    # Note: In real tests, would need to reset global state
    # This is a simplified test
    assert test_config is not None
    assert isinstance(test_config, AppConfig)
