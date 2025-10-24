"""
Pytest configuration and fixtures for comprehensive testing.
Provides mocks, sample data, and test utilities.
"""

import asyncio
from pathlib import Path
from typing import Any, Generator

import pytest
from pydantic import BaseModel

from utils.config import (
    APIConfig,
    AppConfig,
    CacheConfig,
    DataConfig,
    EthicsConfig,
    MLConfig,
    MonitoringConfig,
    NetworkConfig,
    ScraperConfig,
    SourcesConfig,
    TestingConfig,
    PubMedSourceConfig,
    WHOSourceConfig,
)


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> AppConfig:
    """Create test configuration."""
    return AppConfig(
        scraper=ScraperConfig(
            default_query="test query",
            max_pages=5,
            max_results_per_query=50,
            rate_limit_min_seconds=0.1,
            rate_limit_max_seconds=0.2,
            concurrent_requests=10,
            max_retries=3,
            retry_base_delay=0.1,
            retry_max_delay=1.0,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_timeout=10,
        ),
        sources=SourcesConfig(
            enabled=["pubmed"],
            pubmed=PubMedSourceConfig(
                api_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                api_key="",
                use_api=True,
                fallback_scrape=True,
                base_url="https://pubmed.ncbi.nlm.nih.gov/",
            ),
            who=WHOSourceConfig(
                api_url="https://ghoapi.azureedge.net/api/",
                base_url="https://www.who.int/",
                use_api=True,
            ),
        ),
        network=NetworkConfig(
            user_agents=[
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ],
            use_proxies=False,
            proxy_rotation="sticky",
            max_proxy_failures=3,
            verify_ssl=True,
            timeout_seconds=10,
        ),
        cache=CacheConfig(
            enabled=True,
            ttl_hours=24,
            max_size_mb=100,
            auto_vacuum=True,
        ),
        data=DataConfig(
            output_dir="./tests/test_output",
            backup_dir="./tests/test_backups",
            cache_dir="./tests/test_cache",
            export_jsonl=True,
            export_parquet=True,
            export_sqlite=True,
            min_abstract_length=200,
            max_abstract_length=3000,
            min_title_length=10,
            max_title_length=250,
            min_relevance_score=0.6,
            min_quality_score=0.9,
        ),
        ml=MLConfig(
            spacy_model="fr_core_news_sm",
            scispacy_model="en_core_sci_sm",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            extract_entities=False,  # Disable for faster tests
            entity_types=["SYMPTOM", "DISEASE", "DRUG"],
            generate_keywords=True,
            max_keywords=10,
            generate_qa_pairs=True,
            qa_pairs_per_entry=3,
            classify_content=False,
            classification_labels=["diagnostic", "treatment"],
        ),
        ethics=EthicsConfig(
            check_robots_txt=False,  # Disable for tests
            respect_tos=True,
            abort_on_violation=False,
            anonymize_pii=True,
            pii_patterns=["email", "phone"],
            max_requests_per_day=1000,
            embed_compliance_metadata=True,
            log_consent=True,
        ),
        monitoring=MonitoringConfig(
            log_level="DEBUG",
            log_format="text",
            log_file="./tests/test.log",
            log_rotation="10 MB",
            show_progress=False,
            progress_style="simple",
            export_metrics=False,
            metrics_file="./tests/metrics.json",
            enable_alerts=False,
            alert_webhook_url="",
            alert_threshold_error_rate=0.1,
        ),
        api=APIConfig(
            enabled=True,
            host="127.0.0.1",
            port=8001,
            reload=False,
            require_auth=False,
            api_key="",
            rate_limit_enabled=False,
            rate_limit_per_minute=100,
        ),
        testing=TestingConfig(
            dry_run=False,
            mock_mode=True,
            test_sample_size=3,
            chaos_testing=False,
        ),
    )


@pytest.fixture
def sample_raw_entry() -> dict[str, Any]:
    """Sample raw scraped entry."""
    return {
        "pmid": "12345678",
        "title": "Diabetes Management in African Populations: A Comprehensive Review",
        "abstract": (
            "This comprehensive review examines diabetes management strategies "
            "in African populations. The study analyzes treatment protocols, "
            "patient outcomes, and healthcare delivery systems across multiple "
            "countries. Results indicate significant variations in access to care "
            "and treatment efficacy. The research highlights the need for "
            "culturally adapted interventions and improved healthcare infrastructure. "
            "Key findings include the importance of community-based programs and "
            "the role of traditional medicine in diabetes management."
        ),
        "authors": ["John Smith", "Jane Doe", "Ahmed Hassan"],
        "date": "2024-01",
        "keywords": ["diabetes", "Africa", "treatment", "healthcare"],
        "doi": "10.1234/example.2024.001",
        "source": "pubmed",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    }


@pytest.fixture
def sample_pubmed_xml() -> str:
    """Sample PubMed XML response."""
    return """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Diabetes Management in African Populations</ArticleTitle>
        <Abstract>
          <AbstractText>This study examines diabetes management strategies in African populations. Results show significant variations in treatment access and efficacy across different regions.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
      </Article>
      <PubDate>
        <Year>2024</Year>
        <Month>Jan</Month>
      </PubDate>
      <KeywordList>
        <Keyword>diabetes</Keyword>
        <Keyword>Africa</Keyword>
        <Keyword>treatment</Keyword>
      </KeywordList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def sample_html_content() -> str:
    """Sample HTML content for parsing tests."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Article Title</title>
    </head>
    <body>
        <article>
            <h1>Understanding Diabetes Symptoms</h1>
            <p>Diabetes is a chronic condition affecting millions worldwide.</p>
            <p>Common symptoms include increased thirst, frequent urination, and fatigue.</p>
            <p>Early detection and proper management are crucial for patient outcomes.</p>
        </article>
    </body>
    </html>
    """


@pytest.fixture
def sample_entries_batch(sample_raw_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Batch of sample entries for testing."""
    entries = []
    for i in range(5):
        entry = sample_raw_entry.copy()
        entry["pmid"] = f"1234567{i}"
        entry["title"] = f"{entry['title']} - Part {i+1}"
        entries.append(entry)
    return entries


@pytest.fixture
def mock_aiohttp_response():
    """Mock aiohttp response."""

    class MockResponse:
        def __init__(self, text: str, status: int = 200):
            self._text = text
            self.status = status
            self.headers = {"Content-Type": "text/html"}

        async def text(self) -> str:
            return self._text

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockResponse


@pytest.fixture
def cleanup_test_dirs(test_config: AppConfig):
    """Cleanup test directories after tests."""
    yield

    # Cleanup after tests
    import shutil

    for dir_path in [
        test_config.data.output_dir,
        test_config.data.backup_dir,
        test_config.data.cache_dir,
    ]:
        path = Path(dir_path)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
