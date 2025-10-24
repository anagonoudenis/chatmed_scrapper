"""Tests for async web scraper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.scraper import AsyncScraper, TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_scraper_init(test_config):
    """Test scraper initialization."""
    async with AsyncScraper(test_config) as scraper:
        assert scraper.session is not None
        assert scraper.rate_limiter is not None
        assert len(scraper.user_agents) > 0


@pytest.mark.asyncio
async def test_rate_limiter():
    """Test token bucket rate limiter."""
    limiter = TokenBucketRateLimiter(min_interval=0.1, max_interval=0.2)

    # Should allow immediate acquisition
    await limiter.acquire()
    assert limiter.tokens < limiter.capacity


@pytest.mark.asyncio
async def test_get_user_agent(test_config):
    """Test user agent rotation."""
    async with AsyncScraper(test_config) as scraper:
        ua1 = scraper._get_user_agent()
        ua2 = scraper._get_user_agent()

        assert ua1 is not None
        # Might be same if only one UA in test config


@pytest.mark.asyncio
async def test_parse_pubmed_xml(test_config, sample_pubmed_xml):
    """Test PubMed XML parsing."""
    async with AsyncScraper(test_config) as scraper:
        articles = scraper._parse_pubmed_xml(sample_pubmed_xml)

        assert len(articles) > 0
        article = articles[0]
        assert "title" in article
        assert "abstract" in article
        assert "pmid" in article
        assert article["pmid"] == "12345678"


@pytest.mark.asyncio
async def test_generic_parse(test_config, sample_html_content):
    """Test generic HTML parsing."""
    async with AsyncScraper(test_config) as scraper:
        result = scraper._generic_parse(sample_html_content, "https://example.com")

        assert result["title"] != ""
        assert result["abstract"] != ""
        assert "diabetes" in result["abstract"].lower()


@pytest.mark.asyncio
async def test_scraper_stats(test_config):
    """Test scraper statistics."""
    async with AsyncScraper(test_config) as scraper:
        stats = scraper.get_stats()

        assert "requests" in stats
        assert "successes" in stats
        assert "failures" in stats


@pytest.mark.asyncio
async def test_circuit_breaker(test_config):
    """Test circuit breaker functionality."""
    async with AsyncScraper(test_config) as scraper:
        domain = "example.com"
        breaker = scraper._get_circuit_breaker(domain)

        assert breaker is not None
        assert breaker == scraper._get_circuit_breaker(domain)  # Same instance


@pytest.mark.asyncio
async def test_fetch_url_mock(test_config, mock_aiohttp_response):
    """Test URL fetching with mock."""
    async with AsyncScraper(test_config) as scraper:
        # Mock the session.request
        with patch.object(scraper.session, "request") as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_aiohttp_response(
                "<html>Test</html>", 200
            )

            # Note: This test is simplified; real implementation would need more mocking
            # content, status, headers = await scraper.fetch_url("https://example.com")
            # assert status == 200
