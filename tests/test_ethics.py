"""Tests for ethics and compliance utilities."""

import pytest
from utils.ethics import PIIAnonymizer, TOSChecker, RateLimiter
from utils.config import EthicsConfig


@pytest.fixture
def ethics_config():
    """Create ethics configuration for testing."""
    return EthicsConfig(
        check_robots_txt=False,
        respect_tos=True,
        abort_on_violation=False,
        anonymize_pii=True,
        pii_patterns=["email", "phone", "ssn"],
        max_requests_per_day=100,
        embed_compliance_metadata=True,
        log_consent=True,
    )


def test_pii_anonymizer_email(ethics_config: EthicsConfig):
    """Test email anonymization."""
    anonymizer = PIIAnonymizer(ethics_config)

    text = "Contact me at john.doe@example.com for more info"
    anonymized, detected = anonymizer.anonymize_text(text)

    assert "[EMAIL_REDACTED]" in anonymized
    assert "email" in detected
    assert "john.doe@example.com" not in anonymized


def test_pii_anonymizer_phone(ethics_config: EthicsConfig):
    """Test phone number anonymization."""
    anonymizer = PIIAnonymizer(ethics_config)

    text = "Call me at 555-123-4567 or (555) 987-6543"
    anonymized, detected = anonymizer.anonymize_text(text)

    assert "[PHONE_REDACTED]" in anonymized
    assert "phone" in detected


def test_pii_anonymizer_ssn(ethics_config: EthicsConfig):
    """Test SSN anonymization."""
    anonymizer = PIIAnonymizer(ethics_config)

    text = "SSN: 123-45-6789"
    anonymized, detected = anonymizer.anonymize_text(text)

    assert "[SSN_REDACTED]" in anonymized
    assert "ssn" in detected
    assert "123-45-6789" not in anonymized


def test_pii_scan_for_pii(ethics_config: EthicsConfig):
    """Test PII scanning without anonymization."""
    anonymizer = PIIAnonymizer(ethics_config)

    text = "Email: test@example.com, Phone: 555-1234"
    pii_counts = anonymizer.scan_for_pii(text)

    assert pii_counts.get("email", 0) >= 1
    # Phone might not match depending on pattern


def test_pii_validate_pii_free(ethics_config: EthicsConfig):
    """Test PII-free validation."""
    anonymizer = PIIAnonymizer(ethics_config)

    clean_text = "This is a clean medical abstract about diabetes treatment"
    assert anonymizer.validate_pii_free(clean_text) is True

    dirty_text = "Contact: doctor@hospital.com"
    assert anonymizer.validate_pii_free(dirty_text) is False


def test_tos_checker_compliance(ethics_config: EthicsConfig):
    """Test TOS compliance checking."""
    checker = TOSChecker(ethics_config)

    # Allowed domain
    assert checker.check_tos_compliance("pubmed.ncbi.nlm.nih.gov") is True

    # Unknown domain
    assert checker.check_tos_compliance("unknown-site.com") is False


def test_tos_checker_log_consent(ethics_config: EthicsConfig):
    """Test consent logging."""
    checker = TOSChecker(ethics_config)

    url = "https://pubmed.ncbi.nlm.nih.gov/12345/"
    metadata = checker.log_consent(url)

    assert metadata["consent_logged"] == "yes"
    assert metadata["source_url"] == url
    assert "timestamp" in metadata


def test_rate_limiter_check_limit(ethics_config: EthicsConfig):
    """Test rate limiter."""
    limiter = RateLimiter(ethics_config)

    # Should allow requests under limit
    for _ in range(50):
        assert limiter.check_limit() is True
        limiter.increment()

    # Check remaining
    assert limiter.get_remaining() == 50


def test_rate_limiter_exceeds_limit(ethics_config: EthicsConfig):
    """Test rate limiter exceeding daily limit."""
    limiter = RateLimiter(ethics_config)

    # Exhaust limit
    for _ in range(100):
        limiter.increment()

    # Should deny further requests
    assert limiter.check_limit() is False
