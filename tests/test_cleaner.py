"""Tests for data cleaning and validation pipeline."""

import pytest
from core.cleaner import DataPipeline, MedicalEntry
from pydantic import ValidationError


def test_clean_text(test_config, sample_raw_entry):
    """Test text cleaning."""
    pipeline = DataPipeline(test_config)

    # Test HTML removal
    dirty_text = "<p>This is <b>bold</b> text</p>"
    clean = pipeline.clean_text(dirty_text)
    assert "<" not in clean
    assert ">" not in clean

    # Test URL removal
    text_with_url = "Check https://example.com for more"
    clean = pipeline.clean_text(text_with_url)
    assert "https://" not in clean

    # Test whitespace normalization
    text_with_spaces = "Too    many     spaces"
    clean = pipeline.clean_text(text_with_spaces)
    assert "  " not in clean


def test_normalize_text(test_config):
    """Test text normalization."""
    pipeline = DataPipeline(test_config)

    # Test unicode normalization
    text = "Smart quotes: \u2019 and \u201c"
    normalized = pipeline.normalize_text(text)
    assert "\u2019" not in normalized
    assert "'" in normalized


def test_calculate_hash(test_config):
    """Test hash calculation."""
    pipeline = DataPipeline(test_config)

    text = "Test content for hashing"
    hash_sha256 = pipeline.calculate_hash(text, "sha256")
    hash_blake3 = pipeline.calculate_hash(text, "blake3")

    assert len(hash_sha256) == 64
    assert len(hash_blake3) == 64
    assert hash_sha256 != hash_blake3


def test_is_duplicate(test_config, sample_raw_entry):
    """Test duplicate detection."""
    pipeline = DataPipeline(test_config)

    # First entry should not be duplicate
    assert pipeline.is_duplicate(sample_raw_entry) is False

    # Same entry should be duplicate
    assert pipeline.is_duplicate(sample_raw_entry) is True

    # Similar title should be detected
    similar_entry = sample_raw_entry.copy()
    similar_entry["title"] = sample_raw_entry["title"] + " Extra"
    # Might or might not be detected depending on similarity threshold


def test_calculate_quality_scores(test_config, sample_raw_entry):
    """Test quality score calculation."""
    pipeline = DataPipeline(test_config)

    scores = pipeline.calculate_quality_scores(sample_raw_entry)

    assert "relevance" in scores
    assert "quality" in scores
    assert "completeness" in scores
    assert 0.0 <= scores["quality"] <= 1.0
    assert 0.0 <= scores["completeness"] <= 1.0


def test_clean_step(test_config, sample_raw_entry):
    """Test cleaning step."""
    pipeline = DataPipeline(test_config)

    cleaned = pipeline.clean_step(sample_raw_entry)

    assert cleaned is not None
    assert "title" in cleaned
    assert "abstract" in cleaned
    assert len(cleaned["title"]) >= 10
    assert len(cleaned["abstract"]) >= 200


def test_clean_step_insufficient_content(test_config):
    """Test cleaning step with insufficient content."""
    pipeline = DataPipeline(test_config)

    bad_entry = {
        "title": "Short",
        "abstract": "Too short",
        "url": "https://example.com",
        "source": "test",
    }

    cleaned = pipeline.clean_step(bad_entry)
    assert cleaned is None


def test_validate_step(test_config, sample_raw_entry):
    """Test validation step."""
    pipeline = DataPipeline(test_config)

    # Clean first
    cleaned = pipeline.clean_step(sample_raw_entry)
    assert cleaned is not None

    # Validate
    validated = pipeline.validate_step(cleaned)
    assert validated is not None
    assert isinstance(validated, MedicalEntry)
    assert validated.hash_sha256 != ""
    assert validated.scraped_at != ""


def test_medical_entry_validation():
    """Test MedicalEntry Pydantic validation."""
    # Valid entry
    entry = MedicalEntry(
        title="Valid Medical Title Here",
        abstract="A" * 250,  # 250 chars
        source="test",
        url="https://example.com",
    )
    assert entry.title == "Valid Medical Title Here"

    # Invalid: title too short
    with pytest.raises(ValidationError):
        MedicalEntry(
            title="Short",
            abstract="A" * 250,
            source="test",
            url="https://example.com",
        )

    # Invalid: abstract too short
    with pytest.raises(ValidationError):
        MedicalEntry(
            title="Valid Title Here",
            abstract="Too short",
            source="test",
            url="https://example.com",
        )

    # Invalid: HTML in title
    with pytest.raises(ValidationError):
        MedicalEntry(
            title="<b>HTML Title</b>",
            abstract="A" * 250,
            source="test",
            url="https://example.com",
        )


def test_process_full_pipeline(test_config, sample_raw_entry):
    """Test full processing pipeline."""
    pipeline = DataPipeline(test_config)

    result = pipeline.process(sample_raw_entry)

    assert result is not None
    assert isinstance(result, MedicalEntry)
    assert result.quality_score > 0
    assert result.hash_sha256 != ""


def test_process_batch(test_config, sample_entries_batch):
    """Test batch processing."""
    pipeline = DataPipeline(test_config)

    results = pipeline.process_batch(sample_entries_batch)

    assert len(results) > 0
    assert all(isinstance(r, MedicalEntry) for r in results)


def test_pipeline_stats(test_config, sample_raw_entry):
    """Test pipeline statistics tracking."""
    pipeline = DataPipeline(test_config)

    pipeline.process(sample_raw_entry)
    stats = pipeline.get_stats()

    assert stats["processed"] >= 1
    assert "cleaned" in stats
    assert "validated" in stats
