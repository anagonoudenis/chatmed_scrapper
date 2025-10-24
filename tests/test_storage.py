"""Tests for data storage and export."""

import pytest
from pathlib import Path
from core.storage import DataStorage
from core.cleaner import MedicalEntry


@pytest.fixture
def sample_medical_entry():
    """Create sample medical entry for testing."""
    return MedicalEntry(
        pmid="12345678",
        title="Test Medical Article About Diabetes Treatment",
        abstract="A" * 250,  # 250 chars
        authors=["John Doe", "Jane Smith"],
        date="2024-01",
        keywords=["diabetes", "treatment"],
        doi="10.1234/test.2024",
        source="test",
        url="https://example.com/article",
        hash_sha256="a" * 64,
        hash_blake3="b" * 64,
        scraped_at="2024-01-01T00:00:00",
        relevance_score=0.95,
        quality_score=0.92,
        completeness_score=0.98,
    )


def test_storage_init(test_config, cleanup_test_dirs):
    """Test storage initialization."""
    storage = DataStorage(test_config.data)

    assert storage.output_dir.exists()
    assert storage.backup_dir.exists()


def test_export_jsonl(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test JSONL export."""
    storage = DataStorage(test_config.data)

    entries = [sample_medical_entry]
    output_path = storage.export_jsonl(entries, "test_export.jsonl")

    assert output_path is not None
    assert output_path.exists()
    assert output_path.suffix == ".jsonl"

    # Verify content
    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1


def test_export_parquet(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test Parquet export."""
    storage = DataStorage(test_config.data)

    entries = [sample_medical_entry]
    output_path = storage.export_parquet(entries, "test_export.parquet")

    assert output_path is not None
    assert output_path.exists()
    assert output_path.suffix == ".parquet"


def test_export_sqlite(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test SQLite export."""
    storage = DataStorage(test_config.data)

    entries = [sample_medical_entry]
    success = storage.export_sqlite(entries)

    assert success is True

    # Verify database exists
    db_path = storage.output_dir / "medical_data.db"
    assert db_path.exists()


def test_export_all(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test exporting to all formats."""
    storage = DataStorage(test_config.data)

    entries = [sample_medical_entry]
    results = storage.export_all(entries)

    assert "jsonl" in results
    assert "parquet" in results
    assert "sqlite" in results


def test_create_backup(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test backup creation."""
    storage = DataStorage(test_config.data)

    # Create some data first
    entries = [sample_medical_entry]
    storage.export_jsonl(entries)

    # Create backup
    backup_path = storage.create_backup("test_backup.zip")

    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path.suffix == ".zip"


def test_load_jsonl(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test loading JSONL data."""
    storage = DataStorage(test_config.data)

    # Export first
    entries = [sample_medical_entry]
    output_path = storage.export_jsonl(entries, "test_load.jsonl")

    # Load back
    loaded = storage.load_jsonl("test_load.jsonl")

    assert len(loaded) == 1
    assert loaded[0]["pmid"] == "12345678"


def test_storage_stats(test_config, sample_medical_entry, cleanup_test_dirs):
    """Test storage statistics."""
    storage = DataStorage(test_config.data)

    # Create some data
    entries = [sample_medical_entry]
    storage.export_jsonl(entries)

    stats = storage.get_stats()

    assert "output_files" in stats
    assert "total_size_mb" in stats
    assert stats["output_files"] > 0
