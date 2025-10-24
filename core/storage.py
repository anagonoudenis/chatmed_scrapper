"""
Local storage and export for medical data.
Supports JSONL, Parquet, and SQLite with atomic writes.
All data stored locally - no cloud dependencies.
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

import polars as pl
from loguru import logger
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from core.cleaner import MedicalEntry
from utils.config import DataConfig

Base = declarative_base()


class MedicalEntryDB(Base):
    """SQLAlchemy model for medical entries."""

    __tablename__ = "medical_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pmid = Column(String(50), index=True)
    title = Column(String(250), nullable=False)
    abstract = Column(Text, nullable=False)
    authors = Column(Text)  # JSON array
    date = Column(String(50))
    keywords = Column(Text)  # JSON array
    doi = Column(String(100))
    source = Column(String(50), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    entities = Column(Text)  # JSON object
    extracted_keywords = Column(Text)  # JSON array
    qa_pairs = Column(Text)  # JSON array
    category = Column(String(100))
    relevance_score = Column(Float)
    quality_score = Column(Float)
    completeness_score = Column(Float)
    hash_sha256 = Column(String(64), unique=True, index=True)
    hash_blake3 = Column(String(64))
    scraped_at = Column(DateTime, nullable=False, index=True)
    pii_free = Column(Integer)  # Boolean as integer
    compliance_metadata = Column(Text)  # JSON object
    error_log = Column(Text)


class DataStorage:
    """
    Local data storage manager.
    Handles JSONL, Parquet, and SQLite exports with atomic writes.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.backup_dir = Path(config.backup_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite if enabled
        self.engine: Optional[Any] = None
        self.session_factory: Optional[Any] = None
        if config.export_sqlite:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        try:
            db_path = self.output_dir / "medical_data.db"
            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
            Base.metadata.create_all(self.engine)
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info(f"Initialized SQLite database: {db_path}")
        except Exception as e:
            logger.error(f"Error initializing SQLite: {e}")
            self.engine = None

    def export_jsonl(
        self, entries: list[MedicalEntry], filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Export entries to JSONL format (streaming write).
        Returns path to exported file or None on error.
        """
        if not self.config.export_jsonl:
            return None

        try:
            if filename is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"medical_data_{timestamp}.jsonl"

            output_path = self.output_dir / filename

            # Write with atomic operation (write to temp, then rename)
            temp_path = output_path.with_suffix(".tmp")

            with open(temp_path, "wb") as f:
                for entry in entries:
                    # Convert to dict and serialize
                    entry_dict = entry.model_dump()
                    if ORJSON_AVAILABLE:
                        json_bytes = orjson.dumps(entry_dict)
                    else:
                        json_bytes = json.dumps(entry_dict).encode('utf-8')
                    f.write(json_bytes)
                    f.write(b"\n")

            # Atomic rename
            temp_path.replace(output_path)

            logger.info(f"Exported {len(entries)} entries to JSONL: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting JSONL: {e}")
            return None

    def export_parquet(
        self, entries: list[MedicalEntry], filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Export entries to Parquet format using Polars.
        Returns path to exported file or None on error.
        """
        if not self.config.export_parquet:
            return None

        try:
            if filename is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"medical_data_{timestamp}.parquet"

            output_path = self.output_dir / filename

            # Convert entries to dict list
            data = [entry.model_dump() for entry in entries]

            # Flatten nested structures for Parquet
            flattened_data = []
            for item in data:
                flat_item = item.copy()
                # Convert lists and dicts to JSON strings
                flat_item["authors"] = json.dumps(flat_item["authors"])
                flat_item["keywords"] = json.dumps(flat_item["keywords"])
                flat_item["entities"] = json.dumps(flat_item["entities"])
                flat_item["extracted_keywords"] = json.dumps(flat_item["extracted_keywords"])
                flat_item["qa_pairs"] = json.dumps(flat_item["qa_pairs"])
                flat_item["compliance_metadata"] = json.dumps(flat_item["compliance_metadata"])
                flattened_data.append(flat_item)

            # Create Polars DataFrame
            df = pl.DataFrame(flattened_data)

            # Write to Parquet (atomic operation)
            temp_path = output_path.with_suffix(".tmp")
            df.write_parquet(temp_path, compression="zstd")
            temp_path.replace(output_path)

            logger.info(f"Exported {len(entries)} entries to Parquet: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting Parquet: {e}")
            return None

    def export_sqlite(self, entries: list[MedicalEntry]) -> bool:
        """
        Export entries to SQLite database with upsert logic.
        Returns True on success, False on error.
        """
        if not self.config.export_sqlite or self.engine is None:
            return False

        try:
            session: Session = self.session_factory()

            try:
                for entry in entries:
                    # Check if entry exists (by hash)
                    existing = (
                        session.query(MedicalEntryDB)
                        .filter_by(hash_sha256=entry.hash_sha256)
                        .first()
                    )

                    if existing:
                        # Update existing entry
                        existing.title = entry.title
                        existing.abstract = entry.abstract
                        existing.authors = json.dumps(entry.authors)
                        existing.date = entry.date
                        existing.keywords = json.dumps(entry.keywords)
                        existing.doi = entry.doi
                        existing.source = entry.source
                        existing.url = entry.url
                        existing.entities = json.dumps(entry.entities)
                        existing.extracted_keywords = json.dumps(entry.extracted_keywords)
                        existing.qa_pairs = json.dumps(entry.qa_pairs)
                        existing.category = entry.category
                        existing.relevance_score = entry.relevance_score
                        existing.quality_score = entry.quality_score
                        existing.completeness_score = entry.completeness_score
                        existing.hash_blake3 = entry.hash_blake3
                        existing.scraped_at = datetime.fromisoformat(entry.scraped_at)
                        existing.pii_free = 1 if entry.pii_free else 0
                        existing.compliance_metadata = json.dumps(entry.compliance_metadata)
                        existing.error_log = entry.error_log
                    else:
                        # Insert new entry
                        db_entry = MedicalEntryDB(
                            pmid=entry.pmid,
                            title=entry.title,
                            abstract=entry.abstract,
                            authors=json.dumps(entry.authors),
                            date=entry.date,
                            keywords=json.dumps(entry.keywords),
                            doi=entry.doi,
                            source=entry.source,
                            url=entry.url,
                            entities=json.dumps(entry.entities),
                            extracted_keywords=json.dumps(entry.extracted_keywords),
                            qa_pairs=json.dumps(entry.qa_pairs),
                            category=entry.category,
                            relevance_score=entry.relevance_score,
                            quality_score=entry.quality_score,
                            completeness_score=entry.completeness_score,
                            hash_sha256=entry.hash_sha256,
                            hash_blake3=entry.hash_blake3,
                            scraped_at=datetime.fromisoformat(entry.scraped_at),
                            pii_free=1 if entry.pii_free else 0,
                            compliance_metadata=json.dumps(entry.compliance_metadata),
                            error_log=entry.error_log,
                        )
                        session.add(db_entry)

                # Commit transaction
                session.commit()
                logger.info(f"Exported {len(entries)} entries to SQLite")
                return True

            except Exception as e:
                session.rollback()
                logger.error(f"Error in SQLite transaction: {e}")
                return False

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error exporting to SQLite: {e}")
            return False

    def export_all(
        self, entries: list[MedicalEntry], base_filename: Optional[str] = None
    ) -> dict[str, Optional[Path]]:
        """
        Export entries to all enabled formats.
        Returns dict mapping format to output path.
        """
        results = {}

        if self.config.export_jsonl:
            jsonl_path = self.export_jsonl(entries, base_filename)
            results["jsonl"] = jsonl_path

        if self.config.export_parquet:
            parquet_path = self.export_parquet(entries, base_filename)
            results["parquet"] = parquet_path

        if self.config.export_sqlite:
            sqlite_success = self.export_sqlite(entries)
            results["sqlite"] = Path("medical_data.db") if sqlite_success else None

        return results

    def create_backup(self, backup_name: Optional[str] = None) -> Optional[Path]:
        """
        Create compressed backup of all output files.
        Returns path to backup file or None on error.
        """
        try:
            if backup_name is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}.zip"

            backup_path = self.backup_dir / backup_name

            # Create zip archive
            with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from output directory
                for file_path in self.output_dir.glob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)

            logger.info(f"Created backup: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        """
        Load entries from JSONL file.
        Returns list of entry dicts.
        """
        try:
            file_path = self.output_dir / filename
            entries = []

            with open(file_path, "rb") as f:
                for line in f:
                    if line.strip():
                        if ORJSON_AVAILABLE:
                            entry = orjson.loads(line)
                        else:
                            entry = json.loads(line.decode('utf-8'))
                        entries.append(entry)

            logger.info(f"Loaded {len(entries)} entries from {filename}")
            return entries

        except Exception as e:
            logger.error(f"Error loading JSONL: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "output_files": len(list(self.output_dir.glob("*"))),
            "backup_files": len(list(self.backup_dir.glob("*"))),
            "total_size_mb": 0.0,
        }

        try:
            # Calculate total size
            total_size = sum(
                f.stat().st_size
                for f in self.output_dir.glob("*")
                if f.is_file()
            )
            stats["total_size_mb"] = total_size / (1024 * 1024)

        except Exception as e:
            logger.error(f"Error calculating stats: {e}")

        return stats
