"""
Data cleaning and validation pipeline for medical data.
Implements: HTML stripping, normalization, deduplication, PII removal.
Pydantic validation with custom validators for 100% data integrity.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import ftfy
try:
    from blake3 import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False
from loguru import logger
from pydantic import BaseModel, Field, validator
from rapidfuzz import fuzz

from utils.config import AppConfig, DataConfig
from utils.ethics import PIIAnonymizer
from utils.ml_utils import EmbeddingCache, MedicalNER, QAGenerator


class MedicalEntry(BaseModel):
    """
    Validated medical data entry with strict constraints.
    Ensures data quality and integrity.
    """

    # Core fields
    pmid: str = ""
    title: str = Field(min_length=10, max_length=250)
    abstract: str = Field(min_length=200, max_length=3000)
    authors: list[str] = Field(default_factory=list)
    date: str = ""
    keywords: list[str] = Field(default_factory=list)
    doi: str = ""
    source: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://.*")

    # Enriched fields
    entities: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    extracted_keywords: list[str] = Field(default_factory=list)
    qa_pairs: list[dict[str, str]] = Field(default_factory=list)
    category: str = ""

    # Quality metrics
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    completeness_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Metadata
    hash_sha256: str = ""
    hash_blake3: str = ""
    scraped_at: str = ""
    pii_free: bool = True
    compliance_metadata: dict[str, str] = Field(default_factory=dict)
    error_log: str = ""

    @validator("title", "abstract")
    def validate_no_html(cls, v: str) -> str:
        """Ensure no HTML tags in text fields."""
        if re.search(r"<[^>]+>", v):
            raise ValueError("HTML tags not allowed in text fields")
        return v

    @validator("abstract")
    def validate_abstract_quality(cls, v: str) -> str:
        """Validate abstract has sufficient content."""
        # Check for minimum word count
        words = v.split()
        if len(words) < 50:
            raise ValueError("Abstract must contain at least 50 words")

        # Check for non-ASCII ratio (should be mostly readable text)
        ascii_chars = sum(1 for c in v if ord(c) < 128)
        if len(v) > 0 and ascii_chars / len(v) < 0.5:
            raise ValueError("Abstract contains too many non-ASCII characters")

        return v

    @validator("url")
    def validate_url_format(cls, v: str) -> str:
        """Validate URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    class Config:
        """Pydantic v1 configuration."""
        validate_assignment = True


class DataPipeline:
    """
    Chainable data cleaning and enrichment pipeline.
    Processes raw scraped data into validated, enriched entries.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.pii_anonymizer = PIIAnonymizer(config.ethics)
        self.ner = MedicalNER(config.ml)
        self.embedding_cache = EmbeddingCache(config.ml, Path(config.data.cache_dir))
        self.qa_generator = QAGenerator(config.ml)

        # Deduplication tracking
        self.seen_hashes: set[str] = set()
        self.seen_titles: dict[str, str] = {}  # title_lower -> hash

        # Statistics
        self.stats = {
            "processed": 0,
            "cleaned": 0,
            "validated": 0,
            "enriched": 0,
            "duplicates": 0,
            "failures": 0,
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text with comprehensive processing.
        Handles encoding issues, HTML, URLs, and special characters.
        """
        if not text:
            return ""

        try:
            # Fix encoding issues
            text = ftfy.fix_text(text)

            # Remove HTML tags (recursive)
            text = re.sub(r"<[^>]+>", " ", text)

            # Remove URLs
            text = re.sub(r"https?://\S+", "", text)

            # Remove email addresses
            text = re.sub(r"\S+@\S+", "", text)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove control characters
            text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

            # Strip and normalize
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistency.
        Handles case, punctuation, and special characters.
        """
        if not text:
            return ""

        try:
            # Normalize unicode
            text = text.replace("\xa0", " ")  # Non-breaking space
            text = text.replace("\u2019", "'")  # Smart quote
            text = text.replace("\u2018", "'")
            text = text.replace("\u201c", '"')
            text = text.replace("\u201d", '"')

            # Normalize whitespace
            text = " ".join(text.split())

            return text

        except Exception as e:
            logger.error(f"Error normalizing text: {e}")
            return text

    def calculate_hash(self, text: str, algorithm: str = "sha256") -> str:
        """Calculate hash of text for deduplication."""
        try:
            if algorithm == "sha256":
                return hashlib.sha256(text.encode()).hexdigest()
            elif algorithm == "blake3" and BLAKE3_AVAILABLE:
                return blake3(text.encode()).hexdigest()
            else:
                # Fallback to sha256 if blake3 not available
                return hashlib.sha256(text.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            return ""

    def is_duplicate(self, entry: dict[str, Any]) -> bool:
        """
        Check if entry is duplicate using multiple strategies.
        Returns True if duplicate detected.
        """
        try:
            # Strategy 1: Exact hash match
            text_for_hash = f"{entry.get('title', '')}|{entry.get('abstract', '')}"
            entry_hash = self.calculate_hash(text_for_hash)

            if entry_hash in self.seen_hashes:
                logger.debug(f"Duplicate detected (exact hash): {entry.get('title', '')[:50]}")
                return True

            # Strategy 2: Fuzzy title match
            title_lower = entry.get("title", "").lower()
            if title_lower:
                for seen_title, seen_hash in self.seen_titles.items():
                    similarity = fuzz.ratio(title_lower, seen_title)
                    if similarity > 96:  # 96% similarity threshold
                        logger.debug(
                            f"Duplicate detected (fuzzy match {similarity}%): {title_lower[:50]}"
                        )
                        return True

            # Not a duplicate - record it
            self.seen_hashes.add(entry_hash)
            if title_lower:
                self.seen_titles[title_lower] = entry_hash

            return False

        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

    def calculate_quality_scores(self, entry: dict[str, Any]) -> dict[str, float]:
        """
        Calculate quality metrics for entry.
        Returns dict with relevance, quality, and completeness scores.
        """
        scores = {"relevance": 0.0, "quality": 0.0, "completeness": 0.0}

        try:
            # Completeness score (0-1)
            completeness_factors = []
            if entry.get("title"):
                completeness_factors.append(1.0)
            if entry.get("abstract") and len(entry["abstract"]) >= 200:
                completeness_factors.append(1.0)
            if entry.get("authors"):
                completeness_factors.append(1.0)
            if entry.get("date"):
                completeness_factors.append(1.0)
            if entry.get("keywords"):
                completeness_factors.append(1.0)

            scores["completeness"] = (
                sum(completeness_factors) / 5.0 if completeness_factors else 0.0
            )

            # Quality score (based on text quality)
            quality_factors = []

            # Check abstract length (optimal: 300-2000 chars)
            abstract_len = len(entry.get("abstract", ""))
            if 300 <= abstract_len <= 2000:
                quality_factors.append(1.0)
            elif 200 <= abstract_len < 300 or 2000 < abstract_len <= 3000:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)

            # Check for medical keywords
            medical_keywords = [
                "patient",
                "treatment",
                "disease",
                "symptom",
                "diagnosis",
                "therapy",
                "clinical",
                "medical",
                "health",
            ]
            text_lower = (entry.get("title", "") + " " + entry.get("abstract", "")).lower()
            keyword_matches = sum(1 for kw in medical_keywords if kw in text_lower)
            quality_factors.append(min(1.0, keyword_matches / 5.0))

            scores["quality"] = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

            # Relevance score (placeholder - would use embeddings in production)
            scores["relevance"] = (scores["completeness"] + scores["quality"]) / 2.0

            return scores

        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
            return scores

    def clean_step(self, raw_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Step 1: Clean raw scraped data.
        Returns cleaned dict or None if cleaning fails.
        """
        try:
            self.stats["processed"] += 1

            # Clean text fields
            cleaned = {
                "pmid": raw_data.get("pmid", ""),
                "title": self.normalize_text(self.clean_text(raw_data.get("title", ""))),
                "abstract": self.normalize_text(self.clean_text(raw_data.get("abstract", ""))),
                "authors": raw_data.get("authors", []),
                "date": raw_data.get("date", ""),
                "keywords": raw_data.get("keywords", []),
                "doi": raw_data.get("doi", ""),
                "source": raw_data.get("source", "unknown"),
                "url": raw_data.get("url", ""),
            }

            # Anonymize PII
            if self.config.ethics.anonymize_pii:
                cleaned["title"], _ = self.pii_anonymizer.anonymize_text(cleaned["title"])
                cleaned["abstract"], pii_types = self.pii_anonymizer.anonymize_text(
                    cleaned["abstract"]
                )
                cleaned["pii_free"] = len(pii_types) == 0
            else:
                cleaned["pii_free"] = True

            # Check for minimum content
            if len(cleaned["title"]) < 10 or len(cleaned["abstract"]) < 200:
                logger.debug("Entry rejected: insufficient content")
                self.stats["failures"] += 1
                return None

            # Check for duplicates
            if self.is_duplicate(cleaned):
                self.stats["duplicates"] += 1
                return None

            self.stats["cleaned"] += 1
            return cleaned

        except Exception as e:
            logger.error(f"Error in clean_step: {e}")
            self.stats["failures"] += 1
            return None

    def validate_step(self, cleaned_data: dict[str, Any]) -> Optional[MedicalEntry]:
        """
        Step 2: Validate cleaned data with Pydantic.
        Returns validated MedicalEntry or None if validation fails.
        """
        try:
            # Calculate hashes
            text_for_hash = f"{cleaned_data['title']}|{cleaned_data['abstract']}"
            cleaned_data["hash_sha256"] = self.calculate_hash(text_for_hash, "sha256")
            cleaned_data["hash_blake3"] = self.calculate_hash(text_for_hash, "blake3")

            # Add timestamp
            cleaned_data["scraped_at"] = datetime.utcnow().isoformat()

            # Calculate quality scores
            scores = self.calculate_quality_scores(cleaned_data)
            cleaned_data["relevance_score"] = scores["relevance"]
            cleaned_data["quality_score"] = scores["quality"]
            cleaned_data["completeness_score"] = scores["completeness"]

            # Validate with Pydantic
            entry = MedicalEntry(**cleaned_data)

            # Check quality thresholds
            if entry.quality_score < self.config.data.min_quality_score:
                logger.debug(
                    f"Entry rejected: quality score {entry.quality_score} < {self.config.data.min_quality_score}"
                )
                self.stats["failures"] += 1
                return None

            if entry.relevance_score < self.config.data.min_relevance_score:
                logger.debug(
                    f"Entry rejected: relevance score {entry.relevance_score} < {self.config.data.min_relevance_score}"
                )
                self.stats["failures"] += 1
                return None

            self.stats["validated"] += 1
            return entry

        except Exception as e:
            logger.error(f"Error in validate_step: {e}")
            self.stats["failures"] += 1
            return None

    def enrich_step(self, entry: MedicalEntry) -> MedicalEntry:
        """
        Step 3: Enrich entry with ML features.
        Adds NER entities, keywords, embeddings, and Q&A pairs.
        """
        try:
            # Extract entities with NER
            if self.config.ml.extract_entities:
                entities = self.ner.extract_entities(entry.abstract, language="fr")
                entry.entities = entities

            # Extract keywords
            if self.config.ml.generate_keywords:
                keywords = self.ner.extract_keywords_tfidf(entry.abstract)
                entry.extracted_keywords = keywords

            # Generate Q&A pairs
            if self.config.ml.generate_qa_pairs:
                qa_pairs = self.qa_generator.generate_qa_pairs(
                    entry.title, entry.abstract, entry.entities
                )
                entry.qa_pairs = qa_pairs

            # Calculate embedding (for relevance scoring)
            # Note: Embedding stored in cache, not in entry to save space
            if self.embedding_cache.model is not None:
                self.embedding_cache.embed(entry.abstract, use_cache=True)

            self.stats["enriched"] += 1
            return entry

        except Exception as e:
            logger.error(f"Error in enrich_step: {e}")
            entry.error_log = str(e)
            return entry

    def process(self, raw_data: dict[str, Any]) -> Optional[MedicalEntry]:
        """
        Full pipeline: clean -> validate -> enrich.
        Returns enriched MedicalEntry or None if processing fails.
        """
        # Step 1: Clean
        cleaned = self.clean_step(raw_data)
        if cleaned is None:
            return None

        # Step 2: Validate
        validated = self.validate_step(cleaned)
        if validated is None:
            return None

        # Step 3: Enrich
        enriched = self.enrich_step(validated)

        return enriched

    def process_batch(self, raw_data_list: list[dict[str, Any]]) -> list[MedicalEntry]:
        """
        Process multiple entries in batch.
        Returns list of successfully processed entries.
        """
        results = []

        for raw_data in raw_data_list:
            result = self.process(raw_data)
            if result is not None:
                results.append(result)

        logger.info(
            f"Batch processing complete: {len(results)}/{len(raw_data_list)} entries processed"
        )
        return results

    def get_stats(self) -> dict[str, int]:
        """Get pipeline statistics."""
        return self.stats.copy()
