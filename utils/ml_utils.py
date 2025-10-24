"""
ML utilities for medical data enrichment.
Includes NER, embeddings, keyword extraction, and Q&A generation.
Optimized for medical domain with caching and batch processing.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from utils.config import MLConfig

# Type hints for optional heavy imports
try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Language = Any  # type: ignore


class MedicalNER:
    """
    Medical Named Entity Recognition using spaCy and scispaCy.
    Extracts symptoms, diseases, drugs, treatments, and anatomy.
    """

    def __init__(self, config: MLConfig):
        self.config = config
        self.nlp_fr: Optional[Language] = None
        self.nlp_sci: Optional[Language] = None
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize spaCy models with error handling."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, NER will be disabled")
            return

        try:
            # Load French medical model
            import spacy

            try:
                self.nlp_fr = spacy.load(self.config.spacy_model)
                logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
            except OSError:
                logger.warning(
                    f"Model {self.config.spacy_model} not found, downloading may be required"
                )
                # Fallback to smaller model
                try:
                    self.nlp_fr = spacy.load("fr_core_news_sm")
                    logger.info("Loaded fallback model: fr_core_news_sm")
                except OSError:
                    logger.error("No French spaCy model available")

            # Load scientific English model
            try:
                self.nlp_sci = spacy.load(self.config.scispacy_model)
                logger.info(f"Loaded scispaCy model: {self.config.scispacy_model}")
            except OSError:
                logger.warning(
                    f"Model {self.config.scispacy_model} not found, using standard English"
                )
                try:
                    self.nlp_sci = spacy.load("en_core_web_sm")
                except OSError:
                    logger.error("No English spaCy model available")

        except Exception as e:
            logger.error(f"Error initializing NER models: {e}")

    def extract_entities(
        self, text: str, language: str = "fr"
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Extract medical entities from text.
        Returns dict mapping entity types to lists of entities with metadata.
        """
        if not self.config.extract_entities:
            return {}

        nlp = self.nlp_fr if language == "fr" else self.nlp_sci
        if nlp is None:
            logger.warning("NER model not available")
            return {}

        try:
            doc = nlp(text[:1000000])  # Limit text length for performance
            entities: dict[str, list[dict[str, Any]]] = {}

            for ent in doc.ents:
                # Map spaCy entity types to medical categories
                entity_type = self._map_entity_type(ent.label_)

                if entity_type in self.config.entity_types:
                    if entity_type not in entities:
                        entities[entity_type] = []

                    entities[entity_type].append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                        }
                    )

            # Deduplicate entities
            for entity_type in entities:
                seen = set()
                unique_entities = []
                for ent in entities[entity_type]:
                    if ent["text"].lower() not in seen:
                        seen.add(ent["text"].lower())
                        unique_entities.append(ent)
                entities[entity_type] = unique_entities

            logger.debug(f"Extracted {sum(len(v) for v in entities.values())} entities")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}

    def _map_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to medical categories."""
        # Mapping for scispaCy and general medical entities
        mapping = {
            # scispaCy labels
            "DISEASE": "DISEASE",
            "CHEMICAL": "DRUG",
            "SYMPTOM": "SYMPTOM",
            "TREATMENT": "TREATMENT",
            "ANATOMY": "ANATOMY",
            # General spaCy labels that might be relevant
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "DATE": "DATE",
        }
        return mapping.get(spacy_label, "OTHER")

    def extract_keywords_tfidf(self, text: str, max_keywords: Optional[int] = None) -> list[str]:
        """
        Extract keywords using simple TF-IDF-like approach.
        Lightweight alternative to full ML pipeline.
        """
        if not self.config.generate_keywords:
            return []

        max_kw = max_keywords or self.config.max_keywords

        try:
            # Tokenize and clean
            words = re.findall(r"\b[a-zàâäéèêëïîôùûüÿæœç]{3,}\b", text.lower())

            # Remove stopwords (French + English medical)
            stopwords = {
                "le",
                "la",
                "les",
                "un",
                "une",
                "des",
                "de",
                "du",
                "et",
                "ou",
                "mais",
                "pour",
                "dans",
                "sur",
                "avec",
                "par",
                "est",
                "sont",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "been",
                "be",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "this",
                "that",
                "these",
                "those",
            }

            filtered_words = [w for w in words if w not in stopwords]

            # Count frequencies
            word_counts = Counter(filtered_words)

            # Get top keywords
            keywords = [word for word, _ in word_counts.most_common(max_kw)]

            logger.debug(f"Extracted {len(keywords)} keywords")
            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []


class EmbeddingCache:
    """
    Embedding generation and caching for semantic similarity.
    Uses sentence-transformers with local caching.
    """

    def __init__(self, config: MLConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[SentenceTransformer] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize embedding model with error handling."""
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None

    def _get_cache_path(self, text: str) -> Path:
        """Generate cache file path for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.npy"

    def embed(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Generate embedding for text with caching.
        Returns numpy array or None on error.
        """
        if self.model is None:
            logger.warning("Embedding model not available")
            return None

        cache_path = self._get_cache_path(text)

        # Check cache
        if use_cache and cache_path.exists():
            try:
                embedding = np.load(cache_path)
                logger.debug("Loaded embedding from cache")
                return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")

        # Generate embedding
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Cache result
            if use_cache:
                try:
                    np.save(cache_path, embedding)
                except Exception as e:
                    logger.warning(f"Error caching embedding: {e}")

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


class QAGenerator:
    """
    Q&A pair generation for LLM fine-tuning.
    Creates structured question-answer pairs from medical abstracts.
    """

    def __init__(self, config: MLConfig):
        self.config = config

    def generate_qa_pairs(
        self,
        title: str,
        abstract: str,
        entities: dict[str, list[dict[str, Any]]],
        max_pairs: Optional[int] = None,
    ) -> list[dict[str, str]]:
        """
        Generate Q&A pairs from medical content.
        Uses template-based approach for reliability.
        """
        if not self.config.generate_qa_pairs:
            return []

        max_p = max_pairs or self.config.qa_pairs_per_entry
        qa_pairs: list[dict[str, str]] = []

        try:
            # Template 1: What is [disease/condition]?
            if "DISEASE" in entities and entities["DISEASE"]:
                disease = entities["DISEASE"][0]["text"]
                qa_pairs.append(
                    {
                        "question": f"Qu'est-ce que {disease} ?",
                        "answer": f"{title}. {abstract[:500]}",
                        "type": "definition",
                    }
                )

            # Template 2: What are the symptoms of [disease]?
            if "SYMPTOM" in entities and entities["SYMPTOM"]:
                symptoms = [e["text"] for e in entities["SYMPTOM"][:3]]
                qa_pairs.append(
                    {
                        "question": f"Quels sont les symptômes mentionnés ?",
                        "answer": f"Les symptômes incluent : {', '.join(symptoms)}. {abstract[:300]}",
                        "type": "symptoms",
                    }
                )

            # Template 3: What treatments are available?
            if "TREATMENT" in entities and entities["TREATMENT"]:
                treatments = [e["text"] for e in entities["TREATMENT"][:3]]
                qa_pairs.append(
                    {
                        "question": "Quels traitements sont disponibles ?",
                        "answer": f"Les traitements mentionnés incluent : {', '.join(treatments)}. {abstract[:300]}",
                        "type": "treatment",
                    }
                )

            # Template 4: What drugs are mentioned?
            if "DRUG" in entities and entities["DRUG"]:
                drugs = [e["text"] for e in entities["DRUG"][:3]]
                qa_pairs.append(
                    {
                        "question": "Quels médicaments sont mentionnés ?",
                        "answer": f"Les médicaments incluent : {', '.join(drugs)}. {abstract[:300]}",
                        "type": "drugs",
                    }
                )

            # Template 5: General summary
            qa_pairs.append(
                {
                    "question": f"Résumez les informations sur : {title[:100]}",
                    "answer": abstract[:800],
                    "type": "summary",
                }
            )

            # Limit to max pairs
            qa_pairs = qa_pairs[:max_p]

            logger.debug(f"Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs

        except Exception as e:
            logger.error(f"Error generating Q&A pairs: {e}")
            return []
