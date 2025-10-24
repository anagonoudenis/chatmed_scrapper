"""Tests for ML utilities."""

import pytest
from pathlib import Path
from utils.ml_utils import MedicalNER, EmbeddingCache, QAGenerator


def test_medical_ner_init(test_config):
    """Test NER initialization."""
    ner = MedicalNER(test_config.ml)
    assert ner.config is not None


def test_extract_keywords(test_config):
    """Test keyword extraction."""
    ner = MedicalNER(test_config.ml)

    text = """
    Diabetes is a chronic disease affecting millions worldwide.
    Treatment includes insulin therapy and lifestyle modifications.
    Patients require regular monitoring and medical supervision.
    """

    keywords = ner.extract_keywords_tfidf(text, max_keywords=5)

    assert len(keywords) <= 5
    assert all(isinstance(kw, str) for kw in keywords)


def test_embedding_cache_init(test_config):
    """Test embedding cache initialization."""
    cache = EmbeddingCache(test_config.ml, Path(test_config.data.cache_dir))
    assert cache.cache_dir.exists()


def test_embedding_generation(test_config):
    """Test embedding generation."""
    cache = EmbeddingCache(test_config.ml, Path(test_config.data.cache_dir))

    if cache.model is None:
        pytest.skip("Embedding model not available")

    text = "Diabetes treatment and management strategies"
    embedding = cache.embed(text, use_cache=False)

    if embedding is not None:
        assert embedding.shape[0] > 0


def test_cosine_similarity(test_config):
    """Test cosine similarity calculation."""
    import numpy as np

    cache = EmbeddingCache(test_config.ml, Path(test_config.data.cache_dir))

    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    emb3 = np.array([0.0, 1.0, 0.0])

    # Identical vectors
    sim1 = cache.cosine_similarity(emb1, emb2)
    assert abs(sim1 - 1.0) < 0.001

    # Orthogonal vectors
    sim2 = cache.cosine_similarity(emb1, emb3)
    assert abs(sim2) < 0.001


def test_qa_generator_init(test_config):
    """Test Q&A generator initialization."""
    generator = QAGenerator(test_config.ml)
    assert generator.config is not None


def test_generate_qa_pairs(test_config):
    """Test Q&A pair generation."""
    generator = QAGenerator(test_config.ml)

    title = "Diabetes Management in African Populations"
    abstract = """
    This study examines diabetes management strategies in African populations.
    Results show significant variations in treatment access and efficacy.
    Common symptoms include increased thirst and frequent urination.
    """

    entities = {
        "DISEASE": [{"text": "diabetes", "label": "DISEASE", "start": 0, "end": 8}],
        "SYMPTOM": [
            {"text": "increased thirst", "label": "SYMPTOM", "start": 0, "end": 16},
            {"text": "frequent urination", "label": "SYMPTOM", "start": 0, "end": 18},
        ],
    }

    qa_pairs = generator.generate_qa_pairs(title, abstract, entities, max_pairs=3)

    assert len(qa_pairs) <= 3
    assert all("question" in qa for qa in qa_pairs)
    assert all("answer" in qa for qa in qa_pairs)
    assert all("type" in qa for qa in qa_pairs)
