"""
DeepSeek API Client for Medical Data Enrichment
Provides intelligent content extraction, Q&A generation, and quality validation.
"""

import asyncio
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek API"""
    api_key: str
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60


class DeepSeekClient:
    """Client for DeepSeek API with medical-focused prompts"""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        await self.init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()
        
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            logger.info("DeepSeek API session initialized")
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("DeepSeek API session closed")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Call DeepSeek API with retry logic"""
        if not self.session:
            await self.init_session()
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }
        
        try:
            async with self.session.post(self.config.api_url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    logger.debug(f"DeepSeek API response: {len(content)} chars")
                    return content
                else:
                    raise ValueError("Invalid API response format")
                    
        except asyncio.TimeoutError as e:
            logger.error(f"DeepSeek API timeout after {self.config.timeout}s: {e}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"DeepSeek API client error: {type(e).__name__} - {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling DeepSeek API: {type(e).__name__} - {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def generate_medical_topics(self, count: int = 50, language: str = "fr") -> List[str]:
        """
        Generate a list of medical topics to scrape
        
        Args:
            count: Number of topics to generate
            language: Language for topics (fr/en)
            
        Returns:
            List of medical topics
        """
        prompt = f"""Tu es un expert médical. Génère une liste de {count} sujets médicaux importants à documenter.

Critères:
- Maladies courantes et importantes
- Variété: infectieuses, chroniques, parasitaires, etc.
- Pertinence pour la santé publique
- Intérêt pour la recherche médicale

Format: Une maladie par ligne, sans numérotation.
Langue: {language}

Exemple:
Diabète de type 2
Paludisme
Hypertension artérielle
..."""

        messages = [
            {"role": "system", "content": "Tu es un expert médical spécialisé dans la classification des maladies."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.8)
            topics = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
            logger.success(f"Generated {len(topics)} medical topics")
            return topics[:count]
        except Exception as e:
            logger.error(f"Failed to generate topics: {e}")
            return []
    
    async def suggest_urls_for_topic(self, topic: str, max_urls: int = 50) -> List[str]:
        """
        Suggest best URLs to scrape for a given medical topic
        
        Args:
            topic: Medical topic (e.g., "Diabète de type 2")
            max_urls: Maximum number of URLs to suggest
            
        Returns:
            List of suggested URLs
        """
        prompt = f"""Pour le sujet médical "{topic}", génère une liste EXHAUSTIVE de {max_urls} URLs de HAUTE QUALITÉ à scraper.

SOURCES PRIORITAIRES (trouve plusieurs pages par source):
1. OMS/WHO - Articles, guides, rapports, statistiques
2. CDC - Fiches maladies, guides, recommandations
3. Institut Pasteur - Fiches maladies, actualités
4. INSERM - Dossiers, articles de recherche
5. Mayo Clinic - Symptômes, causes, traitements, prévention
6. MedlinePlus - Guides patients, encyclopédie
7. PubMed Central - Articles scientifiques en accès libre
8. Santé Publique France - Rapports, surveillance
9. HAS (Haute Autorité de Santé) - Recommandations
10. Universités médicales - Cours, ressources pédagogiques

TYPES DE CONTENU À INCLURE:
- Pages principales sur la maladie
- Symptômes et diagnostic
- Traitements et médicaments
- Prévention et dépistage
- Épidémiologie et statistiques
- Guides pour professionnels
- Guides pour patients
- Articles de recherche
- Rapports officiels
- FAQ et ressources

FORMAT: Une URL complète par ligne (https://...)
IMPORTANT: Génère EXACTEMENT {max_urls} URLs différentes et valides.
Ne pas répéter les mêmes URLs.
Ne pas inclure de numérotation."""

        messages = [
            {"role": "system", "content": "Tu es un expert en sources médicales fiables."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.5)
            urls = [
                line.strip() 
                for line in response.split('\n') 
                if line.strip().startswith('http')
            ]
            logger.info(f"Suggested {len(urls)} URLs for topic: {topic}")
            return urls[:max_urls]
        except Exception as e:
            logger.error(f"Failed to suggest URLs for {topic}: {e}")
            return []
    
    async def extract_clean_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract and clean medical content from HTML
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Dict with cleaned content and metadata
        """
        # Truncate if too long (DeepSeek has token limits)
        max_chars = 15000
        if len(html_content) > max_chars:
            html_content = html_content[:max_chars] + "..."
        
        prompt = f"""Analyse ce contenu médical et extrait les informations structurées.

URL: {url}

Contenu HTML (tronqué):
{html_content}

Extrait et retourne un JSON avec:
{{
  "title": "Titre principal",
  "main_content": "Contenu médical principal (sans menus, pubs, etc.)",
  "summary": "Résumé en 2-3 phrases",
  "key_points": ["Point clé 1", "Point clé 2", ...],
  "medical_entities": {{
    "diseases": ["maladie1", ...],
    "symptoms": ["symptôme1", ...],
    "treatments": ["traitement1", ...]
  }},
  "quality_score": 0.0-1.0,
  "content_type": "article|guide|study|reference"
}}

Retourne UNIQUEMENT le JSON, sans texte supplémentaire."""

        messages = [
            {"role": "system", "content": "Tu es un expert en extraction de contenu médical."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.3, max_tokens=2000)
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                logger.success(f"Extracted clean content from {url}")
                return result
            else:
                logger.warning(f"No valid JSON in response for {url}")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to extract content: {e}")
            return {}
    
    async def generate_multilingual_qa_pairs(
        self, 
        content: str, 
        count: int = 5,
        languages: List[str] = ["fr", "en"]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate Q&A pairs in multiple languages from medical content
        
        Args:
            content: Medical content text
            count: Number of Q&A pairs to generate per language
            languages: List of language codes (fr, en, es, ar, etc.)
            
        Returns:
            Dict with language codes as keys and Q&A lists as values
        """
        # Truncate if too long
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        language_names = {
            "fr": "français", "en": "English", "es": "español", "ar": "العربية",
            "pt": "português", "de": "Deutsch", "it": "italiano", "zh": "中文",
            "ja": "日本語", "ru": "русский", "hi": "हिन्दी", "sw": "Kiswahili",
            "ko": "한국어", "tr": "Türkçe", "nl": "Nederlands", "pl": "polski",
            "vi": "Tiếng Việt", "th": "ไทย", "id": "Bahasa Indonesia"
        }
        
        prompt = f"""À partir de ce contenu médical, génère {count} paires de questions-réponses DANS CHAQUE LANGUE suivante : {', '.join([language_names.get(lang, lang) for lang in languages])}.

Contenu médical:
{content}

IMPORTANT: Génère les Q&A dans TOUTES les langues demandées.

Format JSON:
{{
  "fr": [
    {{"question": "Question en français", "answer": "Réponse en français", "type": "definition"}},
    ...
  ],
  "en": [
    {{"question": "Question in English", "answer": "Answer in English", "type": "definition"}},
    ...
  ],
  "es": [
    {{"question": "Pregunta en español", "answer": "Respuesta en español", "type": "definition"}},
    ...
  ]
  ... (pour toutes les langues)
}}

Types possibles: definition, symptom, treatment, prevention, diagnosis

Retourne UNIQUEMENT le JSON."""

        messages = [
            {"role": "system", "content": "Tu es un expert médical multilingue créant du contenu éducatif dans plusieurs langues."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.7, max_tokens=4000)
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                multilingual_qa = json.loads(json_str)
                total_qa = sum(len(qa_list) for qa_list in multilingual_qa.values())
                logger.success(f"Generated {total_qa} Q&A pairs in {len(multilingual_qa)} languages")
                return multilingual_qa
            else:
                logger.warning("No valid JSON in multilingual Q&A response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse multilingual Q&A JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to generate multilingual Q&A pairs: {e}")
            return {}
    
    async def generate_qa_pairs(self, content: str, count: int = 5) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from medical content
        
        Args:
            content: Medical content text
            count: Number of Q&A pairs to generate
            
        Returns:
            List of Q&A pairs
        """
        # Truncate if too long
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        prompt = f"""À partir de ce contenu médical, génère {count} paires de questions-réponses pertinentes.

Contenu:
{content}

Format JSON:
[
  {{
    "question": "Question claire et précise",
    "answer": "Réponse complète basée sur le contenu",
    "type": "definition|symptom|treatment|prevention|diagnosis"
  }},
  ...
]

Retourne UNIQUEMENT le JSON array."""

        messages = [
            {"role": "system", "content": "Tu es un expert médical créant du contenu éducatif."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.7, max_tokens=2000)
            # Extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                qa_pairs = json.loads(json_str)
                logger.success(f"Generated {len(qa_pairs)} Q&A pairs")
                return qa_pairs
            else:
                logger.warning("No valid JSON array in response")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Q&A JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to generate Q&A pairs: {e}")
            return []
    
    async def validate_medical_content(self, content: str, url: str) -> Dict[str, Any]:
        """
        Validate if content is relevant medical information
        
        Args:
            content: Content to validate
            url: Source URL
            
        Returns:
            Validation result with score and feedback
        """
        # Truncate if too long
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        prompt = f"""Évalue la qualité et la pertinence de ce contenu médical.

URL: {url}
Contenu: {content}

Retourne un JSON:
{{
  "is_medical": true/false,
  "quality_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "issues": ["problème1", ...],
  "strengths": ["point fort1", ...],
  "recommendation": "keep|review|reject"
}}

Critères:
- Contenu médical factuel
- Sources fiables
- Informations complètes
- Pas de publicité excessive

Retourne UNIQUEMENT le JSON."""

        messages = [
            {"role": "system", "content": "Tu es un expert en validation de contenu médical."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_api(messages, temperature=0.3, max_tokens=1000)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                logger.info(f"Validated content: {result.get('recommendation', 'unknown')}")
                return result
            else:
                logger.warning("No valid JSON in validation response")
                return {"is_medical": False, "quality_score": 0.0, "recommendation": "reject"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation JSON: {e}")
            return {"is_medical": False, "quality_score": 0.0, "recommendation": "reject"}
        except Exception as e:
            logger.error(f"Failed to validate content: {e}")
            return {"is_medical": False, "quality_score": 0.0, "recommendation": "reject"}


# Utility function to create client from config
def create_deepseek_client(config: Any) -> DeepSeekClient:
    """Create DeepSeek client from config object"""
    deepseek_config = DeepSeekConfig(
        api_key=config.deepseek.api_key,
        api_url=config.deepseek.api_url,
        model=config.deepseek.model,
        temperature=config.deepseek.temperature,
        max_tokens=config.deepseek.max_tokens,
        timeout=config.deepseek.timeout
    )
    return DeepSeekClient(deepseek_config)
