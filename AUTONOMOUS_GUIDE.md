# 🤖 Guide de l'Agent Autonome

## Vue d'ensemble

L'agent autonome est un système intelligent qui scrape automatiquement des données médicales **sans aucune intervention humaine**. Il utilise l'API DeepSeek pour :

1. ✅ Générer automatiquement des sujets médicaux
2. ✅ Trouver les meilleures sources pour chaque sujet
3. ✅ Scraper et valider le contenu
4. ✅ Enrichir les données avec des Q&A
5. ✅ Organiser les résultats par sujet

## 🚀 Démarrage Rapide

### Étape 1 : Obtenir une clé API DeepSeek

1. Allez sur https://platform.deepseek.com/
2. Créez un compte
3. Obtenez votre clé API (gratuit avec crédits de démarrage)

### Étape 2 : Configurer la clé API

Ouvrez `config.toml` et ajoutez votre clé :

```toml
[deepseek]
api_key = "sk-votre-cle-api-ici"
```

### Étape 3 : Lancer l'agent

```bash
# Mode simple : 50 sujets
python main.py auto-scrape

# Mode personnalisé : 100 sujets
python main.py auto-scrape --topics 100

# Mode continu : tourne indéfiniment
python main.py auto-scrape --continuous
```

## 📊 Ce que fait l'agent

### Phase 1 : Génération de sujets
```
[Agent] Génération de 50 sujets médicaux...
[Agent] ✓ Générés : Diabète de type 2, Paludisme, Hypertension...
```

L'agent utilise DeepSeek pour générer une liste de maladies importantes :
- Maladies infectieuses
- Maladies chroniques
- Maladies parasitaires
- Conditions médicales courantes

### Phase 2 : Recherche de sources
```
[Agent] Sujet 1/50 : Diabète de type 2
[Agent] 🔍 Recherche des meilleures sources...
[Agent] ✓ Trouvé 10 URLs pertinentes
```

Pour chaque sujet, DeepSeek suggère les meilleures URLs :
- Sites officiels (OMS, CDC, Pasteur)
- Sites médicaux de référence (Mayo Clinic, MedlinePlus)
- Guides et fiches médicales

### Phase 3 : Scraping
```
[Agent] 🌐 Scraping de 10 URLs...
[Agent] ✓ 8 pages scrapées avec succès
```

Le scraper universel extrait :
- Titre et contenu principal
- Métadonnées
- Images et liens

### Phase 4 : Validation et enrichissement
```
[Agent] 🤖 Validation et enrichissement...
[Agent] ✓ Gardé 6 pages (qualité > 0.7)
[Agent] ✓ Rejeté 2 pages (qualité faible)
```

DeepSeek valide chaque page :
- Score de qualité (0-1)
- Pertinence médicale
- Génération de Q&A pairs

### Phase 5 : Sauvegarde
```
[Agent] ✓ Sauvegardé : data/output/diabete_type2/autonomous_scrape_20251024.jsonl
```

Chaque sujet a son propre dossier avec :
- Données enrichies en JSONL
- Rapport de qualité

## 📁 Structure des résultats

```
data/output/
├── diabete_type2/
│   └── autonomous_scrape_20251024_193045.jsonl
├── paludisme/
│   └── autonomous_scrape_20251024_193152.jsonl
├── hypertension_arterielle/
│   └── autonomous_scrape_20251024_193258.jsonl
└── reports/
    └── autonomous_report_20251024_195030.json
```

## 📄 Format des données enrichies

Chaque entrée JSONL contient :

```json
{
  "url": "https://www.mayoclinic.org/...",
  "title": "Diabetes - Symptoms and causes",
  "content": "Diabetes mellitus refers to...",
  "quality_score": 0.92,
  "deepseek_validation": {
    "is_medical": true,
    "quality_score": 0.92,
    "relevance_score": 0.95,
    "recommendation": "keep"
  },
  "qa_pairs": [
    {
      "question": "Qu'est-ce que le diabète ?",
      "answer": "Le diabète est une maladie chronique...",
      "type": "definition"
    },
    {
      "question": "Quels sont les symptômes ?",
      "answer": "Les symptômes incluent...",
      "type": "symptom"
    }
  ]
}
```

## ⚙️ Configuration avancée

### Ajuster le seuil de qualité

Dans `config.toml` :

```toml
[autonomous]
quality_threshold = 0.7  # Ne garde que les pages avec score > 0.7
```

### Nombre d'URLs par sujet

```toml
[autonomous]
max_urls_per_topic = 10  # Scrape jusqu'à 10 URLs par sujet
```

### Délai entre sujets

```toml
[autonomous]
sleep_between_topics = 5  # Pause de 5 secondes entre chaque sujet
```

### Traitement parallèle

```toml
[autonomous]
max_concurrent_topics = 3  # Traite 3 sujets en parallèle
```

## 💰 Coûts estimés

DeepSeek est très économique :

- **Prix** : ~$0.14 / 1M tokens input, ~$0.28 / 1M tokens output
- **50 sujets** : ~$2-5 USD
- **100 sujets** : ~$5-10 USD

### Estimation par sujet :
- Génération du sujet : ~100 tokens
- Suggestion d'URLs : ~200 tokens
- Validation (10 pages) : ~5000 tokens
- Génération Q&A (10 pages) : ~3000 tokens
- **Total** : ~8300 tokens ≈ $0.10 par sujet

## 📊 Rapport final

À la fin, l'agent génère un rapport complet :

```
════════════════════════════════════════════════════════════
🎉 AUTONOMOUS SCRAPING COMPLETED
════════════════════════════════════════════════════════════

Topics Processed       48/50
Topics Failed          2
Total Pages Scraped    420
Pages Kept            312
Pages Rejected        108
Avg Quality Score     0.85
Total Duration        3847.2s

✓ Report saved to: data/output/reports/autonomous_report_20251024.json
```

## 🔧 Dépannage

### Erreur : "DeepSeek API key not configured"

**Solution** : Ajoutez votre clé API dans `config.toml` :
```toml
[deepseek]
api_key = "sk-votre-cle-ici"
```

### Erreur : "Rate limit exceeded"

**Solution** : L'agent attend automatiquement. Si le problème persiste, augmentez le délai :
```toml
[autonomous]
sleep_between_topics = 10  # Augmenter à 10 secondes
```

### Peu de pages gardées (beaucoup rejetées)

**Solution** : Réduisez le seuil de qualité :
```toml
[autonomous]
quality_threshold = 0.6  # Réduire de 0.7 à 0.6
```

### L'agent s'arrête après quelques sujets

**Vérifiez** :
1. Votre crédit DeepSeek API
2. Votre connexion internet
3. Les logs dans `logs/scraper.log`

## 🎯 Cas d'usage

### 1. Créer un dataset médical complet

```bash
# Générer 200 sujets médicaux variés
python main.py auto-scrape --topics 200
```

**Résultat** : 200 dossiers avec données enrichies

### 2. Scraping continu pour mise à jour

```bash
# Mode continu : tourne 24/7
python main.py auto-scrape --continuous
```

**Résultat** : Dataset qui s'enrichit automatiquement

### 3. Focus sur des maladies spécifiques

Modifiez `utils/deepseek_client.py` pour cibler des catégories :

```python
prompt = f"""Génère {count} maladies infectieuses tropicales..."""
```

## 📈 Optimisations

### Parallélisation

Pour aller plus vite, augmentez le parallélisme :

```toml
[autonomous]
max_concurrent_topics = 5  # Traite 5 sujets en parallèle
```

### Cache des résultats DeepSeek

L'agent peut être étendu pour cacher les réponses DeepSeek et éviter les appels répétés.

### Filtrage par langue

Modifiez la génération de sujets pour cibler une langue :

```python
topics = await deepseek_client.generate_medical_topics(
    count=50,
    language="en"  # ou "fr"
)
```

## 🔒 Sécurité

### Protection de la clé API

**Ne commitez JAMAIS votre clé API dans Git !**

Le fichier `config.toml` avec la clé doit rester local.

### Alternative : Variable d'environnement

```bash
export DEEPSEEK_API_KEY="sk-votre-cle"
```

Puis dans le code, lisez depuis l'environnement.

## 📞 Support

Pour toute question :
1. Vérifiez les logs : `logs/scraper.log`
2. Consultez le rapport : `data/output/reports/`
3. Vérifiez votre crédit DeepSeek : https://platform.deepseek.com/

---

**🎉 Profitez du scraping autonome ! Lancez une fois, laissez tourner, récupérez des données de qualité !**
