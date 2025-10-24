# 🚀 Démarrage Rapide - Agent Autonome

## En 4 étapes simples

### 1️⃣ Installer les dépendances

```bash
pip install -r requirements-minimal.txt
```

### 2️⃣ Obtenir une clé API DeepSeek

1. Allez sur https://platform.deepseek.com/
2. Créez un compte (gratuit)
3. Obtenez votre clé API
4. Vous recevez des crédits gratuits pour commencer !

### 3️⃣ Configurer la clé API

Ouvrez `config.toml` et ajoutez votre clé :

```toml
[deepseek]
api_key = "sk-votre-cle-api-ici"
```

### 4️⃣ Lancer l'agent !

```bash
python main.py auto-scrape
```

## 🎉 C'est tout !

L'agent va maintenant :
- ✅ Générer 50 sujets médicaux automatiquement
- ✅ Trouver les meilleures sources pour chaque sujet
- ✅ Scraper et valider le contenu
- ✅ Enrichir avec des Q&A
- ✅ Sauvegarder dans `data/output/`

## 📊 Résultats

Après quelques minutes, vous aurez :

```
data/output/
├── diabete_type2/
│   └── autonomous_scrape_YYYYMMDD_HHMMSS.jsonl
├── paludisme/
│   └── autonomous_scrape_YYYYMMDD_HHMMSS.jsonl
├── hypertension_arterielle/
│   └── autonomous_scrape_YYYYMMDD_HHMMSS.jsonl
└── ... (50 sujets)
```

Chaque fichier JSONL contient :
- Contenu médical nettoyé
- Score de qualité
- Paires de questions-réponses
- Métadonnées complètes

## ⚙️ Options avancées

```bash
# Scraper 100 sujets au lieu de 50
python main.py auto-scrape --topics 100

# Mode continu (tourne indéfiniment)
python main.py auto-scrape --continuous

# Voir toutes les options
python main.py auto-scrape --help
```

## 💰 Coûts

DeepSeek est très économique :
- 50 sujets ≈ $2-5 USD
- 100 sujets ≈ $5-10 USD

## 📖 Documentation complète

Pour plus de détails, consultez :
- [AUTONOMOUS_GUIDE.md](AUTONOMOUS_GUIDE.md) - Guide complet
- [README.md](README.md) - Documentation générale

## 🆘 Besoin d'aide ?

**Erreur : "DeepSeek API key not configured"**
→ Vérifiez que vous avez bien ajouté la clé dans `config.toml`

**Erreur : "Rate limit exceeded"**
→ L'agent attend automatiquement, soyez patient

**Peu de pages gardées**
→ Réduisez le seuil de qualité dans `config.toml` :
```toml
[autonomous]
quality_threshold = 0.6  # Au lieu de 0.7
```

---

**🎯 Profitez du scraping autonome ! Lancez et laissez faire l'IA !**
