# Configuration Securisee avec .env

## Pourquoi utiliser .env ?

- Securite : Les cles API ne sont pas dans le code
- Git-safe : Le fichier .env est automatiquement ignore par Git
- Flexibilite : Differentes cles pour dev/prod

## Configuration en 3 etapes

### Etape 1 : Installer python-dotenv

```bash
pip install python-dotenv==1.0.0
```

Ou reinstallez tout :

```bash
pip install -r requirements-minimal.txt
```

### Etape 2 : Creer le fichier .env

Creez un fichier nomme .env a la racine du projet :

```
DEEPSEEK_API_KEY=sk-votre-cle-api-ici
PUBMED_API_KEY=
```

IMPORTANT : 
- Remplacez sk-votre-cle-api-ici par votre vraie cle DeepSeek
- Ne commitez JAMAIS ce fichier dans Git
- Ne partagez JAMAIS ce fichier

### Etape 3 : Lancer l'agent

```bash
python main.py auto-scrape
```

Le systeme va automatiquement :
1. Lire le fichier .env
2. Charger DEEPSEEK_API_KEY
3. L'utiliser pour l'agent autonome

## Verification

Pour verifier que la cle est bien chargee, lancez :

```bash
python main.py info
```

Vous devriez voir : "DeepSeek API key loaded from environment variable"

## Alternative : config.toml

Si vous preferez, vous pouvez aussi mettre la cle directement dans config.toml :

```toml
[deepseek]
api_key = "sk-votre-cle-ici"
```

Mais .env est plus securise !

## Obtenir une cle DeepSeek

1. Allez sur https://platform.deepseek.com/
2. Creez un compte (gratuit)
3. Allez dans API Keys
4. Creez une nouvelle cle
5. Copiez la cle (commence par sk-)
6. Collez-la dans votre fichier .env

## Couts

DeepSeek est tres economique :
- 50 sujets : environ 2-5 USD
- 100 sujets : environ 5-10 USD
- Credits gratuits au debut

## Securite

Le fichier .env est deja dans .gitignore, donc :
- Il ne sera JAMAIS commite dans Git
- Il reste sur votre machine uniquement
- Vos cles API sont protegees

## Demarrage rapide

```bash
# 1. Installer
pip install -r requirements-minimal.txt

# 2. Creer .env avec votre cle
echo "DEEPSEEK_API_KEY=sk-votre-cle" > .env

# 3. Lancer
python main.py auto-scrape
```

C'est tout !
