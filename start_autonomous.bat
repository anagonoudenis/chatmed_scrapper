@echo off
echo ========================================
echo Agent Autonome ChatMed Scraper
echo ========================================
echo.

REM Verifier si .env existe
if not exist .env (
    echo [ERREUR] Fichier .env introuvable !
    echo.
    echo Creez un fichier .env avec :
    echo DEEPSEEK_API_KEY=sk-votre-cle-ici
    echo.
    pause
    exit /b 1
)

REM Activer l'environnement virtuel si disponible
if exist venv\Scripts\activate.bat (
    echo Activation de l'environnement virtuel...
    call venv\Scripts\activate.bat
)

echo Lancement de l'agent autonome...
echo.
python main.py auto-scrape

pause
