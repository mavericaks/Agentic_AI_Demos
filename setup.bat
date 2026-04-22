@echo off
REM ============================================================
REM  Inbox Intelligence Agent — Windows Setup Script
REM ============================================================
REM  Creates a virtual environment and installs all dependencies.
REM  Run from the project root:   setup.bat
REM ============================================================

echo.
echo  === Inbox Intelligence Agent - Setup ===
echo.

REM ── Step 1: Create virtual environment ─────────────────────
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Is Python installed?
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists.
)

REM ── Step 2: Activate and install dependencies ──────────────
echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

REM ── Step 3: Create .env from template if missing ───────────
if not exist ".env" (
    echo [3/3] Creating .env from template...
    copy .env.example .env >nul
    echo      Edit .env and add your API keys before running demos.
) else (
    echo [3/3] .env already exists.
)

echo.
echo  === Setup Complete! ===
echo.
echo  Next steps:
echo    1. Edit .env with your API keys
echo    2. Place credentials.json in the project root
echo    3. Activate the venv:   venv\Scripts\activate
echo    4. Run first-time auth: python -c "from utils.auth import get_credentials; get_credentials()"
echo    5. Run a demo:          python session_1_vanilla/demo_1a_passive_llm.py
echo.
