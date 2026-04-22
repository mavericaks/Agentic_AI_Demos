#!/usr/bin/env bash
# ============================================================
#  Inbox Intelligence Agent — Linux/macOS Setup Script
# ============================================================
#  Creates a virtual environment and installs all dependencies.
#  Run from the project root:   bash setup.sh
# ============================================================

set -e

echo ""
echo "  === Inbox Intelligence Agent - Setup ==="
echo ""

# ── Step 1: Create virtual environment ─────────────────────
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment already exists."
fi

# ── Step 2: Activate and install dependencies ──────────────
echo "[2/3] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ── Step 3: Create .env from template if missing ───────────
if [ ! -f ".env" ]; then
    echo "[3/3] Creating .env from template..."
    cp .env.example .env
    echo "     Edit .env and add your API keys before running demos."
else
    echo "[3/3] .env already exists."
fi

echo ""
echo "  === Setup Complete! ==="
echo ""
echo "  Next steps:"
echo "    1. Edit .env with your API keys"
echo "    2. Place credentials.json in the project root"
echo "    3. Activate the venv:   source venv/bin/activate"
echo "    4. Run first-time auth: python -c \"from utils.auth import get_credentials; get_credentials()\""
echo "    5. Run a demo:          python session_1_vanilla/demo_1a_passive_llm.py"
echo ""
