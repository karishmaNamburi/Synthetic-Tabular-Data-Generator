#!/usr/bin/env bash
echo "========================================================"
echo " SynTabGAN - Generative Tabular AI Research Platform"
echo " Final-Year B.Tech Engineering Project Demonstration"
echo "========================================================"
echo ""

if [ ! -f ".venv/bin/streamlit" ]; then
    echo "[ERROR] Virtual environment not found. Please run:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "[INFO] Launching Streamlit Dashboard..."
.venv/bin/streamlit run app.py
