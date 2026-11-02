@echo off
echo ========================================================
echo  SynTabGAN - Generative Tabular AI Research Platform
echo  Final-Year B.Tech Engineering Project Demonstration
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv
    echo Please create the virtual environment and install requirements:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Launching Streamlit Dashboard...
echo [INFO] URL: http://localhost:8501
echo.
.\.venv\Scripts\streamlit.exe run app.py --server.headless false

pause
