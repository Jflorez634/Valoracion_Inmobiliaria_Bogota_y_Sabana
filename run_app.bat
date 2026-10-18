@echo off
cd /d "%~dp0"
echo ========================================================
echo  Iniciando Valuador Inmobiliario - Bogota y Sabana
echo ========================================================
set PYTHONPATH=%~dp0
echo Activando entorno virtual...
call .venv\Scripts\activate.bat
echo Ejecutando Streamlit en http://localhost:8501...
streamlit run src\app.py
pause
