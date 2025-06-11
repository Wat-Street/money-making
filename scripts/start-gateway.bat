@echo off
echo Starting WatStreet API Gateway...

cd /d "%~dp0\..\gateway"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting gateway on port 5000...
python app.py

pause 