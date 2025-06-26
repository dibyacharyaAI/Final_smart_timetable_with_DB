@echo off
REM Quick Start Script for Smart Timetable System (Windows)

echo 🚀 Starting Smart Timetable System Setup...

REM Check Python
python --version >nul 2>&1 || (echo Python 3.11+ required & exit /b 1)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Install dependencies
echo ⬇️ Installing dependencies...
pip install -r requirements_complete.txt

REM Setup environment
echo ⚙️ Setting up environment...
copy .env.template .env
echo Please edit .env file with your configuration

REM Initialize database
echo 🗄️ Initializing database...
python -c "from database import DatabaseManager; db = DatabaseManager(); db.initialize_database()"

echo ✅ Setup complete! Run: streamlit run run_streamlit.py --server.port 5000
pause
