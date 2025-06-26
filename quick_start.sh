#!/bin/bash
# Quick Start Script for Smart Timetable System

echo "🚀 Starting Smart Timetable System Setup..."

# Check Python version
python3 --version || { echo "Python 3.11+ required"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "⬇️ Installing dependencies..."
pip install -r requirements_complete.txt

# Setup environment
echo "⚙️ Setting up environment..."
cp .env.template .env
echo "Please edit .env file with your configuration"

# Initialize database
echo "🗄️ Initializing database..."
python3 -c "from database import DatabaseManager; db = DatabaseManager(); db.initialize_database()"

# Test system
echo "🧪 Testing system..."
python3 -c "
from database import DatabaseManager
db = DatabaseManager()
print('Database:', 'Connected' if db.test_connection() else 'Failed')
user = db.authenticate_user('rohit.verma@university.edu', 'admin123')
print('Admin Login:', 'Success' if user else 'Failed')
"

echo "✅ Setup complete! Run: streamlit run run_streamlit.py --server.port 5000"
