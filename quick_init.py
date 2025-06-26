"""
Quick DB Init Script for Smart Timetable System
Creates all tables and inserts default users.
"""

from database import DatabaseManager

def initialize_system():
    db = DatabaseManager()
    print("🔧 Creating tables...")
    db.create_tables()
    print("👥 Inserting default users...")
    db.initialize_default_users()
    print("✅ Database initialized successfully.")

if __name__ == "__main__":
    initialize_system()
