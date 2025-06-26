"""
Configuration settings for Smart Timetable System
"""
import os

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///smart_timetable.db')

# Server Configuration
API_PORT = int(os.getenv('API_PORT', 5001))
ADMIN_PORT = int(os.getenv('ADMIN_PORT', 5002))
TEACHER_PORT = int(os.getenv('TEACHER_PORT', 5003))
STUDENT_PORT = int(os.getenv('STUDENT_PORT', 5004))
MAIN_PORT = int(os.getenv('MAIN_PORT', 5000))

# University Configuration
UNIVERSITY_NAME = os.getenv('UNIVERSITY_NAME', 'Kalinga University')
UNIVERSITY_DOMAIN = os.getenv('UNIVERSITY_DOMAIN', 'kalinga.edu')

# Admin Credentials (Environment Variables)
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@kalinga.edu')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# Security Settings
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
JWT_SECRET = os.getenv('JWT_SECRET', 'jwt-secret-key-here')

# File Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SCHEDULE_FILE = os.path.join(DATA_DIR, 'kalinga_schedule_final.csv')
ROOMS_FILE = os.path.join(DATA_DIR, 'kalinga_rooms_final.csv')
ACTIVITIES_FILE = os.path.join(DATA_DIR, 'kalinga_activities_final.csv')
STUDENTS_FILE = os.path.join(DATA_DIR, 'updated_students_72_batches.csv')
TRANSIT_FILE = os.path.join(DATA_DIR, 'final_transit_data_1750319692453.xlsx')
SUBJECT_MAPPINGS_FILE = os.path.join(DATA_DIR, 'subject_room_mappings.csv')

# System Settings
DEFAULT_TIMEZONE = 'Asia/Kolkata'
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.json'}

# AI/ML Settings
MODEL_FILE = os.path.join(DATA_DIR, 'smart_timetable_model.json')
ANOMALY_THRESHOLD = 0.5
RECONSTRUCTION_EPOCHS = 50