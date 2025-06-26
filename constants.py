"""
Constants and Configuration for Smart Timetable System
All system-wide constants, configurations, and settings
"""

import os

# ==================== DATABASE CONFIGURATION ====================
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///timetable.db')
DATABASE_TIMEOUT = 30

# ==================== PORTAL CONFIGURATION ====================
PORTAL_PORTS = {
    'main': 5000,
    'admin': 5002,
    'teacher': 5003,
    'student': 5004,
    'api': 8000
}

PORTAL_ADDRESSES = {
    'host': '0.0.0.0',
    'local_host': 'localhost'
}

# ==================== AUTHENTICATION SETTINGS ====================
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'smart_timetable_secret_key_2025')
JWT_EXPIRATION_HOURS = 24
PASSWORD_HASH_ROUNDS = 12

# Default user accounts for testing
DEFAULT_USERS = {
    'admin': {
        'email': 'rohit.verma@university.edu',
        'password': 'admin123',
        'role': 'admin',
        'name': 'Rohit Verma'
    },
    'teacher': {
        'email': 'dr.agarwal@kalinga.edu',
        'password': 'teacher123',
        'role': 'teacher',
        'name': 'Dr. Agarwal'
    },
    'student': {
        'email': 'aarav.sharma@kalinga.edu',
        'password': 'student123',
        'role': 'student',
        'name': 'Aarav Sharma',
        'section': 'CSE-A2'
    }
}

# ==================== SYSTEM CONFIGURATION ====================
TOTAL_SECTIONS = 72
SECTIONS_A = [f'A{i}' for i in range(1, 37)]  # A1 to A36
SECTIONS_B = [f'B{i}' for i in range(1, 37)]  # B1 to B36
ALL_SECTIONS = SECTIONS_A + SECTIONS_B

# Campus configuration
CAMPUS_INFO = {
    'Campus_3': {
        'name': 'Main Campus (Campus 3)',
        'theory_rooms': 25,
        'labs': 5,
        'walking_distance': '0m',
        'walking_time': '0 min',
        'description': 'Main academic building with theory rooms'
    },
    'Campus_8': {
        'name': 'Workshop Campus (Campus 8)',
        'theory_rooms': 10,
        'labs': 3,
        'walking_distance': '550m',
        'walking_time': '7 min',
        'description': 'Secondary building with workshop facilities'
    },
    'Campus_15B': {
        'name': 'Engineering Campus (Campus 15B)',
        'theory_rooms': 18,
        'labs': 8,
        'walking_distance': '700m',
        'walking_time': '10 min',
        'description': 'Engineering building with programming labs'
    },
    'Stadium': {
        'name': 'Sports Stadium',
        'theory_rooms': 0,
        'labs': 0,
        'walking_distance': '900m',
        'walking_time': '12 min',
        'description': 'Sports and yoga facilities'
    }
}

# ==================== AI SYSTEM CONFIGURATION ====================
AI_CONFIG = {
    'anomaly_threshold': 0.75,
    'learning_rate': 1e-3,
    'training_epochs': 100,
    'sequence_length': 42,  # Weekly time slots
    'embedding_dim': 64,
    'hidden_dim': 128,
    'param_dim': 10
}

# ==================== TIME SLOTS CONFIGURATION ====================
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
TIME_SLOTS = ['08_00', '09_00', '10_00', '11_00', '12_00', '14_00', '15_00']

# Generate all time slot combinations
ALL_TIME_SLOTS = []
for day in DAYS:
    for time in TIME_SLOTS:
        ALL_TIME_SLOTS.append(f"{day}_{time}")

# ==================== SUBJECT CONFIGURATION ====================
SUBJECTS = [
    'Chemistry',
    'Chemistry Lab',
    'Mathematics',
    'Transform and Numerical Methods',
    'Physics',
    'Physics Lab',
    'Electronics',
    'Programming',
    'Programming Lab',
    'Mechanics',
    'English',
    'Environmental Science',
    'Workshop',
    'Sports and Yoga'
]

SUBJECT_TYPES = {
    'theory': ['Chemistry', 'Mathematics', 'Transform and Numerical Methods', 
               'Physics', 'Electronics', 'Programming', 'Mechanics', 
               'English', 'Environmental Science'],
    'lab': ['Chemistry Lab', 'Physics Lab', 'Programming Lab', 'Workshop'],
    'activity': ['Sports and Yoga']
}

# ==================== FILE PATHS ====================
DATA_PATHS = {
    'teacher_data': 'data/teacher_data.csv',
    'transit_data': 'data/final_transit_data_1750319692453.xlsx',
    'models': 'models/',
    'exports': 'exports/',
    'logs': 'logs/'
}

# ==================== API ENDPOINTS ====================
API_ENDPOINTS = {
    'auth': {
        'login': '/api/auth/login',
        'register': '/api/auth/register',
        'refresh': '/api/auth/refresh',
        'logout': '/api/auth/logout'
    },
    'events': {
        'all': '/api/events',
        'by_id': '/api/events/<int:event_id>',
        'create': '/api/events',
        'update': '/api/events/<int:event_id>',
        'delete': '/api/events/<int:event_id>',
        'swap': '/api/events/swap'
    },
    'users': {
        'profile': '/api/user/profile',
        'update': '/api/user/profile',
        'all': '/api/users',
        'teacher_schedule': '/api/users/teacher/<teacher_id>/schedule',
        'swap_request': '/api/users/teacher/swap-request'
    },
    'system': {
        'status': '/api/status',
        'root': '/'
    }
}

# ==================== RESPONSE MESSAGES ====================
MESSAGES = {
    'success': {
        'login': 'Login successful',
        'logout': 'Logout successful',
        'registration': 'User registered successfully',
        'profile_updated': 'Profile updated successfully',
        'event_created': 'Event created successfully',
        'event_updated': 'Event updated successfully',
        'event_deleted': 'Event deleted successfully',
        'events_swapped': 'Events swapped successfully'
    },
    'error': {
        'invalid_credentials': 'Invalid email or password',
        'user_exists': 'User already exists',
        'user_not_found': 'User not found',
        'event_not_found': 'Event not found',
        'unauthorized': 'Unauthorized access',
        'forbidden': 'Access forbidden',
        'invalid_token': 'Invalid or expired token',
        'missing_fields': 'Required fields missing',
        'invalid_data': 'Invalid data provided',
        'server_error': 'Internal server error'
    }
}

# ==================== ENVIRONMENT SETTINGS ====================
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG_MODE = ENVIRONMENT == 'development'

CORS_SETTINGS = {
    'origins': ['http://localhost:3000', 'http://localhost:5000', 
                'http://localhost:5002', 'http://localhost:5003', 
                'http://localhost:5004'],
    'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    'headers': ['Content-Type', 'Authorization']
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'level': 'DEBUG' if DEBUG_MODE else 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/timetable_system.log'
}

# ==================== PAGINATION SETTINGS ====================
PAGINATION = {
    'default_page_size': 50,
    'max_page_size': 200,
    'default_page': 1
}

# ==================== CACHE SETTINGS ====================
CACHE_CONFIG = {
    'schedule_cache_ttl': 3600,  # 1 hour
    'user_cache_ttl': 1800,     # 30 minutes
    'static_cache_ttl': 86400   # 24 hours
}

# ==================== VERSION INFO ====================
VERSION_INFO = {
    'system_version': '2.0.0',
    'api_version': 'v1',
    'last_updated': '2025-06-26',
    'build': 'production'
}