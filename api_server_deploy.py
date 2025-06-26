"""
Clean API Server for Smart Timetable System
Deployment के लिए अलग API server - बिना Streamlit dependencies के
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
from datetime import datetime, timedelta
from functools import wraps
import os
import json
from constants import (
    JWT_SECRET_KEY, JWT_EXPIRATION_HOURS, API_ENDPOINTS, MESSAGES,
    PORTAL_PORTS, CORS_SETTINGS, DEBUG_MODE
)
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = JWT_SECRET_KEY

# Configure CORS
CORS(app, 
     origins=CORS_SETTINGS['origins'],
     methods=CORS_SETTINGS['methods'],
     allow_headers=CORS_SETTINGS['headers'])

# Initialize database and system
db = DatabaseManager()
timetable_system = SmartTimetableSystem()

# Generate schedule on startup
print("🚀 Initializing Smart Timetable API Server...")
timetable_system.generate_complete_schedule()
print("✓ Schedule generated and ready")

# ==================== AUTHENTICATION DECORATORS ====================

def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': MESSAGES['error']['unauthorized']}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token.split(' ')[1]
            
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            request.current_user = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': MESSAGES['error']['invalid_token']}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': MESSAGES['error']['invalid_token']}), 401
    
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(request, 'current_user') or request.current_user.get('role') != 'admin':
            return jsonify({'error': MESSAGES['error']['forbidden']}), 403
        return f(*args, **kwargs)
    return decorated_function

# ==================== HELPER FUNCTIONS ====================

def get_current_user_from_token(token):
    """Extract user info from token and verify against database"""
    try:
        if token.startswith('Bearer '):
            token = token.split(' ')[1]
        
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user = db.get_user_by_email(payload.get('email'))
        
        if user:
            return {
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'role': user['role']
            }
        return None
    except:
        return None

def convert_schedule_to_events():
    """Convert system schedule to API event format"""
    events = []
    event_id = 1
    
    if hasattr(timetable_system, 'schedule') and timetable_system.schedule:
        for section_id, section_schedule in timetable_system.schedule.items():
            for time_slot, class_info in section_schedule.items():
                # Parse time slot
                day, time = time_slot.split('_')
                start_time = f"{time[:2]}:{time[2:]}"
                
                # Calculate end time (assuming 1-hour slots)
                hour = int(time[:2])
                end_hour = hour + 1
                end_time = f"{end_hour:02d}:{time[2:]}"
                
                # Determine campus
                room = class_info.get('room', '')
                campus = 'Campus_3'  # Default
                if 'C8' in room or 'Workshop' in room:
                    campus = 'Campus_8'
                elif 'C15B' in room or 'Programming' in room:
                    campus = 'Campus_15B'
                elif 'Stadium' in room or 'SY' in room or class_info.get('subject') == 'Sports and Yoga':
                    campus = 'Stadium'
                
                event = {
                    'id': event_id,
                    'title': class_info.get('subject', 'N/A'),
                    'description': f"Section: {section_id}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'day_of_week': day,
                    'location': class_info.get('room', 'N/A'),
                    'campus': campus,
                    'instructor': class_info.get('teacher', 'N/A'),
                    'section': section_id,
                    'subject_type': class_info.get('activity_type', 'Lecture'),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                events.append(event)
                event_id += 1
    
    return events

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.route(API_ENDPOINTS['auth']['login'], methods=['POST'])
def login():
    """User login authentication"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': MESSAGES['error']['missing_fields']}), 400
        
        user = db.authenticate_user(email, password)
        if not user:
            return jsonify({'error': MESSAGES['error']['invalid_credentials']}), 401
        
        # Generate JWT token
        payload = {
            'email': user['email'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        }
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            'message': MESSAGES['success']['login'],
            'token': token,
            'user': {
                'email': user['email'],
                'name': user['name'],
                'role': user['role']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(API_ENDPOINTS['auth']['register'], methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        role = data.get('role', 'student')
        
        if not email or not password or not name:
            return jsonify({'error': MESSAGES['error']['missing_fields']}), 400
        
        success = db.create_user(email, password, name, role)
        if not success:
            return jsonify({'error': MESSAGES['error']['user_exists']}), 409
        
        return jsonify({'message': MESSAGES['success']['registration']}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== USER ENDPOINTS ====================

@app.route(API_ENDPOINTS['users']['profile'], methods=['GET'])
@auth_required
def get_user_profile():
    """Get current user's profile"""
    try:
        user = db.get_user_by_email(request.current_user['email'])
        if not user:
            return jsonify({'error': MESSAGES['error']['user_not_found']}), 404
        
        return jsonify({
            'user': {
                'email': user['email'],
                'name': user['name'],
                'role': user['role']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== EVENTS ENDPOINTS ====================

@app.route(API_ENDPOINTS['events']['all'], methods=['GET'])
@auth_required
def get_all_events():
    """Get all events based on filters"""
    try:
        # Get query parameters
        section = request.args.get('section')
        instructor = request.args.get('instructor')
        day = request.args.get('day')
        campus = request.args.get('campus')
        
        # Get all events
        events = convert_schedule_to_events()
        
        # Apply filters
        if section:
            events = [e for e in events if e.get('section') == section]
        if instructor:
            events = [e for e in events if instructor.lower() in e.get('instructor', '').lower()]
        if day:
            events = [e for e in events if e.get('day_of_week').lower() == day.lower()]
        if campus:
            events = [e for e in events if e.get('campus') == campus]
        
        return jsonify({
            'events': events,
            'total': len(events),
            'filters_applied': {
                'section': section,
                'instructor': instructor,
                'day': day,
                'campus': campus
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(API_ENDPOINTS['events']['by_id'], methods=['GET'])
@auth_required
def get_event_by_id(event_id):
    """Get specific event by ID"""
    try:
        events = convert_schedule_to_events()
        event = next((e for e in events if e['id'] == event_id), None)
        
        if not event:
            return jsonify({'error': MESSAGES['error']['event_not_found']}), 404
        
        return jsonify({'event': event}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== TEACHER ENDPOINTS ====================

@app.route(API_ENDPOINTS['users']['teacher_schedule'], methods=['GET'])
@auth_required
def get_teacher_schedule(teacher_id):
    """Get schedule for a specific teacher"""
    try:
        events = convert_schedule_to_events()
        teacher_events = [e for e in events if teacher_id.lower() in e.get('instructor', '').lower()]
        
        return jsonify({
            'teacher_id': teacher_id,
            'events': teacher_events,
            'total_classes': len(teacher_events)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== SYSTEM ENDPOINTS ====================

@app.route(API_ENDPOINTS['system']['status'], methods=['GET'])
def get_status():
    """Get API status and system information"""
    try:
        return jsonify({
            'status': 'operational',
            'api_version': 'v1',
            'system_version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'total_sections': 72,
            'total_events': len(convert_schedule_to_events()),
            'database_status': 'connected',
            'ai_pipeline_status': 'active'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(API_ENDPOINTS['system']['root'], methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Smart Timetable System API',
        'version': 'v1',
        'status': 'operational',
        'endpoints': {
            'authentication': '/api/auth/*',
            'events': '/api/events/*',
            'users': '/api/users/*',
            'status': '/api/status'
        },
        'documentation': 'See API_DOCUMENTATION.md for complete endpoint details'
    }), 200

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': MESSAGES['error']['server_error']}), 500

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': MESSAGES['error']['unauthorized']}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({'error': MESSAGES['error']['forbidden']}), 403

# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    print("🌐 Starting Smart Timetable API Server for Deployment...")
    print(f"📊 System loaded with {len(convert_schedule_to_events())} events")
    print(f"🔐 Authentication enabled with JWT")
    print(f"🏗️ CORS configured for cross-origin requests")
    
    # Use environment variable for port, default to 8000
    port = int(os.getenv('PORT', PORTAL_PORTS['api']))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"🚀 API Server starting on {host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=DEBUG_MODE,
        threaded=True
    )