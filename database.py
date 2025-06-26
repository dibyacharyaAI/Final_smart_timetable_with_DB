"""
Database Models and Setup for Smart Timetable System
"""

import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)  # admin, teacher, student
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Additional profile fields for API compatibility (will be added via migration)
    # employee_id, profile_image, phone, address will be added when needed
    
    # Teacher specific fields
    teacher_id = Column(String(50), nullable=True)
    department = Column(String(100), nullable=True)
    
    # Student specific fields
    student_id = Column(String(50), nullable=True)
    section = Column(String(50), nullable=True)
    batch = Column(String(50), nullable=True)

class Schedule(Base):
    __tablename__ = 'schedules'
    
    id = Column(Integer, primary_key=True)
    section_id = Column(String(50), nullable=False)
    subject = Column(String(255), nullable=False)
    teacher_id = Column(String(50), nullable=False)
    room = Column(String(100), nullable=False)
    time_slot = Column(String(50), nullable=False)
    activity_type = Column(String(50), nullable=False)
    day_of_week = Column(String(20), nullable=False)
    
    # Additional fields for API compatibility (will be added via migration)
    # start_hour, end_hour, day_number, scheme, campus will be added when needed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'))

class TimetableVersion(Base):
    __tablename__ = 'timetable_versions'
    
    id = Column(Integer, primary_key=True)
    version_name = Column(String(255), nullable=False)
    schedule_data = Column(Text, nullable=False)  # JSON data
    created_by = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    pipeline_results = Column(Text, nullable=True)  # AI pipeline results

class TeacherRequest(Base):
    __tablename__ = 'teacher_requests'
    
    id = Column(Integer, primary_key=True)
    teacher_id = Column(String(50), nullable=False)
    request_type = Column(String(50), nullable=False)  # swap, change_room, etc.
    details = Column(Text, nullable=False)  # JSON details
    status = Column(String(20), default='pending')  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    admin_response = Column(Text, nullable=True)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
        
    def initialize_default_users(self):
        """Initialize default users for testing"""
        session = self.get_session()
        try:
            # Check if users already exist
            if session.query(User).count() > 0:
                return
                
            # Create default admin - matching app.py credentials
            admin = User(
                email="rohit.verma@university.edu",
                password="admin123",
                role="admin",
                name="Rohit Verma",
                department="Administration"
            )
            
            # Create authentic Kalinga teachers
            teachers = [
                User(email="dr.agarwal@kalinga.edu", password="teacher123", role="teacher", 
                     name="Dr. Agarwal", teacher_id="KT001", department="Computer Science"),
                User(email="prof.sharma@kalinga.edu", password="teacher123", role="teacher", 
                     name="Prof. Sharma", teacher_id="KT002", department="Mathematics"),
                User(email="dr.patel@kalinga.edu", password="teacher123", role="teacher", 
                     name="Dr. Patel", teacher_id="KT003", department="Chemistry")
            ]
            
            # Create sample students
            students = [
                User(email="aarav.sharma@kalinga.edu", password="student123", role="student", 
                     name="Aarav Sharma", student_id="S001", section="CSE-A1", batch="2024"),
                User(email="anjali.gupta@kalinga.edu", password="student123", role="student", 
                     name="Anjali Gupta", student_id="S002", section="CSE-A2", batch="2024"),
                User(email="vikram.singh@kalinga.edu", password="student123", role="student", 
                     name="Vikram Singh", student_id="S003", section="CSE-B1", batch="2024")
            ]
            
            session.add(admin)
            for teacher in teachers:
                session.add(teacher)
            for student in students:
                session.add(student)
                
            session.commit()
            print("✓ Default users created successfully")
            
        except Exception as e:
            session.rollback()
            print(f"Error creating default users: {e}")
        finally:
            session.close()
            
    def save_schedule_to_db(self, schedule_data, user_id, version_name="Current"):
        """Save schedule data to database"""
        session = self.get_session()
        try:
            # Deactivate previous versions
            session.query(TimetableVersion).update({'is_active': False})
            
            # Create new version
            new_version = TimetableVersion(
                version_name=version_name,
                schedule_data=json.dumps(schedule_data),
                created_by=user_id,
                is_active=True
            )
            session.add(new_version)
            session.commit()
            return new_version.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    def get_active_schedule(self):
        """Get active schedule from database"""
        session = self.get_session()
        try:
            version = session.query(TimetableVersion).filter_by(is_active=True).first()
            if version:
                return json.loads(version.schedule_data)
            return None
        finally:
            session.close()
            
    def authenticate_user(self, email, password):
        """Authenticate user with email and password"""
        session = self.get_session()
        try:
            user = session.query(User).filter_by(email=email, password=password, is_active=True).first()
            if user:
                return {
                    'id': user.id,
                    'email': user.email,
                    'role': user.role,
                    'name': user.name,
                    'teacher_id': user.teacher_id,
                    'student_id': user.student_id,
                    'department': user.department,
                    'section': user.section,
                    'batch': user.batch
                }
            return None
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
        finally:
            session.close()
    
    def register_user(self, email, password, role, name, **kwargs):
        """Register a new user in the database"""
        session = self.get_session()
        try:
            # Check if user already exists
            existing_user = session.query(User).filter_by(email=email).first()
            if existing_user:
                return {'success': False, 'message': 'User already exists with this email'}
            
            # Create new user
            new_user = User(
                email=email,
                password=password,
                role=role,
                name=name,
                teacher_id=kwargs.get('teacher_id'),
                department=kwargs.get('department'),
                student_id=kwargs.get('student_id'),
                section=kwargs.get('section'),
                batch=kwargs.get('batch')
            )
            
            session.add(new_user)
            session.commit()
            
            return {
                'success': True, 
                'message': 'User registered successfully',
                'user_id': new_user.id
            }
            
        except Exception as e:
            session.rollback()
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
        finally:
            session.close()
    
    def get_user_by_email(self, email):
        """Get user by email"""
        session = self.get_session()
        try:
            user = session.query(User).filter_by(email=email, is_active=True).first()
            if user:
                return {
                    'id': user.id,
                    'email': user.email,
                    'role': user.role,
                    'name': user.name,
                    'teacher_id': user.teacher_id,
                    'student_id': user.student_id,
                    'department': user.department,
                    'section': user.section,
                    'batch': user.batch
                }
            return None
        finally:
            session.close()
