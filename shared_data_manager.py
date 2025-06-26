"""
Shared Data Manager - Ensures consistent data across all portals
Single source of truth for Admin, Teacher, and Student portals
"""

from core_timetable_system import SmartTimetableSystem
from typing import Dict, List, Any, Optional

class SharedDataManager:
    """Singleton class to ensure all portals use the same data"""
    
    _instance = None
    _system = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedDataManager, cls).__new__(cls)
            cls._system = SmartTimetableSystem()
            # Generate schedule once for all portals
            cls._system.generate_complete_schedule()
        return cls._instance
    
    @classmethod
    def get_system(cls):
        """Get the shared system instance"""
        if cls._instance is None:
            cls()
        return cls._system
    
    @classmethod
    def get_campus_from_room(cls, room: str, subject: str = None) -> str:
        """Standardized campus mapping for all portals"""
        if 'C3' in room or 'T1' in room or 'T2' in room or 'Lab-02' in room:
            return 'Campus_3'
        elif 'C8' in room or 'Workshop' in room:
            return 'Campus_8'
        elif 'C15B' in room or 'Programming' in room:
            return 'Campus_15B'
        elif 'Stadium' in room or 'SY' in room:
            return 'Stadium'
        elif subject == 'Sports and Yoga':
            return 'Stadium'
        return 'N/A'
    
    @classmethod
    def get_section_schedule(cls, section_id: str) -> Dict:
        """Get schedule for a specific section - used by all portals"""
        system = cls.get_system()
        
        if hasattr(system, 'schedule') and system.schedule and section_id in system.schedule:
            section_schedule = system.schedule[section_id]
            schedule_data = []
            
            for time_slot, class_info in section_schedule.items():
                room = class_info.get('room', 'N/A')
                subject = class_info.get('subject', 'N/A')
                campus = cls.get_campus_from_room(room, subject)
                
                schedule_data.append({
                    'time_slot': time_slot,
                    'subject': subject,
                    'teacher': class_info.get('teacher', 'N/A'),
                    'room': room,
                    'activity_type': class_info.get('activity_type', 'Lecture'),
                    'campus': campus
                })
            
            return {'success': True, 'classes': schedule_data}
        else:
            return {'success': False, 'classes': []}
    
    @classmethod
    def get_teacher_schedule(cls, teacher_name: str, teacher_id: str = None) -> Dict:
        """Get schedule for a specific teacher - used by Teacher Portal"""
        system = cls.get_system()
        teacher_classes = []
        
        if hasattr(system, 'schedule') and system.schedule:
            for section_id, section_schedule in system.schedule.items():
                for time_slot, class_info in section_schedule.items():
                    teacher = class_info.get('teacher', '')
                    
                    # Match teacher by name or ID
                    if (teacher_name in teacher or 
                        (teacher_id and teacher_id in teacher)):
                        
                        room = class_info.get('room', 'N/A')
                        subject = class_info.get('subject', 'N/A')
                        campus = cls.get_campus_from_room(room, subject)
                        
                        teacher_classes.append({
                            'section_id': section_id,
                            'time_slot': time_slot,
                            'subject': subject,
                            'teacher': teacher,
                            'room': room,
                            'activity_type': class_info.get('activity_type', 'Lecture'),
                            'campus': campus
                        })
        
        return {'success': True, 'classes': teacher_classes}
    
    @classmethod
    def get_all_sections(cls) -> List[str]:
        """Get list of all available sections"""
        system = cls.get_system()
        if hasattr(system, 'schedule') and system.schedule:
            return list(system.schedule.keys())
        return []
    
    @classmethod
    def get_all_teachers(cls) -> List[str]:
        """Get list of all teachers"""
        system = cls.get_system()
        teachers = set()
        
        if hasattr(system, 'schedule') and system.schedule:
            for section_id, section_schedule in system.schedule.items():
                for time_slot, class_info in section_schedule.items():
                    teachers.add(class_info.get('teacher', 'N/A'))
        
        return sorted(list(teachers))
    
    @classmethod
    def get_all_subjects(cls) -> List[str]:
        """Get list of all subjects"""
        system = cls.get_system()
        subjects = set()
        
        if hasattr(system, 'schedule') and system.schedule:
            for section_id, section_schedule in system.schedule.items():
                for time_slot, class_info in section_schedule.items():
                    subjects.add(class_info.get('subject', 'N/A'))
        
        return sorted(list(subjects))
    
    @classmethod
    def get_campus_statistics(cls) -> Dict:
        """Get campus distribution statistics"""
        system = cls.get_system()
        campus_count = {'Campus_3': 0, 'Campus_8': 0, 'Campus_15B': 0, 'Stadium': 0, 'N/A': 0}
        total_classes = 0
        
        if hasattr(system, 'schedule') and system.schedule:
            for section_id, section_schedule in system.schedule.items():
                for time_slot, class_info in section_schedule.items():
                    room = class_info.get('room', 'N/A')
                    subject = class_info.get('subject', 'N/A')
                    campus = cls.get_campus_from_room(room, subject)
                    campus_count[campus] += 1
                    total_classes += 1
        
        return {
            'campus_distribution': campus_count,
            'total_classes': total_classes,
            'total_sections': len(system.schedule) if hasattr(system, 'schedule') and system.schedule else 0
        }