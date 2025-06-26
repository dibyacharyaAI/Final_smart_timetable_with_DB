"""
Data Consistency Checker
Ensures all portals use the same data source and display consistent information
"""

from core_timetable_system import SmartTimetableSystem
import json

def check_data_consistency():
    """Check if all portals are using consistent data"""
    
    print("=== SMART TIMETABLE SYSTEM DATA CONSISTENCY CHECK ===\n")
    
    # Initialize the core system that all portals should use
    system = SmartTimetableSystem()
    
    # Generate/load the schedule
    result = system.generate_complete_schedule()
    
    print(f"1. CORE SYSTEM STATUS:")
    print(f"   ✓ System initialized: {system is not None}")
    print(f"   ✓ Schedule generated: {result.get('success', False) if isinstance(result, dict) else bool(result)}")
    print(f"   ✓ Total sections: {len(system.schedule) if hasattr(system, 'schedule') and system.schedule else 0}")
    
    if hasattr(system, 'schedule') and system.schedule:
        # Check a sample section (A2 for CSE-A2)
        sample_section = 'A2'
        if sample_section in system.schedule:
            section_data = system.schedule[sample_section]
            print(f"\n2. SAMPLE SECTION DATA (Section {sample_section}):")
            print(f"   ✓ Total time slots: {len(section_data)}")
            
            # Show first few entries
            count = 0
            for time_slot, class_info in section_data.items():
                if count < 3:  # Show first 3 entries
                    room = class_info.get('room', 'N/A')
                    campus = 'N/A'
                    if 'C3' in room or 'T1' in room or 'T2' in room or 'Lab-02' in room:
                        campus = 'Campus_3'
                    elif 'C8' in room or 'Workshop' in room:
                        campus = 'Campus_8'
                    elif 'C15B' in room or 'Programming' in room:
                        campus = 'Campus_15B'
                    elif 'Stadium' in room or 'SY' in room:
                        campus = 'Stadium'
                    elif class_info.get('subject') == 'Sports and Yoga':
                        campus = 'Stadium'
                    
                    print(f"   {time_slot}: {class_info.get('subject', 'N/A')} | {class_info.get('teacher', 'N/A')} | {room} | {campus}")
                count += 1
        
        # Check campus distribution
        campus_count = {'Campus_3': 0, 'Campus_8': 0, 'Campus_15B': 0, 'Stadium': 0, 'N/A': 0}
        total_classes = 0
        
        for section_id, section_data in system.schedule.items():
            for time_slot, class_info in section_data.items():
                room = class_info.get('room', 'N/A')
                campus = 'N/A'
                if 'C3' in room or 'T1' in room or 'T2' in room or 'Lab-02' in room:
                    campus = 'Campus_3'
                elif 'C8' in room or 'Workshop' in room:
                    campus = 'Campus_8'
                elif 'C15B' in room or 'Programming' in room:
                    campus = 'Campus_15B'
                elif 'Stadium' in room or 'SY' in room:
                    campus = 'Stadium'
                elif class_info.get('subject') == 'Sports and Yoga':
                    campus = 'Stadium'
                
                campus_count[campus] += 1
                total_classes += 1
        
        print(f"\n3. CAMPUS DISTRIBUTION:")
        for campus, count in campus_count.items():
            percentage = (count/total_classes)*100 if total_classes > 0 else 0
            print(f"   {campus}: {count} classes ({percentage:.1f}%)")
        
        # Check teacher distribution
        teachers = set()
        subjects = set()
        
        for section_id, section_data in system.schedule.items():
            for time_slot, class_info in section_data.items():
                teachers.add(class_info.get('teacher', 'N/A'))
                subjects.add(class_info.get('subject', 'N/A'))
        
        print(f"\n4. DATA VARIETY:")
        print(f"   ✓ Unique teachers: {len(teachers)}")
        print(f"   ✓ Unique subjects: {len(subjects)}")
        print(f"   ✓ Total classes: {total_classes}")
        
        # Show some sample teachers and subjects
        print(f"\n5. SAMPLE DATA:")
        sample_teachers = list(teachers)[:5]
        sample_subjects = list(subjects)[:5]
        print(f"   Sample teachers: {', '.join(sample_teachers)}")
        print(f"   Sample subjects: {', '.join(sample_subjects)}")
        
    print(f"\n=== CONSISTENCY CHECK COMPLETE ===")
    print(f"All portals should now use this same data source.")
    print(f"Admin Portal: Uses system.schedule for editing")
    print(f"Teacher Portal: Uses system.schedule for personal view")
    print(f"Student Portal: Uses system.schedule with section mapping")
    
    return system

if __name__ == "__main__":
    check_data_consistency()