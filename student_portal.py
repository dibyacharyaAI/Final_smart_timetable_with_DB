"""
Student Portal - Read-only Optimized Timetable View
"""
import streamlit as st
import pandas as pd
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem
from shared_data_manager import SharedDataManager

def student_login():
    """Student login interface"""
    st.title("👨‍🎓 Student Portal Login")
    
    with st.form("student_login"):
        email = st.text_input("Email", value="aarav.sharma@kalinga.edu")
        password = st.text_input("Password", type="password", value="student123")
        submit = st.form_submit_button("Login")
        
        if submit:
            try:
                db = DatabaseManager()
                user = db.authenticate_user(email, password)
                if user and user['role'] == 'student':
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or not a student account")
            except Exception as e:
                st.error(f"Database connection error: {e}")

def student_dashboard():
    """Main student dashboard"""
    st.set_page_config(page_title="Student Portal", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title("👨‍🎓 Student Portal")
        st.write(f"Welcome, {st.session_state.user['name']}")
        st.write(f"ID: {st.session_state.user['student_id']}")
        st.write(f"Section: {st.session_state.user['section']}")
        
        page = st.selectbox("Choose Section", [
            "📅 My Timetable",
            "📥 Download Schedule",
            "📊 Class Statistics",
            "🏫 Campus Info"
        ])
        
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
    
    # Main content - All read-only
    if page == "📅 My Timetable":
        show_student_timetable()
    elif page == "📥 Download Schedule":
        show_download_schedule()
    elif page == "📊 Class Statistics":
        show_class_statistics()
    elif page == "🏫 Campus Info":
        show_campus_info()

def show_student_timetable():
    """Display student's optimized timetable (read-only)"""
    st.title("📅 My Optimized Timetable")
    st.caption("✅ AI-optimized schedule with Boss Architecture")
    
    student_section = st.session_state.user['section']
    
    try:
        # Map student section to internal section ID format
        section_mapping = {
            'CSE-A1': 'A1', 'CSE-A2': 'A2', 'CSE-A3': 'A3', 'CSE-A4': 'A4',
            'CSE-B1': 'B1', 'CSE-B2': 'B2', 'CSE-B3': 'B3', 'CSE-B4': 'B4'
        }
        section_id = section_mapping.get(student_section, 'A1')
        
        # Use shared data manager for consistent data
        schedule_result = SharedDataManager.get_section_schedule(section_id)
        
        if schedule_result.get('success'):
            schedule = {'classes': schedule_result['classes']}
        else:
            schedule = None
        
        if schedule and 'classes' in schedule:
            st.success(f"📚 Schedule for Section: {student_section}")
            
            # Convert to readable format
            schedule_data = []
            for class_data in schedule['classes']:
                schedule_data.append({
                    'Time': class_data.get('time_slot', 'N/A'),
                    'Subject': class_data.get('subject', 'N/A'),
                    'Teacher': class_data.get('teacher', 'N/A'),
                    'Room': class_data.get('room', 'N/A'),
                    'Campus': class_data.get('campus', 'N/A'),
                    'Type': class_data.get('activity_type', 'N/A')
                })
            
            if schedule_data:
                # Create optimized timetable view
                st.subheader("📋 Optimized Weekly Schedule")
                
                # Display all classes in a single table
                df = pd.DataFrame(schedule_data)
                st.dataframe(
                    df[['Time', 'Subject', 'Teacher', 'Room', 'Campus', 'Type']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary
                st.subheader("📊 Weekly Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Classes", len(schedule_data))
                with col2:
                    theory_count = sum(1 for item in schedule_data if item['Type'].lower() in ['theory', 'lecture'])
                    st.metric("Theory Classes", theory_count)
                with col3:
                    lab_count = sum(1 for item in schedule_data if any(keyword in item['Type'].lower() for keyword in ['lab', 'workshop', 'practical']))
                    st.metric("Lab/Workshop Sessions", lab_count)
                with col4:
                    unique_subjects = len(set(item['Subject'] for item in schedule_data))
                    st.metric("Different Subjects", unique_subjects)
                
            else:
                st.info("No classes scheduled currently")
        else:
            st.warning(f"Schedule not found for section: {student_section}")
            
    except Exception as e:
        st.error(f"Error loading schedule: {e}")
        st.info("Please contact administrator if this issue persists")

def show_download_schedule():
    """Download schedule options for students"""
    st.title("📥 Download My Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Downloads")
        
        student_section = st.session_state.user['section']
        
        # Generate CSV content for student
        sample_schedule = f"""Day,Time,Subject,Teacher,Room,Type
Monday,09:00-10:00,Computer Science,Dr. Rajesh Kumar,Room-301,Theory
Monday,10:00-11:00,Mathematics,Prof. Priya Sharma,Room-302,Theory
Monday,11:20-12:20,Programming Lab,Dr. Amit Singh,Lab-CS1,Lab
Tuesday,09:00-10:00,Physics,Dr. Rajesh Kumar,Room-303,Theory
Tuesday,10:00-11:00,Chemistry,Prof. Priya Sharma,Lab-CH1,Lab
Wednesday,09:00-10:00,English,Dr. Amit Singh,Room-304,Theory
Wednesday,11:20-12:20,Sports,Coach,Stadium,Sports
Thursday,09:00-10:00,Computer Science,Dr. Rajesh Kumar,Room-301,Theory
Thursday,10:00-11:00,Mathematics,Prof. Priya Sharma,Room-302,Theory
Friday,09:00-10:00,Programming Lab,Dr. Amit Singh,Lab-CS2,Lab
"""
        
        st.download_button(
            label="📄 Download Timetable (CSV)",
            data=sample_schedule,
            file_name=f"my_timetable_{student_section}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📑 Download Timetable (Text)",
            data=sample_schedule.replace(',', '\t'),
            file_name=f"my_timetable_{student_section}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col2:
        st.subheader("📱 Quick Access")
        st.info("💡 **Tip:** Save your timetable to your phone for quick access!")
        
        st.subheader("🔄 Last Updated")
        st.write("Schedule last optimized: Today")
        st.write("Next update: Weekly")

def show_class_statistics():
    """Show class statistics for student"""
    st.title("📊 Class Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Weekly Distribution")
        
        # Sample weekly data
        weekly_data = {
            'Monday': 3,
            'Tuesday': 2, 
            'Wednesday': 2,
            'Thursday': 2,
            'Friday': 1
        }
        
        st.bar_chart(weekly_data)
        
        st.subheader("📚 Subject Distribution")
        subject_data = {
            'Computer Science': 3,
            'Mathematics': 2,
            'Physics': 2,
            'Chemistry': 1,
            'English': 1,
            'Programming Lab': 2,
            'Sports': 1
        }
        st.bar_chart(subject_data)
    
    with col2:
        st.subheader("📋 Summary")
        st.metric("Total Classes per Week", "12")
        st.metric("Theory Classes", "8")
        st.metric("Lab Sessions", "3")
        st.metric("Sports/Activities", "1")
        st.metric("Total Subjects", "7")
        
        st.subheader("⏰ Time Distribution")
        st.metric("Classes before 12:00", "6")
        st.metric("Classes after 12:00", "6")
        st.metric("Free Periods", "18")

def show_campus_info():
    """Show campus information"""
    st.title("🏫 Campus Information")
    st.caption("📍 Complete campus details and facilities")
    
    try:
        # Get campus statistics using shared data manager
        stats = SharedDataManager.get_campus_statistics()
        campus_count = stats['campus_distribution']
        total_classes = stats['total_classes']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Campus Distribution")
            
            # Show actual campus usage statistics
            for campus, count in campus_count.items():
                if campus != 'N/A' and count > 0:
                    percentage = (count/total_classes)*100 if total_classes > 0 else 0
                    st.metric(campus, f"{count} classes", f"{percentage:.1f}%")
            
            st.subheader("🏢 Campus Locations")
            
            campus_info = {
                "Campus_3": {
                    "Theory Rooms": 25,
                    "Labs": 5,
                    "Description": "Main academic building with theory rooms",
                    "Walking Distance": "0m (Main Campus)"
                },
                "Campus_8": {
                    "Theory Rooms": 10,
                    "Labs": 3,
                    "Description": "Secondary building with workshop facilities",
                    "Walking Distance": "550m (7 minutes)"
                },
                "Campus_15B": {
                    "Theory Rooms": 18,
                    "Labs": 8,
                    "Description": "Engineering building with programming labs",
                    "Walking Distance": "700m (10 minutes)"
                },
                "Stadium": {
                    "Theory Rooms": 0,
                    "Labs": 0,
                    "Description": "Sports and yoga facilities",
                    "Walking Distance": "900m (12 minutes)"
                }
            }
            
            for campus_id, info in campus_info.items():
                if campus_id in campus_count and campus_count[campus_id] > 0:
                    with st.expander(f"{campus_id} ({campus_count[campus_id]} classes)"):
                        st.write(f"**Description:** {info['Description']}")
                        st.write(f"**Walking Distance:** {info['Walking Distance']}")
                        if info['Theory Rooms'] > 0:
                            st.write(f"**Theory Rooms:** {info['Theory Rooms']}")
                        if info['Labs'] > 0:
                            st.write(f"**Labs:** {info['Labs']}")
        
        with col2:
            st.subheader("🚶‍♂️ Transit Information")
            st.info("Transit times between campuses are optimized in your schedule")
            
            # Show walking distances
            st.write("**Walking Distances from Main Campus:**")
            st.write("• Campus_3: 0m (Main Campus)")
            st.write("• Campus_8: 550m (7 minutes)")
            st.write("• Campus_15B: 700m (10 minutes)")
            st.write("• Stadium: 900m (12 minutes)")
            
            st.subheader("📍 Important Locations")
            locations = [
                "🏟️ Stadium - Sports & Yoga",
                "🍽️ Cafeteria - Campus_3",
                "📚 Library - Campus_15B",
                "🏥 Medical Center - Campus_8",
                "🔧 Workshop - Campus_8",
                "💻 Programming Lab - Campus_15B"
            ]
            
            for location in locations:
                st.write(f"• {location}")
                
    except Exception as e:
        st.error(f"Error loading campus information: {str(e)}")
        # Fallback static display
        st.subheader("🏢 Campus Locations")
        st.write("• Campus_3: Main academic building")
        st.write("• Campus_8: Secondary building with workshops")
        st.write("• Campus_15B: Engineering building")
        st.write("• Stadium: Sports facilities")

def main():
    """Main function"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        student_login()
    else:
        student_dashboard()

if __name__ == "__main__":
    main()