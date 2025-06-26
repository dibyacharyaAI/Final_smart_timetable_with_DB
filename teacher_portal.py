"""
Teacher Portal - Authentic Teacher Data Integration
Uses teacher_data.csv for signup and personalized schedule management
"""
import streamlit as st
import pandas as pd
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem
import json
import os
import random

def load_teacher_data():
    """Load authentic teacher data from CSV"""
    try:
        if os.path.exists('data/teacher_data.csv'):
            return pd.read_csv('data/teacher_data.csv')
        else:
            st.error("Teacher data file not found!")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading teacher data: {e}")
        return pd.DataFrame()

def assign_demo_classes_to_teacher(system, teacher_id, teacher_name, teacher_expertise):
    """Assign demo classes to teacher based on their expertise"""
    try:
        assigned_count = 0
        preferred_campus = st.session_state.user_data.get('preferred_campus', 'Any')
        
        # Define subject mapping based on expertise
        subject_mapping = {
            'Transform and Numerical Methods': ['Mathematics', 'Transform and Numerical Methods', 'Numerical Analysis'],
            'Chemistry': ['Chemistry', 'Applied Chemistry', 'Engineering Chemistry'],
            'English': ['English', 'Technical English', 'Communication Skills'],
            'Basic Electronics': ['Electronics', 'Basic Electronics', 'Digital Electronics'],
            'Physics': ['Physics', 'Applied Physics', 'Engineering Physics'],
            'Mechanical': ['Mechanical Engineering', 'Thermodynamics', 'Fluid Mechanics'],
            'Computer Science': ['Programming', 'Data Structures', 'Computer Networks']
        }
        
        # Get relevant subjects for this teacher
        relevant_subjects = []
        for expertise, subjects in subject_mapping.items():
            if expertise in teacher_expertise or teacher_expertise in expertise:
                relevant_subjects.extend(subjects)
        
        if not relevant_subjects:
            relevant_subjects = [teacher_expertise]
        
        # Time slots for assignment
        time_slots = ['Monday_8', 'Monday_10', 'Tuesday_9', 'Wednesday_11', 'Thursday_8', 'Friday_10']
        sections_to_assign = ['A1', 'A2', 'B1', 'B2']
        
        rooms = ['T01', 'T02', 'T03', 'L01', 'L02']
        campuses = ['Campus_3', 'Campus_8', 'Campus_15B']
        
        # Assign classes
        for i, section in enumerate(sections_to_assign[:4]):  # Limit to 4 sections
            if section in system.schedule and assigned_count < 6:
                time_slot = time_slots[assigned_count % len(time_slots)]
                subject = relevant_subjects[assigned_count % len(relevant_subjects)]
                room = rooms[assigned_count % len(rooms)]
                campus = campuses[assigned_count % len(campuses)] if preferred_campus == 'Any' else preferred_campus
                
                # Create assignment
                system.schedule[section][time_slot] = {
                    'teacher': teacher_name,
                    'teacher_id': teacher_id,
                    'subject': subject,
                    'room': room,
                    'campus': campus,
                    'activity_type': 'theory',
                    'building': f'Building-{campus.split("_")[1]}',
                    'walking_distance': '0m' if campus == 'Campus_3' else '550m'
                }
                assigned_count += 1
        
        return assigned_count
        
    except Exception as e:
        st.error(f"Error assigning demo classes: {e}")
        return 0

def teacher_login():
    """Teacher login and signup interface with authentic teacher data"""
    st.title("👨‍🏫 Teacher Portal")
    
    # Load authentic teacher data
    teacher_df = load_teacher_data()
    
    if teacher_df.empty:
        st.error("Could not load teacher database. Please contact administrator.")
        return
    
    # Login/Signup tabs
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("teacher_login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if email and password:
                    # Authenticate with database
                    db_manager = DatabaseManager()
                    user_data = db_manager.authenticate_user(email, password)
                    
                    if user_data and user_data.get('role') == 'teacher':
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        st.session_state.user_role = 'teacher'
                        st.success(f"Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials! Please sign up first.")
                else:
                    st.error("Please fill in all fields.")
    
    with tab2:
        st.subheader("Register Your Account")
        st.info("Select your profile from authentic teacher database:")
        
        # Show teacher selection popup
        teacher_options = []
        for _, teacher in teacher_df.iterrows():
            option = f"{teacher['Name']} ({teacher['TeacherID']}) - {teacher['SubjectExpertise']} - {teacher['Department']}"
            teacher_options.append(option)
        
        selected_teacher = st.selectbox(
            "Choose Your Teacher Profile:",
            options=["Select Teacher..."] + teacher_options,
            help="Select your profile from the authentic teacher database"
        )
        
        if selected_teacher and selected_teacher != "Select Teacher...":
            # Extract teacher ID from selection
            try:
                teacher_id = selected_teacher.split('(')[1].split(')')[0]
                selected_teacher_data = teacher_df[teacher_df['TeacherID'] == teacher_id].iloc[0]
            except Exception as e:
                st.error(f"Error processing teacher selection: {e}")
                return
            
            # Display selected teacher info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Selected Teacher Profile:**")
                st.write(f"**Name:** {selected_teacher_data['Name']}")
                st.write(f"**Teacher ID:** {selected_teacher_data['TeacherID']}")
                st.write(f"**Department:** {selected_teacher_data['Department']}")
            
            with col2:
                st.write(f"**Subject Expertise:** {selected_teacher_data['SubjectExpertise']}")
                st.write(f"**Preferred Campus:** {selected_teacher_data['PreferredCampus']}")
                st.write(f"**Max Sections/Day:** {selected_teacher_data['MaxSectionsPerDay']}")
            
            # Registration form
            with st.form("teacher_signup"):
                st.write("**Create Account Credentials:**")
                email = st.text_input("Email Address", value=f"{selected_teacher_data['TeacherID'].lower()}@kalinga.edu")
                password = st.text_input("Create Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Create Account")
                
                if submit:
                    if password and confirm_password and email:
                        if password == confirm_password:
                            # Create teacher account
                            teacher_account = {
                                'teacher_id': selected_teacher_data['TeacherID'],
                                'name': selected_teacher_data['Name'],
                                'email': email,
                                'password': password,
                                'subject_expertise': selected_teacher_data['SubjectExpertise'],
                                'department': selected_teacher_data['Department'],
                                'preferred_campus': selected_teacher_data['PreferredCampus'],
                                'max_sections_per_day': selected_teacher_data['MaxSectionsPerDay'],
                                'availability_slots': selected_teacher_data['AvailabilitySlots']
                            }
                            
                            # Register in database
                            db_manager = DatabaseManager()
                            try:
                                user_id = db_manager.register_user(
                                    email=email,
                                    password=password,
                                    role='teacher',
                                    name=selected_teacher_data['Name'],
                                    teacher_id=selected_teacher_data['TeacherID'],
                                    department=selected_teacher_data['Department']
                                )
                                
                                if user_id:
                                    st.success(f"Account created successfully for {selected_teacher_data['Name']}! Please login now.")
                                    # Also store in session state for backward compatibility
                                    if 'registered_teachers' not in st.session_state:
                                        st.session_state.registered_teachers = []
                                    st.session_state.registered_teachers.append(teacher_account)
                                else:
                                    st.error("Registration failed: Email might already exist")
                            except Exception as e:
                                st.error(f"Registration error: {e}")
                        else:
                            st.error("Passwords don't match!")
                    else:
                        st.error("Please fill in all fields.")

def teacher_dashboard():
    """Teacher dashboard with personalized schedule management"""
    user_data = st.session_state.user_data
    teacher_name = user_data.get('name', 'Teacher')
    st.title(f"👨‍🏫 Welcome, {teacher_name}")
    
    # Teacher profile sidebar
    with st.sidebar:
        st.subheader("Your Profile")
        st.write(f"**Name:** {teacher_name}")
        st.write(f"**Teacher ID:** {user_data.get('teacher_id', 'N/A')}")
        st.write(f"**Department:** {user_data.get('department', 'N/A')}")
        st.write(f"**Email:** {user_data.get('email', 'N/A')}")
        st.write(f"**Role:** {user_data.get('role', 'teacher')}")
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.rerun()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 My Schedule", "✏️ Edit Schedule", "🤖 AI Pipeline", "📈 Analytics"])
    
    with tab1:
        show_teacher_schedule()
    
    with tab2:
        show_editable_teacher_schedule()
    
    with tab3:
        show_teacher_ai_pipeline()
    
    with tab4:
        show_teacher_analytics()

def show_teacher_schedule():
    """Display teacher's personal schedule"""
    st.subheader("📅 Your Current Schedule")
    
    user_data = st.session_state.user_data
    teacher_id = user_data.get('teacher_id', 'Unknown')
    teacher_name = user_data.get('name', 'Unknown')
    
    # Initialize timetable system
    if 'timetable_system' not in st.session_state:
        st.session_state.timetable_system = SmartTimetableSystem()
    
    system = st.session_state.timetable_system
    
    # Try to get schedule data from different possible sources  
    import pandas as pd
    schedule_data = None
    
    # Try to get schedule data from available sources
    try:
        # First try kalinga_schedule if available
        if hasattr(system, 'kalinga_schedule') and system.kalinga_schedule:
            schedule_data = pd.DataFrame(system.kalinga_schedule)
        # Otherwise try to generate schedule
        elif hasattr(system, 'generate_complete_schedule'):
            result = system.generate_complete_schedule()
            if result.get('success') and result.get('schedule_data'):
                schedule_data = pd.DataFrame(result['schedule_data'])
    except Exception as e:
        st.error(f"Error accessing schedule data: {e}")
        schedule_data = None
    
    # Fallback to schedule dict conversion
    if schedule_data is None and hasattr(system, 'schedule') and system.schedule is not None:
        rows = []
        for section_id, section_data in system.schedule.items():
            for time_slot, slot_data in section_data.items():
                rows.append({
                    'SectionID': section_id,
                    'TimeSlot': time_slot,
                    'Subject': slot_data.get('subject', 'Unknown'),
                    'Teacher': slot_data.get('teacher', 'Unknown'),
                    'Room': slot_data.get('room', 'Unknown'),
                    'Type': slot_data.get('type', 'Unknown')
                })
        schedule_data = pd.DataFrame(rows)
    
    if schedule_data is None or len(schedule_data) == 0:
        st.warning("No schedule data available. Please contact administrator.")
        return
    
    # Filter schedule for this teacher
    teacher_schedule = {}
    total_classes = 0
    
    # Get teacher's subject expertise for better matching
    teacher_expertise = user_data.get('subject_expertise', '')
    
    # Filter schedule based on available data format
    if hasattr(system, 'schedule') and system.schedule:
        # Use schedule dict format
        for section_id, section_schedule in system.schedule.items():
            for time_slot, slot_data in section_schedule.items():
                # Multiple ways to match teacher
                teacher_match = (
                    slot_data.get('teacher_id') == teacher_id or 
                    slot_data.get('teacher') == teacher_name or
                    teacher_name in str(slot_data.get('teacher', '')) or
                    teacher_id in str(slot_data.get('teacher_id', '')) or
                    (teacher_expertise and teacher_expertise in str(slot_data.get('subject', '')))
                )
                
                if teacher_match:
                    if section_id not in teacher_schedule:
                        teacher_schedule[section_id] = {}
                    teacher_schedule[section_id][time_slot] = slot_data
                    total_classes += 1
    else:
        # Use DataFrame format
        teacher_classes = schedule_data[
            (schedule_data['Teacher'].str.contains(teacher_name, case=False, na=False)) |
            (schedule_data['Teacher'].str.contains(teacher_id, case=False, na=False)) |
            (teacher_expertise != '' and schedule_data['Subject'].str.contains(teacher_expertise, case=False, na=False))
        ]
        
        for _, row in teacher_classes.iterrows():
            section_id = row['SectionID']
            time_slot = row['TimeSlot']
            if section_id not in teacher_schedule:
                teacher_schedule[section_id] = {}
            teacher_schedule[section_id][time_slot] = {
                'subject': row['Subject'],
                'teacher': row['Teacher'],
                'room': row['Room'],
                'type': row['Type']
            }
            total_classes += 1
    
    if not teacher_schedule:
        st.info("No classes currently assigned to you.")
        
        # Offer to assign demo classes based on teacher's expertise
        if st.button("🎯 Assign Demo Classes Based on Your Expertise"):
            with st.spinner("Assigning classes based on your expertise..."):
                # Assign demo classes to teacher based on their subject expertise
                demo_assigned = assign_demo_classes_to_teacher(system, teacher_id, teacher_name, teacher_expertise)
                
                if demo_assigned > 0:
                    st.success(f"✅ Assigned {demo_assigned} demo classes! Refresh to see your schedule.")
                    st.rerun()
                else:
                    st.warning("Could not assign demo classes. Please contact administrator.")
        return
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Classes", total_classes)
    with col2:
        st.metric("Sections Teaching", len(teacher_schedule))
    with col3:
        weekly_hours = total_classes  # Assuming 1-hour slots
        st.metric("Weekly Hours", f"{weekly_hours}h")
    
    # Display schedule in tabular format
    st.subheader("📋 Weekly Schedule")
    
    # Create schedule table
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    time_slots = ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00']
    
    schedule_df = pd.DataFrame(index=time_slots, columns=days)
    
    for section_id, section_schedule in teacher_schedule.items():
        for time_slot, slot_data in section_schedule.items():
            parts = time_slot.split('_')
            if len(parts) >= 2:
                day = parts[0]
                hour = f"{int(parts[1]):02d}:00"
                
                if day in days and hour in time_slots:
                    cell_content = f"{slot_data.get('subject', 'Unknown')}\n{section_id}\n{slot_data.get('room', 'TBD')}"
                    schedule_df.loc[hour, day] = cell_content
    
    # Display the table
    st.dataframe(schedule_df.fillna(''), use_container_width=True, height=400)
    
    # Export option
    if st.button("📥 Download My Schedule"):
        csv_data = schedule_df.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"teacher_schedule_{teacher_id}.csv",
            mime="text/csv"
        )

def show_editable_teacher_schedule():
    """Show editable version of teacher's schedule"""
    st.subheader("✏️ Edit Your Schedule")
    
    teacher_id = st.session_state.user_data['teacher_id']
    teacher_name = st.session_state.user_data['name']
    
    # Initialize timetable system
    if 'timetable_system' not in st.session_state:
        st.session_state.timetable_system = SmartTimetableSystem()
    
    system = st.session_state.timetable_system
    
    # Get teacher's current schedule
    teacher_schedule_data = []
    teacher_expertise = st.session_state.user_data.get('subject_expertise', '')
    
    for section_id, section_schedule in system.schedule.items():
        for time_slot, slot_data in section_schedule.items():
            # Multiple ways to match teacher
            teacher_match = (
                slot_data.get('teacher_id') == teacher_id or 
                slot_data.get('teacher') == teacher_name or
                teacher_name in str(slot_data.get('teacher', '')) or
                teacher_id in str(slot_data.get('teacher_id', '')) or
                (teacher_expertise and teacher_expertise in str(slot_data.get('subject', '')))
            )
            
            if teacher_match:
                parts = time_slot.split('_')
                day = parts[0] if parts else 'Monday'
                time = f"{int(parts[1]):02d}:00" if len(parts) > 1 else '08:00'
                
                teacher_schedule_data.append({
                    'SectionID': section_id,
                    'Day': day,
                    'Time': time,
                    'Subject': slot_data.get('subject', ''),
                    'Room': slot_data.get('room', ''),
                    'Type': slot_data.get('activity_type', 'theory'),
                    'Campus': slot_data.get('campus', 'Campus_3'),
                    'Building': slot_data.get('building', ''),
                    'WalkingDistance': slot_data.get('walking_distance', '0m')
                })
    
    if not teacher_schedule_data:
        st.info("No classes assigned to you yet.")
        return
    
    # Convert to DataFrame for editing
    df = pd.DataFrame(teacher_schedule_data)
    
    st.write("**Your Current Schedule (Editable):**")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=400,
        key="teacher_schedule_editor"
    )
    
    # Save changes button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Changes"):
            # Convert edited DataFrame back to CSV format for processing
            csv_string = edited_df.to_csv(index=False)
            
            try:
                # Save changes to file
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"teacher_edited_schedule_{teacher_id}_{timestamp}.csv"
                
                with open(filename, 'w') as f:
                    f.write(csv_string)
                
                st.success("✅ Schedule updated successfully!")
                st.info("Changes saved to file for processing.")
                
                # Update session state
                st.session_state.last_edit_result = filename
                
            except Exception as e:
                st.error(f"Error saving changes: {e}")
    
    with col2:
        if st.button("🔄 Reset to Original"):
            st.rerun()

def show_teacher_ai_pipeline():
    """Show AI pipeline processing for teacher's schedule"""
    st.subheader("🤖 AI Pipeline & Optimization")
    
    teacher_id = st.session_state.user_data['teacher_id']
    teacher_name = st.session_state.user_data['name']
    
    # Initialize timetable system
    if 'timetable_system' not in st.session_state:
        st.session_state.timetable_system = SmartTimetableSystem()
    
    system = st.session_state.timetable_system
    
    # Show current teacher assignment status
    st.subheader("📊 Current Assignment Status")
    
    # Get teacher's current workload
    teacher_classes = 0
    teacher_sections = set()
    teacher_subjects = set()
    teacher_campuses = set()
    
    for section_id, section_schedule in system.schedule.items():
        for time_slot, slot_data in section_schedule.items():
            if (slot_data.get('teacher_id') == teacher_id or 
                slot_data.get('teacher') == teacher_name or
                teacher_name in slot_data.get('teacher', '')):
                
                teacher_classes += 1
                teacher_sections.add(section_id)
                teacher_subjects.add(slot_data.get('subject', 'Unknown'))
                teacher_campuses.add(slot_data.get('campus', 'Campus_3'))
    
    # Display current stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Classes", teacher_classes)
    with col2:
        st.metric("Sections", len(teacher_sections))
    with col3:
        st.metric("Subjects", len(teacher_subjects))
    with col4:
        st.metric("Campuses", len(teacher_campuses))
    
    # Calculate optimization score based on actual workload
    max_sections_per_day = st.session_state.user_data.get('max_sections_per_day', 4)
    daily_average = teacher_classes / 6 if teacher_classes > 0 else 0
    
    # Calculate optimization score
    if teacher_classes == 0:
        optimization_score = 0
        load_status = "No Classes Assigned"
    elif daily_average <= max_sections_per_day:
        optimization_score = min(100, int((teacher_classes / (max_sections_per_day * 6)) * 100))
        load_status = "Balanced"
    else:
        optimization_score = max(30, int(100 - ((daily_average - max_sections_per_day) * 20)))
        load_status = "Overloaded"
    
    # Run AI pipeline for teacher's schedule
    if st.button("🚀 Run AI Analysis on My Schedule"):
        with st.spinner("Running AI pipeline for your schedule..."):
            try:
                # Run complete pipeline
                pipeline_result = system.run_complete_pipeline()
                
                if pipeline_result.get('success', True):
                    st.success("✅ AI Pipeline completed successfully!")
                    
                    # Display results with calculated optimization score
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Optimization Score", f"{optimization_score}/100")
                        st.metric("Anomalies Detected", pipeline_result.get('anomalies_count', 0))
                    
                    with col2:
                        st.metric("Constraint Violations", pipeline_result.get('constraint_violations', 0))
                        st.metric("Teacher Load Balance", load_status)
                    
                    # Show detailed teacher analysis
                    st.subheader("📊 Your Schedule Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Current Workload:**")
                        st.write(f"- Total Classes: {teacher_classes}")
                        st.write(f"- Daily Average: {daily_average:.1f} classes")
                        st.write(f"- Recommended Max: {max_sections_per_day} classes/day")
                        st.write(f"- Weekly Load: {teacher_classes}/{max_sections_per_day * 6} classes")
                    
                    with col2:
                        st.write("**Distribution:**")
                        st.write(f"- Teaching Sections: {len(teacher_sections)}")
                        st.write(f"- Subject Variety: {len(teacher_subjects)}")
                        st.write(f"- Campus Coverage: {len(teacher_campuses)}")
                        
                        if teacher_subjects:
                            st.write("**Subjects Teaching:**")
                            for subject in list(teacher_subjects)[:3]:  # Show first 3
                                st.write(f"  • {subject}")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    if teacher_classes == 0:
                        st.warning("⚠️ No classes currently assigned. Contact admin for schedule allocation.")
                    elif daily_average > max_sections_per_day:
                        st.warning(f"⚠️ Workload exceeds recommended limit by {daily_average - max_sections_per_day:.1f} classes/day")
                        st.info("Consider requesting schedule redistribution or additional faculty support.")
                    elif len(teacher_campuses) > 2:
                        st.info("📍 Multi-campus assignment detected. Review transit times between classes.")
                    else:
                        st.success("✅ Your schedule appears well-balanced!")
                    
                    # Store pipeline results
                    st.session_state.pipeline_results = {
                        'optimization_score': optimization_score,
                        'load_status': load_status,
                        'teacher_classes': teacher_classes,
                        'daily_average': daily_average
                    }
                    
                else:
                    st.error("AI Pipeline encountered issues. Results may be incomplete.")
                    
            except Exception as e:
                st.error(f"Error running AI pipeline: {e}")
                st.info("Pipeline may still be processing. Try again in a moment.")
    
    # Show previous results if available
    if 'pipeline_results' in st.session_state:
        st.subheader("📈 Latest Pipeline Results")
        results = st.session_state.pipeline_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last Score", f"{results.get('optimization_score', 0)}/100")
        with col2:
            st.metric("Status", results.get('load_status', 'Unknown'))
        with col3:
            st.metric("Classes", results.get('teacher_classes', 0))

def show_teacher_analytics():
    """Show analytics for teacher's performance and schedule"""
    st.subheader("📈 Your Teaching Analytics")
    
    teacher_id = st.session_state.user_data['teacher_id']
    teacher_name = st.session_state.user_data['name']
    
    # Initialize timetable system
    if 'timetable_system' not in st.session_state:
        st.session_state.timetable_system = SmartTimetableSystem()
    
    system = st.session_state.timetable_system
    
    if not system.schedule:
        st.warning("No data available for analytics.")
        return
    
    # Analyze teacher's schedule
    teacher_stats = {
        'total_classes': 0,
        'sections': set(),
        'subjects': set(),
        'campuses': set(),
        'time_distribution': {},
        'day_distribution': {}
    }
    
    for section_id, section_schedule in system.schedule.items():
        for time_slot, slot_data in section_schedule.items():
            if (slot_data.get('teacher_id') == teacher_id or 
                slot_data.get('teacher') == teacher_name):
                
                teacher_stats['total_classes'] += 1
                teacher_stats['sections'].add(section_id)
                teacher_stats['subjects'].add(slot_data.get('subject', 'Unknown'))
                teacher_stats['campuses'].add(slot_data.get('campus', 'Campus_3'))
                
                # Time and day distribution
                parts = time_slot.split('_')
                if len(parts) >= 2:
                    day = parts[0]
                    hour = parts[1]
                    
                    teacher_stats['day_distribution'][day] = teacher_stats['day_distribution'].get(day, 0) + 1
                    teacher_stats['time_distribution'][hour] = teacher_stats['time_distribution'].get(hour, 0) + 1
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Classes", teacher_stats['total_classes'])
    with col2:
        st.metric("Sections Teaching", len(teacher_stats['sections']))
    with col3:
        st.metric("Subjects", len(teacher_stats['subjects']))
    with col4:
        st.metric("Campuses", len(teacher_stats['campuses']))
    
    # Workload Status
    max_classes_per_day = 6
    daily_average = teacher_stats['total_classes'] / 6 if teacher_stats['total_classes'] > 0 else 0
    
    st.subheader("📊 Workload Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate proper workload status
        if teacher_stats['total_classes'] == 0:
            status = "No Classes"
            status_color = "blue"
        elif daily_average <= 4:
            status = "Balanced"
            status_color = "green"
        elif daily_average <= 6:
            status = "Moderate"
            status_color = "orange"
        else:
            status = "Heavy Load"
            status_color = "red"
        
        st.metric("Daily Average", f"{daily_average:.1f} classes")
        st.metric("Load Status", status)
        
        # Show workload breakdown
        st.write("**Workload Breakdown:**")
        st.write(f"- Total weekly classes: {teacher_stats['total_classes']}")
        st.write(f"- Average per day: {daily_average:.1f}")
        st.write(f"- Recommended max: 4-6 classes/day")
    
    with col2:
        # Day-wise distribution
        if teacher_stats['day_distribution']:
            st.write("**Day-wise Distribution:**")
            for day, count in sorted(teacher_stats['day_distribution'].items()):
                st.write(f"- {day}: {count} classes")
        
        # Campus distribution
        if teacher_stats['campuses']:
            st.write("**Campus Distribution:**")
            for campus in sorted(teacher_stats['campuses']):
                st.write(f"- {campus}")
    
    # Subject breakdown
    if teacher_stats['subjects']:
        st.subheader("📚 Subject Portfolio")
        subjects_list = list(teacher_stats['subjects'])
        for subject in subjects_list:
            st.write(f"• {subject}")
    
    # Time distribution chart
    if teacher_stats['time_distribution']:
        st.subheader("⏰ Time Distribution")
        
        import plotly.express as px
        import pandas as pd
        
        # Convert to DataFrame for plotting
        time_df = pd.DataFrame(
            list(teacher_stats['time_distribution'].items()),
            columns=['Time', 'Classes']
        )
        
        fig = px.bar(time_df, x='Time', y='Classes', 
                     title='Classes by Time Slot')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if teacher_stats['total_classes'] == 0:
        st.info("No classes assigned. Contact administration for schedule allocation.")
    elif daily_average > 6:
        st.warning(f"Heavy workload detected ({daily_average:.1f} classes/day). Consider workload redistribution.")
    elif len(teacher_stats['campuses']) > 2:
        st.info("Multi-campus teaching detected. Review transit times between locations.")
    else:
        st.success("Your teaching load appears well-balanced.")

def main():
    """Main function for Teacher Portal"""
    st.set_page_config(
        page_title="Teacher Portal - Smart Timetable",
        page_icon="👨‍🏫",
        layout="wide"
    )
    
    # Initialize database
    try:
        db = DatabaseManager()
        db.create_tables()
        db.initialize_default_users()
    except Exception as e:
        st.error(f"Database initialization error: {e}")
    
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        teacher_login()
    else:
        teacher_dashboard()

if __name__ == "__main__":
    main()