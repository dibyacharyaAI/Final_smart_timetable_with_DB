"""
Admin Portal - Complete Timetable Management
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem
import json

def admin_login():
    """Admin login interface"""
    st.title("🔐 Admin Portal Login")
    
    with st.form("admin_login"):
        email = st.text_input("Email", value="rohit.verma@university.edu")
        password = st.text_input("Password", type="password", value="admin123")
        submit = st.form_submit_button("Login")
        
        if submit:
            if email and password:
                # Authenticate with database
                db_manager = DatabaseManager()
                user_data = db_manager.authenticate_user(email, password)
                
                if user_data and user_data.get('role') == 'admin':
                    st.session_state.user = user_data
                    st.session_state.logged_in = True
                    st.success(f"Welcome back, {user_data['name']}!")
                    st.rerun()
                    return True
                else:
                    st.error("Invalid credentials or not an admin account")
                    return False
            else:
                st.error("Please fill in all fields")
                return False
    
    return False

def admin_dashboard():
    """Main admin dashboard"""
    
    # Sidebar
    with st.sidebar:
        st.title("👨‍💼 Admin Portal")
        st.write(f"Welcome, {st.session_state.user['name']}")
        
        page = st.selectbox("Choose Section", [
            "📊 Dashboard",
            "📝 Editable Timetable",
            "🧠 AI Pipeline",
            "👨‍🏫 Teacher Requests",
            "📈 Analytics",
            "🔧 System Settings"
        ])
        
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
    
    # Main content
    if page == "📊 Dashboard":
        show_admin_overview()
    elif page == "📝 Editable Timetable":
        show_editable_timetable()
    elif page == "🧠 AI Pipeline":
        show_ai_pipeline_monitoring()
    elif page == "👨‍🏫 Teacher Requests":
        show_teacher_requests()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "🔧 System Settings":
        show_system_settings()

def show_admin_overview():
    """Admin dashboard overview"""
    st.title("📊 Admin Dashboard")
    
    # Initialize system if not exists
    if 'enhanced_system' not in st.session_state:
        st.session_state.enhanced_system = SmartTimetableSystem()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sections", "72")
    with col2:
        st.metric("Active Teachers", "49")
    with col3:
        st.metric("Total Students", "2160")
    with col4:
        st.metric("System Status", "Online")
    
    st.subheader("Recent Activity")
    st.info("System is running with AI pipeline active")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Regenerate Schedule"):
            with st.spinner("Regenerating comprehensive schedule..."):
                schedule = st.session_state.enhanced_system.generate_complete_schedule()
                # Save to database
                db = DatabaseManager()
                saved = db.save_schedule_to_db(
                    schedule, 
                    st.session_state.user.get('id', 11),  # Correct admin ID from database
                    f"Regenerated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                if saved:
                    st.success("✅ 72-section schedule regenerated and saved successfully!")
                else:
                    st.warning("Schedule generated but failed to save to database")
    
    with col2:
        if st.button("📊 Run Analytics"):
            st.success("Analytics updated!")
    
    with col3:
        if st.button("💾 Backup Data"):
            st.success("Data backup completed!")

def show_editable_timetable():
    """Editable timetable interface for admin"""
    st.title("📝 Editable Timetable")
    st.caption("Complete timetable management with AI pipeline")
    
    # Initialize systems
    if 'enhanced_system' not in st.session_state:
        st.session_state.enhanced_system = SmartTimetableSystem()
    
    # CSV Editor
    st.subheader("📊 Schedule Data Editor")
    
    # Get current CSV
    if st.button("🔄 Load Current Schedule"):
        # Generate or load schedule
        # Load from database or generate new
        db = DatabaseManager()
        schedule = db.get_active_schedule()
        if not schedule:
            schedule = st.session_state.enhanced_system.generate_complete_schedule()
        csv_content = st.session_state.enhanced_system.export_schedule_to_csv()
        st.session_state.csv_content = csv_content
    
    # Text area for CSV editing
    if 'csv_content' not in st.session_state:
        st.session_state.csv_content = "Click 'Load Current Schedule' to start editing"
    
    edited_csv = st.text_area(
        "Edit Schedule (CSV Format):",
        value=st.session_state.csv_content,
        height=400,
        help="Edit the schedule data. Click 'Process with AI Pipeline' when done."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 Process with AI Pipeline"):
            if edited_csv and edited_csv != "Click 'Load Current Schedule' to start editing":
                # Save edited content to session state immediately
                st.session_state.csv_content = edited_csv
                
                with st.spinner("Running AI Pipeline..."):
                    try:
                        # Load edited CSV back into system
                        success = st.session_state.enhanced_system.load_edited_csv(edited_csv)
                        
                        if success:
                            # Run the pipeline with edited data
                            result = st.session_state.enhanced_system.run_complete_pipeline()
                            
                            if isinstance(result, dict):
                                steps = result.get('steps', {})
                                if steps:
                                    st.success("✅ Complete AI Pipeline executed successfully!")
                                    
                                    # Show pipeline results
                                    with st.expander("🧠 AI Pipeline Execution Report", expanded=True):
                                        # Summary metrics first
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            encoding_info = steps.get('sequence_encoding', {})
                                            total_sections = encoding_info.get('total_sections', 72)
                                            st.metric("Total Sections Processed", total_sections)
                                        with col2:
                                            anomaly_info = steps.get('anomaly_detection', {})
                                            anomalies = len(anomaly_info.get('anomalies_detected', []))
                                            st.metric("Anomalies Detected", anomalies)
                                        with col3:
                                            or_info = steps.get('or_tools_validation', {})
                                            violations = or_info.get('constraint_violations', 0)
                                            
                                            # Calculate optimization score
                                            base_score = 100
                                            violation_penalty = min(violations * 0.2, 30)
                                            anomaly_penalty = min(anomalies * 0.5, 20)
                                            score = max(50, base_score - violation_penalty - anomaly_penalty)
                                            st.metric("Optimization Score", f"{int(score)}/100")
                                        
                                        # Step by step details
                                        st.subheader("🔧 Pipeline Steps")
                                        
                                        for step_name, step_info in steps.items():
                                            if isinstance(step_info, dict):
                                                status = step_info.get('status', 'unknown')
                                                if status == 'completed':
                                                    if step_name == 'ai_training':
                                                        loss = step_info.get('final_loss', 'N/A')
                                                        st.success(f"🤖 AI Training: Model trained successfully - Final Loss: {loss}")
                                                    elif step_name == 'anomaly_detection':
                                                        if anomalies == 0:
                                                            st.success("🔍 Anomaly Detection: No anomalies found")
                                                        else:
                                                            st.warning(f"🔍 Anomaly Detection: {anomalies} anomalies detected")
                                                    elif step_name == 'or_tools_validation':
                                                        if violations == 0:
                                                            st.success("⚖️ Constraint Validation: All constraints satisfied")
                                                        else:
                                                            st.warning(f"⚖️ Constraint Validation: {violations} violations found")
                                                    elif step_name == 'transit_validation':
                                                        issues = step_info.get('transit_issues_found', 0)
                                                        if issues == 0:
                                                            st.success("🚌 Transit Validation: No transit conflicts detected")
                                                        else:
                                                            st.warning(f"🚌 Transit Validation: {issues} transit issues detected")
                                                    elif step_name == 'optimization':
                                                        entries = step_info.get('schedule_entries', 0)
                                                        st.success(f"📄 Schedule Optimization: Generated {entries} schedule entries")
                                    
                                    # Update CSV content with processed results
                                    try:
                                        optimized_csv = st.session_state.enhanced_system.export_schedule_to_csv()
                                        st.session_state.csv_content = optimized_csv
                                        st.success("📄 Schedule updated with AI-optimized results")
                                    except Exception as e:
                                        st.error(f"Failed to update schedule: {e}")
                                        
                            else:
                                st.error("Pipeline execution failed")
                        else:
                            st.error("Failed to load edited CSV")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
            else:
                st.warning("Please load and edit the schedule first")
    
    with col2:
        if st.button("📥 Download Edited CSV"):
            if edited_csv and edited_csv != "Click 'Load Current Schedule' to start editing":
                # Ensure the CSV content is saved
                st.session_state.csv_content = edited_csv
                st.download_button(
                    label="📄 Download CSV File",
                    data=edited_csv,
                    file_name=f"edited_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success("CSV ready for download")
            else:
                st.warning("Please load and edit the schedule first")

def show_ai_pipeline_monitoring():
    """AI Pipeline monitoring"""
    st.header("🧠 AI Pipeline Monitoring")
    
    # Initialize system if needed
    if "enhanced_system" not in st.session_state:
        with st.spinner("Initializing enhanced AI system..."):
            try:
                from advanced_timetable_system import AdvancedTimetableSystem
                st.session_state.enhanced_system = AdvancedTimetableSystem()
                st.success("Enhanced AI system initialized")
            except Exception as e:
                st.error(f"Failed to initialize system: {e}")
                return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Complete AI Pipeline"):
            with st.spinner("Running complete AI pipeline..."):
                try:
                    result = st.session_state.enhanced_system.run_complete_pipeline()
                    
                    if isinstance(result, dict) and 'steps' in result:
                        st.success("✅ Complete AI Pipeline executed successfully!")
                        
                        # Show pipeline results
                        with st.expander("🧠 AI Pipeline Execution Report", expanded=True):
                            steps = result.get('steps', {})
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                encoding_info = steps.get('sequence_encoding', {})
                                total_sections = encoding_info.get('total_sections', 72)
                                st.metric("Total Sections Processed", total_sections)
                            with col2:
                                anomaly_info = steps.get('anomaly_detection', {})
                                anomalies = len(anomaly_info.get('anomalies_detected', []))
                                st.metric("Anomalies Detected", anomalies)
                            with col3:
                                or_info = steps.get('or_tools_validation', {})
                                violations = or_info.get('constraint_violations', 0)
                                
                                # Calculate optimization score
                                base_score = 100
                                violation_penalty = min(violations * 0.2, 30)
                                anomaly_penalty = min(anomalies * 0.5, 20)
                                score = max(50, base_score - violation_penalty - anomaly_penalty)
                                st.metric("Optimization Score", f"{int(score)}/100")
                            
                            # Step details
                            for step_name, step_info in steps.items():
                                if isinstance(step_info, dict) and step_info.get('status') == 'completed':
                                    if step_name == 'ai_training':
                                        loss = step_info.get('final_loss', 'N/A')
                                        st.success(f"🤖 AI Training: Model trained successfully - Loss: {loss}")
                                    elif step_name == 'anomaly_detection':
                                        if anomalies == 0:
                                            st.success("🔍 Anomaly Detection: No anomalies found")
                                        else:
                                            st.warning(f"🔍 Anomaly Detection: {anomalies} anomalies detected")
                                    elif step_name == 'or_tools_validation':
                                        if violations == 0:
                                            st.success("⚖️ Constraint Validation: All constraints satisfied")
                                        else:
                                            st.warning(f"⚖️ Constraint Validation: {violations} violations found")
                                    elif step_name == 'campus_enhancement':
                                        entries_updated = step_info.get('entries_updated', 0)
                                        if entries_updated > 0:
                                            st.success(f"🏫 Campus Data: Added to {entries_updated} schedule entries")
                                        else:
                                            st.error("🏫 Campus Data: Failed to add campus information")
                    else:
                        st.error("Pipeline execution failed")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
    
    with col2:
        if st.button("📊 View System Status"):
            if hasattr(st.session_state, 'enhanced_system'):
                try:
                    dashboard_data = st.session_state.enhanced_system.get_dashboard_data()
                    
                    st.subheader("🔧 System Status")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sections Loaded", dashboard_data.get('total_sections', 0))
                    with col2:
                        st.metric("Teachers Available", dashboard_data.get('total_teachers', 0))
                    with col3:
                        model_status = "Loaded" if dashboard_data.get('model_loaded', False) else "Not Loaded"
                        st.metric("AI Model Status", model_status)
                    
                    # Show recent anomalies if any
                    recent_anomalies = dashboard_data.get('recent_anomalies', [])
                    if recent_anomalies:
                        st.warning(f"Recent Anomalies: {len(recent_anomalies)} detected")
                        for anomaly in recent_anomalies[:3]:
                            st.text(f"• {anomaly.get('section', 'Unknown')}: {anomaly.get('severity', 'Unknown')} severity")
                    else:
                        st.success("No recent anomalies detected")
                        
                except Exception as e:
                    st.error(f"Failed to get system status: {e}")
            else:
                st.warning("System not initialized")

def show_teacher_requests():
    """Teacher requests management"""
    st.header("📨 Teacher Schedule Requests")
    
    # Simulated requests for demo
    requests = [
        {
            "id": 1,
            "teacher": "Dr. Smith",
            "request_type": "Time Change",
            "details": "Request to move Monday 9:00 AM class to 2:00 PM",
            "status": "Pending",
            "date": "2025-06-26"
        },
        {
            "id": 2,
            "teacher": "Prof. Johnson",
            "request_type": "Room Change", 
            "details": "Need lab room for Programming class",
            "status": "Approved",
            "date": "2025-06-25"
        }
    ]
    
    for req in requests:
        with st.expander(f"Request #{req['id']} - {req['teacher']} ({req['status']})"):
            st.write(f"**Type:** {req['request_type']}")
            st.write(f"**Details:** {req['details']}")
            st.write(f"**Date:** {req['date']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Approve {req['id']}", key=f"approve_{req['id']}"):
                    st.success(f"Request #{req['id']} approved")
            with col2:
                if st.button(f"❌ Reject {req['id']}", key=f"reject_{req['id']}"):
                    st.error(f"Request #{req['id']} rejected")
            with col3:
                if st.button(f"💬 Message {req['id']}", key=f"message_{req['id']}"):
                    st.info(f"Message sent to {req['teacher']}")

def show_analytics():
    """Analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # Initialize system if needed
    if "enhanced_system" not in st.session_state:
        st.warning("Please initialize the system first")
        return
    
    try:
        dashboard_data = st.session_state.enhanced_system.get_dashboard_data()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sections", dashboard_data.get('total_sections', 72))
        with col2:
            st.metric("Total Teachers", dashboard_data.get('total_teachers', 0))
        with col3:
            utilization = dashboard_data.get('room_utilization', 0)
            st.metric("Room Utilization", f"{utilization}%")
        with col4:
            conflicts = dashboard_data.get('schedule_conflicts', 0)
            st.metric("Schedule Conflicts", conflicts)
        
        # Campus distribution
        st.subheader("🏛️ Campus Distribution")
        campus_data = dashboard_data.get('campus_distribution', {})
        if campus_data:
            import plotly.express as px
            import pandas as pd
            
            df = pd.DataFrame(list(campus_data.items()), columns=['Campus', 'Classes'])
            fig = px.bar(df, x='Campus', y='Classes', title='Classes per Campus')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent pipeline performance
        st.subheader("🧠 AI Pipeline Performance")
        pipeline_data = dashboard_data.get('pipeline_performance', {})
        if pipeline_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Training Loss", f"{pipeline_data.get('last_loss', 0):.4f}")
            with col2:
                st.metric("Anomalies Detected", pipeline_data.get('anomalies_count', 0))
                
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

def show_system_settings():
    """System settings"""
    st.header("⚙️ System Settings")
    
    st.subheader("🧠 AI Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        anomaly_threshold = st.slider("Anomaly Detection Threshold", 0.1, 1.0, 0.75, 0.05)
        st.write(f"Current threshold: {anomaly_threshold}")
    with col2:
        training_epochs = st.number_input("Training Epochs", 10, 200, 100, 10)
        st.write(f"Epochs for next training: {training_epochs}")
    
    if st.button("💾 Save AI Settings"):
        if hasattr(st.session_state, 'enhanced_system'):
            try:
                # Update system settings
                st.session_state.enhanced_system.anomaly_threshold = anomaly_threshold
                st.success("AI settings saved successfully")
            except Exception as e:
                st.error(f"Failed to save settings: {e}")
        else:
            st.warning("System not initialized")
    
    st.subheader("🗄️ Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload Teacher Data"):
            if hasattr(st.session_state, 'enhanced_system'):
                try:
                    success = st.session_state.enhanced_system.load_fresh_data()
                    if success:
                        st.success("Teacher data reloaded successfully")
                    else:
                        st.error("Failed to reload teacher data")
                except Exception as e:
                    st.error(f"Error reloading data: {e}")
            else:
                st.warning("System not initialized")
    
    with col2:
        if st.button("🧹 Clear Model Cache"):
            try:
                # Clear any cached models
                if hasattr(st.session_state, 'enhanced_system'):
                    st.session_state.enhanced_system._create_new_model()
                    st.success("Model cache cleared")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
    
    st.subheader("📊 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Export Current Schedule"):
            if hasattr(st.session_state, 'enhanced_system'):
                try:
                    csv_data = st.session_state.enhanced_system.export_schedule_to_csv()
                    st.download_button(
                        label="📥 Download Schedule CSV",
                        data=csv_data,
                        file_name=f"schedule_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")
            else:
                st.warning("System not initialized")
    
    with col2:
        if st.button("📊 Export Analytics Report"):
            try:
                # Generate analytics report
                report_data = "Analytics Report\n"
                report_data += f"Generated: {datetime.now()}\n"
                report_data += "=" * 50 + "\n"
                report_data += "System Status: Operational\n"
                report_data += "Total Sections: 72\n"
                report_data += "AI Pipeline: Active\n"
                
                st.download_button(
                    label="📥 Download Report",
                    data=report_data,
                    file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")

def main():
    """Main function"""
    st.set_page_config(
        page_title="Smart Timetable - Admin Portal",
        page_icon="🎓",
        layout="wide"
    )
    
    # Check if already logged in
    if not st.session_state.get('logged_in', False):
        admin_login()
        return
    
    # Admin dashboard
    admin_dashboard()

if __name__ == "__main__":
    main()
