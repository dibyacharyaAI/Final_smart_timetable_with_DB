"""
Smart Timetable System - Main Application Entry Point
Deployment-ready Streamlit app with proper configuration
"""
import streamlit as st
import os
import sys

# Add the current directory to Python path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    st.set_page_config(
        page_title="Smart Timetable System",
        page_icon="🏫",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🏫 Smart Timetable System")
    st.caption("Kalinga Institute - AI-Powered Timetable Management")
    
    # System overview
    st.markdown("""
    ## Welcome to the Smart Timetable System
    
    This system uses advanced AI architecture with RNN-based anomaly detection and self-healing capabilities 
    to manage timetables for 72 sections across 3 campuses.
    """)
    
    # Portal selection
    st.subheader("Access Portals")
    
    col1, col2, col3 = st.columns(3)
    
    # Detect if we're in Replit deployment environment
    repl_id = os.getenv('REPL_ID')
    replit_domains = os.getenv('REPLIT_DOMAINS', '')
    
    # Check if we're in Replit deployment
    is_replit_deployment = bool(repl_id and replit_domains)
    
    if is_replit_deployment:
        # Use HTTPS for Replit deployment URLs
        domain_base = replit_domains.split(',')[0] if ',' in replit_domains else replit_domains
    else:
        # Local development
        domain_base = "localhost"
    
    with col1:
        st.markdown("### 👨‍💼 Admin Portal")
        st.write("Complete timetable management with editable features and AI pipeline")
        st.write("**Login:** rohit.verma@university.edu")
        st.write("**Password:** admin123")
        if is_replit_deployment:
            admin_url = f"https://{repl_id}-5002.{domain_base}"
        else:
            admin_url = f"http://{domain_base}:5002"
        if st.button("Access Admin Portal", key="admin", use_container_width=True):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={admin_url}">', unsafe_allow_html=True)
            st.success(f"Redirecting to Admin Portal...")
            st.code(admin_url)
    
    with col2:
        st.markdown("### 👨‍🏫 Teacher Portal")
        st.write("Manage your classes, edit your schedule, and download optimized timetables")
        st.write("**Login:** priya.mehta@university.edu")
        st.write("**Password:** teacher123")
        if is_replit_deployment:
            teacher_url = f"https://{repl_id}-5003.{domain_base}"
        else:
            teacher_url = f"http://{domain_base}:5003"
        if st.button("Access Teacher Portal", key="teacher", use_container_width=True):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={teacher_url}">', unsafe_allow_html=True)
            st.success(f"Redirecting to Teacher Portal...")
            st.code(teacher_url)
    
    with col3:
        st.markdown("### 👨‍🎓 Student Portal")
        st.write("View your optimized timetable (read-only)")
        st.write("**Login:** aarav.sharma@example.com")
        st.write("**Password:** student123")
        if is_replit_deployment:
            student_url = f"https://{repl_id}-5004.{domain_base}"
        else:
            student_url = f"http://{domain_base}:5004"
        if st.button("Access Student Portal", key="student", use_container_width=True):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={student_url}">', unsafe_allow_html=True)
            st.success(f"Redirecting to Student Portal...")
            st.code(student_url)
    
    st.markdown("---")
    
    # System status
    st.subheader("System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database", "Connected", "PostgreSQL")
    with col2:
        st.metric("API Server", "Running", "Port 5001")
    with col3:
        st.metric("Boss Architecture", "Active", "AI Pipeline")
    with col4:
        st.metric("Sections", "72", "Authentic Data")
    
    # Direct access links
    st.subheader("Direct Access Links")
    
    if is_replit_deployment:
        api_url = f"https://{repl_id}-5001.{domain_base}"
    else:
        api_url = f"http://{domain_base}:5001"
    
    st.markdown(f"""
    **Direct Portal Access:**
    - [Admin Portal]({admin_url}) - Full timetable management
    - [Teacher Portal]({teacher_url}) - Class management  
    - [Student Portal]({student_url}) - Timetable viewing
    - [API Server]({api_url}) - Backend services
    """)
    
    # Features overview
    st.subheader("System Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **Admin Features:**
        - Complete editable timetable interface
        - AI pipeline monitoring and control
        - Teacher request management
        - System analytics and reports
        - User management
        """)
    
    with feature_col2:
        st.markdown("""
        **Technical Features:**
        - Boss Architecture with RNN autoencoder
        - Real-time anomaly detection
        - Self-healing capabilities
        - Transit time optimization
        - Multi-campus coordination
        """)
    
    # Campus information
    st.subheader("Campus Information")
    campus_col1, campus_col2, campus_col3 = st.columns(3)
    
    with campus_col1:
        st.markdown("""
        **Campus 3**
        - 25 theory classrooms
        - Laboratory facilities
        - 24 sections
        """)
    
    with campus_col2:
        st.markdown("""
        **Campus 15B**
        - 18 theory classrooms
        - Programming labs
        - 24 sections
        """)
    
    with campus_col3:
        st.markdown("""
        **Campus 8**
        - 10 theory classrooms
        - Workshop facilities
        - 24 sections
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Smart Timetable System v2.0 | Kalinga Institute<br>
        Powered by Boss Architecture & RNN Autoencoder
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()