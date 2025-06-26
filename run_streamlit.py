"""
Streamlit Runner for Smart Timetable System
Testing के लिए streamlit portals को run करने के लिए
"""

import streamlit as st
import subprocess
import sys
import os
from constants import PORTAL_PORTS, PORTAL_ADDRESSES

def main():
    """Main Streamlit runner interface"""
    st.set_page_config(
        page_title="Smart Timetable System - Portal Runner",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Smart Timetable System - Portal Runner")
    st.caption("Testing के लिए अलग-अलग portals को run करें")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Available Portals")
        
        portal_options = {
            "Main Portal": {
                "file": "app.py",
                "port": PORTAL_PORTS['main'],
                "description": "Central navigation hub"
            },
            "Admin Portal": {
                "file": "admin_portal.py", 
                "port": PORTAL_PORTS['admin'],
                "description": "Complete timetable management"
            },
            "Teacher Portal": {
                "file": "teacher_portal.py",
                "port": PORTAL_PORTS['teacher'], 
                "description": "Teacher schedule management"
            },
            "Student Portal": {
                "file": "student_portal.py",
                "port": PORTAL_PORTS['student'],
                "description": "Read-only student view"
            }
        }
        
        selected_portal = st.selectbox(
            "Portal चुनें:",
            options=list(portal_options.keys()),
            index=0
        )
        
        if selected_portal:
            portal_info = portal_options[selected_portal]
            st.info(f"**File**: {portal_info['file']}")
            st.info(f"**Port**: {portal_info['port']}")
            st.info(f"**Description**: {portal_info['description']}")
    
    with col2:
        st.subheader("⚙️ Portal Controls")
        
        if st.button("🚀 Start Selected Portal", type="primary"):
            if selected_portal:
                portal_info = portal_options[selected_portal]
                start_portal(portal_info['file'], portal_info['port'])
        
        st.divider()
        
        if st.button("🔥 Start All Portals"):
            start_all_portals(portal_options)
        
        if st.button("⚡ Quick Test - Main Portal"):
            start_portal("app.py", PORTAL_PORTS['main'])
            
        if st.button("🔧 Admin Portal Quick Start"):
            start_portal("admin_portal.py", PORTAL_PORTS['admin'])
    
    st.divider()
    
    # System Status
    st.subheader("📊 System Information")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Portals", "4")
        st.metric("API Server Port", PORTAL_PORTS['api'])
    
    with col4:
        st.metric("Host Address", PORTAL_ADDRESSES['host'])
        st.metric("Main Portal", PORTAL_PORTS['main'])
    
    with col5:
        st.metric("Database", "PostgreSQL")
        st.metric("Environment", "Development")
    
    # Instructions
    st.subheader("📋 Instructions")
    st.write("""
    **Portal Testing के लिए:**
    1. ऊपर से कोई portal select करें
    2. "Start Selected Portal" button दबाएं
    3. New tab में portal खुल जाएगा
    4. Login credentials:
       - **Admin**: rohit.verma@university.edu / admin123
       - **Teacher**: dr.agarwal@kalinga.edu / teacher123
       - **Student**: aarav.sharma@kalinga.edu / student123
    
    **सभी Portals एक साथ run करने के लिए:**
    - "Start All Portals" button का use करें
    
    **Note**: API Server अलग से चल रहा है port 8000 पर
    """)

def start_portal(file_name, port):
    """Start a specific portal"""
    try:
        if not os.path.exists(file_name):
            st.error(f"Error: {file_name} file नहीं मिली!")
            return
        
        command = [
            "streamlit", "run", file_name,
            "--server.port", str(port),
            "--server.address", PORTAL_ADDRESSES['host'],
            "--server.enableCORS", "false"
        ]
        
        st.success(f"🚀 Starting {file_name} on port {port}...")
        st.info(f"Portal URL: http://localhost:{port}")
        st.balloons()
        
        # Run in background
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except Exception as e:
        st.error(f"Error starting portal: {str(e)}")

def start_all_portals(portal_options):
    """Start all portals simultaneously"""
    try:
        st.success("🔥 Starting all portals...")
        
        for portal_name, portal_info in portal_options.items():
            file_name = portal_info['file']
            port = portal_info['port']
            
            if os.path.exists(file_name):
                command = [
                    "streamlit", "run", file_name,
                    "--server.port", str(port),
                    "--server.address", PORTAL_ADDRESSES['host'],
                    "--server.enableCORS", "false"
                ]
                
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.success(f"✅ {portal_name} started on port {port}")
            else:
                st.warning(f"⚠️ {file_name} not found, skipping {portal_name}")
        
        st.balloons()
        st.success("🎉 All available portals started successfully!")
        
        # Show portal URLs
        st.subheader("🔗 Portal URLs:")
        for portal_name, portal_info in portal_options.items():
            if os.path.exists(portal_info['file']):
                st.write(f"- **{portal_name}**: http://localhost:{portal_info['port']}")
                
    except Exception as e:
        st.error(f"Error starting portals: {str(e)}")

def check_portal_status():
    """Check if portals are running"""
    # This could be enhanced to actually check if ports are in use
    pass

if __name__ == "__main__":
    main()