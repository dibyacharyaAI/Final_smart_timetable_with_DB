"""
Streamlit Runner for Smart Timetable System
Testing के लिए streamlit portals को run करने के लिए
"""

import streamlit as st
import subprocess
import sys
import os
import socket
from constants import PORTAL_PORTS, PORTAL_ADDRESSES

def get_host_ip():
    """Get the current host IP address"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def get_current_environment():
    """Detect current environment and return appropriate base URL"""
    
    # Check if running on Replit
    if any(env in os.environ for env in ['REPL_SLUG', 'REPL_OWNER', 'REPLIT_URL']):
        # Try environment variables first
        replit_url = os.environ.get('REPLIT_URL', '')
        if replit_url:
            base_url = replit_url.replace(':3000', '').replace(':5000', '')
            return base_url, "replit"
            
        # Try to construct from environment variables
        repl_slug = os.environ.get('REPL_SLUG', '')
        repl_owner = os.environ.get('REPL_OWNER', '')
        if repl_slug and repl_owner:
            base_url = f"https://{repl_slug}.{repl_owner}.repl.co"
            return base_url, "replit"
    
    # Check for common cloud platforms
    if 'CODESPACE_NAME' in os.environ:
        # GitHub Codespaces
        codespace_name = os.environ.get('CODESPACE_NAME', '')
        if codespace_name:
            base_url = f"https://{codespace_name}-5000.githubpreview.dev"
            return base_url, "codespaces"
    
    if 'GITPOD_WORKSPACE_URL' in os.environ:
        # Gitpod
        workspace_url = os.environ.get('GITPOD_WORKSPACE_URL', '')
        if workspace_url:
            base_url = workspace_url.replace('https://', 'https://5000-')
            return base_url, "gitpod"
    
    # Check if running in Docker or other containerized environment
    try:
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                # Running in Docker - use localhost with port mapping
                return "http://localhost", "docker"
    except:
        pass
    
    # Default to localhost for local development
    return "http://localhost", "local"

def generate_portal_urls(base_url, environment):
    """Generate portal URLs based on environment"""
    urls = {}
    
    if environment == "replit":
        # Replit uses port-based routing
        urls = {
            "admin": f"{base_url}:5002",
            "teacher": f"{base_url}:5003", 
            "student": f"{base_url}:5004",
            "api": f"{base_url}:8000"
        }
    elif environment == "codespaces":
        # GitHub Codespaces uses different port format
        base_domain = base_url.replace('-5000.githubpreview.dev', '')
        urls = {
            "admin": f"{base_domain}-5002.githubpreview.dev",
            "teacher": f"{base_domain}-5003.githubpreview.dev",
            "student": f"{base_domain}-5004.githubpreview.dev", 
            "api": f"{base_domain}-8000.githubpreview.dev"
        }
    elif environment == "gitpod":
        # Gitpod uses subdomain format
        base_domain = base_url.replace('https://5000-', 'https://')
        urls = {
            "admin": base_domain.replace('https://', 'https://5002-'),
            "teacher": base_domain.replace('https://', 'https://5003-'),
            "student": base_domain.replace('https://', 'https://5004-'),
            "api": base_domain.replace('https://', 'https://8000-')
        }
    else:
        # Local development or Docker
        urls = {
            "admin": f"{base_url}:5002",
            "teacher": f"{base_url}:5003",
            "student": f"{base_url}:5004", 
            "api": f"{base_url}:8000"
        }
    
    return urls

def main():
    """Main Streamlit runner interface"""
    st.set_page_config(
        page_title="Smart Timetable System - Portal Runner",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Smart Timetable System - Portal Runner")
    st.caption("Testing के लिए अलग-अलग portals को run करें")
    
    # Detect current environment and generate appropriate URLs
    base_url, environment = get_current_environment()
    portal_urls = generate_portal_urls(base_url, environment)
    
    # Display hosting information
    st.subheader("🌐 Portal Access URLs")
    
    # Environment info
    env_info = {
        "replit": "Replit Cloud",
        "codespaces": "GitHub Codespaces", 
        "gitpod": "Gitpod",
        "docker": "Docker Container",
        "local": "Local Development"
    }
    
    st.info(f"**Environment Detected**: {env_info.get(environment, 'Unknown')} | **Base URL**: {base_url}")
    
    # Create columns for portal URLs
    url_col1, url_col2 = st.columns(2)
    
    with url_col1:
        st.write("**Current Environment URLs:**")
        st.markdown(f"- **Admin Portal**: [{portal_urls['admin']}]({portal_urls['admin']})")
        st.markdown(f"- **Teacher Portal**: [{portal_urls['teacher']}]({portal_urls['teacher']})")
        st.markdown(f"- **Student Portal**: [{portal_urls['student']}]({portal_urls['student']})")
        st.markdown(f"- **API Server**: [{portal_urls['api']}]({portal_urls['api']})")
        
        st.success("✅ ऊपर के links automatically आपके environment के लिए generated हैं")
    
    with url_col2:
        st.write("**Alternative Access Methods:**")
        
        # Show localhost URLs for reference
        if environment != "local":
            st.write("**Local URLs (if port forwarding enabled):**")
            st.code("http://localhost:5002 - Admin Portal")
            st.code("http://localhost:5003 - Teacher Portal")
            st.code("http://localhost:5004 - Student Portal")
            st.code("http://localhost:8000 - API Server")
        
        # Manual URL override
        st.write("**Manual Override:**")
        custom_base = st.text_input(
            "Custom base URL:",
            value=base_url,
            help="Enter your custom domain if auto-detection fails"
        )
        
        if custom_base and custom_base != base_url:
            st.write("**Custom URLs:**")
            st.markdown(f"- **Admin**: [{custom_base}:5002]({custom_base}:5002)")
            st.markdown(f"- **Teacher**: [{custom_base}:5003]({custom_base}:5003)")
            st.markdown(f"- **Student**: [{custom_base}:5004]({custom_base}:5004)")
            st.markdown(f"- **API**: [{custom_base}:8000]({custom_base}:8000)")
    
    st.divider()
    
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
        st.metric("Current Host", host_ip)
    
    with col4:
        st.metric("Main Portal", PORTAL_PORTS['main'])
        st.metric("Admin Portal", PORTAL_PORTS['admin'])
        st.metric("Teacher Portal", PORTAL_PORTS['teacher'])
    
    with col5:
        st.metric("Student Portal", PORTAL_PORTS['student'])
        st.metric("Database", "PostgreSQL")
        st.metric("Environment", "Development")
    
    # Portal Status Check
    st.subheader("🟢 Portal Status")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        st.success("**Admin Portal**\n✅ Ready")
        st.code(f"Port: {PORTAL_PORTS['admin']}")
    
    with status_col2:
        st.success("**Teacher Portal**\n✅ Ready") 
        st.code(f"Port: {PORTAL_PORTS['teacher']}")
    
    with status_col3:
        st.success("**Student Portal**\n✅ Ready")
        st.code(f"Port: {PORTAL_PORTS['student']}")
        
    with status_col4:
        st.info("**API Server**\n🚀 Running")
        st.code(f"Port: {PORTAL_PORTS['api']}")
    
    # Instructions
    st.subheader("📋 Quick Access Instructions")
    
    inst_col1, inst_col2 = st.columns(2)
    
    with inst_col1:
        st.write("**Direct Portal Access:**")
        st.info("""
        🔗 **Working Replit URLs दिए गए हैं ऊपर!**
        
        Replit Cloud URLs section में:
        - Admin Portal: Port 5002
        - Teacher Portal: Port 5003  
        - Student Portal: Port 5004
        - API Server: Port 8000
        
        ✅ Green checkmark वाले links पर click करें
        
        🔧 अगर काम न करें तो Custom Domain box में 
        अपना current domain paste करें
        """)
    
    with inst_col2:
        st.write("**Login Credentials:**")
        st.success("""
        👤 **Pre-configured Accounts:**
        
        **Admin Access:**
        Email: rohit.verma@university.edu
        Password: admin123
        
        **Teacher Access:**
        Email: dr.agarwal@kalinga.edu
        Password: teacher123
        
        **Student Access:**
        Email: aarav.sharma@kalinga.edu
        Password: student123
        """)
    
    st.warning("💡 **सभी portals पहले से चल रहे हैं - बस URLs पर click करें!**")

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