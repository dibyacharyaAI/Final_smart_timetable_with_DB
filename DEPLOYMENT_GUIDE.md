# Deployment Guide - Smart Timetable System

Complete deployment instructions for local development and production environments.

## Quick Start (Local Development)

### 1. Clone and Setup
```bash
git clone https://github.com/dibyacharyaAI/Final_smart_timetable_with_DB.git
cd Final_smart_timetable_with_DB
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements_complete.txt
```

### 2. Database Setup
```bash
# SQLite (Development)
python -c "from database import DatabaseManager; db = DatabaseManager(); db.initialize_database()"

# PostgreSQL (Production)
createdb timetable_db
export DATABASE_URL="postgresql://username:password@localhost:5432/timetable_db"
python -c "from database import DatabaseManager; db = DatabaseManager(); db.initialize_database()"
```

### 3. Test Database Connection
```bash
python -c "
from database import DatabaseManager
db = DatabaseManager()
print('Database:', 'Connected' if db.test_connection() else 'Failed')
user = db.authenticate_user('rohit.verma@university.edu', 'admin123')
print('Admin Login:', 'Success' if user else 'Failed')
"
```

### 4. Run System
```bash
# Method 1: Testing Interface (Recommended)
streamlit run run_streamlit.py --server.port 5000 --server.address 0.0.0.0

# Method 2: Individual Components
python api_server.py &
streamlit run admin_portal.py --server.port 5002 --server.address 0.0.0.0 &
streamlit run teacher_portal.py --server.port 5003 --server.address 0.0.0.0 &
streamlit run student_portal.py --server.port 5004 --server.address 0.0.0.0 &
```

## Production Deployment

### Environment Variables
Create `.env` file:
```env
DATABASE_URL=postgresql://user:password@host:5432/database
JWT_SECRET_KEY=your_production_secret_key
ENVIRONMENT=production
DEBUG_MODE=false
HOST=0.0.0.0
PORT=8000
```

### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE timetable_db;
CREATE USER timetable_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE timetable_db TO timetable_user;
```

### Deployment Commands
```bash
# Production API Server
python api_server_deploy.py

# Production Portals with PM2
npm install -g pm2
pm2 start "streamlit run admin_portal.py --server.port 5002" --name admin-portal
pm2 start "streamlit run teacher_portal.py --server.port 5003" --name teacher-portal
pm2 start "streamlit run student_portal.py --server.port 5004" --name student-portal
```

## Testing Procedures

### 1. System Health Check
```bash
python -c "
import sys
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem

print('=== System Health Check ===')

# Database test
try:
    db = DatabaseManager()
    db_status = 'OK' if db.test_connection() else 'FAILED'
    print(f'Database: {db_status}')
except Exception as e:
    print(f'Database: ERROR - {e}')

# AI Pipeline test
try:
    system = SmartTimetableSystem()
    system.generate_complete_schedule()
    print('AI Pipeline: OK - 72 sections processed')
except Exception as e:
    print(f'AI Pipeline: ERROR - {e}')

# Authentication test
try:
    user = db.authenticate_user('rohit.verma@university.edu', 'admin123')
    auth_status = 'OK' if user else 'FAILED'
    print(f'Authentication: {auth_status}')
except Exception as e:
    print(f'Authentication: ERROR - {e}')

print('=== Health Check Complete ===')
"
```

### 2. API Testing
```bash
# Start API server
python api_server.py &

# Test endpoints
curl -X GET http://localhost:8000/api/status
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"rohit.verma@university.edu","password":"admin123"}'
```

### 3. Portal Testing
```bash
# Use testing interface
streamlit run run_streamlit.py --server.port 5000
# Visit http://localhost:5000 and test each portal
```

## Performance Optimization

### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
```

### System Monitoring
```bash
# Monitor system resources
htop
# Monitor database connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
# Monitor application logs
tail -f logs/timetable_system.log
```

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
lsof -i :5000
kill -9 <PID>
```

**Database Connection Failed**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql
# Check connection parameters
psql postgresql://username:password@localhost:5432/timetable_db
```

**Module Import Errors**
```bash
pip install -r requirements_complete.txt --force-reinstall
```

**Streamlit Cache Issues**
```bash
streamlit cache clear
rm -rf ~/.streamlit/
```

## Security Configuration

### JWT Settings
- Use strong secret key (minimum 32 characters)
- Set appropriate token expiration (24 hours recommended)
- Enable HTTPS in production

### Database Security
- Use strong passwords
- Limit database user permissions
- Enable SSL connections
- Regular backup procedures

### Firewall Configuration
```bash
# Allow required ports
sudo ufw allow 5000:5004/tcp
sudo ufw allow 8000/tcp
sudo ufw enable
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
pg_dump timetable_db > backup_$(date +%Y%m%d).sql

# Restore backup
psql timetable_db < backup_20250626.sql
```

### Application Backup
```bash
# Backup configuration and data
tar -czf timetable_backup_$(date +%Y%m%d).tar.gz \
  *.py data/ logs/ .env
```

## Monitoring and Maintenance

### Log Rotation
```bash
# Setup logrotate for application logs
sudo nano /etc/logrotate.d/timetable
```

### Regular Maintenance Tasks
```bash
# Weekly database maintenance
python -c "
from database import DatabaseManager
db = DatabaseManager()
# Add maintenance procedures here
"

# Monthly system health check
python -c "
from core_timetable_system import SmartTimetableSystem
system = SmartTimetableSystem()
system.generate_complete_schedule()
print('System health: OK')
"
```

## Support and Documentation

- Technical Documentation: `replit.md`
- API Documentation: Check API endpoints in `constants.py`
- Database Schema: Defined in `database.py`
- Configuration: `constants.py` and `config.py`

## Version Control

### Git Workflow
```bash
# Development workflow
git checkout -b feature-branch
# Make changes
git add .
git commit -m "Description of changes"
git push origin feature-branch
# Create pull request
```

### Release Process
```bash
# Create release
git tag v2.0.0
git push origin v2.0.0
```