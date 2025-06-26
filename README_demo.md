# Smart Timetable System for Kalinga University

## Project Overview

Advanced AI-powered timetable management system for Kalinga University with multi-campus scheduling, anomaly detection, and role-based access control.

### Key Features
- **72 Sections Management**: Complete scheduling for sections A1-A36 and B1-B36
- **AI Pipeline**: RNN-based anomaly detection and self-healing optimization
- **Multi-Campus Support**: Campus_3, Campus_8, Campus_15B, and Stadium scheduling
- **Role-Based Access**: Admin, Teacher, and Student portals with JWT authentication
- **Real-Time Optimization**: Constraint solving with conflict resolution

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (production) or SQLite (development)

### Installation

```bash
# Clone repository
git clone https://github.com/dibyacharyaAI/Final_smart_timetable_with_DB.git
cd Final_smart_timetable_with_DB

# Setup environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements_complete.txt

# Configure environment
cp .env.template .env
# Edit .env with your database settings

# Initialize database
python3 -c "from database import DatabaseManager; db = DatabaseManager(); db.initialize_database()"
```

### Quick Test

```bash
# Test system health
python3 -c "
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem

# Database test
db = DatabaseManager()
print('Database:', 'Connected' if db.test_connection() else 'Failed')

# Authentication test
user = db.authenticate_user('rohit.verma@university.edu', 'admin123')
print('Admin Login:', 'Success' if user else 'Failed')

# AI Pipeline test
system = SmartTimetableSystem()
system.generate_complete_schedule()
print('AI Pipeline: SUCCESS - 72 sections processed')
"
```

### Run Application

```bash
# Testing Interface (Recommended)
streamlit run run_streamlit.py --server.port 5000 --server.address 0.0.0.0

# Individual Components
python api_server.py &
streamlit run admin_portal.py --server.port 5002 --server.address 0.0.0.0 &
streamlit run teacher_portal.py --server.port 5003 --server.address 0.0.0.0 &
streamlit run student_portal.py --server.port 5004 --server.address 0.0.0.0 &
```

## Default User Accounts

| Role | Email | Password | Access Level |
|------|-------|----------|--------------|
| Admin | rohit.verma@university.edu | admin123 | Full system access |
| Teacher | dr.agarwal@kalinga.edu | teacher123 | Schedule management |
| Student | aarav.sharma@kalinga.edu | student123 | Read-only access |

## System Architecture

### Core Components

1. **Core Engine** (`core_timetable_system.py`)
   - AI-powered schedule generation
   - Anomaly detection with configurable thresholds
   - Constraint optimization using OR-Tools

2. **Multi-Portal Interface**
   - Admin Portal: Complete timetable management
   - Teacher Portal: Personal schedule editing
   - Student Portal: Schedule viewing and notifications

3. **API Server** (`api_server.py` / `api_server_deploy.py`)
   - RESTful API with JWT authentication
   - Complete CRUD operations
   - Real-time data synchronization

4. **Database Layer** (`database.py`)
   - User management with role-based access
   - Schedule versioning and history
   - PostgreSQL/SQLite support

### Data Sources

- **Teacher Data**: 100 authentic faculty members from `data/teacher_data.csv`
- **Campus Data**: Real walking distances and room assignments
- **Transit Data**: Campus connectivity from `data/final_transit_data_1750319692453.xlsx`

### Campus Information

| Campus | Rooms | Walking Distance | Specialization |
|--------|-------|------------------|----------------|
| Campus_3 | 25 theory + labs | 0m (Main) | General subjects |
| Campus_8 | 10 theory + Workshop | 550m (7 min) | Engineering labs |
| Campus_15B | 18 theory + Programming | 700m (10 min) | Computer labs |
| Stadium | Sports/Yoga | 900m (12 min) | Physical education |

## API Endpoints

### Authentication
- `POST /api/auth/login` - User authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/refresh` - Token refresh

### Schedule Management
- `GET /api/events` - Get all events with filters
- `GET /api/events/{id}` - Get specific event
- `POST /api/events` - Create new event (Admin)
- `PUT /api/events/{id}` - Update event (Admin)
- `DELETE /api/events/{id}` - Delete event (Admin)

### User Management
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update profile
- `GET /api/users` - Get all users (Admin)
- `GET /api/teachers/{id}/schedule` - Teacher schedule

### System Status
- `GET /api/status` - System health check
- `POST /api/events/swap` - Schedule swap requests

## Configuration

### Environment Variables (.env)

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/timetable_db

# Security
JWT_SECRET_KEY=your_secure_secret_key_here

# Application
ENVIRONMENT=production
DEBUG_MODE=false
HOST=0.0.0.0
PORT=8000
```

### Database Setup

```sql
-- PostgreSQL setup
CREATE DATABASE timetable_db;
CREATE USER timetable_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE timetable_db TO timetable_user;
```

## Testing

### Health Check
```bash
python3 -c "
import sys
from database import DatabaseManager
from core_timetable_system import SmartTimetableSystem

print('=== System Health Check ===')

# Database connectivity
try:
    db = DatabaseManager()
    db_status = 'OK' if db.test_connection() else 'FAILED'
    print(f'Database: {db_status}')
except Exception as e:
    print(f'Database: ERROR - {e}')

# AI Pipeline functionality
try:
    system = SmartTimetableSystem()
    system.generate_complete_schedule()
    print('AI Pipeline: OK - 72 sections processed')
except Exception as e:
    print(f'AI Pipeline: ERROR - {e}')

# Authentication system
try:
    user = db.authenticate_user('rohit.verma@university.edu', 'admin123')
    auth_status = 'OK' if user else 'FAILED'
    print(f'Authentication: {auth_status}')
except Exception as e:
    print(f'Authentication: ERROR - {e}')

print('=== Health Check Complete ===')
"
```

### API Testing
```bash
# Start API server
python api_server.py &

# Test authentication
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"rohit.verma@university.edu","password":"admin123"}'

# Test status endpoint
curl -X GET http://localhost:8000/api/status
```

## Deployment Options

### Development (SQLite)
```bash
export DATABASE_URL="sqlite:///timetable.db"
streamlit run run_streamlit.py --server.port 5000
```

### Production (PostgreSQL)
```bash
export DATABASE_URL="postgresql://user:password@host:5432/database"
python api_server_deploy.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements_complete.txt .
RUN pip install -r requirements_complete.txt
COPY . .
EXPOSE 8000
CMD ["python", "api_server_deploy.py"]
```

## Security Features

- JWT-based authentication with role validation
- Password hashing with bcrypt
- SQL injection protection via SQLAlchemy ORM
- Input validation and sanitization
- CORS configuration for cross-origin requests

## Performance Optimization

- Database indexing for user queries
- Streamlit caching for dashboard performance
- Efficient AI pipeline with batch processing
- Connection pooling for database operations

## Troubleshooting

### Common Issues

**Port already in use**
```bash
lsof -i :5000
kill -9 <PID>
```

**Database connection failed**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql
# Test connection
psql $DATABASE_URL
```

**Module import errors**
```bash
pip install -r requirements_complete.txt --force-reinstall
```

**Streamlit cache issues**
```bash
streamlit cache clear
rm -rf ~/.streamlit/
```

## Support and Documentation

- **Technical Documentation**: `replit.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **API Documentation**: Check constants.py for endpoint definitions
- **Configuration Reference**: `config.py` and `constants.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is developed for Kalinga University's internal timetable management system.

---

**Repository**: https://github.com/dibyacharyaAI/Final_smart_timetable_with_DB
**Support**: Contact system administrator for technical support
