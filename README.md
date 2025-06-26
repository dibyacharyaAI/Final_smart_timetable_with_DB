# 🚀 Smart Timetable System

This is a complete Smart Timetable Management System that includes:

- ✅ Streamlit portals for Admin, Teacher, Student
- ✅ Flask-based API server with JWT Authentication
- ✅ PostgreSQL database integration
- ✅ Role-based login system
- ✅ CLI and cURL testing ready

---

## 📦 Clone the Repository

```bash
git clone https://github.com/dibyacharyaAI/Final_smart_timetable_with_DB.git
cd Final_smart_timetable_with_DB
```

---

## 🧪 Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements_complete.txt
```

---

## 🔐 Environment Variables (.env)

Create `.env` file by copying the template:

```bash
cp .env.template .env
```

Edit `.env` and set your PostgreSQL DB credentials:

```
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/smart_timetable_db
JWT_SECRET_KEY=any_secure_long_string
```

---

## 🗃️ PostgreSQL Setup

1. Start PostgreSQL (macOS example):
```bash
brew services start postgresql
```

2. Create the database:
```bash
psql -U postgres
CREATE DATABASE smart_timetable_db;
\q
```

---

## ⚙️ Initialize the Database

```bash
python quick_init.py
```

This creates necessary tables and inserts default users (admin, teachers, students).

source venv/bin/activate

pip install python-dotenv

pip freeze > requirements.txt



---

## 🧑‍💼 Streamlit Portals (Run Individually)

### 🅰️ Admin Portal
```bash
streamlit run admin_portal.py --server.port 5002
```

### 🅱️ Teacher Portal
```bash
streamlit run teacher_portal.py --server.port 5003
```

### 🅲️ Student Portal
```bash
streamlit run student_portal.py --server.port 5004
```

> Visit in browser:
> - Admin: [http://localhost:5002](http://localhost:5002)
> - Teacher: [http://localhost:5003](http://localhost:5003)
> - Student: [http://localhost:5004](http://localhost:5004)

---

## 🔀 Unified Streamlit Portal (Optional)

```bash
streamlit run run_streamlit.py
```

Use sidebar to switch between roles.

---

## 🌐 Run API Server

```bash
python api_server.py
```

> API runs on: [http://localhost:5001](http://localhost:5001)

---

## 🔐 Test API with cURL

### 🔑 Login (Admin)
```bash
curl -X POST http://localhost:5001/api/auth/login \
-H "Content-Type: application/json" \
-d '{
  "email": "rohit.verma@university.edu",
  "password": "admin123"
}'
```

### 🎓 Login (Student)
```bash
curl -X POST http://localhost:5001/api/auth/login \
-H "Content-Type: application/json" \
-d '{
  "email": "aarav.sharma@kalinga.edu",
  "password": "student123"
}'
```

### 👨‍🏫 Login (Teacher)
```bash
curl -X POST http://localhost:5001/api/auth/login \
-H "Content-Type: application/json" \
-d '{
  "email": "dr.agarwal@kalinga.edu",
  "password": "teacher123"
}'
```

---

## 📊 API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/api/auth/login` | Login and get JWT token |
| POST   | `/api/auth/register` | Register a new user |
| GET    | `/api/events` | Get all schedule events |
| GET    | `/api/users/profile` | Get logged-in user profile |
| GET    | `/api/status` | API server status check |

---

## 🧠 Default Users

| Role    | Email                          | Password     |
|---------|--------------------------------|--------------|
| Admin   | rohit.verma@university.edu     | admin123     |
| Teacher | dr.agarwal@kalinga.edu         | teacher123   |
| Student | aarav.sharma@kalinga.edu       | student123   |

---

## 🧹 Tip: Stop All Streamlit Servers

```bash
lsof -i :5002 -t | xargs kill -9
lsof -i :5003 -t | xargs kill -9
lsof -i :5004 -t | xargs kill -9
```

---

## 📬 Contact

Email: support@kalinga.edu
