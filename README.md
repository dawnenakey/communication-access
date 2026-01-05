# SignSync AI - ASL Communication Access Platform

**Author:** Dawnena Key / SonZo AI
**License:** Proprietary - Patent Pending

A full-stack application for American Sign Language (ASL) translation and learning, featuring real-time sign recognition, a sign dictionary, and translation history tracking.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Testing](#testing)
- [EC2 Deployment Guide](#ec2-deployment-guide)
- [API Documentation](#api-documentation)
- [ML Pipeline](#ml-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React Frontend│────▶│  FastAPI Backend│────▶│    MongoDB      │
│   (Port 3000)   │     │   (Port 8000)   │     │   Database      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  ML Pipeline    │
         └─────────────▶│  (TensorFlow/   │
                        │   MediaPipe)    │
                        └─────────────────┘
```

### Project Structure

```
communication-access/
├── backend/
│   ├── server.py           # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React app with routing
│   │   ├── pages/          # Landing, Dashboard, Dictionary, History
│   │   └── components/     # Reusable UI components
│   └── package.json        # Node dependencies
├── asl_handshapes.py       # ASL handshape definitions (MANO parameters)
├── dataset_loader.py       # PyTorch dataset loader for ML training
├── backend_test.py         # Backend API tests
└── README.md               # This file
```

---

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** - High-performance async web framework
- **MongoDB** + **Motor** - Async MongoDB driver
- **Pydantic** - Data validation
- **uvicorn** - ASGI server

### Frontend
- **React 19** - UI framework
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first CSS
- **Radix UI** - Accessible component primitives
- **MediaPipe** - Hand tracking for ASL recognition
- **TensorFlow.js** - ML inference in browser

### ML Pipeline
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **PIL/Pillow** - Image processing

---

## Prerequisites

- **Node.js** >= 18.x
- **Python** >= 3.10
- **MongoDB** (local or MongoDB Atlas)
- **Yarn** or **npm**
- **Git**

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd communication-access
```

### 2. Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Create environment file
cat > backend/.env << 'EOF'
MONGO_URL=mongodb://localhost:27017
DB_NAME=signsync_dev
CORS_ORIGINS=http://localhost:3000
EOF

# Start MongoDB (if local)
# On macOS: brew services start mongodb-community
# On Ubuntu: sudo systemctl start mongod

# Run backend server
cd backend
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
# In a new terminal
cd frontend

# Install dependencies
yarn install  # or npm install

# Create environment file
echo "REACT_APP_BACKEND_URL=http://localhost:8000" > .env

# Start development server
yarn start  # or npm start
```

### 4. Verify Setup

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/api
- Health check: http://localhost:8000/api/health

---

## Testing

### Backend Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio httpx

# Run all tests
pytest backend_test.py -v

# Run with coverage
pip install pytest-cov
pytest backend_test.py -v --cov=backend --cov-report=html

# Run specific test
pytest backend_test.py -v -k "test_health_check"
```

#### Quick API Test (curl)

```bash
# Health check
curl http://localhost:8000/api/health

# Get all signs (public endpoint)
curl http://localhost:8000/api/signs

# Search signs
curl http://localhost:8000/api/signs/search/hello
```

### Frontend Tests

```bash
cd frontend

# Run test suite
yarn test

# Run tests with coverage
yarn test --coverage

# Run tests in watch mode
yarn test --watch
```

### End-to-End Testing

```bash
# Install Playwright (optional)
npm install -g playwright
npx playwright install

# Run E2E tests
npx playwright test
```

### Test the ML Pipeline

```bash
# Test ASL handshape definitions
python asl_handshapes.py

# Test dataset loader (requires synthetic data)
python dataset_loader.py --synthetic-dir /path/to/synthetic_data --visualize
```

---

## EC2 Deployment Guide

### Step 1: Launch EC2 Instance

1. **Login to AWS Console** → EC2 → Launch Instance

2. **Configure Instance:**
   - **Name:** `signsync-production`
   - **AMI:** Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance type:** `t3.medium` (minimum) or `t3.large` (recommended)
   - **Key pair:** Create or select existing key pair
   - **Network settings:**
     - Allow SSH (port 22) from your IP
     - Allow HTTP (port 80) from anywhere
     - Allow HTTPS (port 443) from anywhere
     - Allow Custom TCP (port 8000) from anywhere (API)
     - Allow Custom TCP (port 3000) from anywhere (Frontend dev)

3. **Storage:** 20 GB gp3 (minimum)

4. **Launch the instance**

### Step 2: Connect to EC2

```bash
# Set permissions for key file
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Yarn
sudo npm install -g yarn

# Install Nginx (reverse proxy)
sudo apt install -y nginx

# Install PM2 (process manager)
sudo npm install -g pm2

# Install Git
sudo apt install -y git
```

### Step 4: Install MongoDB

```bash
# Import MongoDB GPG key
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt update
sudo apt install -y mongodb-org

# Start and enable MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify MongoDB is running
sudo systemctl status mongod
```

### Step 5: Clone and Setup Application

```bash
# Clone repository
cd /home/ubuntu
git clone <repository-url> signsync
cd signsync

# Backend setup
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Create backend environment file
cat > backend/.env << 'EOF'
MONGO_URL=mongodb://localhost:27017
DB_NAME=signsync_production
CORS_ORIGINS=http://<EC2-PUBLIC-IP>,https://yourdomain.com
EOF

# Frontend setup
cd frontend
yarn install

# Build frontend for production
echo "REACT_APP_BACKEND_URL=http://<EC2-PUBLIC-IP>:8000" > .env
yarn build
```

### Step 6: Configure PM2 Process Manager

```bash
cd /home/ubuntu/signsync

# Create PM2 ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'signsync-backend',
      cwd: '/home/ubuntu/signsync/backend',
      script: '/home/ubuntu/signsync/venv/bin/uvicorn',
      args: 'server:app --host 0.0.0.0 --port 8000',
      interpreter: 'none',
      env: {
        NODE_ENV: 'production',
      },
    },
    {
      name: 'signsync-frontend',
      cwd: '/home/ubuntu/signsync/frontend',
      script: 'npx',
      args: 'serve -s build -l 3000',
      interpreter: 'none',
      env: {
        NODE_ENV: 'production',
      },
    }
  ]
};
EOF

# Install serve for frontend
cd frontend && npm install -g serve

# Start applications with PM2
cd /home/ubuntu/signsync
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Setup PM2 to start on boot
pm2 startup systemd -u ubuntu --hp /home/ubuntu
# Run the command it outputs
```

### Step 7: Configure Nginx Reverse Proxy

```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/signsync << 'EOF'
server {
    listen 80;
    server_name _;  # Replace with your domain

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/signsync /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### Step 8: Setup SSL (Optional but Recommended)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### Step 9: Verify Deployment

```bash
# Check PM2 status
pm2 status

# Check Nginx status
sudo systemctl status nginx

# Check MongoDB status
sudo systemctl status mongod

# Test endpoints
curl http://localhost:8000/api/health
curl http://localhost/api/health

# View logs
pm2 logs signsync-backend
pm2 logs signsync-frontend
```

### Step 10: Useful Commands

```bash
# Restart services
pm2 restart all

# View real-time logs
pm2 logs

# Monitor resources
pm2 monit

# Update application
cd /home/ubuntu/signsync
git pull
source venv/bin/activate
pip install -r backend/requirements.txt
cd frontend && yarn install && yarn build
pm2 restart all
```

---

## API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/session` | Exchange session_id for session_token |
| GET | `/api/auth/me` | Get current authenticated user |
| POST | `/api/auth/logout` | Logout and clear session |

### Sign Dictionary Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/signs` | Get all signs |
| GET | `/api/signs/{sign_id}` | Get specific sign |
| POST | `/api/signs` | Create new sign (auth required) |
| PUT | `/api/signs/{sign_id}` | Update sign (auth required) |
| DELETE | `/api/signs/{sign_id}` | Delete sign (auth required) |
| GET | `/api/signs/search/{word}` | Search signs by word |

### Translation History Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/history` | Get user's translation history |
| POST | `/api/history` | Save translation to history |
| DELETE | `/api/history/{history_id}` | Delete history entry |
| DELETE | `/api/history` | Clear all history |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/` | API info |
| GET | `/api/health` | Health check |

---

## ML Pipeline

The project includes files for training ASL recognition models:

### ASL Handshape Definitions (`asl_handshapes.py`)

- 26 alphabet handshapes (A-Z)
- 10 number handshapes (0-9)
- 10 common signs (ILY, classifiers, etc.)
- MANO pose parameter format

### Dataset Loader (`dataset_loader.py`)

- PyTorch Dataset implementation
- Supports synthetic and real data
- Curriculum learning phases
- Data augmentation transforms

See [ML Pipeline Documentation](./docs/ml-pipeline.md) for detailed usage.

---

## Environment Variables

### Backend (`backend/.env`)

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=signsync_dev
CORS_ORIGINS=http://localhost:3000
```

### Frontend (`frontend/.env`)

```env
REACT_APP_BACKEND_URL=http://localhost:8000
```

---

## Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
sudo systemctl status mongod

# View MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log

# Restart MongoDB
sudo systemctl restart mongod
```

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :8000
sudo lsof -i :3000

# Kill process
kill -9 <PID>
```

### PM2 Issues

```bash
# View detailed logs
pm2 logs --lines 100

# Restart with fresh state
pm2 delete all
pm2 start ecosystem.config.js
```

### Frontend Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules
rm -rf .cache
yarn install
yarn build
```

---

## Contact

**Dawnena Key**
SonZo AI - Founder/Chief AI Officer
dawnena@sonzo.io
