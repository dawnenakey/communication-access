#!/bin/bash
# ============================================================================
# SonZo AI - EC2 Deployment Script
# ============================================================================
# Deploy the full SonZo AI stack to an EC2 instance with Docker Compose.
#
# Services: Nginx, Frontend (React), Backend (FastAPI), Avatar (GPU),
#           Recognition (GPU), Legacy UI, MongoDB
#
# Usage:
#   ./deploy.sh setup      # First-time EC2 setup (Docker, NVIDIA, etc.)
#   ./deploy.sh env        # Create .env from template
#   ./deploy.sh build      # Build all Docker images
#   ./deploy.sh start      # Start all services
#   ./deploy.sh stop       # Stop all services
#   ./deploy.sh restart    # Restart all services
#   ./deploy.sh logs       # View logs (all services)
#   ./deploy.sh logs <svc> # View logs for a specific service
#   ./deploy.sh ssl        # Setup SSL certificates with Let's Encrypt
#   ./deploy.sh status     # Show service status
#   ./deploy.sh update     # Pull latest code, rebuild, and restart
#   ./deploy.sh backup     # Backup MongoDB data
#
# ============================================================================

set -e

# Configuration
DOMAIN="sonzo.io"
APP_DOMAIN="app.sonzo.io"
EMAIL="dawnena@sonzo.ai"
PROJECT_DIR="/opt/sonzo"
DEPLOY_DIR="$PROJECT_DIR/deploy"
BACKUP_DIR="/opt/sonzo-backups"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# ============================================================================
# First-Time EC2 Setup
# ============================================================================

setup() {
    log_info "========================================="
    log_info "  SonZo AI - EC2 Setup"
    log_info "========================================="

    # Create project directory
    log_step "Creating project directories..."
    sudo mkdir -p $PROJECT_DIR
    sudo mkdir -p $BACKUP_DIR
    sudo chown -R $USER:$USER $PROJECT_DIR $BACKUP_DIR

    # Copy project files
    log_step "Copying project files..."
    rsync -av --exclude='.git' --exclude='node_modules' --exclude='__pycache__' \
        --exclude='.env' --exclude='*.pyc' \
        ../ $PROJECT_DIR/

    # Install Docker
    if ! command -v docker &> /dev/null; then
        log_step "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        log_info "Docker installed. You may need to log out and back in for group changes."
    else
        log_info "Docker already installed: $(docker --version)"
    fi

    # Install Docker Compose plugin
    if ! docker compose version &> /dev/null; then
        log_step "Installing Docker Compose plugin..."
        sudo apt-get update
        sudo apt-get install -y docker-compose-plugin
    else
        log_info "Docker Compose already installed: $(docker compose version)"
    fi

    # Install NVIDIA Container Toolkit (if GPU available)
    if command -v nvidia-smi &> /dev/null; then
        log_step "GPU detected! Setting up NVIDIA Container Toolkit..."
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
            log_info "NVIDIA Container Toolkit installed."
        else
            log_info "NVIDIA Container Toolkit already installed."
        fi
        nvidia-smi
    else
        log_warn "No NVIDIA GPU detected. Avatar and Recognition services will need GPU."
        log_warn "You can still run backend, frontend, ui, and mongodb without GPU."
    fi

    # Create SSL directory
    mkdir -p $DEPLOY_DIR/ssl

    # Setup .env if not exists
    if [ ! -f "$DEPLOY_DIR/.env" ]; then
        log_step "Creating .env from template..."
        cp $DEPLOY_DIR/.env.example $DEPLOY_DIR/.env
        log_warn "Edit $DEPLOY_DIR/.env with your actual values before starting services!"
    fi

    log_info "========================================="
    log_info "  Setup complete!"
    log_info "========================================="
    echo ""
    echo "  Next steps:"
    echo "  1. Edit environment:  nano $DEPLOY_DIR/.env"
    echo "  2. Get SSL certs:     ./deploy.sh ssl"
    echo "  3. Build images:      ./deploy.sh build"
    echo "  4. Start services:    ./deploy.sh start"
    echo ""
    echo "  Or for non-GPU (no avatar/recognition):"
    echo "     ./deploy.sh start-core"
    echo ""
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_env() {
    if [ -f "$DEPLOY_DIR/.env" ]; then
        log_warn ".env already exists at $DEPLOY_DIR/.env"
        read -p "Overwrite? (y/N): " confirm
        if [ "$confirm" != "y" ]; then
            log_info "Keeping existing .env"
            return
        fi
    fi

    cp $DEPLOY_DIR/.env.example $DEPLOY_DIR/.env
    log_info "Created .env from template."
    log_warn "Edit $DEPLOY_DIR/.env with your values:"
    echo "  nano $DEPLOY_DIR/.env"
}

# ============================================================================
# SSL Setup
# ============================================================================

setup_ssl() {
    log_info "Setting up SSL certificates..."

    # Install certbot
    if ! command -v certbot &> /dev/null; then
        log_step "Installing Certbot..."
        sudo apt-get update
        sudo apt-get install -y certbot
    fi

    # Stop nginx if running
    cd $DEPLOY_DIR
    docker compose stop nginx 2>/dev/null || true

    # Get certificates
    log_step "Obtaining SSL certificates for $DOMAIN and $APP_DOMAIN..."
    sudo certbot certonly --standalone \
        -d $DOMAIN \
        -d www.$DOMAIN \
        -d $APP_DOMAIN \
        -d api.$DOMAIN \
        --email $EMAIL \
        --agree-tos \
        --non-interactive

    # Copy certificates
    log_step "Copying certificates..."
    sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $DEPLOY_DIR/ssl/
    sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $DEPLOY_DIR/ssl/
    sudo chown $USER:$USER $DEPLOY_DIR/ssl/*.pem

    # Setup auto-renewal cron
    log_step "Setting up auto-renewal..."
    (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && cp /etc/letsencrypt/live/$DOMAIN/*.pem $DEPLOY_DIR/ssl/ && cd $DEPLOY_DIR && docker compose restart nginx") | sort -u | crontab -

    log_info "SSL setup complete!"
}

# ============================================================================
# Build
# ============================================================================

build() {
    log_info "Building Docker images..."
    cd $DEPLOY_DIR

    if [ -n "$1" ]; then
        log_step "Building $1..."
        docker compose build "$1"
    else
        log_step "Building all services..."
        docker compose build
    fi

    log_info "Build complete!"
}

# ============================================================================
# Service Management
# ============================================================================

start() {
    log_info "Starting all SonZo AI services..."
    cd $DEPLOY_DIR

    # Check .env exists
    if [ ! -f ".env" ]; then
        log_error ".env file not found! Run: ./deploy.sh env"
        exit 1
    fi

    # Check SSL certificates
    if [ ! -f "ssl/fullchain.pem" ]; then
        log_warn "SSL certificates not found. Starting without nginx (HTTP only)."
        log_warn "Run './deploy.sh ssl' to enable HTTPS."
        docker compose up -d mongodb backend frontend ui avatar recognition
    else
        docker compose up -d
    fi

    log_info "Services started!"
    echo ""
    docker compose ps
    echo ""
    if [ -f "ssl/fullchain.pem" ]; then
        log_info "App:  https://$APP_DOMAIN"
        log_info "API:  https://api.$DOMAIN"
    else
        log_info "App:  http://$(curl -s ifconfig.me):3000 (direct to frontend)"
        log_info "API:  http://$(curl -s ifconfig.me):8000 (direct to backend)"
    fi
}

start_core() {
    log_info "Starting core services (no GPU required)..."
    cd $DEPLOY_DIR

    if [ ! -f ".env" ]; then
        log_error ".env file not found! Run: ./deploy.sh env"
        exit 1
    fi

    docker compose up -d mongodb backend frontend ui

    # Start nginx only if SSL is available
    if [ -f "ssl/fullchain.pem" ]; then
        docker compose up -d nginx
    fi

    log_info "Core services started (without avatar/recognition GPU services)."
    docker compose ps
}

stop() {
    log_info "Stopping SonZo AI services..."
    cd $DEPLOY_DIR
    docker compose down
    log_info "Services stopped."
}

restart() {
    log_info "Restarting SonZo AI services..."
    cd $DEPLOY_DIR

    if [ -n "$1" ]; then
        log_step "Restarting $1..."
        docker compose restart "$1"
    else
        docker compose down
        docker compose up -d
    fi

    log_info "Restart complete."
}

# ============================================================================
# Logs
# ============================================================================

logs() {
    cd $DEPLOY_DIR
    if [ -n "$1" ]; then
        docker compose logs -f --tail=100 "$1"
    else
        docker compose logs -f --tail=100
    fi
}

# ============================================================================
# Status
# ============================================================================

status() {
    log_info "Service Status:"
    cd $DEPLOY_DIR
    docker compose ps
    echo ""

    # Show resource usage
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || true
}

# ============================================================================
# Update (pull, rebuild, restart)
# ============================================================================

update() {
    log_info "Updating SonZo AI..."
    cd $PROJECT_DIR

    # Pull latest code
    log_step "Pulling latest code..."
    git pull origin main

    # Rebuild images
    log_step "Rebuilding Docker images..."
    cd $DEPLOY_DIR
    docker compose build

    # Restart with zero-downtime approach
    log_step "Restarting services..."
    docker compose up -d --remove-orphans

    # Clean up old images
    log_step "Cleaning up old images..."
    docker image prune -f

    log_info "Update complete!"
    docker compose ps
}

# ============================================================================
# Backup MongoDB
# ============================================================================

backup() {
    log_info "Backing up MongoDB..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/mongodb_$TIMESTAMP"

    mkdir -p $BACKUP_PATH

    docker exec sonzo-mongodb mongodump --out /dump
    docker cp sonzo-mongodb:/dump $BACKUP_PATH/

    # Compress
    tar -czf "$BACKUP_PATH.tar.gz" -C $BACKUP_DIR "mongodb_$TIMESTAMP"
    rm -rf $BACKUP_PATH

    log_info "Backup saved to: $BACKUP_PATH.tar.gz"

    # Keep only last 7 backups
    ls -t $BACKUP_DIR/mongodb_*.tar.gz 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null || true
    log_info "Backup rotation: keeping last 7 backups."
}

# ============================================================================
# Development Mode
# ============================================================================

dev() {
    log_info "Starting in development mode..."
    cd $PROJECT_DIR

    # Start only MongoDB in Docker, run app locally
    cd $DEPLOY_DIR
    docker compose up -d mongodb
    log_info "MongoDB running on port 27017"
    echo ""
    echo "  Start backend:   cd $PROJECT_DIR/backend && uvicorn server:app --reload --port 8000"
    echo "  Start frontend:  cd $PROJECT_DIR/frontend && npm run dev"
}

# ============================================================================
# Main
# ============================================================================

case "$1" in
    setup)
        setup
        ;;
    env)
        setup_env
        ;;
    ssl)
        setup_ssl
        ;;
    build)
        build "$2"
        ;;
    start)
        start
        ;;
    start-core)
        start_core
        ;;
    stop)
        stop
        ;;
    restart)
        restart "$2"
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    update)
        update
        ;;
    backup)
        backup
        ;;
    dev)
        dev
        ;;
    *)
        echo ""
        echo "  SonZo AI - EC2 Deployment Script"
        echo "  ================================="
        echo ""
        echo "  Usage: $0 <command> [options]"
        echo ""
        echo "  Setup Commands:"
        echo "    setup         First-time EC2 setup (Docker, NVIDIA, directories)"
        echo "    env           Create .env from template"
        echo "    ssl           Setup SSL certificates with Let's Encrypt"
        echo ""
        echo "  Service Commands:"
        echo "    build [svc]   Build Docker images (optionally a specific service)"
        echo "    start         Start all services"
        echo "    start-core    Start without GPU services (no avatar/recognition)"
        echo "    stop          Stop all services"
        echo "    restart [svc] Restart all or a specific service"
        echo "    update        Pull latest code, rebuild, and restart"
        echo ""
        echo "  Monitoring:"
        echo "    logs [svc]    View logs (all or specific service)"
        echo "    status        Show service status and resource usage"
        echo ""
        echo "  Maintenance:"
        echo "    backup        Backup MongoDB data"
        echo "    dev           Start MongoDB only (for local development)"
        echo ""
        echo "  Services: nginx, frontend, backend, ui, avatar, recognition, mongodb"
        echo ""
        exit 1
        ;;
esac
