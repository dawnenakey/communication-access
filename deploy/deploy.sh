#!/bin/bash
# ============================================================================
# SonZo AI - Deployment Script
# ============================================================================
# Deploy SonZo to EC2 with SSL/HTTPS
#
# Usage:
#   ./deploy.sh setup      # First-time setup
#   ./deploy.sh start      # Start services
#   ./deploy.sh stop       # Stop services
#   ./deploy.sh restart    # Restart services
#   ./deploy.sh logs       # View logs
#   ./deploy.sh ssl        # Setup SSL certificates
#
# ============================================================================

set -e

# Configuration
DOMAIN="sonzo.io"
APP_DOMAIN="app.sonzo.io"
EMAIL="dawnena@sonzo.ai"
PROJECT_DIR="/opt/sonzo"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# ============================================================================
# Setup
# ============================================================================

setup() {
    log_info "Setting up SonZo AI..."

    # Create project directory
    sudo mkdir -p $PROJECT_DIR
    sudo chown $USER:$USER $PROJECT_DIR

    # Copy files
    log_info "Copying project files..."
    cp -r ../* $PROJECT_DIR/

    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        log_info "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi

    # Install Docker Compose if not present
    if ! command -v docker-compose &> /dev/null; then
        log_info "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi

    # Install NVIDIA Container Toolkit if GPU available
    if command -v nvidia-smi &> /dev/null; then
        log_info "Setting up NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
    fi

    # Create SSL directory
    mkdir -p $PROJECT_DIR/deploy/ssl

    log_info "Setup complete!"
    log_info "Next steps:"
    echo "  1. Run: ./deploy.sh ssl     # Get SSL certificates"
    echo "  2. Run: ./deploy.sh start   # Start services"
}

# ============================================================================
# SSL Setup
# ============================================================================

setup_ssl() {
    log_info "Setting up SSL certificates..."

    # Install certbot
    if ! command -v certbot &> /dev/null; then
        log_info "Installing Certbot..."
        sudo apt-get update
        sudo apt-get install -y certbot
    fi

    # Stop nginx if running
    docker-compose -f $PROJECT_DIR/deploy/docker-compose.yml down nginx 2>/dev/null || true

    # Get certificates
    log_info "Obtaining SSL certificates for $DOMAIN and $APP_DOMAIN..."
    sudo certbot certonly --standalone \
        -d $DOMAIN \
        -d www.$DOMAIN \
        -d $APP_DOMAIN \
        -d api.$DOMAIN \
        --email $EMAIL \
        --agree-tos \
        --non-interactive

    # Copy certificates
    log_info "Copying certificates..."
    sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $PROJECT_DIR/deploy/ssl/
    sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $PROJECT_DIR/deploy/ssl/
    sudo chown $USER:$USER $PROJECT_DIR/deploy/ssl/*.pem

    # Setup auto-renewal
    log_info "Setting up auto-renewal..."
    (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && cp /etc/letsencrypt/live/$DOMAIN/*.pem $PROJECT_DIR/deploy/ssl/ && docker-compose -f $PROJECT_DIR/deploy/docker-compose.yml restart nginx") | crontab -

    log_info "SSL setup complete!"
}

# ============================================================================
# Service Management
# ============================================================================

start() {
    log_info "Starting SonZo AI services..."
    cd $PROJECT_DIR/deploy

    # Check if SSL certificates exist
    if [ ! -f "ssl/fullchain.pem" ]; then
        log_warn "SSL certificates not found. Running in HTTP mode."
        log_warn "Run './deploy.sh ssl' to set up HTTPS."

        # Use development config without SSL
        docker-compose up -d ui avatar recognition
    else
        docker-compose up -d
    fi

    log_info "Services started!"
    log_info "Access the app at: https://$APP_DOMAIN"
}

stop() {
    log_info "Stopping SonZo AI services..."
    cd $PROJECT_DIR/deploy
    docker-compose down
    log_info "Services stopped."
}

restart() {
    log_info "Restarting SonZo AI services..."
    stop
    start
}

logs() {
    cd $PROJECT_DIR/deploy
    docker-compose logs -f --tail=100
}

status() {
    log_info "Service Status:"
    cd $PROJECT_DIR/deploy
    docker-compose ps
}

# ============================================================================
# Development Mode
# ============================================================================

dev() {
    log_info "Starting in development mode..."
    cd $PROJECT_DIR

    # Start with demo recognition
    python launch.py --demo
}

# ============================================================================
# Main
# ============================================================================

case "$1" in
    setup)
        setup
        ;;
    ssl)
        setup_ssl
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    dev)
        dev
        ;;
    *)
        echo "SonZo AI Deployment Script"
        echo ""
        echo "Usage: $0 {setup|ssl|start|stop|restart|logs|status|dev}"
        echo ""
        echo "Commands:"
        echo "  setup    - First-time setup (install Docker, copy files)"
        echo "  ssl      - Setup SSL certificates with Let's Encrypt"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - View service logs"
        echo "  status   - Show service status"
        echo "  dev      - Start in development mode (local)"
        exit 1
        ;;
esac
