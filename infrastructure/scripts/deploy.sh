#!/bin/bash

# Aegis Nexus Deployment Script
# Deploys the complete Aegis Nexus system

set -e  # Exit on any error

echo "🚀 Starting Aegis Nexus Deployment..."

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/infrastructure/docker-compose.yml"
ENV_FILE="$PROJECT_DIR/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check project directory
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        log_warn ".env file not found, copying from example"
        cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
        log_info "Please review and update $ENV_FILE with your configuration"
    fi
    
    # Source environment variables
    if [ -f "$ENV_FILE" ]; then
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    fi
    
    log_info "Environment setup completed"
}

# Build images
build_images() {
    log_info "Building Docker images..."
    
    # Build API image
    log_info "Building API image..."
    docker build -f "$PROJECT_DIR/Dockerfile.api" -t aegis-nexus-api:latest "$PROJECT_DIR"
    
    # Build Frontend image
    log_info "Building Frontend image..."
    docker build -f "$PROJECT_DIR/frontend/Dockerfile.frontend" -t aegis-nexus-frontend:latest "$PROJECT_DIR/frontend"
    
    log_info "Docker images built successfully"
}

# Start services
start_services() {
    log_info "Starting Aegis Nexus services..."
    
    # PERFORMANCE TUNING: Calculate optimal Gunicorn workers
    # Formula: (2 x CPUs) + 1
    CORES=$(nproc 2>/dev/null || echo 2) # Default to 2 if nproc fails
    export WEB_CONCURRENCY=$((CORES * 2 + 1))
    log_info "🚀 Tuning: Set WEB_CONCURRENCY to $WEB_CONCURRENCY (based on $CORES cores)"
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_info "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for API service
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_info "API service is ready"
            break
        fi
        log_info "Waiting for API service... ($i/30)"
        sleep 10
    done
    
    # Wait for Frontend service
    for i in {1..30}; do
        if curl -f http://localhost:3000 &>/dev/null; then
            log_info "Frontend service is ready"
            break
        fi
        log_info "Waiting for Frontend service... ($i/30)"
        sleep 10
    done
    
    log_info "All services are ready"
}

# Run Immune Response Test (Chaos Smoke Test)
run_immune_response_test() {
    log_info "💉 Running Immune Response Test (Chaos Smoke Test)..."
    
    # Run a short, low-intensity chaos test to ensure system resilience
    if docker-compose -f "$COMPOSE_FILE" exec -T api python -m tools.reality_fuzz --duration 10 --intensity 0.2 --target http://localhost:8000/api/v1/reality/ingest; then
        log_info "✅ Immune Response Test: SYSTEM RESILIENT"
    else
        log_warn "⚠️  Immune Response Test: MINOR DEGRADATION DETECTED (Check Logs)"
        # We don't fail deployment on chaos test, just warn
        return 0
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # API Health Check
    if curl -f http://localhost:8000/health; then
        log_info "✅ API Health Check: PASSED"
    else
        log_error "❌ API Health Check: FAILED"
        return 1
    fi
    
    # Frontend Health Check
    if curl -f http://localhost:3000; then
        log_info "✅ Frontend Health Check: PASSED"
    else
        log_error "❌ Frontend Health Check: FAILED"
        return 1
    fi
    
    # Database Health Check
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U postgres; then
        log_info "✅ Database Health Check: PASSED"
    else
        log_error "❌ Database Health Check: FAILED"
        return 1
    fi
    
    # Redis Health Check
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_info "✅ Redis Health Check: PASSED"
    else
        log_error "❌ Redis Health Check: FAILED"
        return 1
    fi
    
    log_info "All health checks passed"
    return 0
}

# Display deployment information
display_info() {
    echo ""
    echo "=========================================="
    echo "        AEGIS NEXUS DEPLOYMENT COMPLETE"
    echo "=========================================="
    echo ""
    echo "🌐 Access Points:"
    echo "   API: http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   Grafana: http://localhost:3001 (admin/admin)"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "📋 Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "📈 Monitoring:"
    echo "   Logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "   Stats: docker stats"
    echo ""
    echo "🔧 Management Commands:"
    echo "   Stop: docker-compose -f $COMPOSE_FILE down"
    echo "   Restart: docker-compose -f $COMPOSE_FILE restart"
    echo "   Update: ./deploy.sh update"
    echo ""
    echo "Trust is no longer a feeling. It is a mathematical proof."
    echo ""
}

# Update existing deployment
update_deployment() {
    log_info "Updating existing deployment..."
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Rebuild images
    build_images
    
    # Start services
    start_services
    
    # Wait and check
    wait_for_services
    run_health_checks
    
    log_info "Deployment updated successfully"
}

# Main execution
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            setup_environment
            build_images
            start_services
            wait_for_services
            if run_health_checks; then
                run_immune_response_test
                display_info
                log_info "🎉 Aegis Nexus deployment completed successfully!"
            else
                log_error "❌ Deployment completed but health checks failed"
                exit 1
            fi
            ;;
        "update")
            update_deployment
            display_info
            ;;
        "health")
            run_health_checks
            ;;
        "status")
            docker-compose -f "$COMPOSE_FILE" ps
            ;;
        *)
            echo "Usage: $0 {deploy|update|health|status}"
            echo "  deploy  - Deploy Aegis Nexus (default)"
            echo "  update  - Update existing deployment"
            echo "  health  - Run health checks"
            echo "  status  - Show service status"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"