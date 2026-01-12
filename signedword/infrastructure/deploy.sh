#!/bin/bash
# =============================================================================
# SignedWord AWS Infrastructure Deployment Script
# =============================================================================
# Usage:
#   ./deploy.sh create    - Create new stack
#   ./deploy.sh update    - Update existing stack
#   ./deploy.sh delete    - Delete stack
#   ./deploy.sh status    - Check stack status
#   ./deploy.sh outputs   - Get stack outputs
# =============================================================================

set -e

# Configuration
STACK_NAME="signedword-infrastructure"
REGION="${AWS_REGION:-us-east-1}"
TEMPLATE_FILE="$(dirname "$0")/cloudformation.yaml"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not installed. Install with: pip install awscli"
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run: aws configure"
        exit 1
    fi

    log_info "AWS CLI configured for account: $(aws sts get-caller-identity --query Account --output text)"
}

# Get parameters from environment or prompt
get_parameters() {
    # EC2 Key Pair
    if [ -z "$EC2_KEY_PAIR" ]; then
        log_info "Available EC2 key pairs:"
        aws ec2 describe-key-pairs --query 'KeyPairs[].KeyName' --output table --region $REGION
        read -p "Enter EC2 key pair name: " EC2_KEY_PAIR
    fi

    # MongoDB connection string
    if [ -z "$MONGODB_URI" ]; then
        read -p "Enter MongoDB connection string: " MONGODB_URI
    fi

    # SonZo SLR bucket
    if [ -z "$SONZO_SLR_BUCKET" ]; then
        SONZO_SLR_BUCKET="sonzo-slr-training"
        log_info "Using default SonZo SLR bucket: $SONZO_SLR_BUCKET"
    fi
}

# Create stack
create_stack() {
    log_info "Creating CloudFormation stack: $STACK_NAME"

    get_parameters

    aws cloudformation create-stack \
        --stack-name $STACK_NAME \
        --template-body file://$TEMPLATE_FILE \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $REGION \
        --parameters \
            ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
            ParameterKey=EC2KeyPair,ParameterValue=$EC2_KEY_PAIR \
            ParameterKey=MongoDBConnectionString,ParameterValue=$MONGODB_URI \
            ParameterKey=SonZoSLRBucketName,ParameterValue=$SONZO_SLR_BUCKET \
        --tags \
            Key=Project,Value=SignedWord \
            Key=Environment,Value=$ENVIRONMENT \
            Key=ManagedBy,Value=CloudFormation

    log_info "Stack creation initiated. Waiting for completion..."

    aws cloudformation wait stack-create-complete \
        --stack-name $STACK_NAME \
        --region $REGION

    log_info "Stack created successfully!"
    show_outputs
}

# Update stack
update_stack() {
    log_info "Updating CloudFormation stack: $STACK_NAME"

    get_parameters

    aws cloudformation update-stack \
        --stack-name $STACK_NAME \
        --template-body file://$TEMPLATE_FILE \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $REGION \
        --parameters \
            ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
            ParameterKey=EC2KeyPair,ParameterValue=$EC2_KEY_PAIR \
            ParameterKey=MongoDBConnectionString,ParameterValue=$MONGODB_URI \
            ParameterKey=SonZoSLRBucketName,ParameterValue=$SONZO_SLR_BUCKET

    log_info "Stack update initiated. Waiting for completion..."

    aws cloudformation wait stack-update-complete \
        --stack-name $STACK_NAME \
        --region $REGION

    log_info "Stack updated successfully!"
    show_outputs
}

# Delete stack
delete_stack() {
    log_warn "This will delete all SignedWord AWS resources!"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Cancelled."
        exit 0
    fi

    # Empty S3 buckets first (required before deletion)
    log_info "Emptying S3 buckets..."

    CONTENT_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].Outputs[?OutputKey=='ContentBucketName'].OutputValue" \
        --output text --region $REGION 2>/dev/null || echo "")

    USER_DATA_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].Outputs[?OutputKey=='UserDataBucketName'].OutputValue" \
        --output text --region $REGION 2>/dev/null || echo "")

    if [ -n "$CONTENT_BUCKET" ]; then
        aws s3 rm s3://$CONTENT_BUCKET --recursive --region $REGION || true
    fi

    if [ -n "$USER_DATA_BUCKET" ]; then
        aws s3 rm s3://$USER_DATA_BUCKET --recursive --region $REGION || true
    fi

    log_info "Deleting CloudFormation stack: $STACK_NAME"

    aws cloudformation delete-stack \
        --stack-name $STACK_NAME \
        --region $REGION

    log_info "Stack deletion initiated. Waiting for completion..."

    aws cloudformation wait stack-delete-complete \
        --stack-name $STACK_NAME \
        --region $REGION

    log_info "Stack deleted successfully!"
}

# Show stack status
show_status() {
    log_info "Stack status for: $STACK_NAME"

    aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].{Status:StackStatus,Created:CreationTime,Updated:LastUpdatedTime}" \
        --output table \
        --region $REGION
}

# Show stack outputs
show_outputs() {
    log_info "Stack outputs for: $STACK_NAME"

    aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].Outputs[*].{Key:OutputKey,Value:OutputValue}" \
        --output table \
        --region $REGION
}

# SSH into EC2 instance
ssh_to_ec2() {
    EC2_IP=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].Outputs[?OutputKey=='EC2PublicIP'].OutputValue" \
        --output text --region $REGION)

    if [ -z "$EC2_IP" ]; then
        log_error "Could not find EC2 IP. Is the stack deployed?"
        exit 1
    fi

    log_info "Connecting to EC2 at $EC2_IP..."
    ssh -i ~/.ssh/$EC2_KEY_PAIR.pem ubuntu@$EC2_IP
}

# Main
check_prerequisites

case "${1:-}" in
    create)
        create_stack
        ;;
    update)
        update_stack
        ;;
    delete)
        delete_stack
        ;;
    status)
        show_status
        ;;
    outputs)
        show_outputs
        ;;
    ssh)
        ssh_to_ec2
        ;;
    *)
        echo "SignedWord Infrastructure Deployment"
        echo ""
        echo "Usage: $0 {create|update|delete|status|outputs|ssh}"
        echo ""
        echo "Commands:"
        echo "  create  - Create new CloudFormation stack"
        echo "  update  - Update existing stack"
        echo "  delete  - Delete stack (with confirmation)"
        echo "  status  - Show stack status"
        echo "  outputs - Show stack outputs (IPs, bucket names, etc.)"
        echo "  ssh     - SSH into the EC2 instance"
        echo ""
        echo "Environment variables:"
        echo "  AWS_REGION       - AWS region (default: us-east-1)"
        echo "  ENVIRONMENT      - Environment name (default: production)"
        echo "  EC2_KEY_PAIR     - EC2 SSH key pair name"
        echo "  MONGODB_URI      - MongoDB connection string"
        echo "  SONZO_SLR_BUCKET - SonZo SLR training bucket name"
        exit 1
        ;;
esac
