# SignedWord AWS Infrastructure

ASL Bible Devotions App - Infrastructure as Code

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SignedWord                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ React    │───▶│ CloudFront   │───▶│ S3: signedword-content│  │
│  │ Native   │    │ CDN          │    │ (devotional videos)   │  │
│  │ App      │    └──────────────┘    └───────────────────────┘  │
│  │          │                                                    │
│  │          │    ┌──────────────┐    ┌───────────────────────┐  │
│  │          │───▶│ EC2 API      │───▶│ MongoDB               │  │
│  │          │    │ (Node.js)    │    │ (shared with SonZo)   │  │
│  └──────────┘    └──────────────┘    └───────────────────────┘  │
│       │                                                          │
│       │          ┌──────────────┐    ┌───────────────────────┐  │
│       └─────────▶│ S3: user-data│───▶│ Lambda: Video         │  │
│      (uploads)   │ (user videos)│    │ Processing            │  │
│                  └──────────────┘    └───────────────────────┘  │
│                         │                       │                │
│                         │    ┌──────────────────▼──────────┐    │
│                         │    │ EventBridge (nightly)       │    │
│                         │    └──────────────────┬──────────┘    │
│                         │                       │                │
│                         ▼                       ▼                │
│                  ┌─────────────────────────────────────────┐    │
│                  │ S3: sonzo-slr-training                  │    │
│                  │ (feeds SonZo SLR model training)        │    │
│                  └─────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Resources Created

| Resource | Type | Purpose |
|----------|------|---------|
| EC2 Instance | t3.medium | API server (Node.js/Express) |
| S3: signedword-content | Bucket | Devotional videos (via CloudFront) |
| S3: signedword-user-data | Bucket | User uploads |
| CloudFront Distribution | CDN | Video delivery |
| Lambda: video-processor | Function | Process uploads, add metadata |
| Lambda: sonzo-sync | Function | Nightly sync to SonZo SLR |
| EventBridge Rule | Schedule | Triggers nightly sync at 4 AM UTC |
| IAM Roles | Policies | Least-privilege access |

## Quick Start

### Prerequisites

1. AWS CLI installed and configured
2. EC2 key pair created
3. MongoDB connection string (can use existing SonZo cluster)

### Deploy

```bash
# Set environment variables
export AWS_REGION=us-east-1
export EC2_KEY_PAIR=your-key-pair-name
export MONGODB_URI="mongodb+srv://..."
export SONZO_SLR_BUCKET=sonzo-slr-training

# Deploy infrastructure
./deploy.sh create
```

### Post-Deployment

1. **Get EC2 IP:**
   ```bash
   ./deploy.sh outputs
   ```

2. **SSH into server:**
   ```bash
   ./deploy.sh ssh
   ```

3. **Initial EC2 Setup** (run once after first deployment):
   ```bash
   # Update system
   sudo yum update -y

   # Install Node.js 20 LTS
   curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
   sudo yum install -y nodejs

   # Install PM2 globally
   sudo npm install -g pm2

   # Install nginx
   sudo amazon-linux-extras install nginx1 -y
   sudo systemctl enable nginx
   sudo systemctl start nginx

   # Install certbot for SSL
   sudo yum install -y certbot python3-certbot-nginx

   # Create app directory
   sudo mkdir -p /opt/signedword
   sudo chown ec2-user:ec2-user /opt/signedword
   ```

4. **Deploy API code:**
   ```bash
   cd /opt/signedword
   git clone <api-repo> api
   cd api
   npm install --production

   # Create environment file
   cat > .env << EOF
   NODE_ENV=production
   PORT=3000
   MONGODB_URI=mongodb+srv://...
   JWT_SECRET=$(openssl rand -base64 32)
   AWS_REGION=us-east-1
   S3_CONTENT_BUCKET=signedword-content-<account-id>
   S3_USER_BUCKET=signedword-user-data-<account-id>
   EOF

   # Start with PM2
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```

5. **Configure nginx:**
   ```bash
   # Create nginx config
   sudo cat > /etc/nginx/conf.d/signedword.conf << 'EOF'
   server {
       listen 80;
       server_name signedword.sonzo.io;

       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_cache_bypass $http_upgrade;
       }

       # Increase upload size for video uploads
       client_max_body_size 100M;
   }
   EOF

   # Test and reload
   sudo nginx -t
   sudo systemctl reload nginx

   # Get SSL certificate
   sudo certbot --nginx -d signedword.sonzo.io
   ```

6. **Verify deployment:**
   ```bash
   # Check API health
   curl http://localhost:3000/health

   # Check PM2 status
   pm2 status

   # View logs
   pm2 logs signedword-api
   ```

## IAM Policies (Least Privilege)

### EC2 Role
- Read/write to signedword S3 buckets
- Read from sonzo-slr-training bucket
- CloudWatch logs

### Lambda Role
- Read/write to user-data bucket
- Write to sonzo-slr-training bucket
- CloudWatch logs

## S3 → SonZo Pipeline

User uploads flow to SonZo SLR training:

1. **User uploads video** → `signedword-user-data/uploads/`
2. **Lambda trigger** → Adds metadata (consent flag, timestamp)
3. **Nightly sync** → Copies consented videos to `sonzo-slr-training/signedword/`

### Consent Handling

Videos only sync to SonZo if:
- User grants consent in app
- Metadata `consent_flag = 'granted'`

## Costs (Estimated)

| Resource | Monthly Cost |
|----------|-------------|
| EC2 t3.medium | ~$30 |
| S3 (50GB) | ~$1 |
| CloudFront (100GB transfer) | ~$9 |
| Lambda (10k invocations) | ~$0.01 |
| **Total** | **~$40/month** |

## Management Commands

```bash
# Check status
./deploy.sh status

# View outputs
./deploy.sh outputs

# Update stack
./deploy.sh update

# Delete everything
./deploy.sh delete

# SSH to EC2
./deploy.sh ssh
```

## DNS Configuration

Add to your DNS (Route 53 or other):

```
signedword.sonzo.io  A     <EC2-Elastic-IP>
cdn.signedword.sonzo.io  CNAME  <CloudFront-Domain>
```

## App Deployment (Expo)

The SignedWord app is built with Expo for cross-platform deployment (Web, iOS, Android).

### Web Deployment

```bash
cd signedword/app

# Install dependencies
npm install

# Build for web
npx expo export --platform web

# Deploy to S3 (static hosting) or serve via nginx
aws s3 sync dist/ s3://signedword-web-<account-id>/ --delete
```

### Mobile Deployment

```bash
# Build for iOS (requires Apple Developer account)
eas build --platform ios

# Build for Android
eas build --platform android

# Submit to app stores
eas submit --platform ios
eas submit --platform android
```

### Development

```bash
# Start development server (all platforms)
npx expo start

# Web only
npx expo start --web

# With Expo Go on device
npx expo start --tunnel
```

### Environment Configuration

Create `app.config.js` for environment-specific settings:

```javascript
export default {
  expo: {
    extra: {
      apiUrl: process.env.API_URL || 'https://signedword.sonzo.io/api',
      cdnUrl: process.env.CDN_URL || 'https://cdn.signedword.sonzo.io',
    }
  }
}
```

## Security Notes

1. **EC2 Security Group**: Restrict SSH (port 22) to your IP in production
2. **MongoDB**: Use VPC peering or IP whitelist
3. **S3**: All buckets block public access
4. **API**: JWT authentication required
5. **User Data**: Encrypted at rest (S3 default)
6. **Camera/Video**: Uses secure MediaRecorder API with user permission

## Troubleshooting

### EC2 Connection Issues

```bash
# Check EC2 instance status
aws ec2 describe-instance-status --instance-ids <instance-id>

# Check security group rules
aws ec2 describe-security-groups --group-ids <sg-id>

# Check nginx status
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

### API Issues

```bash
# Check PM2 logs
pm2 logs signedword-api --lines 100

# Check environment
pm2 env 0

# Restart API
pm2 restart signedword-api
```

### S3 Upload Issues

```bash
# Check Lambda logs
aws logs tail /aws/lambda/signedword-video-processor --follow

# Test S3 permissions
aws s3 ls s3://signedword-user-data-<account-id>/
```
