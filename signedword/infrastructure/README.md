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

3. **Deploy API code:**
   ```bash
   cd /opt/signedword
   git clone <api-repo>
   npm install
   pm2 start ecosystem.config.js
   ```

4. **Configure nginx:**
   ```bash
   sudo certbot --nginx -d signedword.sonzo.io
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

## Security Notes

1. **EC2 Security Group**: Restrict SSH (port 22) to your IP in production
2. **MongoDB**: Use VPC peering or IP whitelist
3. **S3**: All buckets block public access
4. **API**: JWT authentication required
5. **User Data**: Encrypted at rest (S3 default)
