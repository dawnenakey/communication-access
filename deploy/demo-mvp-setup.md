# Demo EC2 MVP Setup Guide

Get **demo.sonzo.io** running on your deep learning EC2 with minimal setup.

## Prerequisites

- Deep learning EC2 (Virginia region)
- `demo.sonzo.io` DNS pointing to EC2 public IP
- SSH access

---

## Step 1: SSH and Update Repo

```bash
ssh ubuntu@YOUR_DEMO_EC2_IP
cd /home/ubuntu/communication-access
git pull origin main
```

---

## Step 2: Python Environment

```bash
cd /home/ubuntu/communication-access

# Use Python 3.11 if available (mediapipe works better)
python3 --version

# Create venv if not exists
python3 -m venv .venv
source .venv/bin/activate

# Core deps for launch.py --demo
pip install fastapi uvicorn opencv-python-headless numpy pillow

# Required by launch.py check (use CPU torch to save space)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: mediapipe for hand detection (Python 3.11 recommended)
# pip install mediapipe

# Optional: avatar face swap (heavy - add if avatar create fails)
# pip install insightface onnxruntime
```

---

## Step 3: Test Services

```bash
cd /home/ubuntu/communication-access
source .venv/bin/activate
python launch.py --check-deps
python launch.py --demo
```

You should see:
- UI: http://localhost:8081
- Avatar: http://localhost:8080
- Recognition: http://localhost:8082

Press Ctrl+C to stop. If any service fails, check the error and install missing deps.

---

## Step 4: Nginx Config

```bash
# Copy nginx config
sudo cp /home/ubuntu/communication-access/deploy/demo.sonzo.io.conf /etc/nginx/sites-available/demo.sonzo.io
sudo ln -sf /etc/nginx/sites-available/demo.sonzo.io /etc/nginx/sites-enabled/

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

---

## Step 5: SSL (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d demo.sonzo.io
```

---

## Step 6: Run as Systemd Service

```bash
sudo tee /etc/systemd/system/sonzo-demo.service << 'EOF'
[Unit]
Description=SonZo AI Demo Services
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/communication-access
Environment="PATH=/home/ubuntu/communication-access/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/communication-access/.venv/bin/python launch.py --demo
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sonzo-demo
sudo systemctl start sonzo-demo
sudo systemctl status sonzo-demo
```

---

## Step 7: Verify

1. Open **https://demo.sonzo.io**
2. Complete onboarding
3. Test sign recognition (webcam)
4. Test avatar (upload photo, generate sign)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port in use | `sudo lsof -i :8080` then kill or change port |
| MediaPipe error | Use `--demo` mode (works without mediapipe) |
| Avatar fails | Avatar needs insightface/onnx – optional for MVP |
| 502 Bad Gateway | Check `systemctl status sonzo-demo` and `journalctl -u sonzo-demo -f` |

---

## Quick Commands

```bash
# Restart demo services
sudo systemctl restart sonzo-demo

# View logs
journalctl -u sonzo-demo -f

# Stop
sudo systemctl stop sonzo-demo
```
