# 🚀 Speech-to-Symbol API Deployment Guide

This guide shows you how to deploy your Speech-to-Symbol API to production so anyone can access it.

## 📋 Prerequisites

- Python 3.8+ installed
- Git installed
- Audio files for testing (optional)

## 🎯 Quick Start - Choose Your Platform

### Option 1: Heroku (Recommended for Beginners)

**Pros:** Free tier, easy setup, automatic HTTPS  
**Cons:** Limited resources, 30-minute timeout

#### Step 1: Install Heroku CLI
```bash
# Windows
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# macOS
brew install heroku/brew/heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

#### Step 2: Deploy
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-speech2symbol-api

# Add files to git
git init
git add .
git commit -m "Initial deployment"

# Deploy
git push heroku main

# Open your app
heroku open
```

**Your API will be available at:** `https://your-speech2symbol-api.herokuapp.com`

---

### Option 2: Railway (Modern Alternative)

**Pros:** Free tier, fast deployment, good performance  
**Cons:** Limited free tier hours

#### Step 1: Setup Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

**Your API will be available at:** `https://your-app-name.railway.app`

---

### Option 3: Render (Free Tier Available)

**Pros:** Free tier, good performance, easy setup  
**Cons:** Limited free tier hours

#### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New Web Service"

#### Step 2: Connect Repository
1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Deploy!

**Your API will be available at:** `https://your-app-name.onrender.com`

---

### Option 4: Google Cloud Run (Free Tier)

**Pros:** Generous free tier, scalable, good performance  
**Cons:** Requires Google Cloud account

#### Step 1: Setup Google Cloud
```bash
# Install Google Cloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

#### Step 2: Deploy
```bash
# Build and deploy
gcloud run deploy speech2symbol-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

**Your API will be available at:** `https://speech2symbol-api-xxxxx-uc.a.run.app`

---

### Option 5: AWS Lambda + API Gateway

**Pros:** Pay-per-use, highly scalable  
**Cons:** More complex setup, cold starts

#### Step 1: Install AWS CLI
```bash
# Download from: https://aws.amazon.com/cli/
aws configure
```

#### Step 2: Deploy with Serverless Framework
```bash
# Install Serverless Framework
npm install -g serverless

# Deploy
serverless deploy
```

---

## 🐳 Docker Deployment (Advanced)

### Local Docker Build
```bash
# Build image
docker build -t speech2symbol-api .

# Run locally
docker run -p 5000:5000 speech2symbol-api

# Test
curl http://localhost:5000/health
```

### Deploy to Cloud Platforms

#### Google Cloud Run
```bash
# Build and push
docker build -t gcr.io/YOUR_PROJECT_ID/speech2symbol-api .
docker push gcr.io/YOUR_PROJECT_ID/speech2symbol-api

# Deploy
gcloud run deploy speech2symbol-api \
  --image gcr.io/YOUR_PROJECT_ID/speech2symbol-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker build -t speech2symbol-api .
docker tag speech2symbol-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/speech2symbol-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/speech2symbol-api:latest
```

#### Azure Container Instances
```bash
# Build and push to ACR
az acr build --registry YOUR_REGISTRY_NAME --image speech2symbol-api .

# Deploy
az container create \
  --resource-group YOUR_RESOURCE_GROUP \
  --name speech2symbol-api \
  --image YOUR_REGISTRY_NAME.azurecr.io/speech2symbol-api:latest \
  --ports 5000 \
  --dns-name-label your-app-name
```

---

## 🌐 Domain & SSL Setup

### Custom Domain (Optional)
1. **Heroku:** `heroku domains:add yourdomain.com`
2. **Railway:** Add custom domain in dashboard
3. **Render:** Add custom domain in dashboard
4. **Cloud Run:** Use Cloud Load Balancer

### SSL Certificate
- **Heroku:** Automatic HTTPS
- **Railway:** Automatic HTTPS
- **Render:** Automatic HTTPS
- **Cloud Run:** Automatic HTTPS

---

## 📊 Monitoring & Analytics

### Health Checks
```bash
# Check API health
curl https://your-api-url.com/health

# Get statistics
curl https://your-api-url.com/stats
```

### Logs
```bash
# Heroku
heroku logs --tail

# Railway
railway logs

# Render
# View in dashboard

# Cloud Run
gcloud logs read --service=speech2symbol-api
```

---

## 🔧 Environment Variables

Set these in your deployment platform:

```bash
# Optional: Custom model path
CUSTOM_MODEL_PATH=/path/to/custom/model

# Optional: Confidence threshold
CONFIDENCE_THRESHOLD=0.7

# Optional: Enable debug mode
FLASK_ENV=production
```

---

## 🧪 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-api-url.com/health
```

### 2. Text Conversion
```bash
curl -X POST https://your-api-url.com/convert/text \
  -H "Content-Type: application/json" \
  -d '{"text": "two plus three equals five"}'
```

### 3. Audio Upload
```bash
curl -X POST https://your-api-url.com/convert/audio \
  -F "audio=@test1.mp3"
```

### 4. Web Interface
Open `https://your-api-url.com` in your browser!

---

## 🚨 Troubleshooting

### Common Issues

#### 1. "Module not found" errors
```bash
# Make sure all dependencies are in requirements.txt
pip freeze > requirements.txt
```

#### 2. Memory issues
- Increase memory allocation in your platform
- Use smaller models (base instead of large)

#### 3. Timeout errors
- Reduce model size
- Optimize processing pipeline
- Use async processing for large files

#### 4. CORS errors
- CORS is already enabled in the app
- Check if your frontend is making requests correctly

### Debug Commands
```bash
# Check logs
heroku logs --tail

# Restart app
heroku restart

# Check app status
heroku ps
```

---

## 📈 Scaling Considerations

### For High Traffic
1. **Use larger instances** (2GB+ RAM)
2. **Enable caching** (Redis)
3. **Use CDN** for static files
4. **Implement rate limiting**
5. **Add load balancing**

### Cost Optimization
1. **Use free tiers** when possible
2. **Monitor usage** regularly
3. **Scale down** during low traffic
4. **Use spot instances** (AWS)

---

## 🔒 Security Best Practices

1. **Change default secret key**
2. **Enable HTTPS** (automatic on most platforms)
3. **Add rate limiting**
4. **Validate file uploads**
5. **Monitor logs** for suspicious activity
6. **Keep dependencies updated**

---

## 📞 Support

If you encounter issues:

1. Check the logs: `heroku logs --tail`
2. Verify environment variables
3. Test locally first
4. Check platform status pages
5. Review this deployment guide

---

## 🎉 Success!

Your Speech-to-Symbol API is now live and accessible to everyone! 

**Next Steps:**
- Share your API URL with users
- Monitor usage and performance
- Add more features as needed
- Consider adding authentication for production use

**Example Usage:**
```bash
# Your API is now available at:
https://your-app-name.herokuapp.com

# Users can access the web interface at:
https://your-app-name.herokuapp.com

# API documentation at:
https://your-app-name.herokuapp.com/api
``` 