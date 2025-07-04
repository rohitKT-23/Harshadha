#!/usr/bin/env python3
"""
Deployment script for Speech-to-Symbol API
Automates deployment to various cloud platforms
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success: {result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr.strip()}")
        return None

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    tools = {
        'python': 'python --version',
        'pip': 'pip --version',
        'git': 'git --version'
    }
    
    missing = []
    for tool, command in tools.items():
        if run_command(command, f"Checking {tool}") is None:
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing tools: {', '.join(missing)}")
        print("Please install the missing tools and try again.")
        return False
    
    print("✅ All prerequisites met!")
    return True

def deploy_heroku():
    """Deploy to Heroku"""
    print("\n🚀 Deploying to Heroku...")
    
    # Check if Heroku CLI is installed
    if run_command("heroku --version", "Checking Heroku CLI") is None:
        print("❌ Heroku CLI not found. Please install it first:")
        print("   Windows: Download from https://devcenter.heroku.com/articles/heroku-cli")
        print("   macOS: brew install heroku/brew/heroku")
        print("   Linux: curl https://cli-assets.heroku.com/install.sh | sh")
        return False
    
    # Login to Heroku
    if run_command("heroku login", "Logging into Heroku") is None:
        return False
    
    # Create app name
    app_name = input("Enter Heroku app name (or press Enter for auto-generated): ").strip()
    if not app_name:
        app_name = "speech2symbol-api"
    
    # Create app
    if run_command(f"heroku create {app_name}", "Creating Heroku app") is None:
        return False
    
    # Initialize git if not already done
    if not Path(".git").exists():
        run_command("git init", "Initializing git repository")
        run_command("git add .", "Adding files to git")
        run_command('git commit -m "Initial deployment"', "Making initial commit")
    
    # Deploy
    if run_command("git push heroku main", "Deploying to Heroku") is None:
        return False
    
    # Open app
    run_command("heroku open", "Opening app in browser")
    
    print("✅ Heroku deployment complete!")
    return True

def deploy_railway():
    """Deploy to Railway"""
    print("\n🚀 Deploying to Railway...")
    
    # Check if Railway CLI is installed
    if run_command("railway --version", "Checking Railway CLI") is None:
        print("❌ Railway CLI not found. Please install it first:")
        print("   npm install -g @railway/cli")
        return False
    
    # Login to Railway
    if run_command("railway login", "Logging into Railway") is None:
        return False
    
    # Initialize project
    if run_command("railway init", "Initializing Railway project") is None:
        return False
    
    # Deploy
    if run_command("railway up", "Deploying to Railway") is None:
        return False
    
    print("✅ Railway deployment complete!")
    return True

def deploy_render():
    """Deploy to Render"""
    print("\n🚀 Deploying to Render...")
    print("📝 Manual steps required:")
    print("1. Go to https://render.com")
    print("2. Sign up with GitHub")
    print("3. Click 'New Web Service'")
    print("4. Connect your GitHub repository")
    print("5. Set build command: pip install -r requirements.txt")
    print("6. Set start command: gunicorn app:app")
    print("7. Click 'Create Web Service'")
    
    input("Press Enter when you've completed the manual steps...")
    print("✅ Render deployment instructions provided!")
    return True

def deploy_google_cloud():
    """Deploy to Google Cloud Run"""
    print("\n🚀 Deploying to Google Cloud Run...")
    
    # Check if gcloud is installed
    if run_command("gcloud --version", "Checking Google Cloud CLI") is None:
        print("❌ Google Cloud CLI not found. Please install it first:")
        print("   Download from: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Login to Google Cloud
    if run_command("gcloud auth login", "Logging into Google Cloud") is None:
        return False
    
    # Set project
    project_id = input("Enter your Google Cloud project ID: ").strip()
    if not project_id:
        print("❌ Project ID is required")
        return False
    
    run_command(f"gcloud config set project {project_id}", "Setting project")
    
    # Deploy
    if run_command(
        f"gcloud run deploy speech2symbol-api --source . --platform managed --region us-central1 --allow-unauthenticated --memory 2Gi --cpu 2",
        "Deploying to Cloud Run"
    ) is None:
        return False
    
    print("✅ Google Cloud Run deployment complete!")
    return True

def test_deployment(url):
    """Test the deployed API"""
    print(f"\n🧪 Testing deployment at {url}...")
    
    import requests
    
    try:
        # Test health endpoint
        health_url = f"{url}/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            
            # Test text conversion
            text_url = f"{url}/convert/text"
            test_data = {"text": "two plus three equals five"}
            response = requests.post(text_url, json=test_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    print("✅ Text conversion test passed!")
                    print(f"   Input: {result.get('input_text')}")
                    print(f"   Output: {result.get('final_output')}")
                else:
                    print("❌ Text conversion failed")
            else:
                print(f"❌ Text conversion test failed: {response.status_code}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Main deployment function"""
    print("🎤 Speech-to-Symbol API Deployment Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Show deployment options
    print("\n📋 Choose deployment platform:")
    print("1. Heroku (Recommended for beginners)")
    print("2. Railway (Modern alternative)")
    print("3. Render (Free tier available)")
    print("4. Google Cloud Run (Advanced)")
    print("5. Test existing deployment")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    success = False
    deployment_url = None
    
    if choice == "1":
        success = deploy_heroku()
        if success:
            deployment_url = "https://your-app-name.herokuapp.com"
    elif choice == "2":
        success = deploy_railway()
        if success:
            deployment_url = "https://your-app-name.railway.app"
    elif choice == "3":
        success = deploy_render()
        if success:
            deployment_url = "https://your-app-name.onrender.com"
    elif choice == "4":
        success = deploy_google_cloud()
        if success:
            deployment_url = "https://speech2symbol-api-xxxxx-uc.a.run.app"
    elif choice == "5":
        url = input("Enter your deployment URL: ").strip()
        if url:
            test_deployment(url)
        return
    else:
        print("❌ Invalid choice")
        return
    
    if success:
        print(f"\n🎉 Deployment successful!")
        print(f"🌐 Your API is available at: {deployment_url}")
        print(f"📱 Web interface: {deployment_url}")
        print(f"📚 API docs: {deployment_url}/api")
        
        # Test the deployment
        if deployment_url and deployment_url != "https://your-app-name.herokuapp.com":
            test_deployment(deployment_url)
    else:
        print("\n❌ Deployment failed. Please check the errors above.")

if __name__ == "__main__":
    main() 