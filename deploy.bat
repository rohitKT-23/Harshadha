@echo off
echo 🎤 Speech-to-Symbol API Deployment Script
echo ==========================================

echo.
echo 📋 Choose deployment platform:
echo 1. Heroku (Recommended for beginners)
echo 2. Railway (Modern alternative)
echo 3. Render (Free tier available)
echo 4. Google Cloud Run (Advanced)
echo 5. Test existing deployment

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto heroku
if "%choice%"=="2" goto railway
if "%choice%"=="3" goto render
if "%choice%"=="4" goto google
if "%choice%"=="5" goto test
goto invalid

:heroku
echo.
echo 🚀 Deploying to Heroku...
echo.
echo 📝 Steps:
echo 1. Install Heroku CLI from: https://devcenter.heroku.com/articles/heroku-cli
echo 2. Open a new command prompt and run:
echo    heroku login
echo    heroku create your-app-name
echo    git init
echo    git add .
echo    git commit -m "Initial deployment"
echo    git push heroku main
echo    heroku open
echo.
echo Your API will be available at: https://your-app-name.herokuapp.com
pause
goto end

:railway
echo.
echo 🚀 Deploying to Railway...
echo.
echo 📝 Steps:
echo 1. Install Railway CLI: npm install -g @railway/cli
echo 2. Open a new command prompt and run:
echo    railway login
echo    railway init
echo    railway up
echo.
echo Your API will be available at: https://your-app-name.railway.app
pause
goto end

:render
echo.
echo 🚀 Deploying to Render...
echo.
echo 📝 Manual steps:
echo 1. Go to https://render.com
echo 2. Sign up with GitHub
echo 3. Click "New Web Service"
echo 4. Connect your GitHub repository
echo 5. Set build command: pip install -r requirements.txt
echo 6. Set start command: gunicorn app:app
echo 7. Click "Create Web Service"
echo.
echo Your API will be available at: https://your-app-name.onrender.com
pause
goto end

:google
echo.
echo 🚀 Deploying to Google Cloud Run...
echo.
echo 📝 Steps:
echo 1. Install Google Cloud CLI from: https://cloud.google.com/sdk/docs/install
echo 2. Open a new command prompt and run:
echo    gcloud auth login
echo    gcloud config set project YOUR_PROJECT_ID
echo    gcloud run deploy speech2symbol-api --source . --platform managed --region us-central1 --allow-unauthenticated --memory 2Gi --cpu 2
echo.
echo Your API will be available at: https://speech2symbol-api-xxxxx-uc.a.run.app
pause
goto end

:test
echo.
set /p url="Enter your deployment URL: "
if "%url%"=="" goto end
echo.
echo 🧪 Testing deployment at %url%...
echo.
echo Testing health endpoint...
curl -s "%url%/health"
echo.
echo Testing text conversion...
curl -X POST "%url%/convert/text" -H "Content-Type: application/json" -d "{\"text\": \"two plus three equals five\"}"
echo.
pause
goto end

:invalid
echo ❌ Invalid choice. Please run the script again.
pause
goto end

:end
echo.
echo 🎉 Deployment instructions provided!
echo.
echo 📚 For detailed instructions, see DEPLOYMENT.md
echo 📱 For web interface, visit your deployment URL
echo 📖 For API docs, visit your deployment URL/api
echo.
pause 