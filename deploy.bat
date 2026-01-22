@echo off
echo ========================================================
echo NBA Elite Model Deployment Launcher
echo ========================================================

echo.
echo [1/3] Verifying Cloud configuration...
call gcloud config get-value project
if %ERRORLEVEL% NEQ 0 (
    echo Error: gcloud project not set. Run 'gcloud config set project [PROJECT_ID]'
    exit /b 1
)

echo.
echo [2/3] Submitting build to Google Cloud Build...
echo This will build the Docker container and push it to GCR.
echo It avoids the need for local Docker installation.
echo.
call gcloud builds submit --config cloudbuild.yaml .
if %ERRORLEVEL% NEQ 0 (
    echo Build failed! Check the logs above.
    exit /b 1
)

echo.
echo [3/3] Deployment triggered in Cloud Build steps.
echo Verifying service URL...
call gcloud run services describe nba-elite-app --platform managed --region us-central1 --format "value(status.url)"

echo.
echo ========================================================
echo DEPLOYMENT COMPLETE!
echo ========================================================
pause
