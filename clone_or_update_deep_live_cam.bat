@echo off
REM clone_or_update_deep_live_cam.bat - Clone or update Deep-Live-Cam repo in a separate folder and sync to local working folder
SET REPO_URL=https://github.com/hacksider/Deep-Live-Cam.git
SET TARGET_DIR=Deep-Live-Cam-remote
SET LOCAL_DIR=Deep-Live-Cam

IF EXIST %TARGET_DIR% (
    echo Updating existing repo in %TARGET_DIR% ...
    cd %TARGET_DIR%
    git pull
    cd ..
) ELSE (
    echo Cloning repo to %TARGET_DIR% ...
    git clone %REPO_URL% %TARGET_DIR%
)

REM Sync updated code to local working folder (excluding .git and models)
xcopy %TARGET_DIR% %LOCAL_DIR% /E /H /Y /EXCLUDE:exclude.txt

echo Done. Latest code is in %LOCAL_DIR%.
