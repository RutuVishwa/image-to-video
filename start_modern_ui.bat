@echo off
echo 🎬 Talking Head Studio - Modern UI
echo ================================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm not found. Please install npm.
    pause
    exit /b 1
)

echo ✅ Node.js and npm found

REM Install frontend dependencies
echo 📦 Installing frontend dependencies...
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)

REM Install backend dependencies
echo 📦 Installing backend dependencies...
pip install fastapi uvicorn python-multipart
if %errorlevel% neq 0 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)

echo 🎉 Setup completed successfully!
echo.
echo 🚀 Starting servers...

REM Start backend server in background
start "Backend Server" cmd /k "python api_server.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo 🚀 Starting frontend server...
npm run dev

pause

