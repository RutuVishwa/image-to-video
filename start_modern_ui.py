#!/usr/bin/env python3
"""
Setup and start script for Talking Head Studio Modern UI
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org/")
    return False

def check_npm_installed():
    """Check if npm is installed"""
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ npm not found. Please install npm.")
    return False

def install_frontend_dependencies():
    """Install frontend dependencies"""
    print("📦 Installing frontend dependencies...")
    
    try:
        result = subprocess.run(['npm', 'install'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Frontend dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install frontend dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing frontend dependencies: {e}")
        return False

def install_backend_dependencies():
    """Install backend dependencies"""
    print("📦 Installing backend dependencies...")
    
    try:
        packages = ['fastapi', 'uvicorn', 'python-multipart']
        for package in packages:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
                return False
        return True
    except Exception as e:
        print(f"❌ Error installing backend dependencies: {e}")
        return False

def start_backend_server():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    
    try:
        # Start the backend server in a subprocess
        backend_process = subprocess.Popen([
            sys.executable, 'api_server.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if the process is still running
        if backend_process.poll() is None:
            print("✅ Backend server started successfully on http://localhost:7860")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate()
            print(f"❌ Backend server failed to start: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Error starting backend server: {e}")
        return None

def start_frontend_server():
    """Start the frontend server"""
    print("🚀 Starting frontend server...")
    
    try:
        # Start the frontend server in a subprocess
        frontend_process = subprocess.Popen([
            'npm', 'run', 'dev'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        # Check if the process is still running
        if frontend_process.poll() is None:
            print("✅ Frontend server started successfully on http://localhost:3000")
            return frontend_process
        else:
            stdout, stderr = frontend_process.communicate()
            print(f"❌ Frontend server failed to start: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")
        return None

def main():
    """Main setup and start function"""
    print("🎬 Talking Head Studio - Modern UI Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_node_installed():
        return
    if not check_npm_installed():
        return
    
    # Install dependencies
    if not install_frontend_dependencies():
        return
    if not install_backend_dependencies():
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nStarting servers...")
    
    # Start servers
    backend_process = start_backend_server()
    if not backend_process:
        return
    
    frontend_process = start_frontend_server()
    if not frontend_process:
        backend_process.terminate()
        return
    
    print("\n" + "=" * 50)
    print("🎬 Talking Head Studio is now running!")
    print("📱 Frontend: http://localhost:3000")
    print("🔧 Backend API: http://localhost:7860")
    print("\nPress Ctrl+C to stop both servers")
    
    # Open the frontend in the browser
    try:
        webbrowser.open('http://localhost:3000')
    except:
        pass
    
    try:
        # Keep the script running and handle Ctrl+C
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main()

