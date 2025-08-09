#!/usr/bin/env python3
"""
NeuroLM Web Dashboard Launcher
=============================
Easy launcher for the complete NeuroLM web dashboard system
Starts both the web server and NeuroLM processing system
"""

import subprocess
import threading
import time
import webbrowser
import sys
import os
from pathlib import Path

def start_web_server():
    """Start the web server in a separate process"""
    print("🌐 Starting web server...")
    try:
        # Set environment variable for LSL if needed
        env = os.environ.copy()
        env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'
        
        subprocess.run([
            sys.executable, 'web_server.py'
        ], env=env, cwd=Path(__file__).parent)
    except Exception as e:
        print(f"❌ Error starting web server: {e}")

def start_neurolm_system():
    """Start the NeuroLM dashboard system"""
    print("🧠 Starting NeuroLM dashboard system...")
    try:
        # Set environment variable for LSL if needed
        env = os.environ.copy()
        env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'
        
        subprocess.run([
            sys.executable, 'neurolm_web_dashboard_system.py'
        ], env=env, cwd=Path(__file__).parent)
    except Exception as e:
        print(f"❌ Error starting NeuroLM system: {e}")

def main():
    """Main launcher function"""
    print("🎬 NeuroLM Web Dashboard Launcher")
    print("=" * 50)
    print("🧠 Starting complete video experiment system")
    print("📊 Web interface + NeuroLM processing")
    print("=" * 50)
    
    # Start web server in background thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Wait a moment for web server to start
    time.sleep(3)
    
    # Open browser
    print("🌐 Opening web dashboard...")
    try:
        webbrowser.open('http://localhost:8080')
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print("📱 Please open http://localhost:8080 manually")
    
    # Start NeuroLM system (this will block)
    start_neurolm_system()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped by user")
    except Exception as e:
        print(f"❌ Launcher error: {e}")
