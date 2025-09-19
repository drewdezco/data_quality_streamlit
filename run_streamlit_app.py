#!/usr/bin/env python3
"""
Launch script for the Data Quality Streamlit App
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    
    print("🚀 Starting Data Quality Validation Service...")
    print("📊 Streamlit App will open in your browser")
    print("-" * 50)
    
    # Check if running in the correct directory
    app_file = "data_quality_streamlit_app.py"
    if not os.path.exists(app_file):
        print(f"❌ Error: {app_file} not found in current directory")
        print("Please run this script from the same directory as the Streamlit app")
        return
    
    # Check if data_quality_checker.py exists
    checker_file = "data_quality_checker.py"
    if not os.path.exists(checker_file):
        print(f"❌ Error: {checker_file} not found in current directory")
        print("Please ensure data_quality_checker.py is in the same directory")
        return
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            app_file,
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Data Quality Service stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("\n💡 Try installing requirements: pip install -r requirements_streamlit.txt")

if __name__ == "__main__":
    main()

