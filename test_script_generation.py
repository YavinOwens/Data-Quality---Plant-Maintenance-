#!/usr/bin/env python3
"""
Test script to trigger AI agent script generation and create Excel template
"""

import requests
import json
import time

def login():
    """Login to get session cookie"""
    login_data = {
        "username": "admin",
        "password": "Admin@123!"
    }
    
    session = requests.Session()
    response = session.post("http://localhost:8080/login", data=login_data)
    
    if response.status_code == 200:
        print("✅ Login successful")
        return session
    else:
        print(f"❌ Login failed: {response.status_code}")
        return None

def trigger_script_generation(session):
    """Trigger script generation workflow"""
    url = "http://localhost:8080/api/ai-agents/workflow/script_generation"
    data = {"entity_type": "Equipment"}
    
    response = session.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Script generation successful")
        print(f"📄 Template path: {result.get('data', {}).get('template_path', 'N/A')}")
        return result
    else:
        print(f"❌ Script generation failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def check_file_exists():
    """Check if the Excel file was created"""
    import subprocess
    try:
        result = subprocess.run([
            "docker", "exec", "logging-service", 
            "ls", "-la", "/app/templates/equipment_template.xlsx"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Excel template file created successfully!")
            print(f"📁 File details:\n{result.stdout}")
            return True
        else:
            print("❌ Excel template file not found")
            return False
    except Exception as e:
        print(f"❌ Error checking file: {e}")
        return False

def main():
    print("🚀 Testing AI Agent Script Generation")
    print("=" * 50)
    
    # Login
    session = login()
    if not session:
        return
    
    # Trigger script generation
    result = trigger_script_generation(session)
    if not result:
        return
    
    # Wait a moment for file creation
    print("⏳ Waiting for file creation...")
    time.sleep(2)
    
    # Check if file was created
    check_file_exists()
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    main() 