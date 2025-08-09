#!/usr/bin/env python3
"""
Test script to generate Excel templates for all entity types
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

def generate_template(session, entity_type):
    """Generate template for specific entity type"""
    url = "http://localhost:8080/api/ai-agents/workflow/script_generation"
    data = {"entity_type": entity_type}
    
    response = session.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        template_path = result.get('data', {}).get('template_path', 'N/A')
        print(f"✅ {entity_type} template created: {template_path}")
        return template_path
    else:
        print(f"❌ {entity_type} template creation failed: {response.status_code}")
        return None

def check_files():
    """Check all created template files"""
    import subprocess
    templates = ["equipment", "functionallocation", "maintenanceorder"]
    
    for template in templates:
        try:
            result = subprocess.run([
                "docker", "exec", "logging-service", 
                "ls", "-la", f"/app/templates/{template}_template.xlsx"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {template}_template.xlsx exists")
            else:
                print(f"❌ {template}_template.xlsx not found")
        except Exception as e:
            print(f"❌ Error checking {template}: {e}")

def main():
    print("🚀 Generating Excel Templates for All Entity Types")
    print("=" * 60)
    
    # Login
    session = login()
    if not session:
        return
    
    # Generate templates for all entity types
    entity_types = ["Equipment", "FunctionalLocation", "MaintenanceOrder"]
    
    for entity_type in entity_types:
        generate_template(session, entity_type)
        time.sleep(1)  # Small delay between requests
    
    print("\n📁 Checking all created files...")
    check_files()
    
    print("\n🎉 All templates generated!")

if __name__ == "__main__":
    main() 