#!/usr/bin/env python3
"""
Test script to verify frontend changes are working
"""

import requests
import re

def test_frontend_changes():
    """Test that the frontend changes are loaded"""
    print("🚀 Testing Frontend Changes")
    print("=" * 50)
    
    # Create a session and login
    session = requests.Session()
    login_data = {
        "username": "admin",
        "password": "Admin@123!"
    }
    
    # Login
    response = session.post("http://localhost:8080/login", data=login_data)
    if response.status_code != 200:
        print("❌ Login failed")
        return
    
    print("✅ Login successful")
    
    # Get the dashboard HTML
    response = session.get("http://localhost:8080/")
    if response.status_code != 200:
        print("❌ Dashboard access failed")
        return
    
    html_content = response.text
    
    # Check for the enhanced loadTasksData function
    if 'loadTasksData() {' in html_content and 'Load AI agent tasks' in html_content:
        print("✅ Enhanced loadTasksData function found")
    else:
        print("❌ Enhanced loadTasksData function not found")
    
    # Check for the syncAITasksWithTaskManagement function
    if 'syncAITasksWithTaskManagement()' in html_content:
        print("✅ syncAITasksWithTaskManagement function found")
    else:
        print("❌ syncAITasksWithTaskManagement function not found")
    
    # Check for the refreshAITaskMonitoring function
    if 'refreshAITaskMonitoring()' in html_content:
        print("✅ refreshAITaskMonitoring function found")
    else:
        print("❌ refreshAITaskMonitoring function not found")
    
    # Check for the enhanced updateTasksOverview function
    if 'In Progress' in html_content and 'Completed' in html_content:
        print("✅ Enhanced updateTasksOverview function found")
    else:
        print("❌ Enhanced updateTasksOverview function not found")
    
    print("\n🎯 To clear browser cache and see changes:")
    print("1. Open browser developer tools (F12)")
    print("2. Right-click the refresh button")
    print("3. Select 'Empty Cache and Hard Reload'")
    print("4. Or press Ctrl+Shift+R (Cmd+Shift+R on Mac)")
    print("5. Navigate to Task Management section")
    print("6. Check that AI tasks are now visible")

def main():
    test_frontend_changes()
    
    print("\n📋 If changes are not showing:")
    print("1. Hard refresh the browser (Ctrl+Shift+R)")
    print("2. Clear browser cache completely")
    print("3. Try incognito/private browsing mode")
    print("4. Check browser console for any JavaScript errors")

if __name__ == "__main__":
    main() 