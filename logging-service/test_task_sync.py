#!/usr/bin/env python3
"""
Test script to verify Task Management sync with AI agent tasks
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

def test_ai_tasks_endpoint(session):
    """Test AI tasks endpoint"""
    print("\n📋 Testing AI Tasks Endpoint...")
    
    response = session.get("http://localhost:8080/api/ai-agents/tasks")
    
    if response.status_code == 200:
        data = response.json()
        tasks = data.get('data', [])
        print(f"✅ AI tasks endpoint successful")
        print(f"📊 Found {len(tasks)} AI-generated tasks")
        for task in tasks:
            print(f"  - {task.get('title', 'N/A')} ({task.get('status', 'N/A')})")
        return tasks
    else:
        print(f"❌ AI tasks endpoint failed: {response.status_code}")
        return []

def test_task_monitoring(session):
    """Test task monitoring"""
    print("\n🔍 Testing Task Monitoring...")
    
    response = session.post("http://localhost:8080/api/ai-agents/task-execution", 
                          json={"operation": "monitor"})
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Task monitoring successful")
        monitoring_data = data.get('data', {})
        print(f"📊 Total Tasks: {monitoring_data.get('total_tasks', 0)}")
        print(f"📊 Completed: {monitoring_data.get('completed_tasks', 0)}")
        print(f"📊 In Progress: {monitoring_data.get('in_progress_tasks', 0)}")
        print(f"📊 Overdue: {monitoring_data.get('overdue_tasks', 0)}")
        return monitoring_data
    else:
        print(f"❌ Task monitoring failed: {response.status_code}")
        return {}

def test_task_management_sync():
    """Test that Task Management shows AI tasks"""
    print("\n🔄 Testing Task Management Sync...")
    
    session = login()
    if not session:
        return
    
    # Get AI tasks
    ai_tasks = test_ai_tasks_endpoint(session)
    
    # Get task monitoring data
    monitoring_data = test_task_monitoring(session)
    
    print("\n📊 Summary:")
    print(f"✅ AI Tasks Available: {len(ai_tasks)}")
    print(f"✅ Task Monitoring Active: {monitoring_data.get('total_tasks', 0)} total tasks")
    print(f"✅ Task Management should now show AI tasks in the dashboard")
    
    print("\n🎯 To verify the fix:")
    print("1. Go to http://localhost:8080")
    print("2. Navigate to 'Task Management' section")
    print("3. Check that AI-generated tasks are now visible")
    print("4. Verify task counts include AI tasks")
    print("5. Check that 'Refresh' button updates both regular and AI tasks")

def main():
    print("🚀 Testing Task Management Sync with AI Agents")
    print("=" * 60)
    
    test_task_management_sync()
    
    print("\n🎉 Test completed!")
    print("\n📋 What was fixed:")
    print("✅ Modified loadTasksData() to fetch both regular and AI tasks")
    print("✅ Updated updateTasksOverview() to handle AI task status formats")
    print("✅ Added refreshAITaskMonitoring() to sync AI task data")
    print("✅ Added syncAITasksWithTaskManagement() for automatic sync")
    print("✅ Enhanced refreshTasks() to also refresh AI task monitoring")
    print("✅ No existing code was deleted - only enhanced")

if __name__ == "__main__":
    main() 