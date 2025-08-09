#!/usr/bin/env python3
"""
Comprehensive AI Agent Test Suite
Tests all aspects of the AI agent including reports, database queries, and vector search
"""

import requests
import json
import time
import sys
from datetime import datetime

class AIAgentTester:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def login(self):
        """Login to get session cookies"""
        try:
            login_data = {
                'username': 'admin',
                'password': 'admin123'
            }
            response = self.session.post(f"{self.base_url}/login", data=login_data)
            if response.status_code == 200:
                print("✅ Login successful")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def test_chat_endpoint(self, message, expected_keywords=None):
        """Test the chat endpoint with a specific message"""
        try:
            data = {'message': message}
            response = self.session.post(f"{self.base_url}/api/chat", json=data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    response_text = result.get('response', '')
                    print(f"✅ Chat test passed: '{message[:50]}...'")
                    print(f"   Response: {response_text[:200]}...")
                    
                    # Check for expected keywords if provided
                    if expected_keywords:
                        found_keywords = [kw for kw in expected_keywords if kw.lower() in response_text.lower()]
                        if found_keywords:
                            print(f"   ✅ Found expected keywords: {found_keywords}")
                        else:
                            print(f"   ⚠️  Missing expected keywords: {expected_keywords}")
                    
                    return True, response_text
                else:
                    print(f"❌ Chat test failed: {result.get('message', 'Unknown error')}")
                    return False, result.get('message', 'Unknown error')
            else:
                print(f"❌ Chat test failed: HTTP {response.status_code}")
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            print(f"❌ Chat test error: {e}")
            return False, str(e)
    
    def test_vector_endpoints(self):
        """Test vector search and embedding endpoints"""
        print("\n🔍 Testing Vector Search Endpoints...")
        
        # Test vector status
        try:
            response = self.session.get(f"{self.base_url}/api/vector/status")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Vector status: {result}")
            else:
                print(f"❌ Vector status failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Vector status error: {e}")
        
        # Test semantic search
        try:
            search_data = {'query': 'data quality issues'}
            response = self.session.post(f"{self.base_url}/api/vector/search", json=search_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Semantic search: {len(result.get('results', []))} results")
            else:
                print(f"❌ Semantic search failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
    
    def test_report_endpoints(self):
        """Test all report generation endpoints"""
        print("\n📊 Testing Report Generation Endpoints...")
        
        report_endpoints = [
            ('executive', 'Executive Summary Report'),
            ('kpi-dashboard', 'KPI Dashboard Report'),
            ('risk-assessment', 'Risk Assessment Report'),
            ('data-quality', 'Data Quality Report'),
            ('data-architecture', 'Data Architecture Report'),
            ('task-management', 'Task Management Report'),
            ('governance-compliance', 'Governance & Compliance Report'),
            ('system-health', 'System Health Report'),
            ('csuite', 'C-Suite Reports'),
            ('data-management', 'Data Management Report')
        ]
        
        for endpoint, name in report_endpoints:
            try:
                data = {'format': 'pdf'}
                response = self.session.post(f"{self.base_url}/api/reports/{endpoint}", json=data)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'success':
                        print(f"✅ {name}: Generated successfully")
                    else:
                        print(f"❌ {name}: {result.get('message', 'Unknown error')}")
                else:
                    print(f"❌ {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {name} error: {e}")
    
    def test_ai_context_endpoint(self):
        """Test the AI context endpoint"""
        print("\n🧠 Testing AI Context Endpoint...")
        
        try:
            data = {'query': 'test query for context'}
            response = self.session.post(f"{self.base_url}/api/ai/context", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ AI Context: Retrieved {len(result.get('context', {}))} context items")
                return True
            else:
                print(f"❌ AI Context failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ AI Context error: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive AI Agent Test Suite")
        print("=" * 60)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without login")
            return
        
        # Test 1: Basic AI Agent Functionality
        print("\n📝 Test 1: Basic AI Agent Functionality")
        print("-" * 40)
        
        basic_tests = [
            ("Hello, how are you?", None),
            ("What is data quality?", ['data', 'quality', 'metrics']),
            ("Show me the dashboard", ['dashboard', 'overview']),
            ("What are the main features?", ['features', 'capabilities'])
        ]
        
        for message, expected_keywords in basic_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)  # Rate limiting
        
        # Test 2: Report-Related Queries
        print("\n📊 Test 2: Report-Related Queries")
        print("-" * 40)
        
        report_tests = [
            ("What reports are available?", ['report', 'available', 'executive', 'quality']),
            ("Show me the executive summary report", ['executive', 'summary', 'report']),
            ("What data quality reports can I generate?", ['data', 'quality', 'report']),
            ("Tell me about risk assessment reports", ['risk', 'assessment', 'report']),
            ("What governance reports are available?", ['governance', 'compliance', 'report']),
            ("Show me the task management reports", ['task', 'management', 'report']),
            ("What system health reports exist?", ['system', 'health', 'report']),
            ("Tell me about data architecture reports", ['architecture', 'data', 'report'])
        ]
        
        for message, expected_keywords in report_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)
        
        # Test 3: Task and Tracking Queries
        print("\n📋 Test 3: Task and Tracking Queries")
        print("-" * 40)
        
        task_tests = [
            ("What are the overdue tasks?", ['overdue', 'task', 'due']),
            ("Show me task tracking information", ['task', 'tracking', 'status']),
            ("What tasks are due soon?", ['due', 'soon', 'task']),
            ("How many tasks are completed?", ['completed', 'task', 'count']),
            ("What is the task completion rate?", ['completion', 'rate', 'task']),
            ("Show me high priority tasks", ['priority', 'high', 'task'])
        ]
        
        for message, expected_keywords in task_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)
        
        # Test 4: Data Quality and Metrics Queries
        print("\n📈 Test 4: Data Quality and Metrics Queries")
        print("-" * 40)
        
        quality_tests = [
            ("What are the data quality metrics?", ['quality', 'metrics', 'data']),
            ("Show me recent data quality issues", ['recent', 'issues', 'quality']),
            ("What is the overall data quality score?", ['score', 'quality', 'overall']),
            ("Show me validation results", ['validation', 'results', 'check']),
            ("What are the main data quality problems?", ['problems', 'issues', 'quality']),
            ("How is data completeness?", ['completeness', 'data', 'quality'])
        ]
        
        for message, expected_keywords in quality_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)
        
        # Test 5: System and Performance Queries
        print("\n⚙️ Test 5: System and Performance Queries")
        print("-" * 40)
        
        system_tests = [
            ("What is the system health status?", ['system', 'health', 'status']),
            ("Show me performance metrics", ['performance', 'metrics', 'system']),
            ("What are the current system issues?", ['system', 'issues', 'problems']),
            ("How is the system performing?", ['performance', 'system', 'status']),
            ("What are the error rates?", ['error', 'rate', 'system']),
            ("Show me system availability", ['availability', 'system', 'uptime'])
        ]
        
        for message, expected_keywords in system_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)
        
        # Test 6: Risk and Security Queries
        print("\n🛡️ Test 6: Risk and Security Queries")
        print("-" * 40)
        
        risk_tests = [
            ("What are the current risks?", ['risk', 'assessment', 'current']),
            ("Show me the risk matrix", ['risk', 'matrix', 'assessment']),
            ("What security issues exist?", ['security', 'issues', 'threats']),
            ("Show me compliance status", ['compliance', 'status', 'governance']),
            ("What are the high-risk items?", ['high', 'risk', 'items']),
            ("Show me security metrics", ['security', 'metrics', 'status'])
        ]
        
        for message, expected_keywords in risk_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)
        
        # Test 7: Database and Data Source Queries
        print("\n🗄️ Test 7: Database and Data Source Queries")
        print("-" * 40)
        
        database_tests = [
            ("What data sources are available?", ['data', 'sources', 'available']),
            ("Show me database statistics", ['database', 'statistics', 'tables']),
            ("What tables exist in the system?", ['tables', 'database', 'schema']),
            ("Show me data relationships", ['relationships', 'data', 'connections']),
            ("What is the data lineage?", ['lineage', 'data', 'flow']),
            ("Show me data dependencies", ['dependencies', 'data', 'relationships'])
        ]
        
        for message, expected_keywords in database_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(1)
        
        # Test 8: Complex Multi-Aspect Queries
        print("\n🔗 Test 8: Complex Multi-Aspect Queries")
        print("-" * 40)
        
        complex_tests = [
            ("Give me a comprehensive overview of the system including data quality, tasks, and reports", 
             ['comprehensive', 'overview', 'system', 'quality', 'tasks', 'reports']),
            ("What reports would be most relevant for executive review given the current data quality issues?", 
             ['reports', 'executive', 'relevant', 'quality', 'issues']),
            ("How do the overdue tasks relate to data quality problems?", 
             ['overdue', 'tasks', 'quality', 'problems', 'relate']),
            ("What is the impact of system performance on data quality metrics?", 
             ['impact', 'performance', 'quality', 'metrics']),
            ("Show me the relationship between risk assessment and governance compliance", 
             ['risk', 'assessment', 'governance', 'compliance', 'relationship'])
        ]
        
        for message, expected_keywords in complex_tests:
            self.test_chat_endpoint(message, expected_keywords)
            time.sleep(2)  # Longer delay for complex queries
        
        # Test 9: Vector Search and Context
        print("\n🧠 Test 9: Vector Search and Context")
        print("-" * 40)
        
        self.test_vector_endpoints()
        self.test_ai_context_endpoint()
        
        # Test 10: Report Generation
        print("\n📄 Test 10: Report Generation")
        print("-" * 40)
        
        self.test_report_endpoints()
        
        print("\n" + "=" * 60)
        print("🎉 Comprehensive AI Agent Test Suite Completed!")
        print("=" * 60)

if __name__ == "__main__":
    tester = AIAgentTester()
    tester.run_comprehensive_tests() 