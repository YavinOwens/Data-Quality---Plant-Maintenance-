#!/usr/bin/env python3
"""
Comprehensive test script for SAP S/4HANA Data Quality Workflow
Tests the entire workflow with mock data
"""

import requests
import json
import time
import sys
from datetime import datetime

def test_sap_connector():
    """Test SAP connector functionality"""
    print("🔍 Testing Mock SAP Connector...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print("✅ Mock SAP Connector health check passed")
        else:
            print(f"❌ Mock SAP Connector health check failed: {response.status_code}")
            return False
        
        # Test equipment extraction
        response = requests.get('http://localhost:8000/extract/equipment?limit=10')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Equipment extraction successful: {data.get('message', 'Unknown')}")
        else:
            print(f"❌ Equipment extraction failed: {response.status_code}")
            return False
        
        # Test functional locations extraction
        response = requests.get('http://localhost:8000/extract/functional-locations?limit=5')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Functional locations extraction successful: {data.get('message', 'Unknown')}")
        else:
            print(f"❌ Functional locations extraction failed: {response.status_code}")
            return False
        
        # Test maintenance orders extraction
        response = requests.get('http://localhost:8000/extract/maintenance-orders?limit=8')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Maintenance orders extraction successful: {data.get('message', 'Unknown')}")
        else:
            print(f"❌ Maintenance orders extraction failed: {response.status_code}")
            return False
        
        # Test bulk extraction
        response = requests.get('http://localhost:8000/extract/all')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Bulk extraction successful: {len(data.get('results', {}))} data types")
        else:
            print(f"❌ Bulk extraction failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Mock SAP Connector not accessible. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Mock SAP Connector test failed: {str(e)}")
        return False

def test_logging_service():
    """Test logging service functionality"""
    print("\n📊 Testing Logging Service...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8080/health')
        if response.status_code == 200:
            print("✅ Logging Service health check passed")
        else:
            print(f"❌ Logging Service health check failed: {response.status_code}")
            return False
        
        # Test validation summary API
        response = requests.get('http://localhost:8080/api/validation-summary?days=30')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Validation summary API working: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Validation summary API failed: {response.status_code}")
            return False
        
        # Test recent issues API
        response = requests.get('http://localhost:8080/api/recent-issues?limit=10')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recent issues API working: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Recent issues API failed: {response.status_code}")
            return False
        
        # Test quality report API
        response = requests.get('http://localhost:8080/api/quality-report?days=30')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Quality report API working: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Quality report API failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Logging Service not accessible. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Logging Service test failed: {str(e)}")
        return False

def test_database():
    """Test database connectivity"""
    print("\n🗄️  Testing Database...")
    
    try:
        import subprocess
        result = subprocess.run(['docker', 'exec', 'sap-postgres', 'pg_isready', '-U', 'sap_user', '-d', 'sap_data_quality'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Database connection successful")
        else:
            print(f"❌ Database connection failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False

def test_pgadmin():
    """Test pgAdmin functionality"""
    print("\n🛠️  Testing pgAdmin...")
    
    try:
        # Test pgAdmin health endpoint
        response = requests.get('http://localhost:5050/misc/ping')
        if response.status_code == 200:
            print("✅ pgAdmin health check passed")
        else:
            print(f"❌ pgAdmin health check failed: {response.status_code}")
            return False
        
        # Test pgAdmin login page (401 is expected for unauthenticated access)
        response = requests.get('http://localhost:5050/login')
        if response.status_code in [200, 401, 302]:  # 200=OK, 401=Unauthorized, 302=Redirect
            print("✅ pgAdmin login page accessible")
        else:
            print(f"❌ pgAdmin login page failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ pgAdmin not accessible. Is it running?")
        return False
    except Exception as e:
        print(f"❌ pgAdmin test failed: {str(e)}")
        return False

def test_data_files():
    """Test that data files were created"""
    print("\n📁 Testing Data Files...")
    
    try:
        import os
        import glob
        
        # Check for data files
        data_files = glob.glob('data/*.json')
        if data_files:
            print(f"✅ Found {len(data_files)} data files:")
            for file in data_files[:5]:  # Show first 5 files
                print(f"   - {file}")
            if len(data_files) > 5:
                print(f"   ... and {len(data_files) - 5} more")
        else:
            print("⚠️  No data files found. This is normal if no extractions have been run.")
        
        return True
        
    except Exception as e:
        print(f"❌ Data files test failed: {str(e)}")
        return False

def test_validation_engine():
    """Test validation engine functionality"""
    print("\n🔍 Testing Validation Engine...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8001/health')
        if response.status_code == 200:
            print("✅ Validation Engine health check passed")
        else:
            print(f"❌ Validation Engine health check failed: {response.status_code}")
            return False
        
        # Test rules endpoint
        response = requests.get('http://localhost:8001/rules')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Rules API working: {len(data.get('rules', []))} rules available")
        else:
            print(f"❌ Rules API failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Validation Engine not accessible. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Validation Engine test failed: {str(e)}")
        return False

def test_monitoring():
    """Test monitoring components"""
    print("\n📈 Testing Monitoring...")
    
    try:
        # Test Prometheus metrics
        response = requests.get('http://localhost:9090/-/healthy')
        if response.status_code == 200:
            print("✅ Prometheus is healthy")
        else:
            print(f"❌ Prometheus health check failed: {response.status_code}")
            return False
        
        # Test Grafana
        response = requests.get('http://localhost:3000/api/health')
        if response.status_code == 200:
            print("✅ Grafana is healthy")
        else:
            print(f"❌ Grafana health check failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Monitoring services not accessible. Are they running?")
        return False
    except Exception as e:
        print(f"❌ Monitoring test failed: {str(e)}")
        return False

def test_end_to_end_workflow():
    """Test the complete end-to-end workflow"""
    print("\n🔄 Testing End-to-End Workflow...")
    
    try:
        # Step 1: Extract data from Mock SAP
        print("  1. Extracting data from Mock SAP...")
        response = requests.get('http://localhost:8000/extract/equipment?limit=5')
        if response.status_code != 200:
            print("   ❌ Data extraction failed")
            return False
        print("   ✅ Data extraction successful")
        
        # Step 2: Wait a moment for data processing
        time.sleep(2)
        
        # Step 3: Check that data is available in the dashboard
        print("  2. Checking dashboard data...")
        response = requests.get('http://localhost:8080/api/validation-summary')
        if response.status_code == 200:
            print("   ✅ Dashboard data accessible")
        else:
            print("   ❌ Dashboard data not accessible")
            return False
        
        # Step 4: Test export functionality
        print("  3. Testing export functionality...")
        response = requests.get('http://localhost:8080/api/export/csv')
        if response.status_code == 200:
            print("   ✅ CSV export working")
        else:
            print("   ❌ CSV export failed")
            return False
        
        print("   ✅ End-to-end workflow successful")
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end workflow failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 SAP S/4HANA Data Quality Workflow - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Mock SAP Connector", test_sap_connector),
        ("Logging Service", test_logging_service),
        ("Database", test_database),
        ("pgAdmin", test_pgadmin),
        ("Data Files", test_data_files),
        ("Validation Engine", test_validation_engine),
        ("Monitoring", test_monitoring),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is fully operational.")
        print("\n📝 System Status:")
        print("- Dashboard: http://localhost:8080")
        print("- Grafana: http://localhost:3000")
        print("- pgAdmin: http://localhost:5050")
        print("- Prometheus: http://localhost:9090")
        print("- SAP Connector: http://localhost:8000")
        print("- Validation Engine: http://localhost:8001")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. The system is operational with some limitations.")
    else:
        print("❌ Many tests failed. Please check the system configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 