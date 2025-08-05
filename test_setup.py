#!/usr/bin/env python3
"""
Test script to verify SAP S/4HANA Data Quality Workflow setup
"""

import sys
import os
import json
from datetime import datetime

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    
    try:
        import requests
        print(f"✅ Requests {requests.__version__}")
    except ImportError as e:
        print(f"❌ Requests import failed: {e}")
        return False
    
    try:
        import pytest
        print(f"✅ Pytest {pytest.__version__}")
    except ImportError as e:
        print(f"❌ Pytest import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'docker-compose.yml',
        'env.example',
        'README.md',
        'sap-connector/app.py',
        'sap-connector/requirements.txt',
        'validation-engine/app.py',
        'validation-engine/requirements.txt',
        'logging-service/app.py',
        'logging-service/requirements.txt',
        'logging-service/templates/dashboard.html',
        'config/rules/equipment_completeness.yaml',
        'config/rules/maintenance_orders_timeliness.yaml',
        'config/rules/cross_reference_consistency.yaml',
        'init-db/01_create_tables.sql',
        'monitoring/prometheus.yml',
        'monitoring/grafana/dashboards/data-quality-dashboard.json',
        'monitoring/grafana/datasources/prometheus.yml',
        'ARCHITECTURE.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files")
        return False
    else:
        print("✅ All required files present")
        return True

def test_docker():
    """Test Docker functionality"""
    print("\n🐳 Testing Docker...")
    
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print(f"❌ Docker not available: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False
    
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose: {result.stdout.strip()}")
        else:
            print(f"❌ Docker Compose not available: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker Compose test failed: {e}")
        return False
    
    return True

def test_yaml_parsing():
    """Test YAML configuration parsing"""
    print("\n📄 Testing YAML configuration parsing...")
    
    try:
        import yaml
        
        # Test equipment completeness rules
        with open('config/rules/equipment_completeness.yaml', 'r') as f:
            equipment_rules = yaml.safe_load(f)
            print(f"✅ Equipment rules loaded: {equipment_rules.get('rule_name', 'Unknown')}")
        
        # Test maintenance orders rules
        with open('config/rules/maintenance_orders_timeliness.yaml', 'r') as f:
            maintenance_rules = yaml.safe_load(f)
            print(f"✅ Maintenance rules loaded: {maintenance_rules.get('rule_name', 'Unknown')}")
        
        # Test cross-reference rules
        with open('config/rules/cross_reference_consistency.yaml', 'r') as f:
            cross_ref_rules = yaml.safe_load(f)
            print(f"✅ Cross-reference rules loaded: {cross_ref_rules.get('rule_name', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ YAML parsing failed: {e}")
        return False

def test_sample_data():
    """Test with sample data"""
    print("\n📊 Testing with sample data...")
    
    try:
        import pandas as pd
        
        # Create sample equipment data
        sample_equipment = pd.DataFrame({
            'Equipment': ['EQ001', 'EQ002', 'EQ003'],
            'EquipmentName': ['Pump A', 'Motor B', 'Valve C'],
            'EquipmentCategory': ['M', 'M', 'F'],
            'FunctionalLocation': ['FL001', 'FL002', None],
            'ManufacturerSerialNumber': ['SN123456', 'SN789012', 'SN345678']
        })
        
        print(f"✅ Sample equipment data created: {len(sample_equipment)} records")
        
        # Test data quality checks
        missing_fields = sample_equipment.isnull().sum()
        print(f"✅ Missing fields analysis: {missing_fields.to_dict()}")
        
        # Test equipment category validation
        valid_categories = ['M', 'F', 'K', 'P']
        invalid_categories = sample_equipment[~sample_equipment['EquipmentCategory'].isin(valid_categories)]
        print(f"✅ Equipment category validation: {len(invalid_categories)} invalid records")
        
        return True
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

def test_environment():
    """Test environment variables"""
    print("\n🔧 Testing environment setup...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")
    
    # Check Python version
    print(f"✅ Python version: {sys.version}")
    
    # Check current directory
    print(f"✅ Current directory: {os.getcwd()}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 SAP S/4HANA Data Quality Workflow - Environment Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Docker Setup", test_docker),
        ("YAML Configuration", test_yaml_parsing),
        ("Sample Data Processing", test_sample_data),
        ("Environment Setup", test_environment)
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
        print("🎉 All tests passed! Environment is ready for testing.")
        print("\n📝 Next steps:")
        print("1. Copy env.example to .env and configure SAP connection")
        print("2. Run: docker-compose up -d")
        print("3. Access dashboard at: http://localhost:8080")
        print("4. Access Grafana at: http://localhost:3000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 