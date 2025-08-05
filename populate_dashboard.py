#!/usr/bin/env python3
"""
Dashboard Data Population Script
Populates the SAP Data Quality Dashboard with validation results
"""

import requests
import json
import time
from datetime import datetime

def extract_mock_data():
    """Extract all data from Mock SAP"""
    print("📊 Extracting data from Mock SAP...")
    
    # Extract equipment data
    response = requests.get('http://localhost:8000/extract/equipment')
    if response.status_code == 200:
        equipment_data = response.json()['data']
        print(f"✅ Extracted {len(equipment_data)} equipment records")
    else:
        print("❌ Failed to extract equipment data")
        return None
    
    # Extract functional locations data
    response = requests.get('http://localhost:8000/extract/functional-locations')
    if response.status_code == 200:
        fl_data = response.json()['data']
        print(f"✅ Extracted {len(fl_data)} functional location records")
    else:
        print("❌ Failed to extract functional locations data")
        return None
    
    # Extract maintenance orders data
    response = requests.get('http://localhost:8000/extract/maintenance-orders')
    if response.status_code == 200:
        mo_data = response.json()['data']
        print(f"✅ Extracted {len(mo_data)} maintenance order records")
    else:
        print("❌ Failed to extract maintenance orders data")
        return None
    
    # Extract maintenance notifications data
    response = requests.get('http://localhost:8000/extract/maintenance-notifications')
    if response.status_code == 200:
        mn_data = response.json()['data']
        print(f"✅ Extracted {len(mn_data)} maintenance notification records")
    else:
        print("❌ Failed to extract maintenance notifications data")
        return None
    
    # Extract maintenance plans data
    response = requests.get('http://localhost:8000/extract/maintenance-plans')
    if response.status_code == 200:
        mp_data = response.json()['data']
        print(f"✅ Extracted {len(mp_data)} maintenance plan records")
    else:
        print("❌ Failed to extract maintenance plans data")
        return None
    
    return {
        'equipment': equipment_data,
        'functional_locations': fl_data,
        'maintenance_orders': mo_data,
        'maintenance_notifications': mn_data,
        'maintenance_plans': mp_data
    }

def run_equipment_validations(equipment_data):
    """Run equipment data quality validations"""
    print("\n🔍 Running equipment validations...")
    
    # Equipment completeness validation
    response = requests.post(
        'http://localhost:8001/validate/equipment',
        headers={'Content-Type': 'application/json'},
        json={'data': equipment_data}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Equipment completeness: {results['completeness']['success_rate']:.2%}")
        print(f"✅ Equipment accuracy: {results['accuracy']['success_rate']:.2%}")
        return results
    else:
        print(f"❌ Equipment validation failed: {response.status_code}")
        return None

def run_maintenance_orders_validations(mo_data):
    """Run maintenance orders data quality validations"""
    print("\n🔍 Running maintenance orders validations...")
    
    response = requests.post(
        'http://localhost:8001/validate/maintenance-orders',
        headers={'Content-Type': 'application/json'},
        json={'data': mo_data}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Maintenance orders timeliness: {results['timeliness']['success_rate']:.2%}")
        return results
    else:
        print(f"❌ Maintenance orders validation failed: {response.status_code}")
        return None

def run_cross_reference_validations(data):
    """Run cross-reference consistency validations"""
    print("\n🔍 Running cross-reference validations...")
    
    response = requests.post(
        'http://localhost:8001/validate/cross-reference',
        headers={'Content-Type': 'application/json'},
        json={
            'equipment': data['equipment'],
            'functional_locations': data['functional_locations'],
            'maintenance_orders': data['maintenance_orders']
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Cross-reference consistency: {results['consistency']['success_rate']:.2%}")
        return results
    else:
        print(f"❌ Cross-reference validation failed: {response.status_code}")
        return None

def create_additional_test_data():
    """Create additional test data with quality issues"""
    print("\n🔧 Creating additional test data with quality issues...")
    
    # Equipment with issues
    problematic_equipment = [
        {
            "Equipment": "10000006",
            "EquipmentName": "",  # Missing name
            "EquipmentCategory": "X",  # Invalid category
            "FunctionalLocation": "FL-999",  # Non-existent FL
            "Manufacturer": "Test Corp",
            "ModelNumber": "TEST-001",
            "SerialNumber": "123",  # Too short
            "InstallationDate": "2023-06-01",
            "EquipmentStatus": "ACTIVE",
            "MaintenancePlant": "1000",
            "WorkCenter": "WC-TEST",
            "EquipmentDescription": "Test equipment with quality issues"
        },
        {
            "Equipment": "10000007",
            "EquipmentName": "Incomplete Equipment",
            "EquipmentCategory": "M",
            "FunctionalLocation": "FL-001",
            "Manufacturer": "",  # Missing manufacturer
            "ModelNumber": "",
            "SerialNumber": "",
            "InstallationDate": "2023-07-01",
            "EquipmentStatus": "ACTIVE",
            "MaintenancePlant": "1000",
            "WorkCenter": "WC-TEST",
            "EquipmentDescription": "Equipment with missing data"
        }
    ]
    
    # Maintenance orders with issues
    problematic_orders = [
        {
            "OrderNumber": "1000000006",
            "OrderType": "PM01",
            "Equipment": "10000006",  # Non-existent equipment
            "FunctionalLocation": "FL-999",  # Non-existent FL
            "OrderDescription": "Test order with issues",
            "OrderStatus": "RELEASED",
            "Priority": "1",
            "PlannedStartDate": "2023-01-01",  # Overdue
            "PlannedEndDate": "2023-01-02",
            "ActualStartDate": None,
            "ActualEndDate": None,
            "WorkCenter": "WC-TEST",
            "ResponsiblePerson": "Test User",
            "OrderText": "Test order with data quality issues"
        },
        {
            "OrderNumber": "1000000007",
            "OrderType": "PM01",
            "Equipment": "10000001",
            "FunctionalLocation": "FL-001",
            "OrderDescription": "Completed order without end date",
            "OrderStatus": "COMPLETED",
            "Priority": "2",
            "PlannedStartDate": "2024-01-20",
            "PlannedEndDate": "2024-01-21",
            "ActualStartDate": "2024-01-20",
            "ActualEndDate": None,  # Missing end date for completed order
            "WorkCenter": "WC-PUMP",
            "ResponsiblePerson": "John Smith",
            "OrderText": "Completed order with missing end date"
        }
    ]
    
    return {
        'problematic_equipment': problematic_equipment,
        'problematic_orders': problematic_orders
    }

def run_all_validations():
    """Run all validations and populate dashboard"""
    print("🚀 Starting comprehensive data quality validation...")
    
    # Extract all data from Mock SAP
    data = extract_mock_data()
    if not data:
        print("❌ Failed to extract data from Mock SAP")
        return False
    
    # Create additional test data with quality issues
    test_data = create_additional_test_data()
    
    # Combine all equipment data
    all_equipment = data['equipment'] + test_data['problematic_equipment']
    all_orders = data['maintenance_orders'] + test_data['problematic_orders']
    
    # Run equipment validations
    equipment_results = run_equipment_validations(all_equipment)
    
    # Run maintenance orders validations
    orders_results = run_maintenance_orders_validations(all_orders)
    
    # Run cross-reference validations
    cross_ref_results = run_cross_reference_validations(data)
    
    # Wait a moment for database updates
    time.sleep(2)
    
    # Check dashboard data
    print("\n📊 Checking dashboard data...")
    response = requests.get('http://localhost:8080/api/validation-summary')
    if response.status_code == 200:
        dashboard_data = response.json()
        print("✅ Dashboard data populated successfully")
        print(f"📈 Total validations: {len(dashboard_data.get('recent_validations', []))}")
    else:
        print("❌ Dashboard data not accessible")
    
    print("\n🎉 Data quality validation workflow completed!")
    print("📊 Dashboard should now show validation results at: http://localhost:8080")
    
    return True

if __name__ == '__main__':
    run_all_validations() 