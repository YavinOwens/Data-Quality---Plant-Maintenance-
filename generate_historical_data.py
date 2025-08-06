#!/usr/bin/env python3
"""
Historical Data Generation Script
Generates validation results for the last 30 days to demonstrate trends
"""

import requests
import json
import time
from datetime import datetime, timedelta
import random
import psycopg2

def generate_historical_validations():
    """Generate validation results for the last 30 days"""
    print("📊 Generating historical validation data...")
    
    # Database connection - updated for Docker containers
    db_config = {
        'host': 'sap-postgres',  # Docker service name
        'port': 5432,
        'database': 'sap_data_quality',
        'user': 'sap_user',
        'password': 'your_db_password'
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Get today's date
        today = datetime.now()
        
        # Generate data for the last 30 days
        for day in range(30, -1, -1):
            target_date = today - timedelta(days=day)
            print(f"📅 Generating data for {target_date.strftime('%Y-%m-%d')}...")
            
            # Generate equipment validations
            equipment_completeness_rate = random.uniform(0.7, 0.95)  # 70-95%
            equipment_accuracy_rate = random.uniform(0.6, 0.9)      # 60-90%
            
            # Generate maintenance orders validations
            maintenance_timeliness_rate = random.uniform(0.5, 0.85)  # 50-85%
            
            # Generate cross-reference validations
            cross_reference_rate = random.uniform(0.8, 0.98)        # 80-98%
            
            # Create validation results for this date
            validations = [
                {
                    'dataset_name': 'equipment_data',
                    'rule_name': 'equipment_completeness_validation',
                    'status': 'passed' if equipment_completeness_rate > 0.8 else 'failed',
                    'results': {
                        'success_rate': equipment_completeness_rate,
                        'total_records': random.randint(50, 100),
                        'passed_records': int(random.randint(50, 100) * equipment_completeness_rate),
                        'failed_records': int(random.randint(50, 100) * (1 - equipment_completeness_rate)),
                        'error_details': 'Historical data generation'
                    },
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                },
                {
                    'dataset_name': 'equipment_data',
                    'rule_name': 'equipment_accuracy_validation',
                    'status': 'passed' if equipment_accuracy_rate > 0.8 else 'failed',
                    'results': {
                        'success_rate': equipment_accuracy_rate,
                        'total_records': random.randint(50, 100),
                        'passed_records': int(random.randint(50, 100) * equipment_accuracy_rate),
                        'failed_records': int(random.randint(50, 100) * (1 - equipment_accuracy_rate)),
                        'error_details': 'Historical data generation'
                    },
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                },
                {
                    'dataset_name': 'maintenance_orders',
                    'rule_name': 'maintenance_orders_timeliness_validation',
                    'status': 'passed' if maintenance_timeliness_rate > 0.8 else 'failed',
                    'results': {
                        'success_rate': maintenance_timeliness_rate,
                        'total_records': random.randint(30, 80),
                        'passed_records': int(random.randint(30, 80) * maintenance_timeliness_rate),
                        'failed_records': int(random.randint(30, 80) * (1 - maintenance_timeliness_rate)),
                        'error_details': 'Historical data generation'
                    },
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                },
                {
                    'dataset_name': 'cross_reference_data',
                    'rule_name': 'cross_reference_consistency_validation',
                    'status': 'passed' if cross_reference_rate > 0.8 else 'failed',
                    'results': {
                        'success_rate': cross_reference_rate,
                        'total_records': random.randint(40, 90),
                        'passed_records': int(random.randint(40, 90) * cross_reference_rate),
                        'failed_records': int(random.randint(40, 90) * (1 - cross_reference_rate)),
                        'error_details': 'Historical data generation'
                    },
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                }
            ]
            
            # Insert validation results directly into database
            for validation in validations:
                try:
                    cursor.execute("""
                        INSERT INTO validation_results 
                        (dataset_name, rule_name, status, results, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        validation['dataset_name'],
                        validation['rule_name'],
                        validation['status'],
                        json.dumps(validation['results']),
                        validation['created_at'],
                        validation['created_at']  # updated_at same as created_at
                    ))
                    
                    print(f"  ✅ Saved {validation['rule_name']} for {target_date.strftime('%Y-%m-%d')}")
                    
                except Exception as e:
                    print(f"  ❌ Error saving {validation['rule_name']}: {str(e)}")
            
            # Commit after each day
            conn.commit()
        
        cursor.close()
        conn.close()
        
        print("✅ Historical data generation completed!")
        
    except Exception as e:
        print(f"❌ Database connection error: {str(e)}")

def main():
    """Main function"""
    print("🚀 Starting historical data generation...")
    
    # Check if services are running
    try:
        # Check validation engine
        response = requests.get('http://validation-engine:8001/health')  # Updated hostname
        if response.status_code != 200:
            print("❌ Validation engine is not running")
            return
        
        print("✅ Validation engine is running")
        
        # Generate historical data
        generate_historical_validations()
        
        print("\n🎉 Historical data generation completed successfully!")
        print("📊 You can now view the Quality Trends chart with 30 days of data")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure all services are running: docker-compose up -d")

if __name__ == "__main__":
    main() 