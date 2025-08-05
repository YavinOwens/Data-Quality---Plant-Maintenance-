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
    
    # Database connection
    db_config = {
        'host': 'localhost',
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
                    'validation_type': 'equipment_completeness',
                    'validation_name': 'equipment_completeness_validation',
                    'success_rate': equipment_completeness_rate,
                    'total_records': random.randint(50, 100),
                    'passed_records': int(random.randint(50, 100) * equipment_completeness_rate),
                    'failed_records': int(random.randint(50, 100) * (1 - equipment_completeness_rate)),
                    'error_details': 'Historical data generation',
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                },
                {
                    'validation_type': 'equipment_accuracy',
                    'validation_name': 'equipment_accuracy_validation',
                    'success_rate': equipment_accuracy_rate,
                    'total_records': random.randint(50, 100),
                    'passed_records': int(random.randint(50, 100) * equipment_accuracy_rate),
                    'failed_records': int(random.randint(50, 100) * (1 - equipment_accuracy_rate)),
                    'error_details': 'Historical data generation',
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                },
                {
                    'validation_type': 'maintenance_orders_timeliness',
                    'validation_name': 'maintenance_orders_timeliness_validation',
                    'success_rate': maintenance_timeliness_rate,
                    'total_records': random.randint(30, 80),
                    'passed_records': int(random.randint(30, 80) * maintenance_timeliness_rate),
                    'failed_records': int(random.randint(30, 80) * (1 - maintenance_timeliness_rate)),
                    'error_details': 'Historical data generation',
                    'created_at': target_date.replace(hour=random.randint(9, 17), 
                                                    minute=random.randint(0, 59),
                                                    second=random.randint(0, 59))
                },
                {
                    'validation_type': 'cross_reference_consistency',
                    'validation_name': 'cross_reference_consistency_validation',
                    'success_rate': cross_reference_rate,
                    'total_records': random.randint(40, 90),
                    'passed_records': int(random.randint(40, 90) * cross_reference_rate),
                    'failed_records': int(random.randint(40, 90) * (1 - cross_reference_rate)),
                    'error_details': 'Historical data generation',
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
                        (validation_type, validation_name, success_rate, total_records, 
                         passed_records, failed_records, error_details, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        validation['validation_type'],
                        validation['validation_name'],
                        validation['success_rate'],
                        validation['total_records'],
                        validation['passed_records'],
                        validation['failed_records'],
                        json.dumps([validation['error_details']]),
                        validation['created_at']
                    ))
                    
                    print(f"  ✅ Saved {validation['validation_type']} for {target_date.strftime('%Y-%m-%d')}")
                    
                except Exception as e:
                    print(f"  ❌ Error saving {validation['validation_type']}: {str(e)}")
            
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
        response = requests.get('http://localhost:8001/health')
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