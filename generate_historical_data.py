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
        today = datetime.utcnow()
        
        # Validation types and names
        validation_types = ['equipment_data', 'maintenance_orders', 'cross_reference_data']
        validation_names = [
            'completeness_check',
            'accuracy_validation', 
            'consistency_check',
            'timeliness_validation',
            'format_validation'
        ]
        
        # Generate data for the last 30 days
        for day in range(30, -1, -1):
            current_date = today - timedelta(days=day)
            
<<<<<<< HEAD
            # Generate 4 validations per day
            for _ in range(4):
                validation_type = random.choice(validation_types)
                validation_name = random.choice(validation_names)
                
                # Generate realistic success rates
                if random.random() < 0.8:  # 80% chance of good data
                    success_rate = random.uniform(0.85, 1.0)
                else:
                    success_rate = random.uniform(0.0, 0.84)
                
                # Calculate passed and failed records
                total_records = random.randint(100, 1000)
                passed_records = int(total_records * success_rate)
                failed_records = total_records - passed_records
                
                # Generate error details
                if failed_records > 0:
                    error_details = {
                        "errors": [
                            {"field": "equipment_id", "message": "Missing required field"},
                            {"field": "maintenance_date", "message": "Invalid date format"}
                        ],
                        "failed_count": failed_records
                    }
                else:
                    error_details = None
                
                # Insert the record
                cursor.execute("""
                    INSERT INTO validation_results 
                    (validation_type, validation_name, success_rate, passed_records, failed_records, error_details, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    validation_type,
                    validation_name,
                    success_rate,
                    passed_records,
                    failed_records,
                    json.dumps(error_details) if error_details else None,
                    current_date
                ))
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Successfully generated historical validation data!")
        print(f"📅 Data spans from {(today - timedelta(days=30)).strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
        print(f"📊 Total records: {30 * 4} validations")
        
    except Exception as e:
        print(f"❌ Error generating historical data: {str(e)}")
        raise

if __name__ == "__main__":
    generate_historical_validations() 