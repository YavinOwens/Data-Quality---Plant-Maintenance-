#!/usr/bin/env python3
"""
Validation Engine for SAP S/4HANA PM Data Quality
Runs data quality rules and validations
"""

import json
import os
import time
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import structlog
import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
import jsonschema
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = Flask(__name__)
CORS(app)

# Prometheus metrics
VALIDATION_COUNT = Counter('validation_requests_total', 'Total validation requests', ['validation_type'])
VALIDATION_DURATION = Histogram('validation_duration_seconds', 'Validation duration in seconds', ['validation_type'])
VALIDATION_SUCCESS = Counter('validation_success_total', 'Successful validations', ['validation_type'])
VALIDATION_FAILURE = Counter('validation_failure_total', 'Failed validations', ['validation_type'])

class DataQualityValidator:
    """Data quality validation engine"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'database': os.getenv('DB_NAME', 'sap_data_quality'),
            'user': os.getenv('DB_USER', 'sap_user'),
            'password': os.getenv('DB_PASSWORD', 'your_db_password')
        }
        self.rules_path = os.getenv('RULES_PATH', '/app/config/rules')
        self.logger = structlog.get_logger()
        
        # Initialize database connection
        self._init_database()
        
    def _init_database(self):
        """Initialize database connection and create tables if needed"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create validation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id SERIAL PRIMARY KEY,
                    validation_type VARCHAR(100) NOT NULL,
                    validation_name VARCHAR(200) NOT NULL,
                    success_rate DECIMAL(5,4),
                    total_records INTEGER,
                    passed_records INTEGER,
                    failed_records INTEGER,
                    error_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create data quality issues table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_issues (
                    id SERIAL PRIMARY KEY,
                    issue_type VARCHAR(100) NOT NULL,
                    issue_description TEXT NOT NULL,
                    affected_records INTEGER,
                    severity VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error("Database initialization failed", error=str(e))
            raise
    
    def validate_equipment_completeness(self, data: List[Dict]) -> Dict:
        """Validate equipment data completeness"""
        VALIDATION_COUNT.labels(validation_type='equipment_completeness').inc()
        
        with VALIDATION_DURATION.labels(validation_type='equipment_completeness').time():
            try:
                df = pd.DataFrame(data)
                total_records = len(df)
                
                # Check required fields
                required_fields = ['Equipment', 'EquipmentName', 'EquipmentCategory', 'FunctionalLocation']
                missing_fields = []
                
                for field in required_fields:
                    if field not in df.columns:
                        missing_fields.append(field)
                    else:
                        null_count = df[field].isnull().sum()
                        if null_count > 0:
                            missing_fields.append(f"{field} ({null_count} null values)")
                
                # Check for empty strings
                empty_strings = []
                for field in required_fields:
                    if field in df.columns:
                        empty_count = (df[field] == '').sum()
                        if empty_count > 0:
                            empty_strings.append(f"{field} ({empty_count} empty values)")
                
                passed_records = total_records - len([r for r in data if any(
                    not r.get(field) for field in required_fields
                )])
                
                success_rate = passed_records / total_records if total_records > 0 else 0
                
                result = {
                    'validation_type': 'equipment_completeness',
                    'success_rate': success_rate,
                    'total_records': total_records,
                    'passed_records': passed_records,
                    'failed_records': total_records - passed_records,
                    'issues': missing_fields + empty_strings
                }
                
                if success_rate >= 0.95:
                    VALIDATION_SUCCESS.labels(validation_type='equipment_completeness').inc()
                else:
                    VALIDATION_FAILURE.labels(validation_type='equipment_completeness').inc()
                
                return result
                
            except Exception as e:
                VALIDATION_FAILURE.labels(validation_type='equipment_completeness').inc()
                self.logger.error("Equipment completeness validation failed", error=str(e))
                return {
                    'validation_type': 'equipment_completeness',
                    'success_rate': 0,
                    'total_records': 0,
                    'passed_records': 0,
                    'failed_records': 0,
                    'error': str(e)
                }
    
    def validate_equipment_accuracy(self, data: List[Dict]) -> Dict:
        """Validate equipment data accuracy"""
        VALIDATION_COUNT.labels(validation_type='equipment_accuracy').inc()
        
        with VALIDATION_DURATION.labels(validation_type='equipment_accuracy').time():
            try:
                df = pd.DataFrame(data)
                total_records = len(df)
                
                # Check equipment categories
                valid_categories = ['M', 'F', 'K', 'P']
                invalid_categories = df[~df['EquipmentCategory'].isin(valid_categories)]
                
                # Check serial number format (basic validation)
                invalid_serial_numbers = []
                for _, row in df.iterrows():
                    serial = str(row.get('SerialNumber', ''))
                    if len(serial) < 5:  # Basic length check
                        invalid_serial_numbers.append(row.get('Equipment', ''))
                
                passed_records = total_records - len(invalid_categories) - len(invalid_serial_numbers)
                success_rate = passed_records / total_records if total_records > 0 else 0
                
                result = {
                    'validation_type': 'equipment_accuracy',
                    'success_rate': success_rate,
                    'total_records': total_records,
                    'passed_records': passed_records,
                    'failed_records': total_records - passed_records,
                    'issues': [
                        f"Invalid equipment categories: {len(invalid_categories)}",
                        f"Invalid serial numbers: {len(invalid_serial_numbers)}"
                    ]
                }
                
                if success_rate >= 0.95:
                    VALIDATION_SUCCESS.labels(validation_type='equipment_accuracy').inc()
                else:
                    VALIDATION_FAILURE.labels(validation_type='equipment_accuracy').inc()
                
                return result
                
            except Exception as e:
                VALIDATION_FAILURE.labels(validation_type='equipment_accuracy').inc()
                self.logger.error("Equipment accuracy validation failed", error=str(e))
                return {
                    'validation_type': 'equipment_accuracy',
                    'success_rate': 0,
                    'total_records': 0,
                    'passed_records': 0,
                    'failed_records': 0,
                    'error': str(e)
                }
    
    def validate_maintenance_orders_timeliness(self, data: List[Dict]) -> Dict:
        """Validate maintenance orders timeliness"""
        VALIDATION_COUNT.labels(validation_type='maintenance_orders_timeliness').inc()
        
        with VALIDATION_DURATION.labels(validation_type='maintenance_orders_timeliness').time():
            try:
                df = pd.DataFrame(data)
                total_records = len(df)
                
                # Check for overdue orders
                today = datetime.now().date()
                overdue_orders = []
                
                for _, row in df.iterrows():
                    planned_end = row.get('PlannedEndDate')
                    if planned_end:
                        try:
                            planned_date = datetime.strptime(planned_end, '%Y-%m-%d').date()
                            if planned_date < today and row.get('OrderStatus') not in ['COMPLETED', 'CANCELLED']:
                                overdue_orders.append(row.get('OrderNumber', ''))
                        except:
                            pass
                
                # Check for orders without actual end dates when completed
                incomplete_orders = []
                for _, row in df.iterrows():
                    if row.get('OrderStatus') == 'COMPLETED' and not row.get('ActualEndDate'):
                        incomplete_orders.append(row.get('OrderNumber', ''))
                
                passed_records = total_records - len(overdue_orders) - len(incomplete_orders)
                success_rate = passed_records / total_records if total_records > 0 else 0
                
                result = {
                    'validation_type': 'maintenance_orders_timeliness',
                    'success_rate': success_rate,
                    'total_records': total_records,
                    'passed_records': passed_records,
                    'failed_records': total_records - passed_records,
                    'issues': [
                        f"Overdue orders: {len(overdue_orders)}",
                        f"Incomplete orders: {len(incomplete_orders)}"
                    ]
                }
                
                if success_rate >= 0.95:
                    VALIDATION_SUCCESS.labels(validation_type='maintenance_orders_timeliness').inc()
                else:
                    VALIDATION_FAILURE.labels(validation_type='maintenance_orders_timeliness').inc()
                
                return result
                
            except Exception as e:
                VALIDATION_FAILURE.labels(validation_type='maintenance_orders_timeliness').inc()
                self.logger.error("Maintenance orders timeliness validation failed", error=str(e))
                return {
                    'validation_type': 'maintenance_orders_timeliness',
                    'success_rate': 0,
                    'total_records': 0,
                    'passed_records': 0,
                    'failed_records': 0,
                    'error': str(e)
                }
    
    def validate_cross_reference_consistency(self, equipment_data: List[Dict], 
                                          functional_locations: List[Dict],
                                          maintenance_orders: List[Dict]) -> Dict:
        """Validate cross-reference consistency between datasets"""
        VALIDATION_COUNT.labels(validation_type='cross_reference_consistency').inc()
        
        with VALIDATION_DURATION.labels(validation_type='cross_reference_consistency').time():
            try:
                equipment_df = pd.DataFrame(equipment_data)
                fl_df = pd.DataFrame(functional_locations)
                mo_df = pd.DataFrame(maintenance_orders)
                
                total_records = len(equipment_df) + len(fl_df) + len(mo_df)
                
                # Check equipment-functional location consistency
                equipment_fl_mismatch = []
                for _, eq in equipment_df.iterrows():
                    eq_fl = eq.get('FunctionalLocation')
                    if eq_fl and eq_fl not in fl_df['FunctionalLocation'].values:
                        equipment_fl_mismatch.append(eq.get('Equipment', ''))
                
                # Check maintenance order-equipment consistency
                mo_equipment_mismatch = []
                for _, mo in mo_df.iterrows():
                    mo_equipment = mo.get('Equipment')
                    if mo_equipment and mo_equipment not in equipment_df['Equipment'].values:
                        mo_equipment_mismatch.append(mo.get('OrderNumber', ''))
                
                # Check maintenance order-functional location consistency
                mo_fl_mismatch = []
                for _, mo in mo_df.iterrows():
                    mo_fl = mo.get('FunctionalLocation')
                    if mo_fl and mo_fl not in fl_df['FunctionalLocation'].values:
                        mo_fl_mismatch.append(mo.get('OrderNumber', ''))
                
                total_issues = len(equipment_fl_mismatch) + len(mo_equipment_mismatch) + len(mo_fl_mismatch)
                passed_records = total_records - total_issues
                success_rate = passed_records / total_records if total_records > 0 else 0
                
                result = {
                    'validation_type': 'cross_reference_consistency',
                    'success_rate': success_rate,
                    'total_records': total_records,
                    'passed_records': passed_records,
                    'failed_records': total_records - passed_records,
                    'issues': [
                        f"Equipment-FL mismatches: {len(equipment_fl_mismatch)}",
                        f"MO-Equipment mismatches: {len(mo_equipment_mismatch)}",
                        f"MO-FL mismatches: {len(mo_fl_mismatch)}"
                    ]
                }
                
                if success_rate >= 0.95:
                    VALIDATION_SUCCESS.labels(validation_type='cross_reference_consistency').inc()
                else:
                    VALIDATION_FAILURE.labels(validation_type='cross_reference_consistency').inc()
                
                return result
                
            except Exception as e:
                VALIDATION_FAILURE.labels(validation_type='cross_reference_consistency').inc()
                self.logger.error("Cross-reference consistency validation failed", error=str(e))
                return {
                    'validation_type': 'cross_reference_consistency',
                    'success_rate': 0,
                    'total_records': 0,
                    'passed_records': 0,
                    'failed_records': 0,
                    'error': str(e)
                }
    
    def save_validation_results(self, results: Dict, created_at: str = None):
        """Save validation results to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            if created_at:
                cursor.execute("""
                    INSERT INTO validation_results 
                    (validation_type, validation_name, success_rate, total_records, 
                     passed_records, failed_records, error_details, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    results.get('validation_type'),
                    f"{results.get('validation_type')}_validation",
                    results.get('success_rate', 0),
                    results.get('total_records', 0),
                    results.get('passed_records', 0),
                    results.get('failed_records', 0),
                    json.dumps(results.get('issues', [])),
                    created_at
                ))
            else:
                cursor.execute("""
                    INSERT INTO validation_results 
                    (validation_type, validation_name, success_rate, total_records, 
                     passed_records, failed_records, error_details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    results.get('validation_type'),
                    f"{results.get('validation_type')}_validation",
                    results.get('success_rate', 0),
                    results.get('total_records', 0),
                    results.get('passed_records', 0),
                    results.get('failed_records', 0),
                    json.dumps(results.get('issues', []))
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("Validation results saved to database", validation_type=results.get('validation_type'))
            
        except Exception as e:
            self.logger.error("Failed to save validation results", error=str(e))

# Initialize validator
validator = DataQualityValidator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "validation-engine",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/rules', methods=['GET'])
def get_rules():
    """Get available validation rules"""
    rules = [
        {
            "name": "equipment_completeness",
            "description": "Validate equipment data completeness",
            "type": "completeness"
        },
        {
            "name": "equipment_accuracy", 
            "description": "Validate equipment data accuracy",
            "type": "accuracy"
        },
        {
            "name": "maintenance_orders_timeliness",
            "description": "Validate maintenance orders timeliness",
            "type": "timeliness"
        },
        {
            "name": "cross_reference_consistency",
            "description": "Validate cross-reference consistency",
            "type": "consistency"
        }
    ]
    
    return jsonify({"rules": rules})

@app.route('/validate/equipment', methods=['POST'])
def validate_equipment():
    """Validate equipment data"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"error": "No data provided"}), 400
        
        equipment_data = data['data']
        
        # Run completeness validation
        completeness_result = validator.validate_equipment_completeness(equipment_data)
        validator.save_validation_results(completeness_result)
        
        # Run accuracy validation
        accuracy_result = validator.validate_equipment_accuracy(equipment_data)
        validator.save_validation_results(accuracy_result)
        
        results = {
            "completeness": completeness_result,
            "accuracy": accuracy_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Equipment validation completed", results=results)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error("Equipment validation failed", error=str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/validate/maintenance-orders', methods=['POST'])
def validate_maintenance_orders():
    """Validate maintenance orders data"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"error": "No data provided"}), 400
        
        maintenance_orders_data = data['data']
        
        # Run timeliness validation
        timeliness_result = validator.validate_maintenance_orders_timeliness(maintenance_orders_data)
        validator.save_validation_results(timeliness_result)
        
        results = {
            "timeliness": timeliness_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Maintenance orders validation completed", results=results)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error("Maintenance orders validation failed", error=str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/validate/cross-reference', methods=['POST'])
def validate_cross_reference():
    """Validate cross-reference consistency"""
    try:
        data = request.get_json()
        if not data or 'equipment' not in data or 'functional_locations' not in data or 'maintenance_orders' not in data:
            return jsonify({"error": "Missing required data"}), 400
        
        equipment_data = data['equipment']
        functional_locations_data = data['functional_locations']
        maintenance_orders_data = data['maintenance_orders']
        
        # Run cross-reference validation
        consistency_result = validator.validate_cross_reference_consistency(
            equipment_data, functional_locations_data, maintenance_orders_data
        )
        validator.save_validation_results(consistency_result)
        
        results = {
            "consistency": consistency_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Cross-reference validation completed", results=results)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error("Cross-reference validation failed", error=str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Validation Engine", port=8001)
    app.run(host='0.0.0.0', port=8001, debug=True) 