#!/usr/bin/env python3
"""
SAP S/4HANA Connector - Test Version
Simulates data extraction from SAP S/4HANA PM module for testing
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

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
REQUEST_COUNT = Counter('sap_connector_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('sap_connector_request_duration_seconds', 'Request latency')
DATA_EXTRACTION_COUNT = Counter('sap_data_extraction_total', 'Data extraction count', ['entity_type'])

class MockSAPConnector:
    """Mock SAP S/4HANA connector for testing"""
    
    def __init__(self):
        self.sap_host = os.getenv('SAP_HOST', 'sap.example.com')
        self.sap_client = os.getenv('SAP_CLIENT', '100')
        self.sap_username = os.getenv('SAP_USERNAME', 'sap_user')
        self.sap_password = os.getenv('SAP_PASSWORD', 'your_sap_password')
        self.gateway_url = os.getenv('SAP_GATEWAY_URL', 'https://sap.example.com:44300/sap/opu/odata/sap')
        
        logger.info("Mock SAP connector initialized for testing")
    
    def extract_equipment_data(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract mock equipment master data"""
        try:
            # Generate mock equipment data
            equipment_data = []
            for i in range(min(limit, 50)):  # Limit to 50 records for testing
                equipment_data.append({
                    'Equipment': f'EQ{i+1:03d}',
                    'EquipmentName': f'Equipment {i+1}',
                    'EquipmentCategory': 'M' if i % 3 == 0 else 'F' if i % 3 == 1 else 'K',
                    'Manufacturer': f'Manufacturer {i+1}',
                    'ManufacturerCountry': 'US' if i % 2 == 0 else 'DE',
                    'ManufacturerSerialNumber': f'SN{i+1:06d}',
                    'FunctionalLocation': f'FL{i+1:03d}' if i % 2 == 0 else None,
                    'EquipmentIsAvailable': True,
                    'EquipmentIsBlocked': False,
                    'EquipmentIsMarkedForDeletion': False
                })
            
            data = {
                'd': {
                    'results': equipment_data
                }
            }
            
            DATA_EXTRACTION_COUNT.labels(entity_type='equipment').inc()
            
            logger.info(f"Extracted {len(equipment_data)} mock equipment records")
            return data
            
        except Exception as e:
            logger.error(f"Equipment data extraction failed: {str(e)}")
            raise
    
    def extract_functional_locations(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract mock functional location data"""
        try:
            # Generate mock functional location data
            floc_data = []
            for i in range(min(limit, 30)):  # Limit to 30 records for testing
                floc_data.append({
                    'FunctionalLocation': f'FL{i+1:03d}',
                    'FunctionalLocationName': f'Functional Location {i+1}',
                    'FunctionalLocationCategory': 'M' if i % 2 == 0 else 'F',
                    'FunctionalLocationIsDeleted': False,
                    'FunctionalLocationIsMarkedForDeletion': False,
                    'Plant': f'PLANT{i+1:02d}',
                    'PlantName': f'Plant {i+1}',
                    'CompanyCode': '1000',
                    'CompanyCodeName': 'Test Company'
                })
            
            data = {
                'd': {
                    'results': floc_data
                }
            }
            
            DATA_EXTRACTION_COUNT.labels(entity_type='functional_location').inc()
            
            logger.info(f"Extracted {len(floc_data)} mock functional location records")
            return data
            
        except Exception as e:
            logger.error(f"Functional location data extraction failed: {str(e)}")
            raise
    
    def extract_maintenance_orders(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract mock maintenance order data"""
        try:
            # Generate mock maintenance order data
            order_data = []
            for i in range(min(limit, 40)):  # Limit to 40 records for testing
                order_data.append({
                    'MaintenanceOrder': f'MO{i+1:06d}',
                    'MaintenanceOrderType': 'PM' if i % 3 == 0 else 'CM' if i % 3 == 1 else 'EM',
                    'MaintenanceOrderCategory': 'M' if i % 2 == 0 else 'F',
                    'MaintenanceOrderDesc': f'Maintenance Order {i+1}',
                    'FunctionalLocation': f'FL{i+1:03d}' if i % 2 == 0 else None,
                    'Equipment': f'EQ{i+1:03d}' if i % 2 == 1 else None,
                    'Plant': f'PLANT{i+1:02d}',
                    'WorkCenter': f'WC{i+1:03d}',
                    'WorkCenterName': f'Work Center {i+1}',
                    'MaintenanceOrderIsReleased': True if i % 3 == 0 else False,
                    'MaintenanceOrderIsCompleted': True if i % 4 == 0 else False,
                    'MaintenanceOrderIsDeleted': False,
                    'MaintenanceOrderIsMarkedForDeletion': False,
                    'MaintenanceOrderIsClosed': True if i % 5 == 0 else False,
                    'MaintenanceOrderIsLocked': False,
                    'MaintenanceOrderIsUnlocked': True
                })
            
            data = {
                'd': {
                    'results': order_data
                }
            }
            
            DATA_EXTRACTION_COUNT.labels(entity_type='maintenance_order').inc()
            
            logger.info(f"Extracted {len(order_data)} mock maintenance order records")
            return data
            
        except Exception as e:
            logger.error(f"Maintenance order data extraction failed: {str(e)}")
            raise
    
    def extract_notifications(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract mock notification data"""
        try:
            # Generate mock notification data
            notification_data = []
            for i in range(min(limit, 25)):  # Limit to 25 records for testing
                notification_data.append({
                    'Notification': f'NT{i+1:06d}',
                    'NotificationType': 'M1' if i % 3 == 0 else 'M2' if i % 3 == 1 else 'M3',
                    'NotificationCategory': 'M' if i % 2 == 0 else 'F',
                    'NotificationDesc': f'Notification {i+1}',
                    'FunctionalLocation': f'FL{i+1:03d}' if i % 2 == 0 else None,
                    'Equipment': f'EQ{i+1:03d}' if i % 2 == 1 else None,
                    'Plant': f'PLANT{i+1:02d}',
                    'WorkCenter': f'WC{i+1:03d}',
                    'WorkCenterName': f'Work Center {i+1}',
                    'NotificationIsCompleted': True if i % 3 == 0 else False,
                    'NotificationIsDeleted': False,
                    'NotificationIsMarkedForDeletion': False,
                    'NotificationIsClosed': True if i % 4 == 0 else False,
                    'NotificationIsLocked': False,
                    'NotificationIsUnlocked': True
                })
            
            data = {
                'd': {
                    'results': notification_data
                }
            }
            
            DATA_EXTRACTION_COUNT.labels(entity_type='notification').inc()
            
            logger.info(f"Extracted {len(notification_data)} mock notification records")
            return data
            
        except Exception as e:
            logger.error(f"Notification data extraction failed: {str(e)}")
            raise
    
    def extract_maintenance_plans(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract mock maintenance plan data"""
        try:
            # Generate mock maintenance plan data
            plan_data = []
            for i in range(min(limit, 20)):  # Limit to 20 records for testing
                plan_data.append({
                    'MaintenancePlan': f'MP{i+1:06d}',
                    'MaintenancePlanDesc': f'Maintenance Plan {i+1}',
                    'FunctionalLocation': f'FL{i+1:03d}' if i % 2 == 0 else None,
                    'Equipment': f'EQ{i+1:03d}' if i % 2 == 1 else None,
                    'Plant': f'PLANT{i+1:02d}',
                    'WorkCenter': f'WC{i+1:03d}',
                    'WorkCenterName': f'Work Center {i+1}',
                    'MaintenancePlanIsReleased': True if i % 3 == 0 else False,
                    'MaintenancePlanIsCompleted': True if i % 4 == 0 else False,
                    'MaintenancePlanIsDeleted': False,
                    'MaintenancePlanIsMarkedForDeletion': False,
                    'MaintenancePlanIsClosed': True if i % 5 == 0 else False,
                    'MaintenancePlanIsLocked': False,
                    'MaintenancePlanIsUnlocked': True
                })
            
            data = {
                'd': {
                    'results': plan_data
                }
            }
            
            DATA_EXTRACTION_COUNT.labels(entity_type='maintenance_plan').inc()
            
            logger.info(f"Extracted {len(plan_data)} mock maintenance plan records")
            return data
            
        except Exception as e:
            logger.error(f"Maintenance plan data extraction failed: {str(e)}")
            raise

# Initialize mock SAP connector
sap_connector = MockSAPConnector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='health').inc()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'sap-connector-test'
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(endpoint='metrics').inc()
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/extract/equipment', methods=['GET'])
@REQUEST_LATENCY.time()
def extract_equipment():
    """Extract equipment data"""
    REQUEST_COUNT.labels(endpoint='extract_equipment').inc()
    
    try:
        limit = request.args.get('limit', 1000, type=int)
        data = sap_connector.extract_equipment_data(limit)
        
        # Save to file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"/app/data/equipment_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': f'Extracted {len(data.get("d", {}).get("results", []))} equipment records',
            'filename': filename,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Equipment extraction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/extract/functional-locations', methods=['GET'])
@REQUEST_LATENCY.time()
def extract_functional_locations():
    """Extract functional location data"""
    REQUEST_COUNT.labels(endpoint='extract_functional_locations').inc()
    
    try:
        limit = request.args.get('limit', 1000, type=int)
        data = sap_connector.extract_functional_locations(limit)
        
        # Save to file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"/app/data/functional_locations_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': f'Extracted {len(data.get("d", {}).get("results", []))} functional location records',
            'filename': filename,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Functional location extraction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/extract/maintenance-orders', methods=['GET'])
@REQUEST_LATENCY.time()
def extract_maintenance_orders():
    """Extract maintenance order data"""
    REQUEST_COUNT.labels(endpoint='extract_maintenance_orders').inc()
    
    try:
        limit = request.args.get('limit', 1000, type=int)
        data = sap_connector.extract_maintenance_orders(limit)
        
        # Save to file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"/app/data/maintenance_orders_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': f'Extracted {len(data.get("d", {}).get("results", []))} maintenance order records',
            'filename': filename,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Maintenance order extraction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/extract/all', methods=['GET'])
@REQUEST_LATENCY.time()
def extract_all_data():
    """Extract all PM data"""
    REQUEST_COUNT.labels(endpoint='extract_all').inc()
    
    try:
        results = {}
        
        # Extract all data types
        data_types = [
            ('equipment', sap_connector.extract_equipment_data),
            ('functional_locations', sap_connector.extract_functional_locations),
            ('maintenance_orders', sap_connector.extract_maintenance_orders),
            ('notifications', sap_connector.extract_notifications),
            ('maintenance_plans', sap_connector.extract_maintenance_plans)
        ]
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        for data_type, extractor in data_types:
            try:
                data = extractor()
                filename = f"/app/data/{data_type}_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                results[data_type] = {
                    'status': 'success',
                    'filename': filename,
                    'record_count': len(data.get('d', {}).get('results', []))
                }
                
            except Exception as e:
                logger.error(f"{data_type} extraction failed: {str(e)}")
                results[data_type] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        return jsonify({
            'status': 'completed',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Bulk extraction failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False) 