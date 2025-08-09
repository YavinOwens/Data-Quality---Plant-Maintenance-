#!/usr/bin/env python3
"""
SAP S/4HANA Connector
Extracts data from SAP S/4HANA PM module via OData and RFC
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

import requests
from flask import Flask, request, jsonify, g
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

logger = structlog.get_logger(service="sap-connector")

app = Flask(__name__)
# Restrictive CORS in production: allow only configured origins for extract endpoints
_cors_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()]
if _cors_origins:
    CORS(app, resources={r"/extract/*": {"origins": _cors_origins}})
else:
    # Default to no CORS restrictions only in development
    if os.getenv('FLASK_ENV', 'production').lower() == 'development':
        CORS(app)

# Prometheus metrics
REQUEST_COUNT = Counter('sap_connector_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('sap_connector_request_duration_seconds', 'Request latency')
DATA_EXTRACTION_COUNT = Counter('sap_data_extraction_total', 'Data extraction count', ['entity_type'])

class SAPConnector:
    """Handles SAP S/4HANA data extraction"""
    
    def __init__(self):
        def _read_secret(path_env: str, value_env: str) -> Optional[str]:
            path = os.getenv(path_env)
            if path and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception:
                    pass
            return os.getenv(value_env)

        self.sap_host = os.getenv('SAP_HOST')
        self.sap_client = os.getenv('SAP_CLIENT')
        self.sap_username = _read_secret('SAP_USERNAME_FILE', 'SAP_USERNAME')
        self.sap_password = _read_secret('SAP_PASSWORD_FILE', 'SAP_PASSWORD')
        self.gateway_url = os.getenv('SAP_GATEWAY_URL')
        self.session = requests.Session()
        # TLS verification: true by default; can be disabled only via explicit env
        verify_env = os.getenv('SAP_VERIFY_TLS', 'true').lower()
        self.verify_tls: Optional[bool | str]
        if verify_env in ('0', 'false', 'no'):
            self.verify_tls = False
        else:
            ca_bundle = os.getenv('SAP_CA_BUNDLE_FILE')
            self.verify_tls = ca_bundle if ca_bundle and os.path.exists(ca_bundle) else True
        self.authenticated = False
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with SAP S/4HANA"""
        try:
            # OAuth 2.0 authentication
            auth_url = f"https://{self.sap_host}/oauth2/token"
            auth_data = {
                'grant_type': 'password',
                'username': self.sap_username,
                'password': self.sap_password,
                'client_id': 'sap_data_quality'
            }
            
            response = self.session.post(auth_url, data=auth_data, verify=self.verify_tls)
            response.raise_for_status()
            
            token_data = response.json()
            self.session.headers.update({
                'Authorization': f"Bearer {token_data['access_token']}",
                'Content-Type': 'application/json'
            })
            
            self.authenticated = True
            logger.info("SAP authentication successful")
            
        except Exception as e:
            self.authenticated = False
            logger.error(f"SAP authentication failed: {str(e)}")
            raise

# Correlation IDs & error handler
@app.before_request
def _before_request():
    g.correlation_id = request.headers.get('X-Correlation-Id', str(uuid.uuid4()))

@app.after_request
def _after_request(resp):
    try:
        resp.headers['X-Correlation-Id'] = g.correlation_id
    except Exception:
        pass
    return resp
    
    def extract_equipment_data(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract equipment master data"""
        try:
            url = f"{self.gateway_url}/API_PM_EQUIPMENT_SRV/A_Equipment"
            params = {
                '$top': limit,
                '$select': 'Equipment,EquipmentName,EquipmentCategory,Manufacturer,ManufacturerCountry,ManufacturerSerialNumber,FunctionalLocation,EquipmentIsAvailable,EquipmentIsBlocked,EquipmentIsMarkedForDeletion'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            DATA_EXTRACTION_COUNT.labels(entity_type='equipment').inc()
            
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} equipment records")
            return data
            
        except Exception as e:
            logger.error(f"Equipment data extraction failed: {str(e)}")
            raise
    
    def extract_functional_locations(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract functional location data"""
        try:
            url = f"{self.gateway_url}/API_PM_FUNCTIONAL_LOCATION_SRV/A_FunctionalLocation"
            params = {
                '$top': limit,
                '$select': 'FunctionalLocation,FunctionalLocationName,FunctionalLocationCategory,FunctionalLocationIsDeleted,FunctionalLocationIsMarkedForDeletion,Plant,PlantName,CompanyCode,CompanyCodeName'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            DATA_EXTRACTION_COUNT.labels(entity_type='functional_location').inc()
            
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} functional location records")
            return data
            
        except Exception as e:
            logger.error(f"Functional location data extraction failed: {str(e)}")
            raise
    
    def extract_maintenance_orders(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract maintenance order data"""
        try:
            url = f"{self.gateway_url}/API_PM_ORDER_SRV/A_MaintenanceOrder"
            params = {
                '$top': limit,
                '$select': 'MaintenanceOrder,MaintenanceOrderType,MaintenanceOrderCategory,MaintenanceOrderDesc,FunctionalLocation,Equipment,Plant,WorkCenter,WorkCenterName,MaintenanceOrderIsReleased,MaintenanceOrderIsCompleted,MaintenanceOrderIsDeleted,MaintenanceOrderIsMarkedForDeletion,MaintenanceOrderIsClosed,MaintenanceOrderIsLocked,MaintenanceOrderIsUnlocked,MaintenanceOrderIsReleased,MaintenanceOrderIsCompleted,MaintenanceOrderIsDeleted,MaintenanceOrderIsMarkedForDeletion,MaintenanceOrderIsClosed,MaintenanceOrderIsLocked,MaintenanceOrderIsUnlocked'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            DATA_EXTRACTION_COUNT.labels(entity_type='maintenance_order').inc()
            
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} maintenance order records")
            return data
            
        except Exception as e:
            logger.error(f"Maintenance order data extraction failed: {str(e)}")
            raise
    
    def extract_notifications(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract notification data"""
        try:
            url = f"{self.gateway_url}/API_PM_NOTIFICATION_SRV/A_Notification"
            params = {
                '$top': limit,
                '$select': 'Notification,NotificationType,NotificationCategory,NotificationDesc,FunctionalLocation,Equipment,Plant,WorkCenter,WorkCenterName,NotificationIsCompleted,NotificationIsDeleted,NotificationIsMarkedForDeletion,NotificationIsClosed,NotificationIsLocked,NotificationIsUnlocked'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            DATA_EXTRACTION_COUNT.labels(entity_type='notification').inc()
            
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} notification records")
            return data
            
        except Exception as e:
            logger.error(f"Notification data extraction failed: {str(e)}")
            raise
    
    def extract_maintenance_plans(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract maintenance plan data"""
        try:
            url = f"{self.gateway_url}/API_PM_MAINTENANCE_PLAN_SRV/A_MaintenancePlan"
            params = {
                '$top': limit,
                '$select': 'MaintenancePlan,MaintenancePlanDesc,FunctionalLocation,Equipment,Plant,WorkCenter,WorkCenterName,MaintenancePlanIsReleased,MaintenancePlanIsCompleted,MaintenancePlanIsDeleted,MaintenancePlanIsMarkedForDeletion,MaintenancePlanIsClosed,MaintenancePlanIsLocked,MaintenancePlanIsUnlocked'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            DATA_EXTRACTION_COUNT.labels(entity_type='maintenance_plan').inc()
            
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} maintenance plan records")
            return data
            
        except Exception as e:
            logger.error(f"Maintenance plan data extraction failed: {str(e)}")
            raise

# Initialize SAP connector
sap_connector = SAPConnector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='health').inc()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'sap-connector'
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness probe: ensure authentication token is present.
    # TODO: Verify with real SAP by performing a lightweight API call in non-mock environments.
    """
    REQUEST_COUNT.labels(endpoint='ready').inc()
    is_ready = bool(sap_connector.session.headers.get('Authorization')) and sap_connector.authenticated
    status_code = 200 if is_ready else 503
    return jsonify({
        'status': 'ready' if is_ready else 'not_ready',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'sap-connector'
    }), status_code

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