#!/usr/bin/env python3
"""
Mock SAP S/4HANA PM Data Service
Simulates SAP S/4HANA Plant Maintenance data for local development
"""

import json
import random
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import structlog
import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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
REQUEST_COUNT = Counter('mock_sap_requests_total', 'Total requests to mock SAP', ['endpoint'])
REQUEST_LATENCY = Histogram('mock_sap_request_duration_seconds', 'Request latency in seconds', ['endpoint'])

# Mock PM Data
EQUIPMENT_DATA = [
    {
        "Equipment": "10000001",
        "EquipmentName": "Pump Station Alpha",
        "EquipmentCategory": "M",
        "FunctionalLocation": "FL-001",
        "Manufacturer": "Siemens",
        "ModelNumber": "PS-2023-A",
        "SerialNumber": "SN123456789",
        "InstallationDate": "2023-01-15",
        "EquipmentStatus": "ACTIVE",
        "MaintenancePlant": "1000",
        "WorkCenter": "WC-PUMP",
        "EquipmentDescription": "Main pump station for cooling system",
        "TechnicalData": {
            "Power": "75 kW",
            "FlowRate": "500 L/min",
            "Pressure": "8 bar"
        }
    },
    {
        "Equipment": "10000002",
        "EquipmentName": "Compressor Beta",
        "EquipmentCategory": "M",
        "FunctionalLocation": "FL-002",
        "Manufacturer": "Atlas Copco",
        "ModelNumber": "AC-1500",
        "SerialNumber": "SN987654321",
        "InstallationDate": "2023-03-20",
        "EquipmentStatus": "ACTIVE",
        "MaintenancePlant": "1000",
        "WorkCenter": "WC-COMP",
        "EquipmentDescription": "Air compressor for pneumatic systems",
        "TechnicalData": {
            "Power": "110 kW",
            "Capacity": "1500 L/min",
            "Pressure": "10 bar"
        }
    },
    {
        "Equipment": "10000003",
        "EquipmentName": "Conveyor Gamma",
        "EquipmentCategory": "F",
        "FunctionalLocation": "FL-003",
        "Manufacturer": "Bosch Rexroth",
        "ModelNumber": "CV-500",
        "SerialNumber": "SN456789123",
        "InstallationDate": "2023-02-10",
        "EquipmentStatus": "ACTIVE",
        "MaintenancePlant": "1000",
        "WorkCenter": "WC-CONV",
        "EquipmentDescription": "Belt conveyor for material transport",
        "TechnicalData": {
            "Length": "50 m",
            "Speed": "1.5 m/s",
            "Capacity": "1000 kg/h"
        }
    },
    {
        "Equipment": "10000004",
        "EquipmentName": "Control Panel Delta",
        "EquipmentCategory": "K",
        "FunctionalLocation": "FL-004",
        "Manufacturer": "Allen Bradley",
        "ModelNumber": "CP-2000",
        "SerialNumber": "SN789123456",
        "InstallationDate": "2023-04-05",
        "EquipmentStatus": "ACTIVE",
        "MaintenancePlant": "1000",
        "WorkCenter": "WC-CONTROL",
        "EquipmentDescription": "Main control panel for automation",
        "TechnicalData": {
            "Voltage": "24V DC",
            "Current": "5A",
            "IOPoints": "128"
        }
    },
    {
        "Equipment": "10000005",
        "EquipmentName": "Heat Exchanger Epsilon",
        "EquipmentCategory": "P",
        "FunctionalLocation": "FL-005",
        "Manufacturer": "Alfa Laval",
        "ModelNumber": "HE-300",
        "SerialNumber": "SN321654987",
        "InstallationDate": "2023-01-30",
        "EquipmentStatus": "ACTIVE",
        "MaintenancePlant": "1000",
        "WorkCenter": "WC-HEAT",
        "EquipmentDescription": "Plate heat exchanger for process cooling",
        "TechnicalData": {
            "HeatTransfer": "500 kW",
            "Temperature": "80°C",
            "FlowRate": "200 L/min"
        }
    }
]

FUNCTIONAL_LOCATIONS = [
    {
        "FunctionalLocation": "FL-001",
        "FunctionalLocationName": "Pump Station Area",
        "FunctionalLocationCategory": "M",
        "Plant": "1000",
        "WorkCenter": "WC-PUMP",
        "SuperiorFunctionalLocation": "FL-ROOT",
        "FunctionalLocationDescription": "Main pump station area for cooling systems",
        "Location": "Building A, Level 1",
        "ResponsiblePerson": "John Smith"
    },
    {
        "FunctionalLocation": "FL-002",
        "FunctionalLocationName": "Compressor Room",
        "FunctionalLocationCategory": "M",
        "Plant": "1000",
        "WorkCenter": "WC-COMP",
        "SuperiorFunctionalLocation": "FL-ROOT",
        "FunctionalLocationDescription": "Air compressor room for pneumatic systems",
        "Location": "Building B, Level 0",
        "ResponsiblePerson": "Maria Garcia"
    },
    {
        "FunctionalLocation": "FL-003",
        "FunctionalLocationName": "Conveyor System",
        "FunctionalLocationCategory": "F",
        "Plant": "1000",
        "WorkCenter": "WC-CONV",
        "SuperiorFunctionalLocation": "FL-ROOT",
        "FunctionalLocationDescription": "Material handling conveyor system",
        "Location": "Building C, Level 2",
        "ResponsiblePerson": "David Johnson"
    },
    {
        "FunctionalLocation": "FL-004",
        "FunctionalLocationName": "Control Room",
        "FunctionalLocationCategory": "K",
        "Plant": "1000",
        "WorkCenter": "WC-CONTROL",
        "SuperiorFunctionalLocation": "FL-ROOT",
        "FunctionalLocationDescription": "Main control room for automation systems",
        "Location": "Building A, Level 2",
        "ResponsiblePerson": "Lisa Chen"
    },
    {
        "FunctionalLocation": "FL-005",
        "FunctionalLocationName": "Heat Exchange Area",
        "FunctionalLocationCategory": "P",
        "Plant": "1000",
        "WorkCenter": "WC-HEAT",
        "SuperiorFunctionalLocation": "FL-ROOT",
        "FunctionalLocationDescription": "Heat exchange area for process cooling",
        "Location": "Building B, Level 1",
        "ResponsiblePerson": "Robert Wilson"
    }
]

MAINTENANCE_ORDERS = [
    {
        "OrderNumber": "1000000001",
        "OrderType": "PM01",
        "Equipment": "10000001",
        "FunctionalLocation": "FL-001",
        "OrderDescription": "Preventive maintenance - Pump inspection",
        "OrderStatus": "RELEASED",
        "Priority": "2",
        "PlannedStartDate": "2024-01-15",
        "PlannedEndDate": "2024-01-16",
        "ActualStartDate": "2024-01-15",
        "ActualEndDate": None,
        "WorkCenter": "WC-PUMP",
        "ResponsiblePerson": "John Smith",
        "OrderText": "Monthly preventive maintenance for pump station alpha"
    },
    {
        "OrderNumber": "1000000002",
        "OrderType": "PM01",
        "Equipment": "10000002",
        "FunctionalLocation": "FL-002",
        "OrderDescription": "Preventive maintenance - Compressor service",
        "OrderStatus": "COMPLETED",
        "Priority": "1",
        "PlannedStartDate": "2024-01-10",
        "PlannedEndDate": "2024-01-11",
        "ActualStartDate": "2024-01-10",
        "ActualEndDate": "2024-01-11",
        "WorkCenter": "WC-COMP",
        "ResponsiblePerson": "Maria Garcia",
        "OrderText": "Quarterly preventive maintenance for compressor beta"
    },
    {
        "OrderNumber": "1000000003",
        "OrderType": "PM02",
        "Equipment": "10000003",
        "FunctionalLocation": "FL-003",
        "OrderDescription": "Corrective maintenance - Conveyor belt replacement",
        "OrderStatus": "IN_PROGRESS",
        "Priority": "1",
        "PlannedStartDate": "2024-01-12",
        "PlannedEndDate": "2024-01-13",
        "ActualStartDate": "2024-01-12",
        "ActualEndDate": None,
        "WorkCenter": "WC-CONV",
        "ResponsiblePerson": "David Johnson",
        "OrderText": "Emergency repair - Conveyor belt replacement"
    },
    {
        "OrderNumber": "1000000004",
        "OrderType": "PM01",
        "Equipment": "10000004",
        "FunctionalLocation": "FL-004",
        "OrderDescription": "Preventive maintenance - Control panel inspection",
        "OrderStatus": "CREATED",
        "Priority": "3",
        "PlannedStartDate": "2024-01-20",
        "PlannedEndDate": "2024-01-21",
        "ActualStartDate": None,
        "ActualEndDate": None,
        "WorkCenter": "WC-CONTROL",
        "ResponsiblePerson": "Lisa Chen",
        "OrderText": "Monthly preventive maintenance for control panel"
    },
    {
        "OrderNumber": "1000000005",
        "OrderType": "PM03",
        "Equipment": "10000005",
        "FunctionalLocation": "FL-005",
        "OrderDescription": "Predictive maintenance - Heat exchanger analysis",
        "OrderStatus": "RELEASED",
        "Priority": "2",
        "PlannedStartDate": "2024-01-18",
        "PlannedEndDate": "2024-01-19",
        "ActualStartDate": None,
        "ActualEndDate": None,
        "WorkCenter": "WC-HEAT",
        "ResponsiblePerson": "Robert Wilson",
        "OrderText": "Predictive maintenance based on sensor data analysis"
    }
]

MAINTENANCE_NOTIFICATIONS = [
    {
        "NotificationNumber": "2000000001",
        "NotificationType": "M1",
        "Equipment": "10000001",
        "FunctionalLocation": "FL-001",
        "NotificationDescription": "Pump vibration detected",
        "NotificationStatus": "PROCESSED",
        "Priority": "2",
        "ReportedDate": "2024-01-14",
        "ProcessedDate": "2024-01-15",
        "ReportedBy": "John Smith",
        "NotificationText": "High vibration detected in pump station alpha"
    },
    {
        "NotificationNumber": "2000000002",
        "NotificationType": "M2",
        "Equipment": "10000002",
        "FunctionalLocation": "FL-002",
        "NotificationDescription": "Compressor temperature warning",
        "NotificationStatus": "CREATED",
        "Priority": "1",
        "ReportedDate": "2024-01-16",
        "ProcessedDate": None,
        "ReportedBy": "Maria Garcia",
        "NotificationText": "Temperature exceeding normal operating range"
    },
    {
        "NotificationNumber": "2000000003",
        "NotificationType": "M3",
        "Equipment": "10000003",
        "FunctionalLocation": "FL-003",
        "NotificationDescription": "Conveyor belt wear",
        "NotificationStatus": "PROCESSED",
        "Priority": "1",
        "ReportedDate": "2024-01-11",
        "ProcessedDate": "2024-01-12",
        "ReportedBy": "David Johnson",
        "NotificationText": "Conveyor belt showing signs of wear"
    }
]

MAINTENANCE_PLANS = [
    {
        "MaintenancePlan": "MP001",
        "MaintenancePlanDescription": "Monthly Pump Maintenance",
        "Equipment": "10000001",
        "FunctionalLocation": "FL-001",
        "PlanType": "PM01",
        "CycleUnit": "MONTH",
        "CycleValue": "1",
        "WorkCenter": "WC-PUMP",
        "ResponsiblePerson": "John Smith",
        "PlanStatus": "ACTIVE"
    },
    {
        "MaintenancePlan": "MP002",
        "MaintenancePlanDescription": "Quarterly Compressor Service",
        "Equipment": "10000002",
        "FunctionalLocation": "FL-002",
        "PlanType": "PM01",
        "CycleUnit": "MONTH",
        "CycleValue": "3",
        "WorkCenter": "WC-COMP",
        "ResponsiblePerson": "Maria Garcia",
        "PlanStatus": "ACTIVE"
    },
    {
        "MaintenancePlan": "MP003",
        "MaintenancePlanDescription": "Annual Control Panel Inspection",
        "Equipment": "10000004",
        "FunctionalLocation": "FL-004",
        "PlanType": "PM01",
        "CycleUnit": "YEAR",
        "CycleValue": "1",
        "WorkCenter": "WC-CONTROL",
        "ResponsiblePerson": "Lisa Chen",
        "PlanStatus": "ACTIVE"
    }
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='health').inc()
    with REQUEST_LATENCY.labels(endpoint='health').time():
        return jsonify({
            "status": "healthy",
            "service": "mock-sap-pm",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/extract/equipment', methods=['GET'])
def extract_equipment():
    """Extract equipment data"""
    REQUEST_COUNT.labels(endpoint='extract_equipment').inc()
    with REQUEST_LATENCY.labels(endpoint='extract_equipment').time():
        limit = request.args.get('limit', type=int)
        
        if limit:
            data = EQUIPMENT_DATA[:limit]
        else:
            data = EQUIPMENT_DATA
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        logger.info("Equipment data extracted", count=len(data))
        
        return jsonify({
            "status": "success",
            "message": f"Extracted {len(data)} equipment records",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

@app.route('/extract/functional-locations', methods=['GET'])
def extract_functional_locations():
    """Extract functional locations data"""
    REQUEST_COUNT.labels(endpoint='extract_functional_locations').inc()
    with REQUEST_LATENCY.labels(endpoint='extract_functional_locations').time():
        limit = request.args.get('limit', type=int)
        
        if limit:
            data = FUNCTIONAL_LOCATIONS[:limit]
        else:
            data = FUNCTIONAL_LOCATIONS
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        logger.info("Functional locations data extracted", count=len(data))
        
        return jsonify({
            "status": "success",
            "message": f"Extracted {len(data)} functional location records",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

@app.route('/extract/maintenance-orders', methods=['GET'])
def extract_maintenance_orders():
    """Extract maintenance orders data"""
    REQUEST_COUNT.labels(endpoint='extract_maintenance_orders').inc()
    with REQUEST_LATENCY.labels(endpoint='extract_maintenance_orders').time():
        limit = request.args.get('limit', type=int)
        
        if limit:
            data = MAINTENANCE_ORDERS[:limit]
        else:
            data = MAINTENANCE_ORDERS
        
        # Simulate processing time
        time.sleep(random.uniform(0.2, 0.6))
        
        logger.info("Maintenance orders data extracted", count=len(data))
        
        return jsonify({
            "status": "success",
            "message": f"Extracted {len(data)} maintenance order records",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

@app.route('/extract/maintenance-notifications', methods=['GET'])
def extract_maintenance_notifications():
    """Extract maintenance notifications data"""
    REQUEST_COUNT.labels(endpoint='extract_maintenance_notifications').inc()
    with REQUEST_LATENCY.labels(endpoint='extract_maintenance_notifications').time():
        limit = request.args.get('limit', type=int)
        
        if limit:
            data = MAINTENANCE_NOTIFICATIONS[:limit]
        else:
            data = MAINTENANCE_NOTIFICATIONS
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.4))
        
        logger.info("Maintenance notifications data extracted", count=len(data))
        
        return jsonify({
            "status": "success",
            "message": f"Extracted {len(data)} maintenance notification records",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

@app.route('/extract/maintenance-plans', methods=['GET'])
def extract_maintenance_plans():
    """Extract maintenance plans data"""
    REQUEST_COUNT.labels(endpoint='extract_maintenance_plans').inc()
    with REQUEST_LATENCY.labels(endpoint='extract_maintenance_plans').time():
        limit = request.args.get('limit', type=int)
        
        if limit:
            data = MAINTENANCE_PLANS[:limit]
        else:
            data = MAINTENANCE_PLANS
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        logger.info("Maintenance plans data extracted", count=len(data))
        
        return jsonify({
            "status": "success",
            "message": f"Extracted {len(data)} maintenance plan records",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

@app.route('/extract/all', methods=['GET'])
def extract_all():
    """Extract all PM data"""
    REQUEST_COUNT.labels(endpoint='extract_all').inc()
    with REQUEST_LATENCY.labels(endpoint='extract_all').time():
        results = {
            "equipment": EQUIPMENT_DATA,
            "functional_locations": FUNCTIONAL_LOCATIONS,
            "maintenance_orders": MAINTENANCE_ORDERS,
            "maintenance_notifications": MAINTENANCE_NOTIFICATIONS,
            "maintenance_plans": MAINTENANCE_PLANS
        }
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 1.0))
        
        total_records = sum(len(data) for data in results.values())
        logger.info("All PM data extracted", total_records=total_records)
        
        return jsonify({
            "status": "success",
            "message": f"Extracted {total_records} total records",
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

if __name__ == '__main__':
    logger.info("Starting Mock SAP PM Service", port=8000)
    app.run(host='0.0.0.0', port=8000, debug=True) 