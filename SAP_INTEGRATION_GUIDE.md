# SAP S/4HANA Integration Guide

This guide explains how to configure the SAP S/4HANA Data Quality Workflow for real SAP integration, replacing the mock environment with actual SAP S/4HANA connectivity.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [SAP Connection Configuration](#sap-connection-configuration)
3. [Authentication Setup](#authentication-setup)
4. [Dependencies Installation](#dependencies-installation)
5. [Environment Configuration](#environment-configuration)
6. [SAP Connector Configuration](#sap-connector-configuration)
7. [Testing Real SAP Connection](#testing-real-sap-connection)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### SAP System Requirements

- **SAP S/4HANA** system (on-premise or cloud)
- **SAP Gateway** enabled and configured
- **OData services** activated for PM module
- **RFC access** enabled
- **User credentials** with appropriate authorizations

### Required SAP Authorizations

The SAP user needs the following authorizations:

```
S_DEVELOP (Development authorization)
S_RFC (RFC authorization)
S_RS_ADMWB (Administration Workbench)
S_RS_PREM (Premise authorization)
```

### SAP OData Services

Ensure these OData services are activated in your SAP system:

- **Equipment Master Data**: `/sap/opu/odata/sap/ZPM_EQUIPMENT_SRV`
- **Functional Locations**: `/sap/opu/odata/sap/ZPM_FUNCTIONAL_LOCATION_SRV`
- **Maintenance Orders**: `/sap/opu/odata/sap/ZPM_MAINTENANCE_ORDER_SRV`
- **Notifications**: `/sap/opu/odata/sap/ZPM_NOTIFICATION_SRV`
- **Maintenance Plans**: `/sap/opu/odata/sap/ZPM_MAINTENANCE_PLAN_SRV`

## SAP Connection Configuration

### 1. Update Environment Variables

Edit your `.env` file with real SAP connection details:

```bash
# SAP Connection Settings
SAP_HOST=your-sap-server.com
SAP_CLIENT=100
SAP_USERNAME=your_sap_user
SAP_PASSWORD=your_sap_password
SAP_GATEWAY_URL=https://your-sap-server.com:44300/sap/opu/odata/sap

# SAP RFC Settings (for direct RFC calls)
SAP_RFC_DESTINATION=YOUR_RFC_DEST
SAP_ASHOST=your-sap-server.com
SAP_SYSNR=00
SAP_CLIENT=100

# SAP OAuth Settings (if using OAuth 2.0)
SAP_OAUTH_CLIENT_ID=your_oauth_client_id
SAP_OAUTH_CLIENT_SECRET=your_oauth_client_secret
SAP_OAUTH_TOKEN_URL=https://your-sap-server.com:44300/sap/opu/odata/sap/oauth2/token

# Certificate-based Authentication (optional)
SAP_CERT_PATH=/path/to/your/certificate.pem
SAP_KEY_PATH=/path/to/your/private_key.pem
SAP_CERT_PASSWORD=your_cert_password
```

### 2. SAP Gateway Configuration

Ensure your SAP Gateway is configured for external access:

```abap
* In SAP Gateway Administration (transaction /IWFND/MAINT_SERVICE)
* Activate the following services:
- ZPM_EQUIPMENT_SRV
- ZPM_FUNCTIONAL_LOCATION_SRV  
- ZPM_MAINTENANCE_ORDER_SRV
- ZPM_NOTIFICATION_SRV
- ZPM_MAINTENANCE_PLAN_SRV
```

## Authentication Setup

### Option 1: Basic Authentication

For development/testing environments:

```python
# In sap-connector/app.py
import requests
from requests.auth import HTTPBasicAuth

class SAPConnector:
    def __init__(self):
        self.auth = HTTPBasicAuth(
            os.getenv('SAP_USERNAME'),
            os.getenv('SAP_PASSWORD')
        )
```

### Option 2: OAuth 2.0 Authentication

For production environments:

```python
# In sap-connector/app.py
import requests
from requests_oauthlib import OAuth2Session

class SAPConnector:
    def __init__(self):
        self.oauth_client_id = os.getenv('SAP_OAUTH_CLIENT_ID')
        self.oauth_client_secret = os.getenv('SAP_OAUTH_CLIENT_SECRET')
        self.token_url = os.getenv('SAP_OAUTH_TOKEN_URL')
        
    def get_access_token(self):
        oauth = OAuth2Session(self.oauth_client_id)
        token = oauth.fetch_token(
            token_url=self.token_url,
            client_secret=self.oauth_client_secret
        )
        return token['access_token']
```

### Option 3: Certificate-based Authentication

For high-security environments:

```python
# In sap-connector/app.py
import requests

class SAPConnector:
    def __init__(self):
        self.cert_path = os.getenv('SAP_CERT_PATH')
        self.key_path = os.getenv('SAP_KEY_PATH')
        self.cert_password = os.getenv('SAP_CERT_PASSWORD')
        
    def get_session(self):
        return requests.Session().cert=(
            self.cert_path, 
            self.key_path
        )
```

## Dependencies Installation

### 1. Install SAP-Specific Dependencies

```bash
# Activate your virtual environment
source sap_data_quality_env/bin/activate

# Install SAP dependencies
pip install pyrfc==2.8.0
pip install pyodata==2.1.0
pip install requests-oauthlib==1.3.1
pip install cryptography==41.0.7
```

### 2. SAP RFC Library Installation

#### On macOS:
```bash
# Install SAP NW RFC SDK
brew install sap-nw-rfc-sdk

# Or download from SAP Support Portal
# https://support.sap.com/en/product/connectors/nwrfcsdk.html
```

#### On Linux:
```bash
# Download and install SAP NW RFC SDK
wget https://support.sap.com/en/product/connectors/nwrfcsdk.html
tar -xzf nwrfcsdk.tar.gz
cd nwrfcsdk
./install.sh
```

### 3. Update Requirements Files

Update `sap-connector/requirements.txt`:

```txt
# SAP Connectivity
pyrfc==2.8.0
pyodata==2.1.0
requests-oauthlib==1.3.1

# Web Framework
flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0

# Data Processing
pandas==2.2.0
numpy==1.26.4

# Authentication & Security
cryptography==41.0.7
pyjwt==2.8.0

# Logging & Monitoring
structlog==24.1.0
prometheus-client==0.20.0

# Utilities
python-dotenv==1.0.1
click==8.1.7
requests==2.31.0
pydantic==2.4.2
```

## Environment Configuration

### 1. Create Production Environment File

```bash
# Copy the example environment file
cp env.example .env.production

# Edit with your SAP details
nano .env.production
```

### 2. Configure SAP Connection Parameters

```bash
# SAP System Details
SAP_HOST=your-sap-server.com
SAP_CLIENT=100
SAP_USERNAME=your_sap_user
SAP_PASSWORD=your_sap_password
SAP_GATEWAY_URL=https://your-sap-server.com:44300/sap/opu/odata/sap

# SAP RFC Configuration
SAP_RFC_DESTINATION=YOUR_RFC_DEST
SAP_ASHOST=your-sap-server.com
SAP_SYSNR=00
SAP_CLIENT=100

# Security Settings
SAP_USE_SSL=true
SAP_VERIFY_SSL=true
SAP_TIMEOUT=30

# OAuth Configuration (if using OAuth)
SAP_OAUTH_CLIENT_ID=your_oauth_client_id
SAP_OAUTH_CLIENT_SECRET=your_oauth_client_secret
SAP_OAUTH_TOKEN_URL=https://your-sap-server.com:44300/sap/opu/odata/sap/oauth2/token

# Certificate Configuration (if using certificates)
SAP_CERT_PATH=/path/to/your/certificate.pem
SAP_KEY_PATH=/path/to/your/private_key.pem
SAP_CERT_PASSWORD=your_cert_password
```

## SAP Connector Configuration

### 1. Replace Mock Connector with Real Connector

Update `sap-connector/app.py` to use real SAP connectivity:

```python
#!/usr/bin/env python3
"""
SAP S/4HANA Connector - Production Version
Real SAP S/4HANA connectivity for PM module data extraction
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

# SAP-specific imports
try:
    import pyrfc
    from pyodata.v2.service import GetEntitySetResponse
    from pyodata.v2.service import EntitySet
except ImportError:
    print("Warning: SAP dependencies not installed. Install pyrfc and pyodata for real SAP connectivity.")

class RealSAPConnector:
    """Real SAP S/4HANA connector for production"""
    
    def __init__(self):
        self.sap_host = os.getenv('SAP_HOST')
        self.sap_client = os.getenv('SAP_CLIENT', '100')
        self.sap_username = os.getenv('SAP_USERNAME')
        self.sap_password = os.getenv('SAP_PASSWORD')
        self.gateway_url = os.getenv('SAP_GATEWAY_URL')
        
        # Initialize connection
        self.session = self._create_session()
        self.rfc_connection = self._create_rfc_connection()
        
    def _create_session(self):
        """Create authenticated HTTP session"""
        session = requests.Session()
        
        # Add authentication
        if os.getenv('SAP_USE_OAUTH'):
            # OAuth 2.0 authentication
            token = self._get_oauth_token()
            session.headers.update({'Authorization': f'Bearer {token}'})
        else:
            # Basic authentication
            session.auth = (self.sap_username, self.sap_password)
            
        # Add SSL configuration
        if os.getenv('SAP_USE_SSL', 'true').lower() == 'true':
            session.verify = os.getenv('SAP_VERIFY_SSL', 'true').lower() == 'true'
            
        return session
    
    def _create_rfc_connection(self):
        """Create RFC connection for direct SAP calls"""
        try:
            conn = pyrfc.Connection(
                ashost=os.getenv('SAP_ASHOST'),
                sysnr=os.getenv('SAP_SYSNR'),
                client=os.getenv('SAP_CLIENT'),
                user=os.getenv('SAP_USERNAME'),
                passwd=os.getenv('SAP_PASSWORD')
            )
            return conn
        except Exception as e:
            logger.warning(f"RFC connection failed: {e}")
            return None
    
    def extract_equipment_data(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract real equipment master data from SAP"""
        try:
            # OData call to equipment service
            url = f"{self.gateway_url}/ZPM_EQUIPMENT_SRV/EquipmentSet"
            params = {'$top': limit, '$format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} equipment records")
            return data
            
        except Exception as e:
            logger.error(f"Equipment data extraction failed: {str(e)}")
            raise
    
    def extract_functional_locations(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract real functional location data from SAP"""
        try:
            # OData call to functional location service
            url = f"{self.gateway_url}/ZPM_FUNCTIONAL_LOCATION_SRV/FunctionalLocationSet"
            params = {'$top': limit, '$format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} functional location records")
            return data
            
        except Exception as e:
            logger.error(f"Functional location data extraction failed: {str(e)}")
            raise
    
    def extract_maintenance_orders(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract real maintenance order data from SAP"""
        try:
            # OData call to maintenance order service
            url = f"{self.gateway_url}/ZPM_MAINTENANCE_ORDER_SRV/MaintenanceOrderSet"
            params = {'$top': limit, '$format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} maintenance order records")
            return data
            
        except Exception as e:
            logger.error(f"Maintenance order data extraction failed: {str(e)}")
            raise
    
    def extract_notifications(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract real notification data from SAP"""
        try:
            # OData call to notification service
            url = f"{self.gateway_url}/ZPM_NOTIFICATION_SRV/NotificationSet"
            params = {'$top': limit, '$format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} notification records")
            return data
            
        except Exception as e:
            logger.error(f"Notification data extraction failed: {str(e)}")
            raise
    
    def extract_maintenance_plans(self, limit: int = 1000) -> Dict[str, Any]:
        """Extract real maintenance plan data from SAP"""
        try:
            # OData call to maintenance plan service
            url = f"{self.gateway_url}/ZPM_MAINTENANCE_PLAN_SRV/MaintenancePlanSet"
            params = {'$top': limit, '$format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Extracted {len(data.get('d', {}).get('results', []))} maintenance plan records")
            return data
            
        except Exception as e:
            logger.error(f"Maintenance plan data extraction failed: {str(e)}")
            raise

# Initialize real SAP connector
sap_connector = RealSAPConnector()

# Flask app setup (same as before)
app = Flask(__name__)
CORS(app)

# ... rest of the Flask routes remain the same
```

### 2. Update Docker Configuration

Update `sap-connector/Dockerfile` for production:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including SAP NW RFC SDK
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install SAP NW RFC SDK (you'll need to provide the SDK)
COPY sap-nw-rfc-sdk /opt/sap/nwrfcsdk
ENV SAPNWRFC_HOME=/opt/sap/nwrfcsdk
ENV LD_LIBRARY_PATH=/opt/sap/nwrfcsdk/lib:$LD_LIBRARY_PATH

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "app.py"]
```

## Testing Real SAP Connection

### 1. Test SAP Connectivity

Create a test script `test_sap_connection.py`:

```python
#!/usr/bin/env python3
"""
Test script for real SAP connectivity
"""

import os
import sys
sys.path.append('sap-connector')

from app import sap_connector

def test_sap_connection():
    """Test SAP connection and data extraction"""
    print("🔍 Testing SAP Connection...")
    
    try:
        # Test equipment extraction
        print("📊 Testing equipment data extraction...")
        equipment_data = sap_connector.extract_equipment_data(limit=5)
        print(f"✅ Equipment extraction successful: {len(equipment_data.get('d', {}).get('results', []))} records")
        
        # Test functional locations
        print("📊 Testing functional location extraction...")
        floc_data = sap_connector.extract_functional_locations(limit=5)
        print(f"✅ Functional location extraction successful: {len(floc_data.get('d', {}).get('results', []))} records")
        
        # Test maintenance orders
        print("📊 Testing maintenance order extraction...")
        order_data = sap_connector.extract_maintenance_orders(limit=5)
        print(f"✅ Maintenance order extraction successful: {len(order_data.get('d', {}).get('results', []))} records")
        
        print("🎉 All SAP connection tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ SAP connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_sap_connection()
    sys.exit(0 if success else 1)
```

### 2. Run Connection Test

```bash
# Test SAP connection
python test_sap_connection.py
```

## Production Deployment

### 1. Update Docker Compose for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  sap-connector:
    build:
      context: ./sap-connector
      dockerfile: Dockerfile.prod
    container_name: sap-connector-prod
    environment:
      - SAP_HOST=${SAP_HOST}
      - SAP_CLIENT=${SAP_CLIENT}
      - SAP_USERNAME=${SAP_USERNAME}
      - SAP_PASSWORD=${SAP_PASSWORD}
      - SAP_GATEWAY_URL=${SAP_GATEWAY_URL}
      - SAP_USE_SSL=${SAP_USE_SSL}
      - SAP_VERIFY_SSL=${SAP_VERIFY_SSL}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    ports:
      - "8000:8000"
    restart: unless-stopped
    networks:
      - sap-network

  # ... other services remain the same
```

### 2. Production Environment Setup

```bash
# Create production environment
cp .env.example .env.production

# Edit production environment
nano .env.production

# Deploy with production environment
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

## Troubleshooting

### Common Issues and Solutions

#### 1. SAP Connection Issues

**Problem**: Connection refused or authentication failed
```bash
# Check SAP system accessibility
telnet your-sap-server.com 44300

# Verify credentials
# Test with SAP GUI or RFC connection
```

**Solution**: Verify SAP system details and credentials

#### 2. OData Service Issues

**Problem**: 404 errors when accessing OData services
```abap
* In SAP Gateway Administration (transaction /IWFND/MAINT_SERVICE)
* Check if services are activated
* Verify service endpoints
```

**Solution**: Activate required OData services in SAP Gateway

#### 3. RFC Connection Issues

**Problem**: pyrfc connection fails
```bash
# Check SAP NW RFC SDK installation
python -c "import pyrfc; print('RFC SDK installed')"

# Verify RFC destination
# Test with SAP transaction SM59
```

**Solution**: Install SAP NW RFC SDK and configure RFC destination

#### 4. SSL Certificate Issues

**Problem**: SSL verification errors
```python
# For development, disable SSL verification
SAP_VERIFY_SSL=false

# For production, install proper certificates
SAP_CERT_PATH=/path/to/certificate.pem
```

**Solution**: Configure proper SSL certificates or disable verification for development

#### 5. Authorization Issues

**Problem**: 403 Forbidden errors
```abap
* Check user authorizations in SAP
* Transaction SU01 - User maintenance
* Transaction PFCG - Role maintenance
```

**Solution**: Grant appropriate authorizations to the SAP user

### Debug Commands

```bash
# Test SAP connectivity
curl -v -u username:password https://your-sap-server.com:44300/sap/opu/odata/sap/ZPM_EQUIPMENT_SRV/EquipmentSet

# Check Docker logs
docker-compose logs sap-connector

# Test RFC connection
python -c "import pyrfc; conn = pyrfc.Connection(ashost='host', sysnr='00', client='100', user='user', passwd='pass'); print('RFC OK')"
```

## Security Considerations

### 1. Credential Management

- Use environment variables for sensitive data
- Consider using Docker secrets for production
- Implement credential rotation

### 2. Network Security

- Use VPN for SAP connectivity
- Implement firewall rules
- Use SSL/TLS encryption

### 3. Access Control

- Implement least privilege principle
- Use dedicated SAP user for data extraction
- Regular access reviews

## Performance Optimization

### 1. Connection Pooling

```python
# Implement connection pooling for better performance
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

### 2. Batch Processing

```python
# Implement batch processing for large datasets
def extract_equipment_batch(batch_size=1000):
    """Extract equipment data in batches"""
    offset = 0
    while True:
        url = f"{self.gateway_url}/ZPM_EQUIPMENT_SRV/EquipmentSet"
        params = {
            '$top': batch_size,
            '$skip': offset,
            '$format': 'json'
        }
        # ... process batch
        offset += batch_size
```

## Monitoring and Alerting

### 1. SAP Connection Monitoring

```python
# Add SAP-specific metrics
SAP_CONNECTION_STATUS = Gauge('sap_connection_status', 'SAP connection status')
SAP_EXTRACTION_DURATION = Histogram('sap_extraction_duration_seconds', 'SAP data extraction duration')
SAP_ERROR_COUNT = Counter('sap_errors_total', 'SAP connection errors')
```

### 2. Alerting Rules

```yaml
# In monitoring/rules/alerts.yml
- alert: SAPConnectionDown
  expr: sap_connection_status == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "SAP connection is down"
    description: "Cannot connect to SAP S/4HANA system"

- alert: SAPExtractionErrors
  expr: rate(sap_errors_total[5m]) > 0.1
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High SAP extraction error rate"
    description: "SAP data extraction is failing frequently"
```

This guide provides a comprehensive approach to transitioning from the mock environment to real SAP S/4HANA integration. Follow the steps sequentially and test each component before proceeding to the next. 