#!/usr/bin/env python3
"""
Simple test script to run SAP connector locally
"""

import sys
import os

# Add the sap-connector directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sap-connector'))

if __name__ == '__main__':
    print("🚀 Starting SAP Connector Test Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 Available endpoints:")
    print("   - GET /health")
    print("   - GET /extract/equipment")
    print("   - GET /extract/functional-locations")
    print("   - GET /extract/maintenance-orders")
    print("   - GET /extract/all")
    print("   - GET /metrics")
    print("\n🔄 Starting server...")
    
    # Run the app directly
    os.system("cd sap-connector && python app-test.py") 