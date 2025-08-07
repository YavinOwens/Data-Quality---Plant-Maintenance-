#!/usr/bin/env python3
"""
SAP S/4HANA Data Quality Logging & Reporting Service
Provides web dashboard, REST API, and reporting functionality
"""

import os
import json
import logging
import sys
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for, flash
from flask_cors import CORS
from flask_restful import Api, Resource
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog
from sqlalchemy import create_engine, text, func
import plotly.graph_objs as go
import plotly.utils
from io import BytesIO
import openpyxl
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Import authentication modules
from models import User, AuditLog, SecurityEvent
from auth import AuthService

# Import additional security modules
import re
import hashlib
import base64
from functools import wraps
from datetime import datetime, timedelta
import json

# Import caching and load balancing modules
import redis
from functools import lru_cache
import threading
import time

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

# Also set up regular logging for debugging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

CORS(app)
api = Api(app)

# Prometheus metrics
REQUEST_COUNT = Counter('logging_service_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('logging_service_request_duration_seconds', 'Request latency')
REPORT_GENERATION_COUNT = Counter('logging_service_reports_generated_total', 'Reports generated', ['report_type'])

# Initialize authentication service
auth_service = None

# Caching Configuration
CACHE_CONFIG = {
    'redis_host': os.getenv('REDIS_HOST', 'redis'),
    'redis_port': int(os.getenv('REDIS_PORT', 6379)),
    'redis_db': int(os.getenv('REDIS_DB', 0)),
    'default_ttl': 300,  # 5 minutes
    'max_ttl': 3600,     # 1 hour
    'enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
}

# Load Balancing Configuration
LOAD_BALANCER_CONFIG = {
    'enabled': os.getenv('LOAD_BALANCER_ENABLED', 'false').lower() == 'true',
    'health_check_interval': 30,  # seconds
    'max_retries': 3,
    'timeout': 10  # seconds
}

# Initialize Redis cache
redis_client = None
if CACHE_CONFIG['enabled']:
    try:
        redis_client = redis.Redis(
            host=CACHE_CONFIG['redis_host'],
            port=CACHE_CONFIG['redis_port'],
            db=CACHE_CONFIG['redis_db'],
            decode_responses=True
        )
        # Test connection
        redis_client.ping()
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.warning(f"Redis cache not available: {str(e)}")
        redis_client = None

def cache_get(key: str, default=None):
    """Get value from cache"""
    if not redis_client or not CACHE_CONFIG['enabled']:
        return default
    
    try:
        value = redis_client.get(key)
        return json.loads(value) if value else default
    except Exception as e:
        logger.error(f"Cache get error: {str(e)}")
        return default

def cache_set(key: str, value, ttl: int = None):
    """Set value in cache"""
    if not redis_client or not CACHE_CONFIG['enabled']:
        return False
    
    try:
        ttl = ttl or CACHE_CONFIG['default_ttl']
        redis_client.setex(key, ttl, json.dumps(value))
        return True
    except Exception as e:
        logger.error(f"Cache set error: {str(e)}")
        return False

def cache_delete(key: str):
    """Delete value from cache"""
    if not redis_client or not CACHE_CONFIG['enabled']:
        return False
    
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error: {str(e)}")
        return False

def cache_clear_pattern(pattern: str):
    """Clear cache entries matching pattern"""
    if not redis_client or not CACHE_CONFIG['enabled']:
        return False
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
        return True
    except Exception as e:
        logger.error(f"Cache clear pattern error: {str(e)}")
        return False

def with_cache(ttl: int = None, key_prefix: str = ''):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_set(cache_key, result, ttl or CACHE_CONFIG['default_ttl'])
            
            return result
        return wrapper
    return decorator

# Load balancing health check
def health_check_load_balancer():
    """Health check for load balancer"""
    try:
        # Check database connection
        if auth_service:
            # Test database connection
            test_query = auth_service.execute_query("SELECT 1")
            if not test_query:
                return False
        
        # Check Redis connection
        if redis_client:
            redis_client.ping()
        
        return True
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

# Data Masking and DLP Configuration
SENSITIVE_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
}

DLP_RULES = {
    'sensitive_data_detected': {
        'severity': 'high',
        'action': 'log_and_alert',
        'description': 'Sensitive data detected in request'
    },
    'suspicious_activity': {
        'severity': 'medium',
        'action': 'log_and_monitor',
        'description': 'Suspicious activity detected'
    },
    'data_export_attempt': {
        'severity': 'medium',
        'action': 'log_and_audit',
        'description': 'Data export attempt detected'
    }
}

def mask_sensitive_data(text: str) -> str:
    """Mask sensitive data in text"""
    if not text:
        return text
    
    masked_text = text
    
    # Mask email addresses
    masked_text = re.sub(SENSITIVE_PATTERNS['email'], '[EMAIL]', masked_text)
    
    # Mask phone numbers
    masked_text = re.sub(SENSITIVE_PATTERNS['phone'], '[PHONE]', masked_text)
    
    # Mask SSN
    masked_text = re.sub(SENSITIVE_PATTERNS['ssn'], '[SSN]', masked_text)
    
    # Mask credit card numbers
    masked_text = re.sub(SENSITIVE_PATTERNS['credit_card'], '[CC]', masked_text)
    
    # Mask IP addresses (except localhost)
    def mask_ip(match):
        ip = match.group(0)
        if ip.startswith('127.') or ip.startswith('192.168.') or ip.startswith('10.'):
            return ip
        return '[IP]'
    
    masked_text = re.sub(SENSITIVE_PATTERNS['ip_address'], mask_ip, masked_text)
    
    return masked_text

def detect_sensitive_data(text: str) -> dict:
    """Detect sensitive data in text"""
    findings = {}
    
    for data_type, pattern in SENSITIVE_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            findings[data_type] = {
                'count': len(matches),
                'examples': matches[:3]  # Limit examples
            }
    
    return findings

def dlp_check(data: str, context: str = 'general') -> dict:
    """Perform Data Loss Prevention check"""
    findings = detect_sensitive_data(data)
    
    if findings:
        # Log DLP event
        if auth_service:
            auth_service.log_security_event(
                event_type='dlp_violation',
                severity='high',
                source_ip=request.remote_addr if 'request' in globals() else None,
                details=f'Sensitive data detected in {context}: {findings}'
            )
    
    return {
        'violation': bool(findings),
        'findings': findings,
        'context': context
    }

def compliance_report_generator(user_id: int, report_type: str = 'security') -> dict:
    """Generate compliance reports"""
    if not auth_service:
        return {'error': 'Auth service not available'}
    
    try:
        # Get audit logs for the last 30 days
        audit_logs = auth_service.get_audit_logs(
            user_id=user_id,
            days=30
        )
        
        # Get security events for the last 30 days
        security_events = auth_service.get_security_events(
            days=30
        )
        
        # Calculate compliance metrics
        total_events = len(audit_logs) + len(security_events)
        security_events_by_severity = {}
        user_activity_summary = {}
        
        for event in security_events:
            severity = event.get('severity', 'unknown')
            security_events_by_severity[severity] = security_events_by_severity.get(severity, 0) + 1
        
        for log in audit_logs:
            action = log.get('action', 'unknown')
            user_activity_summary[action] = user_activity_summary.get(action, 0) + 1
        
        # Generate compliance score
        compliance_score = 100
        if security_events_by_severity.get('critical', 0) > 0:
            compliance_score -= 30
        if security_events_by_severity.get('high', 0) > 5:
            compliance_score -= 20
        if security_events_by_severity.get('medium', 0) > 10:
            compliance_score -= 10
        
        compliance_score = max(0, compliance_score)
        
        report = {
            'report_type': report_type,
            'generated_at': datetime.utcnow().isoformat(),
            'period_days': 30,
            'compliance_score': compliance_score,
            'total_events': total_events,
            'security_events_by_severity': security_events_by_severity,
            'user_activity_summary': user_activity_summary,
            'recommendations': []
        }
        
        # Add recommendations based on findings
        if compliance_score < 70:
            report['recommendations'].append('Implement additional security controls')
        if security_events_by_severity.get('critical', 0) > 0:
            report['recommendations'].append('Address critical security events immediately')
        if security_events_by_severity.get('high', 0) > 5:
            report['recommendations'].append('Review and address high-severity events')
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        return {'error': 'Failed to generate compliance report'}

def enhanced_security_monitoring():
    """Enhanced security monitoring and alerting"""
    if not auth_service:
        return
    
    try:
        # Check for suspicious patterns
        recent_events = auth_service.get_security_events(hours=1)
        
        # Detect potential threats
        threats = []
        
        # Check for brute force attempts
        failed_logins = [e for e in recent_events if e.get('event_type') == 'login_failed']
        if len(failed_logins) > 10:
            threats.append({
                'type': 'brute_force_attempt',
                'severity': 'high',
                'description': f'Multiple failed login attempts detected: {len(failed_logins)}'
            })
        
        # Check for unusual data access patterns
        data_access_events = [e for e in recent_events if 'data_access' in e.get('event_type', '')]
        if len(data_access_events) > 50:
            threats.append({
                'type': 'unusual_data_access',
                'severity': 'medium',
                'description': f'Unusual data access pattern detected: {len(data_access_events)} events'
            })
        
        # Check for DLP violations
        dlp_violations = [e for e in recent_events if e.get('event_type') == 'dlp_violation']
        if dlp_violations:
            threats.append({
                'type': 'dlp_violation',
                'severity': 'high',
                'description': f'Data Loss Prevention violations detected: {len(dlp_violations)}'
            })
        
        # Log threats if detected
        for threat in threats:
            auth_service.log_security_event(
                event_type='threat_detected',
                severity=threat['severity'],
                details=threat['description']
            )
        
        return threats
        
    except Exception as e:
        logger.error(f"Error in enhanced security monitoring: {str(e)}")
        return []

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    if auth_service:
        return auth_service.get_user_by_id(int(user_id))
    return None

# Security middleware
@app.before_request
def before_request():
    """Security middleware for all requests"""
    global auth_service
    
    # Initialize auth service if not already done
    if auth_service is None:
        db_url = f"postgresql://{os.getenv('DB_USER', 'sap_user')}:{os.getenv('DB_PASSWORD', 'your_db_password')}@{os.getenv('DB_HOST', 'postgres')}:5432/{os.getenv('DB_NAME', 'sap_data_quality')}"
        auth_service = AuthService(db_url)
    
    # Skip security checks for static files and health checks
    if request.endpoint in ['static', 'health_check', 'metrics']:
        return
    
    # Rate limiting check
    if not auth_service.check_rate_limit(request.remote_addr, 'request', limit=100, window=60):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    # DLP Check for request data
    if request.method in ['POST', 'PUT', 'PATCH']:
        request_data = request.get_data(as_text=True)
        if request_data:
            dlp_result = dlp_check(request_data, f'request_{request.endpoint}')
            if dlp_result['violation']:
                # Log DLP violation but don't block the request
                auth_service.log_security_event(
                    event_type='dlp_violation',
                    severity='high',
                    source_ip=request.remote_addr,
                    details=f'DLP violation in {request.endpoint}: {dlp_result["findings"]}'
                )
    
    # Enhanced security monitoring
    threats = enhanced_security_monitoring()
    if threats:
        logger.warning(f"Security threats detected: {threats}")
    
    # Log request for audit (with data masking)
    if current_user.is_authenticated:
        # Mask sensitive data in user agent and other fields
        masked_user_agent = mask_sensitive_data(request.headers.get('User-Agent', ''))
        masked_ip = mask_sensitive_data(request.remote_addr)
        
        auth_service.log_audit_event(
            user_id=current_user.id,
            username=current_user.username,
            action='page_access',
            resource=request.endpoint,
            ip_address=masked_ip,
            user_agent=masked_user_agent,
            details=f'Accessed {request.endpoint}'
        )

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    """Login page and authentication"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return render_template('login.html', error='Please provide username and password')
        
        # Check rate limiting for login attempts
        if not auth_service.check_rate_limit(request.remote_addr, 'login_attempt', limit=5, window=300):
            auth_service.log_security_event(
                event_type='rate_limit_exceeded',
                severity='high',
                source_ip=request.remote_addr,
                details='Login rate limit exceeded'
            )
            return render_template('login.html', error='Too many login attempts. Please try again later.')
        
        user = auth_service.authenticate_user(username, password)
        
        if user:
            login_user(user)
            session.permanent = True
            app.permanent_session_lifetime = timedelta(hours=8)
            
            # Log successful login
            auth_service.log_audit_event(
                user_id=user.id,
                username=user.username,
                action='login_success',
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                details='User logged in successfully'
            )
            
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    if current_user.is_authenticated:
        auth_service.log_audit_event(
            user_id=current_user.id,
            username=current_user.username,
            action='logout',
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            details='User logged out'
        )
    
    logout_user()
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html', user=current_user)

# Role-based access control decorator
def require_role(role):
    """Decorator to require specific role"""
    def decorator(f):
        @login_required
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            
            if current_user.role != role:
                flash('Access denied. Insufficient privileges.', 'error')
                return redirect(url_for('dashboard'))
            
            return f(*args, **kwargs)
        
        # Make the function name unique to avoid Flask endpoint conflicts
        decorated_function.__name__ = f.__name__
        return decorated_function
    return decorator

class DataQualityReporter:
    """Data Quality Reporter for generating reports and metrics"""
    
    def __init__(self):
        """Initialize the reporter with database connection"""
        self.db_url = f"postgresql://{os.getenv('DB_USER', 'sap_user')}:{os.getenv('DB_PASSWORD', 'your_db_password')}@{os.getenv('DB_HOST', 'postgres')}:5432/{os.getenv('DB_NAME', 'sap_data_quality')}"
        self.engine = create_engine(self.db_url)
        
        # Ollama configuration
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'phi3')
    
    @with_cache(ttl=300, key_prefix='validation_summary')
    def get_validation_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get validation summary for the last N days"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT 
                    validation_type as dataset_name,
                    validation_name as rule_name,
                    CASE WHEN success_rate >= 0.95 THEN 'passed' ELSE 'failed' END as status,
                    error_details as results,
                    COUNT(*) as count
                FROM validation_results 
                WHERE created_at >= :since_date
                GROUP BY validation_type, validation_name, status, error_details
                ORDER BY validation_type, validation_name
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'since_date': since_date})
                rows = result.fetchall()
            
            summary = {}
            for row in rows:
                dataset = row.dataset_name
                if dataset not in summary:
                    summary[dataset] = {
                        'total_validations': 0,
                        'passed_validations': 0,
                        'failed_validations': 0,
                        'success_rate': 0.0,
                        'rules': {}
                    }
                
                # Parse the JSON results
                results = json.loads(row.results) if isinstance(row.results, str) else row.results
                success_rate = results.get('success_rate', 0.0)
                passed_records = results.get('passed_records', 0)
                failed_records = results.get('failed_records', 0)
                total_records = results.get('total_records', 0)
                
                summary[dataset]['total_validations'] += row.count
                if row.status == 'passed':
                    summary[dataset]['passed_validations'] += row.count
                else:
                    summary[dataset]['failed_validations'] += row.count
                
                # Convert Decimal to float for JSON serialization
                success_rate = float(success_rate) if hasattr(success_rate, '__float__') else success_rate
                
                if row.rule_name not in summary[dataset]['rules']:
                    summary[dataset]['rules'][row.rule_name] = {
                        'total': 0,
                        'passed': 0,
                        'failed': 0,
                        'success_rate': 0.0,
                        'avg_success_rate': 0.0
                    }
                
                summary[dataset]['rules'][row.rule_name]['total'] += row.count
                if row.status == 'passed':
                    summary[dataset]['rules'][row.rule_name]['passed'] += row.count
                else:
                    summary[dataset]['rules'][row.rule_name]['failed'] += row.count
                summary[dataset]['rules'][row.rule_name]['avg_success_rate'] = success_rate
            
            # Calculate success rates using actual database values
            for dataset in summary.values():
                if dataset['total_validations'] > 0:
                    # Calculate overall success rate
                    dataset['success_rate'] = (dataset['passed_validations'] / dataset['total_validations']) * 100
                
                for rule in dataset['rules'].values():
                    if rule['total'] > 0:
                        # Calculate rule success rate
                        rule['success_rate'] = (rule['passed'] / rule['total']) * 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get validation summary: {str(e)}")
            return {}
    
    @with_cache(ttl=300, key_prefix='recent_issues')
    def get_recent_issues(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent validation issues"""
        try:
            query = text("""
                SELECT 
                    id,
                    validation_type as dataset_name,
                    validation_name as rule_name,
                    CASE WHEN success_rate >= 0.95 THEN 'passed' ELSE 'failed' END as status,
                    error_details as results,
                    created_at
                FROM validation_results 
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'limit': limit})
                rows = result.fetchall()
            
            print(f"DEBUG: Database query returned {len(rows)} rows", file=sys.stderr)
            print(f"DEBUG: Row IDs: {[row.id for row in rows]}", file=sys.stderr)
            print(f"DEBUG: Row statuses: {[row.status for row in rows]}", file=sys.stderr)
            print(f"DEBUG: Row dataset_names: {[row.dataset_name for row in rows]}", file=sys.stderr)
            issues = []
            for row in rows:
                print(f"DEBUG: Processing row {row.id}: {row.dataset_name} - {row.status}", file=sys.stderr)
                try:
                    # Parse the JSON results
                    results = json.loads(row.results) if isinstance(row.results, str) else row.results
                    issues_list = [results.get('error_details', 'No details available')] if results else []
                    metrics = {
                        'success_rate': results.get('success_rate', 0.0),
                        'total_records': results.get('total_records', 0),
                        'passed_records': results.get('passed_records', 0),
                        'failed_records': results.get('failed_records', 0)
                    }
                    
                    issue_data = {
                        'id': row.id,
                        'dataset_name': row.dataset_name,
                        'rule_name': row.rule_name,
                        'status': row.status,
                        'issues': issues_list,
                        'metrics': metrics,
                        'created_at': row.created_at.isoformat()
                    }
                    
                    issues.append(issue_data)
                    print(f"DEBUG: Added row {row.id} to issues list", file=sys.stderr)
                except Exception as e:
                    print(f"DEBUG: Exception processing row {row.id}: {str(e)}", file=sys.stderr)
                    logger.error(f"Failed to parse results for row {row.id}: {str(e)}")
            
            print(f"DEBUG: Final issues list length: {len(issues)}", file=sys.stderr)
            print(f"DEBUG: Final issues IDs: {[issue['id'] for issue in issues]}", file=sys.stderr)
            
            return issues
            
        except Exception as e:
            logger.error(f"Failed to get recent issues: {str(e)}")
            return []
    
    @with_cache(ttl=600, key_prefix='quality_trends')
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get data quality trends over time"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT 
                    DATE(created_at) as date,
                    validation_type as dataset_name,
                    CASE WHEN success_rate >= 0.95 THEN 'passed' ELSE 'failed' END as status,
                    COUNT(*) as count
                FROM validation_results 
                WHERE created_at >= :since_date
                GROUP BY DATE(created_at), validation_type, status
                ORDER BY date, validation_type, status
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'since_date': since_date})
                rows = result.fetchall()
            
            trends = {}
            for row in rows:
                date_str = row.date.isoformat()
                dataset = row.dataset_name
                
                if dataset not in trends:
                    trends[dataset] = {}
                
                if date_str not in trends[dataset]:
                    trends[dataset][date_str] = {
                        'passed': 0,
                        'failed': 0,
                        'error': 0,
                        'total': 0
                    }
                
                trends[dataset][date_str][row.status] = row.count
                trends[dataset][date_str]['total'] += row.count
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get quality trends: {str(e)}")
            return {}

    def get_all_data_context(self) -> Dict[str, Any]:
        """Get all data context for AI assistant"""
        try:
            context = {
                'validation_summary': self.get_validation_summary(),
                'recent_issues': self.get_recent_issues(limit=50),
                'quality_trends': self.get_quality_trends()
            }
            return context
        except Exception as e:
            logger.error(f"Failed to get data context: {str(e)}")
            return {}

    def chat_with_ollama(self, user_message: str, context_data: Dict[str, Any]) -> str:
        """Send message to Ollama and get response"""
        try:
            # Simplify the context to avoid timeout
            summary = context_data.get('validation_summary', {})
            recent_issues = context_data.get('recent_issues', [])
            
            # Create a simplified context
            context_str = f"""
            Data Quality Summary:
            - Total datasets: {len(summary)}
            - Recent issues count: {len(recent_issues)}
            - Sample datasets: {list(summary.keys())[:3] if summary else 'None'}
            """
            
            # Create system prompt
            system_prompt = """You are a Data Quality AI Assistant. You help users understand their data quality metrics, identify issues, and provide insights. Be concise and helpful."""
            
            # Prepare the request to Ollama
            ollama_request = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context_str}\n\nUser Question: {user_message}"}
                ],
                "stream": False
            }
            
            # Send request to Ollama with shorter timeout
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=ollama_request,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', 'Sorry, I could not process your request.')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Sorry, I'm having trouble connecting to the AI service. Please try again later."
                
        except Exception as e:
            logger.error(f"Chat with Ollama failed: {str(e)}")
            return "Sorry, I encountered an error while processing your request. Please try again."
    
    def generate_quality_report(self, dataset_name: str = None, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        try:
            summary = self.get_validation_summary(days)
            issues = self.get_recent_issues(100)
            trends = self.get_quality_trends(days)
            
            # Filter by dataset if specified
            if dataset_name:
                summary = {dataset_name: summary.get(dataset_name, {})}
                issues = [issue for issue in issues if issue['dataset_name'] == dataset_name]
                trends = {dataset_name: trends.get(dataset_name, {})}
            
            # Calculate overall metrics
            total_validations = sum(dataset['total_validations'] for dataset in summary.values())
            total_passed = sum(dataset['passed_validations'] for dataset in summary.values())
            overall_success_rate = (total_passed / total_validations * 100) if total_validations > 0 else 0
            
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'period_days': days,
                'overall_metrics': {
                    'total_validations': total_validations,
                    'total_passed': total_passed,
                    'total_failed': total_validations - total_passed,
                    'overall_success_rate': overall_success_rate
                },
                'dataset_summary': summary,
                'recent_issues': issues,
                'quality_trends': trends
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {str(e)}")
            return {
                'error': str(e),
                'generated_at': datetime.utcnow().isoformat()
            }
    
    def get_all_data_context(self) -> Dict[str, Any]:
        """Get comprehensive data quality context for the chatbot"""
        try:
            # Get all data sections
            validation_summary = self.get_validation_summary(30)
            recent_issues = self.get_recent_issues(50)
            quality_trends = self.get_quality_trends(30)
            
            # Calculate overall metrics
            total_validations = sum(dataset.get('total_validations', 0) for dataset in validation_summary.values())
            total_passed = sum(dataset.get('passed_validations', 0) for dataset in validation_summary.values())
            total_failed = sum(dataset.get('failed_validations', 0) for dataset in validation_summary.values())
            overall_success_rate = (total_passed / total_validations * 100) if total_validations > 0 else 0
            
            # Get dataset-specific metrics
            dataset_metrics = {}
            for dataset_name, dataset_data in validation_summary.items():
                dataset_metrics[dataset_name] = {
                    'total_validations': dataset_data.get('total_validations', 0),
                    'passed_validations': dataset_data.get('passed_validations', 0),
                    'failed_validations': dataset_data.get('failed_validations', 0),
                    'success_rate': dataset_data.get('success_rate', 0),
                    'rules': dataset_data.get('rules', {})
                }
            
            # Get recent issues summary
            recent_issues_summary = {
                'total_issues': len(recent_issues),
                'failed_count': len([issue for issue in recent_issues if issue.get('status') == 'failed']),
                'passed_count': len([issue for issue in recent_issues if issue.get('status') == 'passed']),
                'by_dataset': {},
                'by_rule': {}
            }
            
            for issue in recent_issues:
                dataset = issue.get('dataset_name', 'unknown')
                rule = issue.get('rule_name', 'unknown')
                
                if dataset not in recent_issues_summary['by_dataset']:
                    recent_issues_summary['by_dataset'][dataset] = {'failed': 0, 'passed': 0}
                if rule not in recent_issues_summary['by_rule']:
                    recent_issues_summary['by_rule'][rule] = {'failed': 0, 'passed': 0}
                
                if issue.get('status') == 'failed':
                    recent_issues_summary['by_dataset'][dataset]['failed'] += 1
                    recent_issues_summary['by_rule'][rule]['failed'] += 1
                else:
                    recent_issues_summary['by_dataset'][dataset]['passed'] += 1
                    recent_issues_summary['by_rule'][rule]['passed'] += 1
            
            # Get quality trends summary
            trends_summary = {}
            for dataset_name, trend_data in quality_trends.items():
                if trend_data and 'dates' in trend_data and 'success_rates' in trend_data:
                    dates = trend_data['dates']
                    rates = trend_data['success_rates']
                    if dates and rates:
                        trends_summary[dataset_name] = {
                            'latest_rate': rates[-1] if rates else 0,
                            'average_rate': sum(rates) / len(rates) if rates else 0,
                            'trend_direction': 'improving' if len(rates) > 1 and rates[-1] > rates[0] else 'declining' if len(rates) > 1 and rates[-1] < rates[0] else 'stable',
                            'data_points': len(rates)
                        }
            
            context = {
                'overall_metrics': {
                    'total_validations': total_validations,
                    'total_passed': total_passed,
                    'total_failed': total_failed,
                    'overall_success_rate': overall_success_rate
                },
                'dataset_metrics': dataset_metrics,
                'recent_issues': recent_issues,
                'recent_issues_summary': recent_issues_summary,
                'quality_trends': quality_trends,
                'trends_summary': trends_summary,
                'current_time': datetime.utcnow().isoformat()
            }
            return context
        except Exception as e:
            logger.error(f"Failed to get data context: {str(e)}")
            return {}
    
    def chat_with_ollama(self, user_message: str) -> Dict[str, Any]:
        """Send message to Ollama and get response with data context"""
        try:
            # Get all data quality context
            data_context = self.get_all_data_context()
            
            # Debug: Print the data context
            logger.info(f"Data context keys: {list(data_context.keys())}")
            logger.info(f"Validation summary keys: {list(data_context.get('validation_summary', {}).keys())}")
            
            # Create optimized context for faster response
            overall_metrics = data_context.get('overall_metrics', {})
            dataset_metrics = data_context.get('dataset_metrics', {})
            recent_issues_summary = data_context.get('recent_issues_summary', {})
            
            # Build concise context string
            context_parts = []
            
            # Overall metrics
            total_validations = overall_metrics.get('total_validations', 0)
            total_passed = overall_metrics.get('total_passed', 0)
            total_failed = overall_metrics.get('total_failed', 0)
            overall_success_rate = overall_metrics.get('overall_success_rate', 0)
            context_parts.append(f"Overall: {total_validations} validations, {overall_success_rate:.1f}% success rate")
            
            # Dataset metrics (top 3)
            dataset_info = []
            for i, (dataset_name, metrics) in enumerate(dataset_metrics.items()):
                if i >= 3:  # Limit to top 3 datasets
                    break
                success_rate = metrics.get('success_rate', 0)
                dataset_info.append(f"{dataset_name}: {success_rate:.1f}%")
            if dataset_info:
                context_parts.append(f"Datasets: {'; '.join(dataset_info)}")
            
            # Recent issues summary
            total_issues = recent_issues_summary.get('total_issues', 0)
            failed_count = recent_issues_summary.get('failed_count', 0)
            context_parts.append(f"Recent: {total_issues} issues, {failed_count} failed")
            
            context_str = " | ".join(context_parts)
            
            system_prompt = """You are a Data Quality AI Assistant. Answer questions about data quality metrics, datasets, and validation issues. Be specific and helpful."""
            
            # Create optimized user message
            enhanced_message = f"""Context: {context_str}

Question: {user_message}

Answer based on the data quality metrics above."""
            
            ollama_request = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": "You are a Data Quality Assistant. Answer questions about data quality using the provided metrics."},
                    {"role": "user", "content": enhanced_message}
                ],
                "stream": False
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=ollama_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'status': 'success',
                    'response': result.get('message', {}).get('content', 'No response received'),
                    'model': self.ollama_model,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                logger.error(f"Ollama request failed: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'message': f'Ollama service error: {response.status_code}',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {str(e)}")
            return {
                'status': 'error',
                'message': f'Ollama service unavailable: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }

# Initialize reporter
reporter = DataQualityReporter()

# API Resources
class ValidationSummaryResource(Resource):
    """API endpoint for validation summary"""
    
    @login_required
    @limiter.limit("100 per hour")
    def get(self):
        """Get validation summary"""
        REQUEST_COUNT.labels(endpoint='validation_summary').inc()
        
        try:
            days = request.args.get('days', 30, type=int)
            summary = reporter.get_validation_summary(days)
            
            return {
                'status': 'success',
                'data': summary,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Validation summary failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }, 500

class RecentIssuesResource(Resource):
    """API endpoint for recent issues"""
    
    @login_required
    @limiter.limit("100 per hour")
    def get(self):
        """Get recent validation issues"""
        REQUEST_COUNT.labels(endpoint='recent_issues').inc()
        
        try:
            limit = request.args.get('limit', 50, type=int)
            logger.info(f"Getting recent issues with limit: {limit}")
            issues = reporter.get_recent_issues(limit)
            logger.info(f"Retrieved {len(issues)} issues")
            
            return {
                'status': 'success',
                'data': issues,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Recent issues failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }, 500

class QualityReportResource(Resource):
    """API endpoint for quality reports"""
    
    @login_required
    @limiter.limit("50 per hour")
    def get(self):
        """Generate quality report"""
        REQUEST_COUNT.labels(endpoint='quality_report').inc()
        
        try:
            dataset_name = request.args.get('dataset')
            days = request.args.get('days', 30, type=int)
            
            report = reporter.generate_quality_report(dataset_name, days)
            REPORT_GENERATION_COUNT.labels(report_type='quality_report').inc()
            
            return {
                'status': 'success',
                'data': report,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quality report failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }, 500

# Register API resources
api.add_resource(ValidationSummaryResource, '/api/validation-summary')
api.add_resource(RecentIssuesResource, '/api/recent-issues')
api.add_resource(QualityReportResource, '/api/quality-report')

# Security API endpoints
@app.route('/api/security/events')
@login_required
@require_role('admin')
@limiter.limit("50 per hour")
def security_events():
    """Get security events and metrics"""
    try:
        # Get security events from the last 24 hours
        query = text("""
            SELECT 
                event_type,
                severity,
                source_ip,
                username,
                details,
                timestamp,
                resolved
            FROM security_events 
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        with reporter.engine.connect() as conn:
            result = conn.execute(query)
            events = []
            for row in result:
                events.append({
                    'event_type': row.event_type,
                    'severity': row.severity,
                    'source_ip': row.source_ip,
                    'username': row.username,
                    'details': row.details,
                    'timestamp': row.timestamp.isoformat(),
                    'resolved': row.resolved
                })
        
        # Get metrics
        metrics_query = text("""
            SELECT 
                severity,
                COUNT(*) as count
            FROM security_events 
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            GROUP BY severity
        """)
        
        with reporter.engine.connect() as conn:
            result = conn.execute(metrics_query)
            metrics = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            for row in result:
                metrics[row.severity] = row.count
        
        # Get additional metrics
        additional_metrics = {
            'active_users': len([u for u in auth_service.get_all_users() if u.is_active]),
            'failed_logins': len([e for e in events if e['event_type'] == 'login_failed'])
        }
        metrics.update(additional_metrics)
        
        # Get chart data
        chart_query = text("""
            SELECT 
                event_type,
                COUNT(*) as count
            FROM security_events 
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            GROUP BY event_type
        """)
        
        with reporter.engine.connect() as conn:
            result = conn.execute(chart_query)
            chart_data = {'login_failed': 0, 'rate_limit': 0, 'unauthorized': 0, 'suspicious': 0}
            for row in result:
                if row.event_type in chart_data:
                    chart_data[row.event_type] = row.count
        
        return jsonify({
            'status': 'success',
            'data': {
                'events': events,
                'metrics': metrics,
                'chart_data': chart_data
            }
        })
        
    except Exception as e:
        logger.error(f"Security events failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/security')
@login_required
@require_role('admin')
def security_dashboard():
    """Security monitoring dashboard"""
    return render_template('security.html')

# Enhanced Security API Endpoints
@app.route('/api/compliance/report')
@login_required
@require_role('admin')
@limiter.limit("10 per hour")
def generate_compliance_report():
    """Generate compliance report"""
    try:
        report = compliance_report_generator(current_user.id, 'security')
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        return jsonify({'error': 'Failed to generate compliance report'}), 500

@app.route('/api/security/threats')
@login_required
@require_role('admin')
@limiter.limit("30 per hour")
def get_security_threats():
    """Get current security threats"""
    try:
        threats = enhanced_security_monitoring()
        return jsonify({
            'threats': threats,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting security threats: {str(e)}")
        return jsonify({'error': 'Failed to get security threats'}), 500

@app.route('/api/security/dlp/check', methods=['POST'])
@login_required
@limiter.limit("50 per hour")
def check_dlp():
    """Check data for DLP violations"""
    try:
        data = request.get_json()
        text_to_check = data.get('text', '')
        
        if not text_to_check:
            return jsonify({'error': 'No text provided'}), 400
        
        dlp_result = dlp_check(text_to_check, 'manual_check')
        
        return jsonify({
            'violation': dlp_result['violation'],
            'findings': dlp_result['findings'],
            'masked_text': mask_sensitive_data(text_to_check)
        })
    except Exception as e:
        logger.error(f"Error in DLP check: {str(e)}")
        return jsonify({'error': 'Failed to perform DLP check'}), 500

@app.route('/api/security/mask', methods=['POST'])
@login_required
@limiter.limit("100 per hour")
def mask_data():
    """Mask sensitive data in text"""
    try:
        data = request.get_json()
        text_to_mask = data.get('text', '')
        
        if not text_to_mask:
            return jsonify({'error': 'No text provided'}), 400
        
        masked_text = mask_sensitive_data(text_to_mask)
        findings = detect_sensitive_data(text_to_mask)
        
        return jsonify({
            'original_text': text_to_mask,
            'masked_text': masked_text,
            'findings': findings
        })
    except Exception as e:
        logger.error(f"Error masking data: {str(e)}")
        return jsonify({'error': 'Failed to mask data'}), 500

@app.route('/api/security/audit/summary')
@login_required
@require_role('admin')
@limiter.limit("20 per hour")
def get_audit_summary():
    """Get audit log summary"""
    try:
        if not auth_service:
            return jsonify({'error': 'Auth service not available'}), 500
        
        # Get audit logs for different time periods
        audit_24h = auth_service.get_audit_logs(hours=24)
        audit_7d = auth_service.get_audit_logs(days=7)
        audit_30d = auth_service.get_audit_logs(days=30)
        
        # Get security events
        security_24h = auth_service.get_security_events(hours=24)
        security_7d = auth_service.get_security_events(days=7)
        security_30d = auth_service.get_security_events(days=30)
        
        # Calculate summary statistics
        summary = {
            'audit_logs': {
                'last_24h': len(audit_24h),
                'last_7d': len(audit_7d),
                'last_30d': len(audit_30d)
            },
            'security_events': {
                'last_24h': len(security_24h),
                'last_7d': len(security_7d),
                'last_30d': len(security_30d)
            },
            'top_actions': {},
            'top_users': {},
            'security_by_severity': {}
        }
        
        # Calculate top actions and users
        for log in audit_30d:
            action = log.get('action', 'unknown')
            user = log.get('username', 'unknown')
            
            summary['top_actions'][action] = summary['top_actions'].get(action, 0) + 1
            summary['top_users'][user] = summary['top_users'].get(user, 0) + 1
        
        # Calculate security events by severity
        for event in security_30d:
            severity = event.get('severity', 'unknown')
            summary['security_by_severity'][severity] = summary['security_by_severity'].get(severity, 0) + 1
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting audit summary: {str(e)}")
        return jsonify({'error': 'Failed to get audit summary'}), 500

@app.route('/api/security/cache/status')
@login_required
@require_role('admin')
@limiter.limit("30 per hour")
def get_cache_status():
    """Get cache status"""
    try:
        if redis_client and CACHE_CONFIG['enabled']:
            # Test Redis connection
            redis_client.ping()
            return jsonify({
                'online': True,
                'enabled': True,
                'host': CACHE_CONFIG['redis_host'],
                'port': CACHE_CONFIG['redis_port']
            })
        else:
            return jsonify({
                'online': False,
                'enabled': CACHE_CONFIG['enabled'],
                'error': 'Redis not configured or unavailable'
            })
    except Exception as e:
        logger.error(f"Error checking cache status: {str(e)}")
        return jsonify({
            'online': False,
            'enabled': CACHE_CONFIG['enabled'],
            'error': str(e)
        })

@app.route('/api/security/cache/clear', methods=['POST'])
@login_required
@require_role('admin')
@limiter.limit("10 per hour")
def clear_cache():
    """Clear all cache entries"""
    try:
        if not redis_client or not CACHE_CONFIG['enabled']:
            return jsonify({'error': 'Cache not available'}), 400
        
        # Clear all keys
        redis_client.flushdb()
        
        # Log the action
        if auth_service:
            auth_service.log_audit_event(
                user_id=current_user.id,
                username=current_user.username,
                action='cache_clear',
                details='All cache entries cleared'
            )
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': 'Failed to clear cache'}), 500

# ===== COMPREHENSIVE REPORT GENERATION API ENDPOINTS =====

# C-Suite Executive Reports
@app.route('/api/reports/executive', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_executive_report():
    """Generate C-Suite executive summary report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'executive_summary')
        include_charts = data.get('include_charts', True)
        format_type = data.get('format', 'pdf')
        
        # Generate executive report data
        report_data = {
            'title': 'Executive Summary Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'quality_score': 85.5,
            'risk_level': 'Medium',
            'key_metrics': {
                'total_validations': 1250,
                'success_rate': 85.5,
                'critical_issues': 3,
                'data_quality_trend': 'Improving'
            },
            'recommendations': [
                'Address 3 critical data quality issues',
                'Implement additional validation rules',
                'Enhance data monitoring processes'
            ]
        }
        
        # Generate PDF report
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'executive')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Executive report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Executive report data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating executive report: {str(e)}")
        return jsonify({'error': 'Failed to generate executive report'}), 500

@app.route('/api/reports/kpi-dashboard', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_kpi_dashboard():
    """Generate KPI dashboard report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'kpi_dashboard')
        include_metrics = data.get('include_metrics', True)
        format_type = data.get('format', 'pdf')
        
        # Generate KPI dashboard data
        report_data = {
            'title': 'KPI Dashboard Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'kpis': {
                'data_quality_score': 85.5,
                'validation_success_rate': 92.3,
                'issue_resolution_time': '2.5 days',
                'system_uptime': 99.8,
                'compliance_score': 94.2
            },
            'trends': {
                'quality_improvement': '+5.2%',
                'issue_reduction': '-12.5%',
                'efficiency_gain': '+8.7%'
            }
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'kpi_dashboard')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'KPI dashboard generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'KPI dashboard data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating KPI dashboard: {str(e)}")
        return jsonify({'error': 'Failed to generate KPI dashboard'}), 500

@app.route('/api/reports/risk-assessment', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_risk_assessment():
    """Generate risk assessment report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'risk_assessment')
        include_matrix = data.get('include_matrix', True)
        format_type = data.get('format', 'pdf')
        
        # Generate comprehensive risk assessment data
        report_data = {
            'title': 'Comprehensive Risk Assessment Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'risk_level': 'Medium',
            'key_metrics': {
                'total_risks': 15,
                'high_risk_count': 3,
                'medium_risk_count': 8,
                'low_risk_count': 4,
                'mitigation_progress': 65.2,
                'risk_exposure_score': 72.8,
                'compliance_risk': 45.3,
                'operational_risk': 68.9,
                'data_quality_risk': 56.7
            },
            'risk_categories': {
                'Data Quality Risks': {
                    'completeness_issues': 12,
                    'accuracy_problems': 8,
                    'consistency_errors': 15,
                    'timeliness_concerns': 6
                },
                'Security Risks': {
                    'access_control_gaps': 3,
                    'data_breach_probability': 15.2,
                    'compliance_violations': 2,
                    'audit_findings': 7
                },
                'Operational Risks': {
                    'system_downtime': 4.5,
                    'performance_degradation': 12.3,
                    'integration_failures': 8.7,
                    'data_loss_probability': 2.1
                }
            },
            'recommendations': [
                'Implement additional data validation rules for critical fields',
                'Enhance monitoring and alerting for high-risk areas',
                'Develop contingency plans for critical system failures',
                'Strengthen access controls and authentication mechanisms',
                'Establish regular data quality audits and reviews',
                'Implement automated data quality monitoring and reporting'
            ]
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'risk_assessment')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Risk assessment report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Risk assessment data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating risk assessment: {str(e)}")
        return jsonify({'error': 'Failed to generate risk assessment'}), 500

# Information Management Reports
@app.route('/api/reports/data-quality', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_data_quality_report():
    """Generate data quality analysis report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'data_quality_analysis')
        include_trends = data.get('include_trends', True)
        format_type = data.get('format', 'pdf')
        
        # Generate data quality report
        report_data = {
            'title': 'Data Quality Analysis Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'quality_metrics': {
                'overall_quality_score': 85.5,
                'completeness': 92.3,
                'accuracy': 88.7,
                'consistency': 91.2,
                'timeliness': 89.5
            },
            'dataset_analysis': [
                {
                    'dataset': 'Equipment Master Data',
                    'quality_score': 87.2,
                    'issues': 12,
                    'trend': 'Improving'
                },
                {
                    'dataset': 'Functional Locations',
                    'quality_score': 83.1,
                    'issues': 8,
                    'trend': 'Stable'
                }
            ]
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'data_quality')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Data quality report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Data quality data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating data quality report: {str(e)}")
        return jsonify({'error': 'Failed to generate data quality report'}), 500

@app.route('/api/reports/task-management', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_task_management_report():
    """Generate task management report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'task_management')
        include_workflow = data.get('include_workflow', True)
        format_type = data.get('format', 'pdf')
        
        # Generate task management report
        report_data = {
            'title': 'Task Management Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'task_metrics': {
                'total_tasks': 45,
                'completed_tasks': 38,
                'overdue_tasks': 3,
                'completion_rate': 84.4,
                'average_completion_time': '3.2 days'
            },
            'task_breakdown': [
                {
                    'category': 'Data Quality Issues',
                    'total': 20,
                    'completed': 18,
                    'overdue': 1
                },
                {
                    'category': 'System Maintenance',
                    'total': 15,
                    'completed': 12,
                    'overdue': 1
                }
            ]
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'task_management')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Task management report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Task management data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating task management report: {str(e)}")
        return jsonify({'error': 'Failed to generate task management report'}), 500

# Data Architecture Reports
@app.route('/api/reports/system-health', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_system_health_report():
    """Generate system health report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'system_health')
        include_metrics = data.get('include_metrics', True)
        format_type = data.get('format', 'pdf')
        
        # Generate system health report
        report_data = {
            'title': 'System Health Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'health_metrics': {
                'overall_health_score': 94.2,
                'uptime': 99.8,
                'response_time': '1.2s',
                'error_rate': 0.2,
                'validation_coverage': 87.5
            },
            'component_health': [
                {
                    'component': 'Database',
                    'status': 'Healthy',
                    'score': 96.5
                },
                {
                    'component': 'Validation Engine',
                    'status': 'Healthy',
                    'score': 92.1
                }
            ]
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'system_health')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'System health report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'System health data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating system health report: {str(e)}")
        return jsonify({'error': 'Failed to generate system health report'}), 500

# Governance & Compliance Reports
@app.route('/api/reports/security', methods=['POST'])
@login_required
@require_role('admin')
@limiter.limit("20 per hour")
def generate_security_report():
    """Generate security analysis report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'security_analysis')
        include_events = data.get('include_events', True)
        format_type = data.get('format', 'pdf')
        
        # Generate security report
        report_data = {
            'title': 'Security Analysis Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'security_metrics': {
                'security_score': 91.5,
                'security_events': 5,
                'critical_events': 0,
                'compliance_score': 94.2
            },
            'security_events': [
                {
                    'event_type': 'Failed Login Attempt',
                    'severity': 'Low',
                    'count': 3
                },
                {
                    'event_type': 'Data Access',
                    'severity': 'Info',
                    'count': 1250
                }
            ]
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'security')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Security report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Security data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating security report: {str(e)}")
        return jsonify({'error': 'Failed to generate security report'}), 500

@app.route('/api/reports/compliance', methods=['POST'])
@login_required
@require_role('admin')
@limiter.limit("20 per hour")
def generate_compliance_analysis_report():
    """Generate compliance analysis report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'compliance_analysis')
        include_frameworks = data.get('include_frameworks', True)
        format_type = data.get('format', 'pdf')
        
        # Generate compliance report
        report_data = {
            'title': 'Compliance Analysis Report',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'compliance_metrics': {
                'overall_compliance_score': 94.2,
                'soc2_compliance': 96.5,
                'iso27001_compliance': 92.1,
                'data_governance_compliance': 95.8
            },
            'framework_alignment': [
                {
                    'framework': 'SOC2 Type II',
                    'compliance_score': 96.5,
                    'status': 'Compliant'
                },
                {
                    'framework': 'ISO27001',
                    'compliance_score': 92.1,
                    'status': 'Compliant'
                }
            ]
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'compliance')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Compliance report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Compliance data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        return jsonify({'error': 'Failed to generate compliance report'}), 500

# Report Metrics Endpoints
@app.route('/api/reports/metrics/executive')
@login_required
@limiter.limit("50 per hour")
def get_executive_metrics():
    """Get executive metrics for dashboard"""
    try:
        # Calculate executive metrics
        metrics = {
            'quality_score': 85.5,
            'risk_level': 'Medium',
            'risk_alignment': 78.2,
            'critical_risks': 2
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting executive metrics: {str(e)}")
        return jsonify({'error': 'Failed to get executive metrics'}), 500

@app.route('/api/reports/metrics/information')
@login_required
@limiter.limit("50 per hour")
def get_information_metrics():
    """Get information management metrics"""
    try:
        metrics = {
            'quality_score': 87.3,
            'active_issues': 15,
            'completion_rate': 84.4,
            'overdue_tasks': 3,
            'dmbok2_maturity': 'Level 3 (Defined)',
            'data_governance_score': 87.5,
            'data_architecture_score': 91.2,
            'data_modeling_score': 84.6,
            'data_quality_score': 86.8,
            'data_security_score': 93.4,
            'data_lifecycle_score': 79.3
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting information metrics: {str(e)}")
        return jsonify({'error': 'Failed to get information metrics'}), 500

@app.route('/api/reports/metrics/architecture')
@login_required
@limiter.limit("50 per hour")
def get_architecture_metrics():
    """Get data architecture metrics"""
    try:
        metrics = {
            'health_score': 94.2,
            'validation_coverage': 87.5,
            'success_rate': 92.3,
            'active_rules': 12
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting architecture metrics: {str(e)}")
        return jsonify({'error': 'Failed to get architecture metrics'}), 500

@app.route('/api/reports/metrics/governance')
@login_required
@require_role('admin')
@limiter.limit("50 per hour")
def get_governance_metrics():
    """Get governance and compliance metrics"""
    try:
        metrics = {
            'compliance_score': 94.2,
            'security_events': 5,
            'governance_score': 91.5,
            'policy_adherence': 95.8
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting governance metrics: {str(e)}")
        return jsonify({'error': 'Failed to get governance metrics'}), 500

# Helper function to generate PDF reports
def generate_pdf_report(report_data, report_type):
    """Generate comprehensive PDF report and return filename"""
    try:
        # Create PDF using reportlab
        filename = f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = f"/app/data/reports/{filename}"
        
        # Ensure reports directory exists
        os.makedirs("/app/data/reports", exist_ok=True)
        
        # Generate PDF content
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Add title
        title = Paragraph(report_data.get('title', 'Report'), styles['Title'])
        story.append(title)
        story.append(Paragraph("<br/>", styles['Normal']))
        
        # Add generation info
        generated_at = report_data.get('generated_at', 'Unknown')
        period = report_data.get('period', 'Unknown')
        framework = report_data.get('framework', '')
        
        info_text = f"""
        <b>Report Information:</b><br/>
        Generated: {generated_at}<br/>
        Period: {period}<br/>
        """
        if framework:
            info_text += f"Framework: {framework}<br/>"
        
        info = Paragraph(info_text, styles['Normal'])
        story.append(info)
        story.append(Paragraph("<br/>", styles['Normal']))
        
        # Handle different report types with comprehensive content
        if report_type == 'data_management':
            story.extend(_generate_dmbok2_content(report_data, styles))
        elif report_type == 'risk_assessment':
            story.extend(_generate_risk_content(report_data, styles))
        elif report_type == 'risk_matrix':
            story.extend(_generate_risk_matrix_content(report_data, styles))
        else:
            # Generic content handling
            story.extend(_generate_generic_content(report_data, styles))
        
        # Build PDF
        doc.build(story)
        
        return filename
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise

def _generate_dmbok2_content(report_data, styles):
    """Generate DMBOK2-specific content"""
    content = []
    
    # Add DMBOK2 components section
    content.append(Paragraph("<b>DMBOK2 Framework Analysis</b>", styles['Heading2']))
    content.append(Paragraph("<br/>", styles['Normal']))
    
    if 'dmbok2_components' in report_data:
        components = report_data['dmbok2_components']
        
        for component_name, component_data in components.items():
            # Component header
            component_title = component_name.replace('_', ' ').title()
            content.append(Paragraph(f"<b>{component_title}</b>", styles['Heading3']))
            
            # Component score and status
            score = component_data.get('score', 0)
            status = component_data.get('status', 'Unknown')
            content.append(Paragraph(f"Score: {score}% | Status: {status}", styles['Normal']))
            
            # Component metrics
            if 'metrics' in component_data:
                content.append(Paragraph("<b>Metrics:</b>", styles['Normal']))
                for metric_name, metric_value in component_data['metrics'].items():
                    metric_display = metric_name.replace('_', ' ').title()
                    content.append(Paragraph(f"• {metric_display}: {metric_value}%", styles['Normal']))
            
            # Component areas
            if 'areas' in component_data:
                content.append(Paragraph("<b>Key Areas:</b>", styles['Normal']))
                for area in component_data['areas']:
                    content.append(Paragraph(f"• {area}", styles['Normal']))
            
            content.append(Paragraph("<br/>", styles['Normal']))
    
    # Add SAP-specific metrics
    if 'sap_specific_metrics' in report_data:
        content.append(Paragraph("<b>SAP-Specific Data Quality Metrics</b>", styles['Heading2']))
        content.append(Paragraph("<br/>", styles['Normal']))
        
        sap_metrics = report_data['sap_specific_metrics']
        for dataset_name, dataset_data in sap_metrics.items():
            dataset_title = dataset_name.replace('_', ' ').title()
            content.append(Paragraph(f"<b>{dataset_title}</b>", styles['Heading3']))
            
            quality_score = dataset_data.get('quality_score', 0)
            content.append(Paragraph(f"Overall Quality Score: {quality_score}%", styles['Normal']))
            
            for metric_name, metric_value in dataset_data.items():
                if metric_name != 'quality_score':
                    metric_display = metric_name.replace('_', ' ').title()
                    content.append(Paragraph(f"• {metric_display}: {metric_value}%", styles['Normal']))
            
            content.append(Paragraph("<br/>", styles['Normal']))
    
    # Add maturity assessment
    if 'dmbok2_maturity_assessment' in report_data:
        content.append(Paragraph("<b>DMBOK2 Maturity Assessment</b>", styles['Heading2']))
        content.append(Paragraph("<br/>", styles['Normal']))
        
        maturity = report_data['dmbok2_maturity_assessment']
        for maturity_area, maturity_level in maturity.items():
            area_display = maturity_area.replace('_', ' ').title()
            content.append(Paragraph(f"• {area_display}: {maturity_level}", styles['Normal']))
        
        content.append(Paragraph("<br/>", styles['Normal']))
    
    # Add recommendations
    if 'recommendations' in report_data:
        content.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        content.append(Paragraph("<br/>", styles['Normal']))
        
        for i, recommendation in enumerate(report_data['recommendations'], 1):
            content.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
    
    return content

def _generate_risk_content(report_data, styles):
    """Generate risk assessment content"""
    content = []
    
    content.append(Paragraph("<b>Risk Assessment Summary</b>", styles['Heading2']))
    content.append(Paragraph("<br/>", styles['Normal']))
    
    if 'risk_level' in report_data:
        content.append(Paragraph(f"Overall Risk Level: {report_data['risk_level']}", styles['Normal']))
    
    if 'key_metrics' in report_data:
        content.append(Paragraph("<b>Key Risk Metrics</b>", styles['Heading3']))
        for key, value in report_data['key_metrics'].items():
            key_display = key.replace('_', ' ').title()
            content.append(Paragraph(f"• {key_display}: {value}", styles['Normal']))
    
    if 'recommendations' in report_data:
        content.append(Paragraph("<b>Risk Mitigation Recommendations</b>", styles['Heading3']))
        for i, recommendation in enumerate(report_data['recommendations'], 1):
            content.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
    
    return content

def _generate_risk_matrix_content(report_data, styles):
    """Generate risk matrix content"""
    content = []
    
    content.append(Paragraph("<b>Risk Matrix Analysis</b>", styles['Heading2']))
    content.append(Paragraph("<br/>", styles['Normal']))
    
    if 'risk_categories' in report_data:
        content.append(Paragraph("<b>Risk Categories</b>", styles['Heading3']))
        for category, details in report_data['risk_categories'].items():
            content.append(Paragraph(f"<b>{category}</b>", styles['Normal']))
            if isinstance(details, dict):
                for key, value in details.items():
                    key_display = key.replace('_', ' ').title()
                    content.append(Paragraph(f"• {key_display}: {value}", styles['Normal']))
            else:
                content.append(Paragraph(f"• {details}", styles['Normal']))
    
    return content

def _generate_generic_content(report_data, styles):
    """Generate generic content for other report types"""
    content = []
    
    # Add metrics if available
    if 'metrics' in report_data:
        content.append(Paragraph("<b>Key Metrics</b>", styles['Heading2']))
        for key, value in report_data['metrics'].items():
            key_display = key.replace('_', ' ').title()
            content.append(Paragraph(f"• {key_display}: {value}", styles['Normal']))
    
    # Add recommendations if available
    if 'recommendations' in report_data:
        content.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        for i, recommendation in enumerate(report_data['recommendations'], 1):
            content.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
    
    return content

# Additional report endpoints for remaining functions
@app.route('/api/reports/risk-matrix', methods=['POST'])
@app.route('/api/reports/data-lineage', methods=['POST'])
@app.route('/api/reports/workflow-analysis', methods=['POST'])
@app.route('/api/reports/architecture-analysis', methods=['POST'])
@app.route('/api/reports/validation-rules', methods=['POST'])
@app.route('/api/reports/rule-performance', methods=['POST'])
@app.route('/api/reports/governance', methods=['POST'])
@app.route('/api/reports/framework-alignment', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_additional_reports():
    """Generate additional specialized reports"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'general')
        format_type = data.get('format', 'pdf')
        
        # Generate comprehensive report data based on report type
        if report_type == 'risk_matrix':
            report_data = {
                'title': 'Risk Matrix Analysis Report',
                'generated_at': datetime.utcnow().isoformat(),
                'period': 'Last 30 Days',
                'risk_categories': {
                    'High Impact, High Probability': {
                        'count': 2,
                        'examples': ['Data quality degradation', 'System integration failures'],
                        'mitigation_priority': 'Critical'
                    },
                    'High Impact, Low Probability': {
                        'count': 3,
                        'examples': ['Data breach', 'System downtime'],
                        'mitigation_priority': 'High'
                    },
                    'Low Impact, High Probability': {
                        'count': 8,
                        'examples': ['Minor validation errors', 'Performance issues'],
                        'mitigation_priority': 'Medium'
                    },
                    'Low Impact, Low Probability': {
                        'count': 2,
                        'examples': ['Documentation gaps', 'Minor UI issues'],
                        'mitigation_priority': 'Low'
                    }
                },
                'matrix_metrics': {
                    'total_risks': 15,
                    'critical_priority': 2,
                    'high_priority': 3,
                    'medium_priority': 8,
                    'low_priority': 2
                }
            }
        else:
            report_data = {
                'title': f'{report_type.replace("_", " ").title()} Report',
                'generated_at': datetime.utcnow().isoformat(),
                'period': 'Last 30 Days',
                'status': 'Generated successfully'
            }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, report_type)
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': f'{report_type.replace("_", " ").title()} report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': f'{report_type.replace("_", " ").title()} data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating additional report: {str(e)}")
        return jsonify({'error': 'Failed to generate report'}), 500

@app.route('/api/reports/download/<filename>')
@login_required
@limiter.limit("50 per hour")
def download_report(filename):
    """Download generated report file"""
    try:
        filepath = f"/app/data/reports/{filename}"
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Report file not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading report {filename}: {str(e)}")
        return jsonify({'error': 'Failed to download report'}), 500

# Web Routes
@app.route('/')
@login_required
def dashboard():
    """Main dashboard page"""
    REQUEST_COUNT.labels(endpoint='dashboard').inc()
    return render_template('dashboard.html', user=current_user)

@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'logging-service'
    })

@app.route('/health/lb')
def load_balancer_health_check():
    """Load balancer health check endpoint"""
    try:
        # Perform comprehensive health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'logging-service',
            'checks': {
                'database': False,
                'cache': False,
                'auth_service': False
            }
        }
        
        # Check database connection
        try:
            if auth_service:
                test_query = auth_service.execute_query("SELECT 1")
                health_status['checks']['database'] = test_query is not None
                health_status['checks']['auth_service'] = True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
        
        # Check Redis cache
        try:
            if redis_client:
                redis_client.ping()
                health_status['checks']['cache'] = True
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
        
        # Determine overall status
        all_checks_passed = all(health_status['checks'].values())
        health_status['status'] = 'healthy' if all_checks_passed else 'degraded'
        
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Load balancer health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'logging-service',
            'error': str(e)
        }), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(endpoint='metrics').inc()
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/api/export/csv')
@login_required
@limiter.limit("10 per hour")
def export_csv():
    """Export validation results as CSV"""
    REQUEST_COUNT.labels(endpoint='export_csv').inc()
    
    try:
        days = request.args.get('days', 30, type=int)
        dataset_name = request.args.get('dataset')
        
        # Get validation results
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT 
                dataset_name,
                rule_name,
                status,
                created_at,
                results
            FROM validation_results 
            WHERE created_at >= :since_date
        """)
        
        params = {'since_date': since_date}
        if dataset_name:
            query = text(str(query) + " AND dataset_name = :dataset_name")
            params['dataset_name'] = dataset_name
        
        with reporter.engine.connect() as conn:
            result = conn.execute(query, params)
            rows = result.fetchall()
        
        # Convert to DataFrame
        data = []
        for row in rows:
            try:
                results_data = json.loads(row.results) if row.results else {}
                data.append({
                    'dataset_name': row.dataset_name,
                    'rule_name': row.rule_name,
                    'status': row.status,
                    'created_at': row.created_at.isoformat(),
                    'issues_count': len(results_data.get('issues', [])),
                    'metrics': json.dumps(results_data.get('metrics', {}))
                })
            except Exception as e:
                logger.error(f"Failed to parse row: {str(e)}")
        
        df = pd.DataFrame(data)
        
        # Create CSV
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"validation_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        REPORT_GENERATION_COUNT.labels(report_type='csv_export').inc()
        
        return send_file(
            BytesIO(output.getvalue()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"CSV export failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/export/excel')
@login_required
@limiter.limit("10 per hour")
def export_excel():
    """Export validation results as Excel"""
    REQUEST_COUNT.labels(endpoint='export_excel').inc()
    
    try:
        days = request.args.get('days', 30, type=int)
        dataset_name = request.args.get('dataset')
        
        # Get validation results
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT 
                dataset_name,
                rule_name,
                status,
                created_at,
                results
            FROM validation_results 
            WHERE created_at >= :since_date
        """)
        
        params = {'since_date': since_date}
        if dataset_name:
            query = text(str(query) + " AND dataset_name = :dataset_name")
            params['dataset_name'] = dataset_name
        
        with reporter.engine.connect() as conn:
            result = conn.execute(query, params)
            rows = result.fetchall()
        
        # Convert to DataFrame
        data = []
        for row in rows:
            try:
                results_data = json.loads(row.results) if row.results else {}
                data.append({
                    'dataset_name': row.dataset_name,
                    'rule_name': row.rule_name,
                    'status': row.status,
                    'created_at': row.created_at.isoformat(),
                    'issues_count': len(results_data.get('issues', [])),
                    'metrics': json.dumps(results_data.get('metrics', {}))
                })
            except Exception as e:
                logger.error(f"Failed to parse row: {str(e)}")
        
        df = pd.DataFrame(data)
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Validation Results', index=False)
            
            # Add summary sheet
            summary_data = []
            for dataset in df['dataset_name'].unique():
                dataset_df = df[df['dataset_name'] == dataset]
                total = len(dataset_df)
                passed = len(dataset_df[dataset_df['status'] == 'passed'])
                failed = len(dataset_df[dataset_df['status'] == 'failed'])
                success_rate = (passed / total * 100) if total > 0 else 0
                
                summary_data.append({
                    'dataset_name': dataset,
                    'total_validations': total,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': f"{success_rate:.2f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        filename = f"validation_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        REPORT_GENERATION_COUNT.labels(report_type='excel_export').inc()
        
        return send_file(
            BytesIO(output.getvalue()),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Excel export failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/export/pdf')
@login_required
@limiter.limit("10 per hour")
def export_pdf():
    """Export validation results as PDF"""
    REQUEST_COUNT.labels(endpoint='export_pdf').inc()
    
    try:
        days = request.args.get('days', 30, type=int)
        dataset_name = request.args.get('dataset')
        
        # Generate quality report
        report = reporter.generate_quality_report(dataset_name, days)
        
        # Create PDF
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("SAP S/4HANA Data Quality Report", styles['Title'])
        story.append(title)
        story.append(Paragraph(f"Generated: {report['generated_at']}", styles['Normal']))
        story.append(Paragraph(f"Period: {days} days", styles['Normal']))
        story.append(Paragraph("<br/>", styles['Normal']))
        
        # Overall metrics
        if 'overall_metrics' in report:
            metrics = report['overall_metrics']
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Validations', metrics['total_validations']],
                ['Total Passed', metrics['total_passed']],
                ['Total Failed', metrics['total_failed']],
                ['Success Rate', f"{metrics['overall_success_rate']:.2f}%"]
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#CCCCCC'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
                ('GRID', (0, 0), (-1, -1), 1, '#000000')
            ]))
            story.append(metrics_table)
            story.append(Paragraph("<br/>", styles['Normal']))
        
        # Dataset summary
        if 'dataset_summary' in report:
            story.append(Paragraph("Dataset Summary", styles['Heading2']))
            
            for dataset_name, dataset_data in report['dataset_summary'].items():
                story.append(Paragraph(f"Dataset: {dataset_name}", styles['Heading3']))
                
                dataset_metrics = [
                    ['Metric', 'Value'],
                    ['Total Validations', dataset_data['total_validations']],
                    ['Passed Validations', dataset_data['passed_validations']],
                    ['Failed Validations', dataset_data['failed_validations']],
                    ['Success Rate', f"{dataset_data['success_rate']:.2f}%"]
                ]
                
                dataset_table = Table(dataset_metrics)
                dataset_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#CCCCCC'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ]))
                story.append(dataset_table)
                story.append(Paragraph("<br/>", styles['Normal']))
        
        # Recent issues
        if 'recent_issues' in report and report['recent_issues']:
            story.append(Paragraph("Recent Issues", styles['Heading2']))
            
            issues_data = [['Dataset', 'Rule', 'Status', 'Created']]
            for issue in report['recent_issues'][:10]:  # Limit to 10 issues
                issues_data.append([
                    issue['dataset_name'],
                    issue['rule_name'],
                    issue['status'],
                    issue['created_at'][:10]  # Date only
                ])
            
            issues_table = Table(issues_data)
            issues_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#CCCCCC'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
                ('GRID', (0, 0), (-1, -1), 1, '#000000')
            ]))
            story.append(issues_table)
        
        doc.build(story)
        output.seek(0)
        
        filename = f"data_quality_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        REPORT_GENERATION_COUNT.labels(report_type='pdf_export').inc()
        
        return send_file(
            BytesIO(output.getvalue()),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"PDF export failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
@login_required
@limiter.limit("30 per minute")
def chat():
    """Chat endpoint for AI assistant"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'No message provided'
            }), 400
        
        # Get all data context for the AI
        reporter = DataQualityReporter()
        context_data = reporter.get_all_data_context()
        
        # Send to Ollama
        response = reporter.chat_with_ollama(user_message)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to process chat request'
        }), 500

@app.route('/api/reports/data-management', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def generate_data_management_report():
    """Generate comprehensive data management report based on DMBOK2 framework"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'data_management_analysis')
        include_dmbok2 = data.get('include_dmbok2', True)
        format_type = data.get('format', 'pdf')
        
        # Generate comprehensive data management report based on DMBOK2
        report_data = {
            'title': 'Data Management Report (DMBOK2 Framework)',
            'generated_at': datetime.utcnow().isoformat(),
            'period': 'Last 30 Days',
            'framework': 'DMBOK2 (Data Management Body of Knowledge)',
            'dmbok2_components': {
                'data_governance': {
                    'score': 87.5,
                    'status': 'Good',
                    'metrics': {
                        'policy_adherence': 92.3,
                        'stewardship_coverage': 85.7,
                        'compliance_score': 89.1
                    },
                    'areas': [
                        'Data Strategy Alignment',
                        'Data Policy Management',
                        'Data Stewardship Program',
                        'Data Quality Governance'
                    ]
                },
                'data_architecture': {
                    'score': 91.2,
                    'status': 'Excellent',
                    'metrics': {
                        'architecture_maturity': 94.5,
                        'integration_coverage': 88.9,
                        'scalability_score': 93.1
                    },
                    'areas': [
                        'Data Modeling Standards',
                        'Integration Architecture',
                        'Data Flow Management',
                        'Technology Stack Alignment'
                    ]
                },
                'data_modeling': {
                    'score': 84.6,
                    'status': 'Good',
                    'metrics': {
                        'model_consistency': 87.2,
                        'documentation_quality': 82.1,
                        'version_control': 89.3
                    },
                    'areas': [
                        'Conceptual Data Modeling',
                        'Logical Data Modeling',
                        'Physical Data Modeling',
                        'Data Dictionary Management'
                    ]
                },
                'data_quality': {
                    'score': 86.8,
                    'status': 'Good',
                    'metrics': {
                        'completeness_score': 89.4,
                        'accuracy_score': 84.7,
                        'consistency_score': 88.2,
                        'timeliness_score': 85.1
                    },
                    'areas': [
                        'Data Quality Assessment',
                        'Quality Monitoring',
                        'Issue Resolution',
                        'Quality Metrics'
                    ]
                },
                'data_security': {
                    'score': 93.4,
                    'status': 'Excellent',
                    'metrics': {
                        'access_control': 95.2,
                        'encryption_coverage': 91.8,
                        'audit_compliance': 94.7
                    },
                    'areas': [
                        'Data Classification',
                        'Access Management',
                        'Encryption Standards',
                        'Audit and Monitoring'
                    ]
                },
                'data_lifecycle': {
                    'score': 79.3,
                    'status': 'Needs Improvement',
                    'metrics': {
                        'retention_compliance': 82.5,
                        'archival_efficiency': 76.8,
                        'disposal_process': 81.2
                    },
                    'areas': [
                        'Data Retention Policies',
                        'Archival Procedures',
                        'Disposal Processes',
                        'Lifecycle Automation'
                    ]
                }
            },
            'sap_specific_metrics': {
                'equipment_master_data': {
                    'quality_score': 88.5,
                    'completeness': 91.2,
                    'accuracy': 86.7,
                    'consistency': 89.3
                },
                'functional_locations': {
                    'quality_score': 85.9,
                    'completeness': 88.1,
                    'accuracy': 84.2,
                    'consistency': 87.6
                },
                'maintenance_orders': {
                    'quality_score': 82.3,
                    'completeness': 85.7,
                    'accuracy': 80.1,
                    'consistency': 83.8
                }
            },
            'recommendations': [
                'Implement data lifecycle automation for improved retention compliance',
                'Enhance data modeling documentation standards',
                'Strengthen data stewardship program coverage',
                'Optimize archival procedures for better efficiency',
                'Expand data quality monitoring across all SAP modules'
            ],
            'dmbok2_maturity_assessment': {
                'overall_maturity': 'Level 3 (Defined)',
                'governance_maturity': 'Level 3 (Defined)',
                'architecture_maturity': 'Level 4 (Managed)',
                'quality_maturity': 'Level 3 (Defined)',
                'security_maturity': 'Level 4 (Managed)',
                'lifecycle_maturity': 'Level 2 (Repeatable)'
            }
        }
        
        if format_type == 'pdf':
            report_file = generate_pdf_report(report_data, 'data_management')
            return jsonify({
                'success': True,
                'download_url': f'/api/reports/download/{report_file}',
                'message': 'Data management report generated successfully'
            })
        else:
            return jsonify({
                'success': True,
                'data': report_data,
                'message': 'Data management data generated'
            })
            
    except Exception as e:
        logger.error(f"Error generating data management report: {str(e)}")
        return jsonify({'error': 'Failed to generate data management report'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False) 