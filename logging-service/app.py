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
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_restful import Api, Resource
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
CORS(app)
api = Api(app)

# Prometheus metrics
REQUEST_COUNT = Counter('logging_service_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('logging_service_request_duration_seconds', 'Request latency')
REPORT_GENERATION_COUNT = Counter('logging_service_reports_generated_total', 'Reports generated', ['report_type'])

class DataQualityReporter:
    """Handles data quality reporting and analytics"""
    
    def __init__(self):
        self.db_host = os.getenv('DB_HOST', 'postgres')
        self.db_name = os.getenv('DB_NAME', 'sap_data_quality')
        self.db_user = os.getenv('DB_USER', 'sap_user')
        self.db_password = os.getenv('DB_PASSWORD', 'your_db_password')
        
        # Initialize database connection
        self.db_engine = create_engine(
            f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:5432/{self.db_name}"
        )
        
        # Ollama configuration for AI chat
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'phi3')
    
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
            
            with self.db_engine.connect() as conn:
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
            
            with self.db_engine.connect() as conn:
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
            
            with self.db_engine.connect() as conn:
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



# Web Routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    REQUEST_COUNT.labels(endpoint='dashboard').inc()
    return render_template('dashboard.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='health').inc()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'logging-service'
    })

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(endpoint='metrics').inc()
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/api/export/csv')
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
        
        with reporter.db_engine.connect() as conn:
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
        
        with reporter.db_engine.connect() as conn:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False) 