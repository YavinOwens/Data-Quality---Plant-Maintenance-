#!/usr/bin/env python3
"""
User models and authentication for SAP S/4HANA Data Quality Workflow
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(UserMixin, Base):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default='user')  # admin, user, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class AuditLog(Base):
    """Audit log for security events"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    username = Column(String(80))
    action = Column(String(100), nullable=False)
    resource = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    
    def __repr__(self):
        return f'<AuditLog {self.action} by {self.username}>'

class SecurityEvent(Base):
    """Security events for monitoring"""
    __tablename__ = 'security_events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)  # login_failed, rate_limit, suspicious_activity
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    source_ip = Column(String(45))
    user_id = Column(Integer)
    username = Column(String(80))
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    
    def __repr__(self):
        return f'<SecurityEvent {self.event_type} - {self.severity}>'

# Default users for development
DEFAULT_USERS = [
    {
        'username': 'admin',
        'email': 'admin@company.com',
        'password': 'Admin@123!',
        'role': 'admin'
    },
    {
        'username': 'user',
        'email': 'user@company.com',
        'password': 'User@123!',
        'role': 'user'
    },
    {
        'username': 'viewer',
        'email': 'viewer@company.com',
        'password': 'Viewer@123!',
        'role': 'viewer'
    }
] 