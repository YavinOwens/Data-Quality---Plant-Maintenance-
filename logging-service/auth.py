#!/usr/bin/env python3
"""
Authentication and security services for SAP S/4HANA Data Quality Workflow
"""

import os
import logging
from datetime import datetime, timedelta
from flask import request, session, current_app
from flask_login import login_user, logout_user, current_user, login_required
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import User, AuditLog, SecurityEvent, Base, DEFAULT_USERS
from werkzeug.security import generate_password_hash
import structlog

logger = structlog.get_logger()

class AuthService:
    """Authentication service for user management and security"""
    
    def __init__(self, db_url):
        """Initialize authentication service"""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.setup_default_users()
    
    def setup_default_users(self):
        """Create default users if they don't exist"""
        try:
            db_session = self.Session()
            
            for user_data in DEFAULT_USERS:
                existing_user = db_session.query(User).filter_by(username=user_data['username']).first()
                if not existing_user:
                    user = User(
                        username=user_data['username'],
                        email=user_data['email'],
                        role=user_data['role']
                    )
                    user.set_password(user_data['password'])
                    db_session.add(user)
                    logger.info(f"Created default user: {user_data['username']}")
            
            db_session.commit()
            db_session.close()
            
        except Exception as e:
            logger.error(f"Error setting up default users: {str(e)}")
    
    def authenticate_user(self, username, password):
        """Authenticate user with username and password"""
        try:
            db_session = self.Session()
            user = db_session.query(User).filter_by(username=username, is_active=True).first()
            
            if user and user.check_password(password):
                # Update last login
                user.last_login = datetime.utcnow()
                db_session.commit()
                
                # Log successful login
                self.log_audit_event(
                    user_id=user.id,
                    username=user.username,
                    action='login_success',
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'),
                    success=True
                )
                
                return user
            else:
                # Log failed login attempt
                self.log_audit_event(
                    username=username,
                    action='login_failed',
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent'),
                    success=False
                )
                
                # Log security event
                self.log_security_event(
                    event_type='login_failed',
                    severity='medium',
                    source_ip=request.remote_addr,
                    username=username,
                    details=f'Failed login attempt for user: {username}'
                )
                
                return None
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
        finally:
            db_session.close()
    
    def log_audit_event(self, user_id=None, username=None, action=None, 
                       resource=None, ip_address=None, user_agent=None, 
                       details=None, success=True):
        """Log audit event"""
        try:
            db_session = self.Session()
            audit_log = AuditLog(
                user_id=user_id,
                username=username,
                action=action,
                resource=resource,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                success=success
            )
            db_session.add(audit_log)
            db_session.commit()
            db_session.close()
            
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
    
    def log_security_event(self, event_type=None, severity='medium', 
                          source_ip=None, user_id=None, username=None, 
                          details=None):
        """Log security event"""
        try:
            db_session = self.Session()
            security_event = SecurityEvent(
                event_type=event_type,
                severity=severity,
                source_ip=source_ip,
                user_id=user_id,
                username=username,
                details=details
            )
            db_session.add(security_event)
            db_session.commit()
            db_session.close()
            
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")
    
    def check_rate_limit(self, ip_address, action, limit=5, window=300):
        """Check rate limiting for actions"""
        try:
            db_session = self.Session()
            
            # Count recent events for this IP and action
            recent_events = db_session.query(AuditLog).filter(
                AuditLog.ip_address == ip_address,
                AuditLog.action == action,
                AuditLog.timestamp >= datetime.utcnow() - timedelta(seconds=window)
            ).count()
            
            if recent_events >= limit:
                # Log rate limit event
                self.log_security_event(
                    event_type='rate_limit_exceeded',
                    severity='high',
                    source_ip=ip_address,
                    details=f'Rate limit exceeded for {action}: {recent_events} attempts'
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            return True
        finally:
            db_session.close()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            db_session = self.Session()
            user = db_session.query(User).filter_by(id=user_id).first()
            db_session.close()
            return user
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    def get_user_by_username(self, username):
        """Get user by username"""
        try:
            db_session = self.Session()
            user = db_session.query(User).filter_by(username=username).first()
            db_session.close()
            return user
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    def create_user(self, username, email, password, role='user'):
        """Create new user"""
        try:
            db_session = self.Session()
            
            # Check if user already exists
            existing_user = db_session.query(User).filter_by(username=username).first()
            if existing_user:
                return None
            
            user = User(
                username=username,
                email=email,
                role=role
            )
            user.set_password(password)
            
            db_session.add(user)
            db_session.commit()
            db_session.close()
            
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None
    
    def update_user_password(self, user_id, new_password):
        """Update user password"""
        try:
            db_session = self.Session()
            user = db_session.query(User).filter_by(id=user_id).first()
            
            if user:
                user.set_password(new_password)
                db_session.commit()
                db_session.close()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating password: {str(e)}")
            return False
    
    def deactivate_user(self, user_id):
        """Deactivate user account"""
        try:
            db_session = self.Session()
            user = db_session.query(User).filter_by(id=user_id).first()
            
            if user:
                user.is_active = False
                db_session.commit()
                db_session.close()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deactivating user: {str(e)}")
            return False
    
    def get_all_users(self):
        """Get all users"""
        try:
            db_session = self.Session()
            users = db_session.query(User).all()
            db_session.close()
            return users
        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            return [] 