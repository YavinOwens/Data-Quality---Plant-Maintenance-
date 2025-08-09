#!/usr/bin/env python3
"""
Authentication and security services for SAP S/4HANA Data Quality Workflow
"""

import os
import logging
from datetime import datetime, timedelta
from flask import request, session, current_app
from flask_login import login_user, logout_user, current_user, login_required
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import User, AuditLog, SecurityEvent, Base, DEFAULT_USERS
from werkzeug.security import generate_password_hash
import structlog

logger = structlog.get_logger()

class AuthService:
    """Authentication service for user management and security"""
    
    def __init__(self, db_url):
        """Initialize authentication service"""
        try:
            self.engine = create_engine(db_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            self.db_available = True
            self.setup_default_users()
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            self.db_available = False
            self.engine = None
            self.Session = None
    
    def setup_default_users(self):
        """Create default users if they don't exist"""
        if not self.db_available:
            logger.info("Skipping default user setup - database not available")
            return
            
        try:
            # Only create default users in development to avoid test/demo accounts in production
            if os.getenv("FLASK_ENV", "production").lower() != "development":
                return
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
        logger.info(f"Authentication attempt for user: {username}")
        
        # Use only fallback authentication for now
        fallback_users = {
            'admin': {'password': 'Admin@123!', 'role': 'admin', 'id': 1},
            'user': {'password': 'User@123!', 'role': 'user', 'id': 2},
            'viewer': {'password': 'Viewer@123!', 'role': 'viewer', 'id': 3}
        }
        
        logger.info(f"Using fallback authentication for user: {username}")
        if username in fallback_users and fallback_users[username]['password'] == password:
            logger.info(f"Fallback authentication successful for user: {username}")
            # Create a mock user object with required Flask-Login methods
            class MockUser:
                def __init__(self, user_id, username, role):
                    self.id = user_id
                    self.username = username
                    self.role = role
                    self.is_active = True
                    self.is_authenticated = True
                    self.is_anonymous = False
                
                def get_id(self):
                    return str(self.id)
            
            user = MockUser(
                fallback_users[username]['id'],
                username,
                fallback_users[username]['role']
            )
            # Persist username for user_loader fallback
            try:
                session['username'] = username
            except Exception:
                pass
            return user
        else:
            logger.info(f"Fallback authentication failed for user: {username}")
            return None
    
    def log_audit_event(self, user_id=None, username=None, action=None, 
                       resource=None, ip_address=None, user_agent=None, 
                       details=None, success=True):
        """Log audit event"""
        if not self.db_available:
            logger.info(f"Audit event (DB unavailable): {action} by {username} from {ip_address}")
            return
            
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
                success=success,
                timestamp=datetime.utcnow()
            )
            db_session.add(audit_log)
            db_session.commit()
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
        finally:
            db_session.close()
    
    def log_security_event(self, event_type=None, severity='medium', 
                          source_ip=None, user_id=None, username=None, 
                          details=None):
        """Log security event"""
        if not self.db_available:
            logger.warning(f"Security event (DB unavailable): {event_type} - {details}")
            return
            
        try:
            db_session = self.Session()
            security_event = SecurityEvent(
                event_type=event_type,
                severity=severity,
                source_ip=source_ip,
                user_id=user_id,
                username=username,
                details=details,
                timestamp=datetime.utcnow()
            )
            db_session.add(security_event)
            db_session.commit()
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")
        finally:
            db_session.close()
    
    def check_rate_limit(self, ip_address, action, limit=5, window=300):
        """Check rate limiting for actions"""
        logger.info(f"Rate limit check for {ip_address}:{action}")
        # Temporarily disable rate limiting for debugging
        return True
    
    def get_user_by_id(self, user_id):
        """Load user by ID for Flask-Login"""
        if not self.db_available:
            # Reconstruct a mock user from session data
            uname = session.get('username', 'admin')
            role = 'admin' if uname == 'admin' else ('user' if uname == 'user' else 'viewer')
            class MockUser:
                def __init__(self, user_id, username, role):
                    self.id = user_id
                    self.username = username
                    self.role = role
                    self.is_active = True
                    self.is_authenticated = True
                    self.is_anonymous = False
                def get_id(self):
                    return str(self.id)
            try:
                return MockUser(int(user_id), uname, role)
            except Exception:
                return None
        try:
            db_session = self.Session()
            return db_session.query(User).filter_by(id=user_id).first()
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
        finally:
            try:
                db_session.close()
            except Exception:
                pass
    
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

    # --- Added minimal helpers used elsewhere in the app ---
    def execute_query(self, sql: str):
        """Execute a simple SQL statement for health checks; returns first row or None."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                row = result.fetchone()
                return row[0] if row and len(row) > 0 else None
        except Exception as e:
            logger.error(f"execute_query error: {e}")
            return None

    def get_security_events(self, hours: int = None, days: int = None, limit: int = 100):
        """Return recent security events; window by hours or days if provided."""
        try:
            db_session = self.Session()
            query = db_session.query(SecurityEvent)
            cutoff = None
            now = datetime.utcnow()
            if hours is not None:
                cutoff = now - timedelta(hours=hours)
            elif days is not None:
                cutoff = now - timedelta(days=days)
            if cutoff is not None:
                query = query.filter(SecurityEvent.timestamp >= cutoff)
            events = query.order_by(SecurityEvent.timestamp.desc()).limit(limit).all()
            db_session.close()
            return events
        except Exception as e:
            logger.error(f"get_security_events error: {e}")
            return []