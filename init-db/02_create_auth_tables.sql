-- Authentication and Security Tables
-- This script creates the necessary tables for user authentication and security monitoring

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    username VARCHAR(80),
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT TRUE
);

-- Security events table
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    source_ip VARCHAR(45),
    user_id INTEGER,
    username VARCHAR(80),
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);

-- Insert default users (passwords will be hashed by the application)
-- These are placeholder users that will be replaced by the application
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@company.com', 'placeholder_hash', 'admin'),
('user', 'user@company.com', 'placeholder_hash', 'user'),
('viewer', 'viewer@company.com', 'placeholder_hash', 'viewer')
ON CONFLICT (username) DO NOTHING;

-- Create a view for recent security events
CREATE OR REPLACE VIEW recent_security_events AS
SELECT 
    event_type,
    severity,
    source_ip,
    username,
    details,
    timestamp
FROM security_events 
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Create a view for user activity
CREATE OR REPLACE VIEW user_activity AS
SELECT 
    u.username,
    u.role,
    u.last_login,
    COUNT(a.id) as total_actions,
    COUNT(CASE WHEN a.success = FALSE THEN 1 END) as failed_actions
FROM users u
LEFT JOIN audit_logs a ON u.id = a.user_id
WHERE a.timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY u.id, u.username, u.role, u.last_login;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON users TO sap_user;
-- GRANT SELECT, INSERT ON audit_logs TO sap_user;
-- GRANT SELECT, INSERT ON security_events TO sap_user; 