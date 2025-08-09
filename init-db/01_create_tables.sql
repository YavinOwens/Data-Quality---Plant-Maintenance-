-- SAP S/4HANA Data Quality Database Schema
-- This script creates all necessary tables for the data quality workflow

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Unified Validation Results Table (aligned with validation-engine)
CREATE TABLE IF NOT EXISTS validation_results (
    id SERIAL PRIMARY KEY,
    validation_type VARCHAR(100) NOT NULL,
    validation_name VARCHAR(200) NOT NULL,
    success_rate DECIMAL(5,4),
    total_records INTEGER,
    passed_records INTEGER,
    failed_records INTEGER,
    error_details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Quality Metrics Table
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(50),
    threshold_value DECIMAL(10,4),
    status VARCHAR(20) CHECK (status IN ('passed', 'failed', 'warning')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_name, metric_name, created_at)
);

-- Issues Table
CREATE TABLE IF NOT EXISTS data_quality_issues (
    id SERIAL PRIMARY KEY,
    validation_result_id INTEGER REFERENCES validation_results(id),
    dataset_name VARCHAR(100) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    issue_type VARCHAR(100) NOT NULL,
    issue_description TEXT,
    affected_records_count INTEGER,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    assigned_to VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Data Extraction Logs
CREATE TABLE IF NOT EXISTS data_extraction_logs (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    extraction_type VARCHAR(50) NOT NULL,
    records_extracted INTEGER,
    extraction_duration_ms INTEGER,
    status VARCHAR(20) CHECK (status IN ('success', 'failed', 'partial')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Execution Logs
CREATE TABLE IF NOT EXISTS workflow_execution_logs (
    id SERIAL PRIMARY KEY,
    workflow_name VARCHAR(100) NOT NULL,
    execution_id UUID DEFAULT uuid_generate_v4(),
    status VARCHAR(20) CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    error_message TEXT,
    parameters JSONB
);

-- Notifications Table
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    notification_type VARCHAR(50) NOT NULL,
    recipient VARCHAR(100) NOT NULL,
    subject VARCHAR(200),
    message TEXT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Configuration Table
CREATE TABLE IF NOT EXISTS configuration (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    config_type VARCHAR(50) DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit Logs Table (pluralized to match application)
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    username VARCHAR(80),
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT TRUE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_validation_results_type ON validation_results(validation_type);
CREATE INDEX IF NOT EXISTS idx_validation_results_created_at ON validation_results(created_at);

CREATE INDEX IF NOT EXISTS idx_data_quality_issues_dataset_name ON data_quality_issues(dataset_name);
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_status ON data_quality_issues(status);
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_severity ON data_quality_issues(severity);

CREATE INDEX IF NOT EXISTS idx_data_extraction_logs_dataset_name ON data_extraction_logs(dataset_name);
CREATE INDEX IF NOT EXISTS idx_data_extraction_logs_created_at ON data_extraction_logs(created_at);

CREATE INDEX IF NOT EXISTS idx_workflow_execution_logs_workflow_name ON workflow_execution_logs(workflow_name);
CREATE INDEX IF NOT EXISTS idx_workflow_execution_logs_status ON workflow_execution_logs(status);

CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);

-- Risk Register Table
CREATE TABLE IF NOT EXISTS risk_register (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    likelihood SMALLINT CHECK (likelihood BETWEEN 1 AND 3),
    impact SMALLINT CHECK (impact BETWEEN 1 AND 3),
    status VARCHAR(50),
    owner VARCHAR(100),
    due_date DATE,
    priority VARCHAR(20),
    description TEXT,
    causes TEXT,
    consequences TEXT,
    mitigation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_register_status ON risk_register(status);
CREATE INDEX IF NOT EXISTS idx_risk_register_category ON risk_register(category);

-- (deprecated) Updated_at trigger not required for new validation_results

-- NOTE: removed updated_at triggers for unified schema

-- Insert default configuration
INSERT INTO configuration (config_key, config_value, config_type, description) VALUES
('data_retention_days', '90', 'integer', 'Number of days to retain data quality results'),
('notification_enabled', 'true', 'boolean', 'Enable/disable notifications'),
('auto_correction_enabled', 'false', 'boolean', 'Enable/disable automatic data correction'),
('dashboard_refresh_interval', '300', 'integer', 'Dashboard refresh interval in seconds'),
('max_issues_per_report', '1000', 'integer', 'Maximum number of issues to include in reports'),
('alert_threshold_completeness', '95.0', 'decimal', 'Completeness threshold for alerts'),
('alert_threshold_accuracy', '90.0', 'decimal', 'Accuracy threshold for alerts'),
('alert_threshold_consistency', '98.0', 'decimal', 'Consistency threshold for alerts'),
('alert_threshold_timeliness', '85.0', 'decimal', 'Timeliness threshold for alerts')
ON CONFLICT (config_key) DO NOTHING;

-- Create views for easier reporting
CREATE OR REPLACE VIEW validation_summary_view AS
SELECT 
    vr.validation_type AS dataset_name,
    vr.validation_name AS rule_name,
    CASE WHEN vr.success_rate >= 0.95 THEN 'passed' ELSE 'failed' END AS status,
    COUNT(*) AS count,
    AVG(CASE WHEN vr.success_rate >= 0.95 THEN 1 ELSE 0 END) AS success_rate,
    MIN(vr.created_at) AS first_validation,
    MAX(vr.created_at) AS last_validation
FROM validation_results vr
GROUP BY vr.validation_type, vr.validation_name, status;

CREATE OR REPLACE VIEW recent_issues_view AS
SELECT 
    i.id,
    COALESCE(vr.validation_type, i.dataset_name) AS dataset_name,
    COALESCE(vr.validation_name, i.rule_name) AS rule_name,
    i.issue_type,
    i.issue_description,
    i.affected_records_count,
    i.severity,
    i.status,
    i.created_at,
    vr.error_details AS validation_results
FROM data_quality_issues i
LEFT JOIN validation_results vr ON i.validation_result_id = vr.id
WHERE i.status IN ('open', 'in_progress')
ORDER BY i.created_at DESC;

CREATE OR REPLACE VIEW quality_metrics_dashboard AS
SELECT 
    vr.validation_type AS dataset_name,
    COUNT(*) AS total_validations,
    COUNT(CASE WHEN vr.success_rate >= 0.95 THEN 1 END) AS passed_validations,
    COUNT(CASE WHEN vr.success_rate < 0.95 THEN 1 END) AS failed_validations,
    0 AS error_validations,
    ROUND((COUNT(CASE WHEN vr.success_rate >= 0.95 THEN 1 END)::DECIMAL / COUNT(*) * 100), 2) AS success_rate,
    MAX(vr.created_at) AS last_validation
FROM validation_results vr
WHERE vr.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY vr.validation_type;