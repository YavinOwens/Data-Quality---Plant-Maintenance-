-- SAP S/4HANA Data Quality Database Schema
-- This script creates all necessary tables for the data quality workflow

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Validation Results Table
CREATE TABLE IF NOT EXISTS validation_results (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('passed', 'failed', 'error', 'skipped')),
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
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

-- Audit Log Table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_validation_results_dataset_name ON validation_results(dataset_name);
CREATE INDEX IF NOT EXISTS idx_validation_results_created_at ON validation_results(created_at);
CREATE INDEX IF NOT EXISTS idx_validation_results_status ON validation_results(status);

CREATE INDEX IF NOT EXISTS idx_data_quality_issues_dataset_name ON data_quality_issues(dataset_name);
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_status ON data_quality_issues(status);
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_severity ON data_quality_issues(severity);

CREATE INDEX IF NOT EXISTS idx_data_extraction_logs_dataset_name ON data_extraction_logs(dataset_name);
CREATE INDEX IF NOT EXISTS idx_data_extraction_logs_created_at ON data_extraction_logs(created_at);

CREATE INDEX IF NOT EXISTS idx_workflow_execution_logs_workflow_name ON workflow_execution_logs(workflow_name);
CREATE INDEX IF NOT EXISTS idx_workflow_execution_logs_status ON workflow_execution_logs(status);

CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_validation_results_updated_at 
    BEFORE UPDATE ON validation_results 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_quality_issues_updated_at 
    BEFORE UPDATE ON data_quality_issues 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configuration_updated_at 
    BEFORE UPDATE ON configuration 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

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
    dataset_name,
    rule_name,
    status,
    COUNT(*) as count,
    AVG(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as success_rate,
    MIN(created_at) as first_validation,
    MAX(created_at) as last_validation
FROM validation_results
GROUP BY dataset_name, rule_name, status;

CREATE OR REPLACE VIEW recent_issues_view AS
SELECT 
    i.id,
    i.dataset_name,
    i.rule_name,
    i.issue_type,
    i.issue_description,
    i.affected_records_count,
    i.severity,
    i.status,
    i.created_at,
    vr.results as validation_results
FROM data_quality_issues i
LEFT JOIN validation_results vr ON i.validation_result_id = vr.id
WHERE i.status IN ('open', 'in_progress')
ORDER BY i.created_at DESC;

CREATE OR REPLACE VIEW quality_metrics_dashboard AS
SELECT 
    dataset_name,
    COUNT(*) as total_validations,
    COUNT(CASE WHEN status = 'passed' THEN 1 END) as passed_validations,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_validations,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as error_validations,
    ROUND(
        (COUNT(CASE WHEN status = 'passed' THEN 1 END)::DECIMAL / COUNT(*) * 100), 2
    ) as success_rate,
    MAX(created_at) as last_validation
FROM validation_results
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY dataset_name; 