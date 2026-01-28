#!/bin/bash
# Aegis Nexus Database Initialization Script

set -e

echo "Initializing Aegis Nexus Database..."

# Create tables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create verification requests table
    CREATE TABLE IF NOT EXISTS verification_requests (
        request_id VARCHAR(64) PRIMARY KEY,
        prompt TEXT NOT NULL,
        context JSONB,
        llm_provider VARCHAR(32) NOT NULL,
        priority INTEGER NOT NULL DEFAULT 1,
        metadata JSONB,
        status VARCHAR(32) NOT NULL DEFAULT 'pending',
        submitted_at TIMESTAMP NOT NULL DEFAULT NOW(),
        processing_started_at TIMESTAMP,
        processing_completed_at TIMESTAMP,
        overall_confidence DECIMAL(5,2),
        final_status VARCHAR(32)
    );

    -- Create verification results table
    CREATE TABLE IF NOT EXISTS verification_results (
        result_id SERIAL PRIMARY KEY,
        request_id VARCHAR(64) REFERENCES verification_requests(request_id),
        phase_1_audit JSONB,
        phase_2_cognition JSONB,
        phase_3_reality JSONB,
        detailed_results JSONB,
        error_message TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    -- Create audit trail table
    CREATE TABLE IF NOT EXISTS audit_trail (
        entry_id SERIAL PRIMARY KEY,
        entry_hash VARCHAR(128) UNIQUE NOT NULL,
        previous_hash VARCHAR(128),
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
        event_type VARCHAR(64) NOT NULL,
        component VARCHAR(64) NOT NULL,
        data JSONB NOT NULL,
        pqc_signature TEXT,
        metadata JSONB
    );

    -- Create system metrics table
    CREATE TABLE IF NOT EXISTS system_metrics (
        metric_id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
        metric_name VARCHAR(64) NOT NULL,
        metric_value DECIMAL(10,4) NOT NULL,
        labels JSONB
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_verification_requests_status ON verification_requests(status);
    CREATE INDEX IF NOT EXISTS idx_verification_requests_submitted ON verification_requests(submitted_at);
    CREATE INDEX IF NOT EXISTS idx_verification_results_request ON verification_results(request_id);
    CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_trail_event_type ON audit_trail(event_type);
    CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
    CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

    -- Insert initial system metrics
    INSERT INTO system_metrics (metric_name, metric_value, labels) VALUES
        ('system_initialized', 1.0, '{"version": "1.0.0"}'),
        ('deployment_timestamp', EXTRACT(EPOCH FROM NOW()), '{}');

    -- Grant permissions
    GRANT ALL PRIVILEGES ON TABLE verification_requests TO postgres;
    GRANT ALL PRIVILEGES ON TABLE verification_results TO postgres;
    GRANT ALL PRIVILEGES ON TABLE audit_trail TO postgres;
    GRANT ALL PRIVILEGES ON TABLE system_metrics TO postgres;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres;

EOSQL

echo "Database initialization completed successfully!"