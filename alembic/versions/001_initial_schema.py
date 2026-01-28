"""initial_schema

Revision ID: 001
Revises: 
Create Date: 2024-01-24 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Verification Requests
    op.create_table('verification_requests',
        sa.Column('request_id', sa.String(length=64), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('llm_provider', sa.String(length=32), nullable=False),
        sa.Column('priority', sa.Integer(), server_default='1', nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(length=32), server_default='pending', nullable=False),
        sa.Column('submitted_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('processing_started_at', sa.DateTime(), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(), nullable=True),
        sa.Column('overall_confidence', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('final_status', sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint('request_id')
    )
    op.create_index('idx_verification_requests_status', 'verification_requests', ['status'], unique=False)
    op.create_index('idx_verification_requests_submitted', 'verification_requests', ['submitted_at'], unique=False)

    # 2. Verification Results
    op.create_table('verification_results',
        sa.Column('result_id', sa.Integer(), nullable=False),
        sa.Column('request_id', sa.String(length=64), nullable=True),
        sa.Column('phase_1_audit', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('phase_2_cognition', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('phase_3_reality', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('detailed_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['request_id'], ['verification_requests.request_id'], ),
        sa.PrimaryKeyConstraint('result_id')
    )
    op.create_index('idx_verification_results_request', 'verification_results', ['request_id'], unique=False)

    # 3. Audit Trail
    op.create_table('audit_trail',
        sa.Column('entry_id', sa.Integer(), nullable=False),
        sa.Column('entry_hash', sa.String(length=128), nullable=False),
        sa.Column('previous_hash', sa.String(length=128), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('component', sa.String(length=64), nullable=False),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('pqc_signature', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('entry_id'),
        sa.UniqueConstraint('entry_hash')
    )
    op.create_index('idx_audit_trail_event_type', 'audit_trail', ['event_type'], unique=False)
    op.create_index('idx_audit_trail_timestamp', 'audit_trail', ['timestamp'], unique=False)

    # 4. System Metrics
    op.create_table('system_metrics',
        sa.Column('metric_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('metric_name', sa.String(length=64), nullable=False),
        sa.Column('metric_value', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('labels', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('metric_id')
    )
    op.create_index('idx_system_metrics_name', 'system_metrics', ['metric_name'], unique=False)
    op.create_index('idx_system_metrics_timestamp', 'system_metrics', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_system_metrics_timestamp', table_name='system_metrics')
    op.drop_index('idx_system_metrics_name', table_name='system_metrics')
    op.drop_table('system_metrics')
    op.drop_index('idx_audit_trail_timestamp', table_name='audit_trail')
    op.drop_index('idx_audit_trail_event_type', table_name='audit_trail')
    op.drop_table('audit_trail')
    op.drop_index('idx_verification_results_request', table_name='verification_results')
    op.drop_table('verification_results')
    op.drop_index('idx_verification_requests_submitted', table_name='verification_requests')
    op.drop_index('idx_verification_requests_status', table_name='verification_requests')
    op.drop_table('verification_requests')
