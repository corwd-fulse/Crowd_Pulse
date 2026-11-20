"""Append-only concentration evidence and explicit operator actions."""
from alembic import op
import sqlalchemy as sa
revision='20261117_08'
down_revision='20261117_07'
branch_labels=None
depends_on=None


def upgrade():
    op.create_table('venue_concentration_runs',
        sa.Column('id',sa.Uuid(),primary_key=True),
        sa.Column('venue_id',sa.String(80),nullable=False),
        sa.Column('revision',sa.Integer(),nullable=False),
        sa.Column('request',sa.JSON(),nullable=False),
        sa.Column('result',sa.JSON(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now(),nullable=False),
        sa.ForeignKeyConstraint(['venue_id','revision'],['venue_revisions.venue_id','venue_revisions.revision'],ondelete='RESTRICT'))
    op.create_index('ix_venue_concentration_runs_venue_id','venue_concentration_runs',['venue_id'])
    op.create_table('venue_operator_events',
        sa.Column('id',sa.Uuid(),primary_key=True),
        sa.Column('run_id',sa.Uuid(),sa.ForeignKey('venue_concentration_runs.id',ondelete='RESTRICT'),nullable=False),
        sa.Column('operator',sa.String(100),nullable=False),
        sa.Column('zone_id',sa.String(80)),
        sa.Column('verification',sa.String(30),nullable=False),
        sa.Column('action',sa.String(40),nullable=False),
        sa.Column('note',sa.Text(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now(),nullable=False))
    op.create_index('ix_venue_operator_events_run_id','venue_operator_events',['run_id'])


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve operator evidence')
