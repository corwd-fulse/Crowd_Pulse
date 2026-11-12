
"""prevent duplicate active analysis jobs

Revision ID: bdebb741521f
Revises: 20261112_01
"""

from alembic import op
import sqlalchemy as sa


revision = "bdebb741521f"
down_revision = "20261112_01"
branch_labels = None
depends_on = None


def upgrade():
    # 동일 영상에 queued/running 작업이 하나만 존재하도록 제한
    op.create_index(
        "uq_analysis_jobs_active_video",
        "analysis_jobs",
        ["video_id"],
        unique=True,
        postgresql_where=sa.text(
            "status IN ('queued', 'running')"
        ),
    )


def downgrade():
    # 마이그레이션 롤백 시 인덱스 제거
    op.drop_index(
        "uq_analysis_jobs_active_video",
        table_name="analysis_jobs",
    )
