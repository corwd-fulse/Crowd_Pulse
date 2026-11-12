"""Keep CSV result storage independent of algorithm window settings."""
from alembic import op

revision = '20261112_03'
down_revision = '20261112_02'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint('ck_vision_score_window', 'vision_bottleneck_scores', type_='check')
    op.create_check_constraint('ck_vision_score_window', 'vision_bottleneck_scores', 'window_start_ms >= 0')


def downgrade():
    raise RuntimeError('Do not reimpose an algorithm-specific window on preserved results')
