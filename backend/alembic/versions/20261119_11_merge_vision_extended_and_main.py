"""Merge heads: vision extended results (20261112_06) + ROI/venue/capture chain (20261117_10).

Both branches forked from 20261112_05 and touch unrelated tables, so no schema work is needed.
"""
revision = '20261119_11'
down_revision = ('20261112_06', '20261117_10')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
