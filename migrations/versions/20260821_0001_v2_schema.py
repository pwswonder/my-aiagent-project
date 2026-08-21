"""Create the additive V2 schema.

Revision ID: 20260821_0001
Revises:
"""
from __future__ import annotations

from alembic import op

from paper_agent_v2 import models  # noqa: F401
from paper_agent_v2.db import Base

revision = "20260821_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunks_fts ON chunks "
            "USING gin (to_tsvector('english', text))"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
