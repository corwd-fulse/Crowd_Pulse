"""Append-only capture metadata; original videos/results are never rewritten."""
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Integer, JSON, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class CaptureSessionRevision(Base):
    __tablename__ = 'capture_session_revisions'
    __table_args__ = (ForeignKeyConstraint(['venue_id', 'venue_revision'],
        ['venue_revisions.venue_id', 'venue_revisions.revision'], ondelete='RESTRICT'),)
    session_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    venue_id: Mapped[str] = mapped_column(String(80), index=True)
    venue_revision: Mapped[int] = mapped_column(Integer)
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CaptureVideoVenue(Base):
    """Unique ownership prevents concurrent cross-venue assignment of a video."""
    __tablename__ = 'capture_video_venues'
    __table_args__ = (ForeignKeyConstraint(['venue_id', 'venue_revision'],
        ['venue_revisions.venue_id', 'venue_revisions.revision'], ondelete='RESTRICT'),)
    video_id: Mapped[UUID] = mapped_column(ForeignKey('videos.id', ondelete='RESTRICT'), primary_key=True)
    venue_id: Mapped[str] = mapped_column(String(80), index=True)
    venue_revision: Mapped[int] = mapped_column(Integer)
