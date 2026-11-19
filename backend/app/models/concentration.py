from datetime import datetime
from uuid import UUID,uuid4
from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Integer, JSON, String, Text, Uuid, func
from sqlalchemy.orm import Mapped,mapped_column
from .base import Base


class ConcentrationRun(Base):
    __tablename__='venue_concentration_runs'
    __table_args__=(ForeignKeyConstraint(['venue_id','revision'],['venue_revisions.venue_id','venue_revisions.revision'],ondelete='RESTRICT'),)
    id:Mapped[UUID]=mapped_column(Uuid,primary_key=True,default=uuid4)
    venue_id:Mapped[str]=mapped_column(String(80),index=True)
    revision:Mapped[int]=mapped_column(Integer)
    request:Mapped[dict]=mapped_column(JSON)
    result:Mapped[dict]=mapped_column(JSON)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())


class OperatorEvent(Base):
    __tablename__='venue_operator_events'
    id:Mapped[UUID]=mapped_column(Uuid,primary_key=True,default=uuid4)
    run_id:Mapped[UUID]=mapped_column(ForeignKey('venue_concentration_runs.id',ondelete='RESTRICT'),index=True)
    operator:Mapped[str]=mapped_column(String(100))
    zone_id:Mapped[str|None]=mapped_column(String(80))
    verification:Mapped[str]=mapped_column(String(30))
    action:Mapped[str]=mapped_column(String(40))
    note:Mapped[str]=mapped_column(Text)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
