from datetime import datetime
from uuid import UUID,uuid4
from sqlalchemy import ForeignKey,DateTime,Integer,String,UniqueConstraint,func,Uuid,CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped,mapped_column
from .base import Base

class VideoRoiRevision(Base):
    __tablename__='video_roi_revisions'
    __table_args__=(UniqueConstraint('video_id','ordinal',name='uq_video_roi_revision_ordinal'),
                    CheckConstraint('ordinal>0',name='ck_video_roi_revision_ordinal'))
    id:Mapped[UUID]=mapped_column(Uuid,primary_key=True,default=uuid4)
    video_id:Mapped[UUID]=mapped_column(ForeignKey('videos.id',ondelete='RESTRICT'),index=True)
    ordinal:Mapped[int]=mapped_column(Integer)
    snapshot:Mapped[dict]=mapped_column(JSONB)
    fingerprint:Mapped[str]=mapped_column(String(64))
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
