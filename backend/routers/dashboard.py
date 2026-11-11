
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"],
)


class DashboardSummary(BaseModel):
    person_count: int = Field(ge=0)
    congestion_percent: float = Field(ge=0, le=100)
    bottleneck_count: int = Field(ge=0)
    alert_count: int = Field(ge=0)
    updated_at: datetime
    data_source: str


@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary():
    return DashboardSummary(
        person_count=350,
        congestion_percent=68,
        bottleneck_count=2,
        alert_count=3,
        updated_at=datetime.now(timezone.utc),
        data_source="mock",
    )
