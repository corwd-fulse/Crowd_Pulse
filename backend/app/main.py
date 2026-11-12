
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

from routers.dashboard import router as dashboard_router
from app.database import check_database_connection
from routers.videos import router as videos_router
from routers import analysis
from routers.vision_results import router as vision_results_router
from routers.video_rois import router as video_rois_router


app = FastAPI(
    title="Crowd Pulse AI API",
    description="AI 기반 인파 분석 및 혼잡 예측 시스템",
    version="0.1.0",
)

# 대시보드 라우터 등록
app.include_router(dashboard_router)

app.include_router(videos_router)

app.include_router(analysis.router)
app.include_router(vision_results_router)
app.include_router(video_rois_router)

# React 개발 서버 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Crowd Pulse AI API",
        "version": "0.1.0",
    }


@app.get("/api/database/health", tags=["Database"])
def database_health():
    try:
        connected = check_database_connection()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable",
        ) from exc

    if not connected:
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable",
        )

    return {
        "status": "ok",
        "database": "postgresql",
        "connected": True,
    }
