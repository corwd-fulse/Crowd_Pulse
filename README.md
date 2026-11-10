# CROWD PULSE AI

### AI 기반 실시간 인파 밀집 분석 및 혼잡 예측 시스템

**CROWD PULSE AI**는 영상 분석과 시계열 예측 기술을 활용하여 인파의 밀집도와 이동 흐름을 분석하고, 위험 상황을 사전에 감지하기 위한 AI 기반 모니터링 시스템입니다.

단순히 인원수를 측정하는 데 그치지 않고, **인파 이동 패턴, 병목 구간, 혼잡도 변화 및 미래 혼잡 위험**까지 분석하는 것을 목표로 합니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | CROWD PULSE AI |
| 개발 인원 | 3명 |
| 프로젝트 유형 | AI 기반 웹 모니터링 서비스 |
| 개발 방식 | GitHub Organization + Fork + Pull Request |
| 개발 상태 | 개발 환경 구축 및 초기 개발 진행 중 |

## 주요 기능

### 1. 실시간 인파 모니터링
- YOLO 기반 사람 객체 탐지
- ByteTrack 기반 객체 추적
- 영상 내 인원수 및 밀집도 분석
- 실시간 모니터링 대시보드

### 2. 인파 이동 흐름 및 병목 분석
- Optical Flow 기반 이동 흐름 분석
- 인파 이동 방향 및 속도 분석
- 특정 구역의 정체 및 병목 현상 감지
- 분석 결과 시각화

### 3. 혼잡도 예측
- 과거 인파 데이터 기반 시계열 분석
- LightGBM / LSTM 기반 혼잡도 예측 모델 검토
- 시간대별 혼잡도 변화 분석
- 미래 혼잡 위험 예측

### 4. 위험 알림 및 데이터 시각화
- 위험 단계별 혼잡 상태 표시
- 혼잡도 변화 및 이벤트 기록
- 인파 통계 그래프 제공
- 위험 상황 발생 시 알림 제공

## 기술 스택

| 구분 | 기술 |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Backend | Python, FastAPI |
| Database | PostgreSQL, TimescaleDB |
| Computer Vision | YOLO, ByteTrack, OpenCV, Optical Flow |
| Prediction AI | LightGBM, LSTM (검토 및 개발 예정) |
| Visualization | Recharts |
| API | Axios, REST API, WebSocket (예정) |
| Collaboration | Git, GitHub, Fork, Pull Request |

## 프로젝트 구조

```text
Crowd_Pulse/
├── frontend/             # React + Vite + TypeScript
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/              # FastAPI (예정)
│   └── app/
│
├── ai/                   # AI 모델 및 추론 코드 (예정)
│
└── README.md
```

※ 실제 폴더 구조는 개발 진행에 따라 변경될 수 있습니다.

## 팀 구성

| 인원 | 담당 업무 |
|---|---|
| 팀원 1 | Full Stack (React, FastAPI, DB, AI 모델 연동) |
| 팀원 2 | Vision AI (YOLO, ByteTrack, Optical Flow) |
| 팀원 3 | 시계열 AI (혼잡도 예측 및 분석) |

## 개발 진행 현황

- [x] 프로젝트 기획 및 주제 선정
- [x] GitHub Organization 생성
- [x] React + Vite + TypeScript 환경 구축
- [x] 프론트엔드 기본 라이브러리 설치
- [ ] 프론트엔드 대시보드 개발
- [ ] FastAPI 백엔드 구축
- [ ] PostgreSQL / TimescaleDB 연동
- [ ] Vision AI 모델 개발 및 연동
- [ ] 시계열 AI 모델 개발 및 연동
- [ ] 통합 테스트 및 배포

## 프로젝트 목표

인파 밀집 위험을 단순히 탐지하는 것을 넘어, 인파의 움직임과 변화 추세를 분석하여 **위험 상황을 더 빠르게 인식하고 예측할 수 있는 시스템**을 구현하는 것이 최종 목표입니다.

---

**CROWD PULSE AI — Detect the Flow, Predict the Risk.**
