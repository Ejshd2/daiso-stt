# Daiso Project Setup Guide

이 프로젝트는 Backend(FastAPI)와 Frontend(Next.js)로 구성되어 있습니다.

## 1. Backend 설정 (Python)

### 가상환경 생성 및 활성화
**생성 (최초 1회)**
```bash
python -m venv venv
```

**활성화**
- Bash: `source venv/bin/activate`
- PowerShell: `venv\Scripts\Activate.ps1`
- CMD: `venv\Scripts\activate`

*활성화 시 터미널 프롬프트 앞에 `(venv)`가 표시되어야 합니다.*

### 라이브러리 설치
가상환경이 활성화된 상태에서:
```bash
pip install -r requirements.txt
```

### 서버 실행
프로젝트 루트(`daiso-category-search-dev`)에서 실행하세요:
```bash
uvicorn backend.api:app --reload
```
- API 서버 주소: http://localhost:8000

---

## 2. Frontend 설정 (Next.js)

Frontend 작업은 `frontend` 폴더 내부에서 진행합니다.

### 필수 준비사항
- Node.js 설치 필요 (LTS 버전 권장)

### 패키지 설치
```bash
cd frontend
npm install
```

### 개발 서버 실행
```bash
npm run dev
```
- 웹사이트 주소: http://localhost:3000

## 3. 전체 구조
- **backend/**: FastAPI 서버 로직 (API, AI 모델 연동 등)
- **frontend/**: Next.js 웹 애플리케이션
