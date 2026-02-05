# backend/README.md
# STT Pipeline Backend - PoC Phase 1

본 서버는 Daiso 매장 내 AI 상품 위치 안내 서비스의 STT 파이프라인 백엔드입니다.

## 📁 폴더 구조

```
backend/
├── stt/
│   ├── __init__.py
│   ├── types.py          # Pydantic 타입 정의
│   ├── adapters.py       # WhisperAdapter (faster-whisper)
│   ├── quality_gate.py   # 품질 게이트 (R1→R4)
│   └── policy_gate.py    # 정책 게이트 (키워드 기반)
├── config.yaml           # 설정 파일
├── main.py               # FastAPI 서버
├── runner.py             # CLI 테스트 러너
└── README.md
```

## 🚀 환경 설정

### 1. 패키지 설치

```bash
# 필수 패키지 설치 (requirements.txt 이미 업데이트됨)
pip install faster-whisper pyyaml python-multipart

# 또는 전체 requirements
pip install -r requirements.txt
```

### 2. GPU 확인

```bash
# NVIDIA GPU 확인
nvidia-smi

# CUDA 사용 가능 확인 (Python)
python -c "import torch; print(torch.cuda.is_available())"
```

## ⚙️ 설정 파일

`config.yaml`에서 다음을 설정할 수 있습니다:

- **STT 모델**: medium (기본), fallback: small
- **품질 게이트 임계값**: min_chars, min_confidence, nonsense_patterns
- **고정 위치**: 화장실/계산대/입구/출구 응답
- **비지원 키워드**: 배달/교환/환불 등

## 📝 텍스트 시뮬레이션 테스트 (오디오 파일 불필요)

```bash
# 테스트 케이스 실행
cd c:\Users\301\daiso-category-search-dev\daiso-category-search-dev
python backend/runner.py

# 결과 확인
cat outputs/test_results.json
```

### 테스트 케이스 수정

`data/test_cases.tsv`를 편집하여 케이스 추가/수정 가능 (TSV 형식)

## 🌐 FastAPI 서버 실행

### 로컬 개발 서버

```bash
# 백엔드 폴더에서
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버 실행 후:
- API 문서: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### API 엔드포인트

#### POST `/stt/process`
음성 파일을 업로드하여 STT 파이프라인 실행

**Request**:
- `audio`: 오디오 파일 (WAV 권장)
- `attempt`: 시도 횟수 (1 또는 2)

**Response**:
```json
{
  "request_id": "uuid",
  "stt": {
    "text_raw": "화장실 어디에요",
    "confidence": 0.95,
    "lang": "ko",
    "latency_ms": 1234
  },
  "quality_gate": {
    "status": "OK",
    "is_usable": true,
    "reason": "OK"
  },
  "policy_intent": {
    "intent_type": "FIXED_LOCATION",
    "location_target": "restroom",
    "confidence": 1.0,
    "reason": "Matched fixed location keyword: '화장실'"
  },
  "final_response": "화장실은 매장 뒤쪽 왼편에 있습니다.",
  "processing_time_ms": 1500
}
```

## 🔧 트러블슈팅

### 1. faster-whisper 설치 실패
```bash
# CUDA 버전 확인
nvidia-smi

# cuDNN이 필요한 경우
pip install nvidia-cudnn-cu11
```

### 2. OOM (Out of Memory) 에러
- `config.yaml`에서 `model: "small"`로 변경
- 또는 `device: "cpu"`로 변경 (느림)

### 3. 패키지 import 에러
```bash
# 현재 디렉토리 확인
pwd

# backend 폴더에서 실행하는지 확인
cd c:\Users\301\daiso-category-search-dev\daiso-category-search-dev
```

## 📊 Decision Log 업데이트

모델 변경 또는 임계값 조정 시 `10_DECISION_LOG.md`에 기록:

```markdown
### 2026-01-16 — Whisper 모델 선택
- 계획: large-v3
- 실제: medium (RTX 3050 4GB 제약)
- fallback: small
```

## 🎯 다음 단계

- [ ] NLU/검색 모듈 연동 (PRODUCT_SEARCH 처리)
- [ ] 정규화 모듈 구현 (07_NORMALIZATION.md)
- [ ] 실제 오디오 파일 테스트
- [ ] Google STT 어댑터 구현
- [ ] frontend 연동 (Next.js fetch 예시)
