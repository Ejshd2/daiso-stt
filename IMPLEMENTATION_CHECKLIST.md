# STT Pipeline 구현 완료 체크리스트

## ✅ 완료된 항목

### 백엔드 구조
- [x] backend/stt/types.py - Pydantic 타입 정의
- [x] backend/stt/adapters.py - WhisperAdapter (medium→small fallback)
- [x] backend/stt/quality_gate.py - 품질 게이트 (R1→R4 우선순위)
- [x] backend/stt/policy_gate.py - 정책 게이트 (키워드 기반)
- [x] backend/config.yaml - 설정 파일
- [x] backend/main.py - FastAPI 서버
- [x] backend/runner.py - CLI 테스트 러너
- [x] backend/README.md - 설치/실행 가이드

### 테스트 데이터
- [x] data/test_cases.tsv - 20케이스 TSV
  - FIXED_LOCATION: 4개
  - PRODUCT_SEARCH: 8개
  - UNSUPPORTED: 4개
  - QUALITY_FAIL: 4개

### Frontend
- [x] frontend/src/app/api/stt-example.ts - fetch 호출 예시

## ⚠️ 알려진 이슈

### 패키지 설치 이슈
- tokenizers 패키지 설치 실패 (Rust 컴파일러 필요)
- **해결 방법**: 아래 명령어로 pre-built wheel 사용

```bash
pip install faster-whisper --no-build-isolation
```

## 📋 다음 실행 단계

### 1. 패키지 재설치 (필수)
```bash
cd c:\Users\301\daiso-category-search-dev\daiso-category-search-dev
pip install faster-whisper --no-build-isolation
```

### 2. 텍스트 시뮬레이션 테스트
```bash
python backend/runner.py
```

### 3. FastAPI 서버 실행
```bash
python backend/main.py
```

### 4. API 문서 확인
브라우저에서 http://localhost:8000/docs

## 🔧 추가 구현 필요 (Phase 2)

- [ ] 정규화 모듈 (07_NORMALIZATION.md)
- [ ] Google STT 어댑터
- [ ] 실제 오디오 파일 테스트
- [ ] NLU/검색 연동 (PRODUCT_SEARCH 처리)
- [ ] Frontend UI 페이지 (최소 데모)

## 📊 Decision Log 업데이트 필요

`10_DECISION_LOG.md`에 다음 내용 추가:

```markdown
### 2026-01-16 — Whisper 모델 선택
- 계획: large-v3
- 실제 선택: medium (기본), small (fallback)
- 실행: faster-whisper (GPU cuda, compute_type float16)
- 사유: RTX 3050 4GB VRAM 제약
- 검증 계획: Phase 2에서 CPU large-v3 샘플 비교
```
