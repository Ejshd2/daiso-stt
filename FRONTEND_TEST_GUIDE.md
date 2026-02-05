# 프론트엔드 테스트 실행 가이드

## 🚀 실행 방법

### 1. 백엔드 서버 실행 (필수)
```bash
# 터미널 1
cd c:\Users\301\daiso-category-search-dev\daiso-category-search-dev
python backend/main.py
```

서버 실행 확인:
- 터미널에 `INFO:     Uvicorn running on http://0.0.0.0:8000` 메시지 확인
- 브라우저에서 http://localhost:8000/health 접속 → `{"status":"healthy"}` 확인

### 2. Next.js 프론트엔드 실행
```bash
# 터미널 2 (새 터미널)
cd c:\Users\301\daiso-category-search-dev\daiso-category-search-dev\frontend
npm run dev
```

### 3. 테스트 페이지 접속
브라우저에서:
- http://localhost:3000/stt-test

## 🎯 페이지 기능

### 음성 녹음 모드
1. "⏺️ 녹음 시작" 버튼 클릭
2. 마이크 권한 허용
3. 음성 입력 (예: "화장실 어디에요")
4. "🔴 녹음 중지" 버튼 클릭
5. "🚀 STT 처리 시작" 버튼 클릭

### 파일 업로드 모드
1. "파일 선택" 버튼으로 WAV 파일 업로드
2. "🚀 STT 처리 시작" 버튼 클릭

## 📊 결과 화면

성공 시 표시되는 정보:
- **💬 최종 응답**: 사용자에게 보여줄 메시지
- **🎯 STT 결과**: 인식된 텍스트, 신뢰도, 처리 시간
- **✅ 품질 게이트**: OK/RETRY/FAIL 상태, 사유
- **🎯 정책 게이트**: PRODUCT_SEARCH/FIXED_LOCATION/UNSUPPORTED 분류

## 🧪 테스트 시나리오

### 1. FIXED_LOCATION (고정 위치)
- 입력: "화장실 어디에요"
- 예상 결과: "화장실은 매장 뒤쪽 왼편에 있습니다."

### 2. PRODUCT_SEARCH (상품 검색)
- 입력: "수세미 어디 있어요"
- 예상 결과: "[PRODUCT_SEARCH] 상품 '수세미 어디 있어요'을(를) 검색합니다..."

### 3. UNSUPPORTED (비지원)
- 입력: "배달되나요"
- 예상 결과: "이 서비스는 상품과 매장 내 위치 안내를 도와드리고 있어요..."

### 4. QUALITY_FAIL (품질 실패)
- 짧은 입력: "아"
- 예상 결과: RETRY/FAIL + "말씀을 잘 듣지 못했어요..."

## ⚠️ 문제 해결

### CORS 에러
백엔드 `main.py`의 CORS 설정 확인:
```python
allow_origins=["http://localhost:3000"]
```

### 마이크 권한 거부
브라우저 주소창 왼쪽 🔒 아이콘 → 사이트 설정 → 마이크 허용

### "서버 오류" 메시지
1. 백엔드 서버 실행 중인지 확인 (`python backend/main.py`)
2. http://localhost:8000/health 접속 확인
