# 11. Vibe Coding 프롬프트 세트 (Team2)

※ PoC 1차 단계에서는 동일 조건 비교를 위해  
   Whisper large-v3와 Google Speech-to-Text를  
   **음성 파일 기반(batch)**으로 테스트한다.  
   (실시간 Streaming 연동은 2차 확장 범위)

---

## P1) STT 어댑터 골격
"""
다음 파일을 생성/수정해줘: backend/stt/stt_adapters.py

목표:
- STT 제공자 교체 가능한 어댑터 구조 구현

요구:
- BaseAdapter 인터페이스 정의
  - transcribe(audio_path: str) -> STTResult

- STTResult 타입 정의 (backend/stt/types.py)
  - text_raw: str | None
  - confidence: float | None
  - lang: str | None
  - latency_ms: int
  - error: str | None

- WhisperAdapter, GoogleAdapter 구현
  - 실제 API 호출은 TODO 가능
  - latency_ms 반드시 측정

- 예외 발생 시:
  - text_raw=None
  - confidence=None
  - error에 사유 저장
  - latency_ms는 측정값 유지

- provider factory:
  - get_adapter(provider: str) -> BaseAdapter
  - provider는 "whisper", "google"
  - 미지원 provider 입력 시 ValueError 발생

Done 기준:
- get_adapter("whisper") 호출 시 정상 객체 반환
"""

---

## P2) 품질게이트
"""
다음 파일을 생성/수정해줘: backend/stt/quality_gate.py

입력:
- text_raw: str | None
- confidence: float | None
- attempt: int (1 또는 2만 허용)
- config:
  - min_chars
  - min_confidence
  - nonsense_patterns

출력:
- status: "OK" | "RETRY" | "FAIL"
- is_usable: bool
- reason: str (코드값만 사용)

reason 코드:
- EMPTY_TRANSCRIPT
- TOO_SHORT
- LOW_CONFIDENCE
- NONSENSE_PATTERN
- OK

규칙:
- EMPTY / TOO_SHORT / NONSENSE / LOW_CONF 중 하나라도 해당 시 불가
- attempt=1 → RETRY
- attempt=2 → FAIL
- confidence=None이면 LOW_CONFIDENCE 규칙 skip

- 복수 조건 해당 시 reason 우선순위:
  EMPTY_TRANSCRIPT → TOO_SHORT → NONSENSE_PATTERN → LOW_CONFIDENCE

Done 기준:
- 최소 4가지 케이스에 대해 기대한 status/reason 반환
"""

---

## P3) 정규화
"""
다음 파일을 생성/수정해줘: backend/stt/normalization.py

입력:
- text_raw: str

출력:
- normalized_text: str
- applied_rules: list[str]

규칙:
- 필러 제거
- 연속 공백 축소
- 반복 문자 축약
- 간단 치환 규칙(dict 기반)

주의:
- text_raw는 수정하지 않는다
- normalized_text만 가공

추가 규칙:
- 정규화 결과가 빈 문자열이면
  normalized_text = text_raw 로 fallback

Done 기준:
- 예제 입력에 대해 normalized_text / applied_rules 정상 반환
"""

---

## P4) 배치 러너
"""
다음 파일을 생성/수정해줘: backend/stt/run_batch_stt.py

입력:
- data/audio_manifest.tsv
  - case_id
  - audio_path
  - expected_text (optional)
  - type

흐름:
- provider ("whisper", "google") 순회
- 처리 절차:
  1) STT 호출
  2) quality_gate(attempt=1)
  3) RETRY면 attempt=2 시뮬레이션
  4) OK인 경우에만 normalization

주의:
- attempt=2는 동일 파일 재처리로 시뮬레이션될 수 있음
- 실서비스의 재발화(새 오디오)와는 다름

출력:
- outputs/stt_results.csv
  - case_id, provider, attempt
  - status, reason
  - text_raw, confidence, normalized_text
  - latency_ms

KPI 콘솔 출력:
- success_rate
- retry_rate
- fail_rate
- p95_latency_ms

옵션:
- --dry-run (텍스트 시뮬레이션)

Done 기준:
- outputs/stt_results.csv 생성
- KPI 콘솔 출력 확인
"""
