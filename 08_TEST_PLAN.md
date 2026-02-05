# 08. 테스트 계획 (STT / 품질게이트 PoC)

## 0) 1차 / 2차 범위 정의

- **1차(필수)**  
  동일 음성 파일(오디오) 세트로  
  Whisper large-v3 vs Google Speech-to-Text를 공정 비교

- **2차(확장)**  
  마이크 입력 기반 Streaming (실시간 UX / 지연 / 부분 결과) 검증

- **0차(선검증)**  
  오디오 파일이 준비되지 않은 경우에 한해,  
  텍스트 시뮬레이션으로 품질게이트 / 정규화 로직을 먼저 검증한다.

---

## 1) 테스트 데이터 구성 (최소 120케이스)

- 자연 발화(일반): 30
- 사투리 / 발음 불명확: 30
- 설명형 발화(짧은 문구): 30
- 짧은 발화 / 잡음: 20
- 범위 밖 / 엉뚱한 발화: 10

---

## 2) 테스트 단계 전략

### 0차 — 로직 선검증
- 텍스트 시뮬레이션 20케이스
- 목적:
  - 품질게이트 규칙 검증
  - 정규화 규칙 검증

### 1차 — STT 모델 비교 (필수)
- 동일 음성 파일 세트로 Whisper vs Google STT 비교
- 공정성 확보를 위해 Streaming 미사용

### 2차 — Streaming 확장 (선택)
- 마이크 입력 기반 실시간 UX / 지연 / 부분 결과 평가

---

## 3) attempt 정책 (배치 테스트 기준)

- PoC 배치 테스트에서는 **attempt=1 결과를 중심으로 측정**
- attempt=2 (재발화 기반 재시도)는
  - 배치에서 재현이 어려우므로
  - **수동 시나리오 테스트로 별도 검증**

※ 배치 러너에서 attempt=2가 실행되는 경우,  
   동일 파일 재처리로 시뮬레이션될 수 있으며  
   이는 **로직 검증 목적**이다.  
   실서비스에서는 RETRY 시 **재발화(새 오디오)**를 전제로 한다.

---

## 4) 측정 지표 (KPI)

- **WER (Word Error Rate)**  
  - 1차 자동 측정 메트릭

- **의미 보존 성공률 (보조 메트릭)**  
  - 전체 중 30케이스 샘플링
  - 수동 판정 (핵심 키워드 포함 여부)

- retry_rate  
  - attempt=1에서 RETRY가 발생한 비율

- fail_rate  
  - 수동 시나리오 테스트 기준

- 오판률
  - false_retry_rate: usable인데 RETRY로 간 비율
  - false_ok_rate: unusable인데 OK로 간 비율 (가능 시)

- p95_latency_ms  
  - 오디오 입력 → STT 결과 수신까지

- reason 분포
  - EMPTY / TOO_SHORT / LOW_CONFIDENCE / NONSENSE_PATTERN

---

## 5) 합격 리포트 산출물

- `stt_results.csv`
  - case_id, provider, attempt, status, reason
  - text_raw, confidence, normalized_text
  - latency_ms

- `stt_kpi_summary.md`
  - KPI 요약
  - 모델 선택 결론 및 근거

- `10_DECISION_LOG.md`
  - 임계값 / 모델 / 정책 변경 이력

---

## 6) expected_text 라벨링 기준

- expected_text는 **필러 / 잡음 제거 후의 ‘의도된 발화’**를 기준으로 작성한다.
- 실제 발화 원문과 1:1 매칭을 요구하지 않는다.
