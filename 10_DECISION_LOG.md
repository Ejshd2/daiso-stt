# 10. Decision Log (STT/품질게이트 변경 이력)

## 변경 기록

### 2026-01-15 — 초기 기준선(Baseline) 설정
- providers: whisper vs google
- attempt 정책: 1회 RETRY 후 FAIL

#### 품질 게이트 임계값 (초기 가설값)
- MIN_CONFIDENCE: 0.6
- MIN_CHARS: 2

#### 테스트 규모
- n_cases: 120 (08_TEST_PLAN 기준)

#### 성능 목표 (초기)
- p95_latency_target: 2.0초 (보수적 시작값)

#### 결과 지표 (측정 후 업데이트)
- success_rate: TBD
- retry_rate: TBD
- fail_rate: TBD
- p95_latency_ms: TBD

#### 메모
- 초기 PoC 단계로, 임계값은 보수적 가설값으로 설정
- 실제 측정 결과에 따라 임계값/목표치는 조정 예정