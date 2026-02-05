# PoC Team2 사전 문서 팩 (STT & 품질게이트 / Vibe Coding용)

※ PoC 1차에서는 동일 조건 비교를 위해 음성 파일 기반 테스트만 수행하며,
실시간 Streaming STT는 2차 확장 범위로 둔다.


- 날짜: 2026-01-16
- 범위: Agentic Workflow 중 **STT(음성→텍스트)** + **품질게이트(입력 유효성/재요청 1회)** + **텍스트 정규화(전처리)**

## Team2의 PoC 목표(한 줄)
“매장 소음/사투리/짧은 발화” 상황에서도 **검색 가능한 텍스트 입력**을 안정적으로 제공하고,
실패 시에는 **1회 재요청 또는 안전 종료**가 되도록 만든다.

## 권장 진행 순서
1) `01_MISSION.md`로 합격기준 고정(정확도/지연/재요청 정책)
2) `04_INTERFACE_CONTRACT.md`로 Team1(오디오) ↔ Team2(텍스트) ↔ Team3/4(검색) 계약 고정
3) `05_STT_ADAPTERS.md`로 STT 2개(Whisper vs Google) 어댑터 골격 구현
4) `06_QUALITY_GATE.md`로 품질게이트(1회 재요청) 구현
5) `07_NORMALIZATION.md`로 정규화 최소 규칙 적용
6) `08_TEST_PLAN.md`대로 배치 테스트 → `stt_results.csv`/KPI 출력
7) 최종 선택/임계값 변경은 `10_DECISION_LOG.md`에 기록

## 최종 산출물(체크)
- `stt_results.csv` (케이스별 STT 결과/신뢰도/재요청 여부)
- `stt_kpi_summary.md` (정확도/재요청률/실패율/지연)
- `10_DECISION_LOG.md` (모델 선택/임계값 근거)
