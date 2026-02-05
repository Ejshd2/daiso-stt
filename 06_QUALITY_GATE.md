# 06. 품질 게이트 스펙 (재요청 1회 정책)

## 목적
STT 결과가 “검색 입력으로 사용 가능한지”를 판정한다.  
기준 미달 시 **1회만 재요청(RETRY)**하며,  
2회차(attempt=2)에서도 기준 미달이면 **FAIL로 종료**한다.

본 품질게이트는 PoC 1차 단계 기준이며,  
임계값은 config로 분리하여 실험적으로 조정 가능하도록 설계한다.

---

## 1) 1차 규칙 (저비용 컷, 항상 적용)

아래 조건 중 하나라도 해당하면 **불가 판정**으로 간주한다.

- **R1. EMPTY_TRANSCRIPT**
  - `text_raw`가 None 또는 빈 문자열인 경우

- **R2. TOO_SHORT**
  - `len(text_raw) < MIN_CHARS`

- **R3. NONSENSE_PATTERN**
  - 의미 없는 반복, 잡음성 패턴에 해당
  - 예:
    - "ㅋㅋㅋ", "ㅎㅎㅎ", "ㅇㅇㅇ"
    - 특수문자 과다 (!!!, ??? 등)
    - NONSENSE_PATTERNS 정규식에 매칭되는 경우

---

## 2) 신뢰도 기반 규칙 (가능한 경우에만 적용)

STT provider가 confidence 또는 log-probability 기반 점수를 제공하는 경우에만 적용한다.

- **R4. LOW_CONFIDENCE**
  - `confidence < MIN_CONFIDENCE`

※ confidence가 제공되지 않는 경우(None),  
   해당 규칙은 **skip**하고 다른 규칙만으로 판정한다.

---

## 3) attempt별 분기 규칙 (단일 정책)

불가 판정(R1~R4 중 하나라도 해당) 시:

- `attempt = 1` → **RETRY**
- `attempt = 2` → **FAIL**

※ RETRY는 최대 1회만 허용하며, 루프는 금지한다.

---

## 4) reason 우선순위 (복수 규칙 해당 시)

복수 규칙이 동시에 해당될 경우,  
reason은 **R1 → R2 → R3 → R4** 순으로  
첫 매칭 규칙의 코드값을 사용한다.

---

## 5) 품질게이트 출력 reason (필수)

불가 판정 시 아래 **사전 정의된 코드값만 사용**한다.

- `EMPTY_TRANSCRIPT`
- `TOO_SHORT`
- `LOW_CONFIDENCE`
- `NONSENSE_PATTERN`
- `OK`

※ `NOISE_SUSPECTED`  
- PoC 1차에서는 미사용  
- 2차(Streaming) 단계에서 VAD/잡음 신호 기반 확장 예정

---

## 6) 임계값 설정 (config 분리)

임계값은 코드에 하드코딩하지 않고 config로 관리한다.

- `MIN_CONFIDENCE`
  - 예: `0.6` (PoC 초기 가정값, 10_DECISION_LOG.md 참조)

- `MIN_CHARS`
  - 예: `2` (10_DECISION_LOG.md 초기값 참조)

- `NONSENSE_PATTERNS`
  - 정규식 리스트
  - 예:
    - `(ㅋ|ㅎ){3,}`
    - `(ㅇ){3,}`
    - `[!?]{3,}`

---

## 범위 명시 (중요)

- 본 품질게이트는 **STT 결과의 사용 가능성만 판단**
- FAIL 이후의 사용자 안내, UI 처리, 재입력 유도 메시지는  
  **상위 레이어(Frontend/Integration)의 책임**


- 본 품질 게이트는 STT 결과의 **형식적/통계적 품질만 판단**하며,  
  해당 발화가 서비스 정책상 처리 대상인지 여부에 대한 판단은  
  `03_AI_QUERY_POLICY.md`에서 정의한 질문 허용 정책에서 수행한다.

- 즉, 품질 게이트 통과(OK)는  
  “검색 입력으로 사용 가능함”을 의미할 뿐,  
  “서비스에서 처리해야 함”을 의미하지는 않는다.
