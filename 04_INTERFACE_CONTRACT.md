# 04. 인터페이스 계약 (Team2 I/O)

본 서비스는 매장 내 곳곳에 설치된
태블릿에서 실행되는 웹 기반 AI 상품 위치 안내 서비스이다.


## 목적
Team2는 오디오 입력을 받아 STT 처리 후 품질게이트와 정규화를 거쳐  
**Team3/4가 바로 소비 가능한 텍스트 입력**을 제공한다.

본 인터페이스는 PoC 단계의 모델 비교와 실서비스 확장을 모두 고려한 계약이다.

※ STT 입력은 **PoC 1차 단계에서는 동일 조건 비교를 위해 음성 파일 기반으로 수행**하며,  
※ **실시간 Streaming 입력은 2차 확장 범위**로 정의한다.

---

## 입력 (Request)

```json
{
  "request_id": "uuid-or-string",
  "audio": {
    "content_type": "audio/wav",
    "path": "local-path-or-url-or-bytes-ref",
    "sample_rate": 16000
  },
  "attempt": 1,
  "meta": {
    "vad_applied": false,
    "device_noise_level": "unknown"
  }
}

```

---

## 출력 (Response)
```json
{
  "request_id": "same-as-input",
  "status": "OK | RETRY | FAIL",
  "stt": {
    "provider": "whisper | google | ...",
    "text_raw": "string-or-null",
    "confidence": 0.0 | null,
    "lang": "ko"
  },
  "normalized_text": "string-or-null",
  "quality_gate": {
    "is_usable": true,
    "reason": "EMPTY_TRANSCRIPT | TOO_SHORT | LOW_CONFIDENCE | NONSENSE_PATTERN | OK",
    "thresholds": {
      "min_confidence": 0.0,
      "min_chars": 0
    }
  },
  "latency_ms": {
    "stt": 0,
    "quality_gate": 0,
    "normalization": 0,
    "total": 0
  }
}
```
### 필드 설명
- confidence: STT provider가 제공하지 않는 경우 null 가능



### 상태 정의
- `OK`: `normalized_text` 제공(Team3/4로 전달 가능)
- `RETRY`: 사용자에게 “한 번만 더 말씀해 주세요”를 유도(attempt=2로 재호출)
  - RETRY 시에는 사용자 **재발화로 새 오디오 입력**을 받는다.
- `FAIL`: attempt=2에서도 불가 → 안전 종료(안내 메시지는 상위 레이어에서 처리)

---

## 필수 보장
- status는 항상 OK/RETRY/FAIL
- reason은 항상 비어있지 않음(설명 가능성)


