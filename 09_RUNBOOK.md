# 09. 실행 Runbook (Team2)

## 환경 변수(예시)
- `GOOGLE_STT_API_KEY=...`
- (로컬 Whisper 사용 시) 모델/가중치 경로 또는 설치 안내

## 단건 실행(예시)
- `python run_one_stt.py --audio ./samples/case_001.wav --provider whisper`

## 배치 실행(예시)
- `python run_batch_stt.py --manifest ./testdata/audio_manifest.tsv --out stt_results.csv`

## 디버깅 순서
1) audio 재생해서 실제 발화 확인(라벨 오류 제거)
2) STT provider 별 text_raw 비교
3) confidence/규칙으로 RETRY가 기대대로 발생하는지 확인
4) normalized_text가 검색에 유리하게 바뀌는지 확인(과변환 방지)

## 제출 체크리스트
- [ ] stt_results.csv 생성됨
- [ ] stt_kpi_summary.md 작성됨
- [ ] decision log 업데이트됨
