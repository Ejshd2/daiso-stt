#!/usr/bin/env python3
# backend/test_fallback.py
"""
Whisper Fallback 테스트 스크립트
- 녹음 파일을 WebSocket으로 스트리밍 전송
- 시나리오 1: 정상 Google Streaming (fallback 없이 성공)
- 시나리오 2: 강제 Fallback (빈 오디오 전송 → Google 실패 → Whisper)
- 시나리오 3: 실제 오디오 + 짧은 timeout 시뮬레이션

사용법:
    1) 서버 시작: python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
    2) 테스트:   python backend/test_fallback.py
"""

import asyncio
import json
import base64
import time
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("❌ websockets 패키지가 필요합니다: pip install websockets")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("❌ pydub 패키지가 필요합니다: pip install pydub")
    sys.exit(1)

# 설정
WS_URL = "ws://localhost:8000/ws/stt"
SAMPLE_RATE = 16000
CHUNK_MS = 100  # 100ms per chunk
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # 32 bytes/ms

# 테스트용 오디오 파일 (상대경로)
TEST_AUDIO_DIR = Path("data/test_audio/01_general")


def load_audio_as_pcm16(audio_path: str) -> bytes:
    """오디오 파일을 16kHz mono PCM16 bytes로 변환"""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return audio.raw_data


def chunk_audio(pcm_bytes: bytes, chunk_ms: int = CHUNK_MS) -> list:
    """PCM bytes를 chunk_ms 단위로 분할"""
    chunk_bytes = chunk_ms * BYTES_PER_MS
    chunks = []
    for i in range(0, len(pcm_bytes), chunk_bytes):
        chunks.append(pcm_bytes[i:i + chunk_bytes])
    return chunks


async def run_scenario(
    name: str,
    audio_path: str = None,
    send_real_audio: bool = True,
    send_silence_only: bool = False,
    silence_duration_sec: float = 1.0,
    pacing: bool = True,
    force_fallback: bool = False
):
    """
    단일 테스트 시나리오 실행
    
    Args:
        name: 시나리오 이름
        audio_path: 오디오 파일 경로 (없으면 silence_only)
        send_real_audio: 실제 오디오 전송 여부
        send_silence_only: 묵음만 전송 (Google 실패 유도)
        silence_duration_sec: 묵음 전송 시간
        pacing: 실시간 pacing 여부
    """
    print(f"\n{'='*60}")
    print(f"🧪 시나리오: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        async with websockets.connect(WS_URL, max_size=10**7) as ws:
            # 1. Start 메시지 전송
            start_msg = {
                "type": "start",
                "meta": {
                    "run_id": f"fallback_test",
                    "test_id": f"test_{name}_{int(time.time())}",
                    "utterance_type": "general",
                    "spoken_text": name,
                    "save_audio": True,
                    "force_fallback": force_fallback
                }
            }
            await ws.send(json.dumps(start_msg))
            
            # started 응답 대기
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            resp = json.loads(response)
            print(f"  📡 서버 응답: {resp.get('type', '?')}")
            
            if resp.get("type") != "started":
                print(f"  ❌ 시작 실패: {resp}")
                return None
            
            # 2. 오디오 전송
            if send_silence_only:
                # 묵음만 전송 (Google이 인식할 내용 없음 → Fallback 유도)
                silence_bytes = b'\x00' * int(SAMPLE_RATE * 2 * silence_duration_sec)
                chunks = chunk_audio(silence_bytes)
                print(f"  🔇 묵음 전송: {silence_duration_sec}초 ({len(chunks)} chunks)")
            elif send_real_audio and audio_path:
                pcm = load_audio_as_pcm16(audio_path)
                chunks = chunk_audio(pcm)
                duration = len(pcm) / (SAMPLE_RATE * 2)
                print(f"  🎵 오디오 전송: {audio_path} ({duration:.1f}초, {len(chunks)} chunks)")
            else:
                print("  ❌ 오디오 없음")
                return None
            
            # 청크 전송
            for i, chunk in enumerate(chunks):
                audio_msg = {
                    "type": "audio",
                    "pcm_b64": base64.b64encode(chunk).decode("utf-8"),
                    "seq": i
                }
                await ws.send(json.dumps(audio_msg))
                
                if pacing:
                    await asyncio.sleep(CHUNK_MS / 1000)
            
            print(f"  ✅ 전송 완료 ({len(chunks)} chunks)")
            
            # 3. Stop 전송
            await ws.send(json.dumps({"type": "stop"}))
            print(f"  🛑 Stop 전송")
            
            # 4. 결과 수신 (interim + final)
            final_result = None
            interim_count = 0
            
            while True:
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=30)
                    resp = json.loads(response)
                    
                    if resp.get("type") == "interim":
                        interim_count += 1
                        if interim_count <= 3:
                            print(f"  💬 interim [{interim_count}]: {resp.get('text', '')[:40]}")
                    
                    elif resp.get("type") == "final":
                        final_result = resp
                        break
                    
                    elif resp.get("type") == "error":
                        print(f"  ❌ 에러: {resp.get('message', '')}")
                    
                except asyncio.TimeoutError:
                    print("  ⏰ Timeout (30초) - 응답 없음")
                    break
            
            # 5. 결과 출력
            total_time = int((time.time() - start_time) * 1000)
            
            if final_result:
                text = final_result.get("text", "")
                status = final_result.get("status", "?")
                meta = final_result.get("meta", {})
                text_raw = meta.get("text_raw", text)
                text_processed = meta.get("text_processed", text)
                fallback_used = meta.get("fallback_used", False)
                fallback_provider = meta.get("fallback_provider", "")
                fallback_latency = meta.get("fallback_latency_ms", 0)
                fallback_reason = meta.get("fallback_reason", "")
                confidence = meta.get("confidence", 0)
                
                print(f"\n  📊 결과:")
                print(f"     text_raw:       \"{text_raw}\"")
                print(f"     text_processed: \"{text_processed}\"")
                print(f"     최종 text:      \"{text}\"")
                print(f"     상태: {status}")
                print(f"     신뢰도: {confidence:.4f}")
                print(f"     Fallback 사용: {'✅ Yes' if fallback_used else '❌ No'}")
                if fallback_used:
                    print(f"     Fallback 제공자: {fallback_provider}")
                    print(f"     Fallback 이유: {fallback_reason}")
                    print(f"     Fallback 지연: {fallback_latency}ms")
                print(f"     총 소요: {total_time}ms")
                print(f"     Interim 수: {interim_count}")
                
                return {
                    "scenario": name,
                    "text_raw": text_raw,
                    "text_processed": text_processed,
                    "text": text,
                    "status": status,
                    "fallback_used": fallback_used,
                    "fallback_provider": fallback_provider,
                    "fallback_reason": fallback_reason,
                    "fallback_latency_ms": fallback_latency,
                    "total_time_ms": total_time,
                    "interim_count": interim_count
                }
            else:
                print(f"\n  ❌ Final 결과 없음 (총 {total_time}ms)")
                return None
                
    except ConnectionRefusedError:
        print(f"  ❌ 서버 연결 실패! 서버가 실행 중인지 확인하세요.")
        print(f"     python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000")
        return None
    except Exception as e:
        print(f"  ❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    print("="*60)
    print("🧪 Whisper Fallback 테스트")
    print("="*60)
    print(f"WebSocket URL: {WS_URL}")
    print(f"Chunk: {CHUNK_MS}ms, Sample Rate: {SAMPLE_RATE}Hz")
    
    # 테스트 오디오 파일
    audio_files = sorted(TEST_AUDIO_DIR.glob("*.m4a"))[:3]
    #audio_files = [Path(r"data/test_audio/06_phase3_extra/이선영_긴음성01.m4a")]
    
    if not audio_files or not audio_files[0].exists():
        print(f"\n❌ 테스트 오디오 파일이 없습니다: {audio_files[0]}")
        return
    
    print(f"\n📁 테스트 파일: {[f.name for f in audio_files]}")
    
    results = []
    
    # ============================================
    # 시나리오 1: 정상 Google Streaming
    # ============================================
    result = await run_scenario(
        name="정상_Google_Streaming",
        audio_path=str(audio_files[0]),
        send_real_audio=True,
        pacing=True,
        force_fallback=False
    )
    results.append(result)
    
    await asyncio.sleep(2)  # 세션 간 간격
    
    # ============================================
    # 시나리오 2: 강제 Fallback (같은 오디오, Whisper로)
    # ============================================
    result = await run_scenario(
        name="강제_Whisper_Fallback",
        audio_path=str(audio_files[0]),
        send_real_audio=True,
        pacing=True,
        force_fallback=True  # ⭐ Google 결과 무시 → Whisper로
    )
    results.append(result)
    
    # ============================================
    # 최종 요약
    # ============================================
    print(f"\n\n{'='*60}")
    print("📊 테스트 결과 비교")
    print(f"{'='*60}")
    
    for r in results:
        if r:
            fb = "✅ Fallback" if r.get("fallback_used") else "🟢 Google"
            raw = r.get("text_raw", "")[:40]
            processed = r.get("text_processed", "")[:40]
            ms = r.get("total_time_ms", 0)
            fb_ms = r.get("fallback_latency_ms", 0)
            fb_reason = r.get("fallback_reason", "")
            print(f"  [{r['scenario']}]")
            print(f"    프로바이더: {fb}")
            print(f"    raw:       \"{raw}\"")
            print(f"    processed: \"{processed}\"")
            print(f"    상태: {r.get('status', '?')}")
            print(f"    총 소요: {ms}ms")
            if fb_ms:
                print(f"    Fallback: {fb_reason} | {fb_ms}ms")
            print()
        else:
            print(f"  [실패] 결과 없음\n")
    
    print(f"{'='*60}")
    print("✅ 테스트 완료!")
    print(f"📄 서버 로그 및 CSV 확인: outputs/streaming_poc_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
