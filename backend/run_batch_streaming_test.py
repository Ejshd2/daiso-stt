#!/usr/bin/env python3
# backend/run_batch_streaming_test.py
"""
2차 PoC: Google Streaming STT 배치 테스트 (전처리/후처리 포함)
- 녹음 파일을 청크로 분할하여 Streaming API에 전송
- pacing 옵션: 실시간 유사(True) vs 순수 처리량(False)
- 전처리: 볼륨 정규화, VAD(무음 제거), denoise
- 후처리: 추임새 제거, 단위 정규화
- interim/final latency 측정
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Google Cloud Speech
try:
    from google.cloud import speech
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    print("[WARN] google-cloud-speech not installed")

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("[WARN] pydub not installed")

# Text postprocessor
try:
    from stt.text_postprocessor import TextPostprocessor
    POSTPROCESSOR_AVAILABLE = True
except ImportError:
    POSTPROCESSOR_AVAILABLE = False
    print("[WARN] TextPostprocessor not available")

# Noise reduction (optional)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

# VAD - silero or webrtcvad (optional)
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


def normalize_ko(text: str) -> str:
    """Normalize Korean text for comparison"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text


class StreamingBatchTestRunner:
    """Google Streaming STT 배치 테스트 러너 (전처리/후처리 포함)"""
    
    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    
    def __init__(
        self,
        input_dir: str = "data/test_audio/01_general",
        output_dir: str = "outputs",
        manifest_path: Optional[str] = None,
        credentials_path: str = "backend/daisoproject-sst.json",
        pacing: bool = True,
        chunk_ms: int = 100,
        config_path: str = "backend/config.yaml",
        # 전처리/후처리 옵션 (PoC 배치 전용)
        enable_postprocess: bool = True,
        enable_volume_normalize: bool = True,
        enable_vad: bool = False,
        enable_denoise: bool = False,
        target_dBFS: float = -20.0
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_path
        self.manifest_data = {}
        
        # Streaming 설정
        self.pacing = pacing
        self.chunk_ms = chunk_ms
        self.credentials_path = credentials_path
        
        # 전처리/후처리 옵션
        self.enable_postprocess = enable_postprocess
        self.enable_volume_normalize = enable_volume_normalize
        self.enable_vad = enable_vad
        self.enable_denoise = enable_denoise
        self.target_dBFS = target_dBFS
        
        # Load config
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Google client
        if GOOGLE_SPEECH_AVAILABLE:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            self.client = speech.SpeechClient()
            print(f"[OK] Google Speech client initialized")
        else:
            self.client = None
            print("[FAIL] Google Speech client not available")
        
        # Initialize postprocessor
        if POSTPROCESSOR_AVAILABLE and enable_postprocess:
            self.postprocessor = TextPostprocessor(
                config=self.config.get("postprocessing", {})
            )
            print("[OK] TextPostprocessor initialized")
        else:
            self.postprocessor = None
        
        # Load manifest
        if manifest_path:
            self.load_manifest(manifest_path)
        
        # Print settings
        print(f"[CONFIG] pacing={self.pacing}, chunk_ms={self.chunk_ms}")
        print(f"[CONFIG] postprocess={self.enable_postprocess}, volume_norm={self.enable_volume_normalize}")
        print(f"[CONFIG] vad={self.enable_vad}, denoise={self.enable_denoise}")
        print(f"[CONFIG] input_dir={self.input_dir}")
    
    def load_manifest(self, manifest_path: str):
        """Load test manifest TSV file"""
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if not lines:
                return
            
            headers = lines[0].strip().split("\t")
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                values = line.strip().split("\t")
                row = dict(zip(headers, values))
                
                audio_path = row.get("audio_path", "")
                if audio_path:
                    expected_text = row.get("expected_text", "")
                    gt_keywords_raw = row.get("gt_keywords", "")
                    gt_keywords = [k.strip() for k in gt_keywords_raw.split(",") if k.strip()]
                    utterance_type = row.get("utterance_type", "general")
                    
                    self.manifest_data[audio_path] = {
                        "expected_text": expected_text,
                        "gt_keywords": gt_keywords,
                        "utterance_type": utterance_type
                    }
            
            print(f"[OK] Loaded {len(self.manifest_data)} manifest entries")
            
        except Exception as e:
            print(f"[WARN] Failed to load manifest: {e}")
    
    def find_audio_files(self) -> List[Path]:
        """Find all audio files in input directory"""
        audio_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            audio_files.extend(self.input_dir.rglob(f"*{ext}"))
        return sorted(audio_files)
    
    def apply_volume_normalization(self, audio: AudioSegment) -> AudioSegment:
        """볼륨 정규화 적용"""
        if not self.enable_volume_normalize:
            return audio
        
        try:
            current_dBFS = audio.dBFS
            change_in_dBFS = self.target_dBFS - current_dBFS
            
            # 너무 큰 변화 제한
            if abs(change_in_dBFS) > 30:
                change_in_dBFS = 30 if change_in_dBFS > 0 else -30
            
            return audio.apply_gain(change_in_dBFS)
        except Exception:
            return audio
    
    def apply_denoise(self, audio: AudioSegment) -> AudioSegment:
        """노이즈 제거 적용 (파일 단위, 실시간 청크 X)"""
        if not self.enable_denoise or not NOISEREDUCE_AVAILABLE:
            return audio
        
        try:
            # AudioSegment → numpy array
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate
            
            # 노이즈 제거
            reduced = nr.reduce_noise(
                y=samples.astype(np.float32),
                sr=sample_rate,
                prop_decrease=0.8,
                stationary=True
            )
            
            # numpy array → AudioSegment
            return audio._spawn(reduced.astype(np.int16).tobytes())
        except Exception:
            return audio
    
    def apply_vad(self, audio: AudioSegment) -> AudioSegment:
        """VAD 적용 - 무음 구간 제거 (파일 단위)"""
        if not self.enable_vad:
            return audio
        
        # webrtcvad 사용 가능하면 사용
        if WEBRTCVAD_AVAILABLE:
            return self._apply_webrtc_vad(audio)
        
        # fallback: 간단한 임계값 기반 VAD
        return self._apply_simple_vad(audio)
    
    def _apply_webrtc_vad(self, audio: AudioSegment) -> AudioSegment:
        """WebRTC VAD 적용"""
        try:
            # 16kHz mono 16-bit로 변환
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            vad = webrtcvad.Vad(2)  # 0-3, higher = more aggressive
            frame_duration_ms = 30  # 10, 20, or 30
            bytes_per_frame = int(16000 * 2 * frame_duration_ms / 1000)
            
            raw = audio.raw_data
            speech_frames = []
            
            for i in range(0, len(raw) - bytes_per_frame, bytes_per_frame):
                frame = raw[i:i + bytes_per_frame]
                if len(frame) == bytes_per_frame:
                    is_speech = vad.is_speech(frame, 16000)
                    if is_speech:
                        speech_frames.append(frame)
            
            if not speech_frames:
                return audio  # VAD 결과 없으면 원본 반환
            
            # 말 구간만 합치기
            speech_audio = b"".join(speech_frames)
            return audio._spawn(speech_audio)
        except Exception:
            return audio
    
    def _apply_simple_vad(self, audio: AudioSegment) -> AudioSegment:
        """간단한 임계값 기반 VAD (fallback)"""
        try:
            # dBFS 기준으로 조용한 구간 제거
            silence_thresh = audio.dBFS - 16
            
            from pydub.silence import split_on_silence
            chunks = split_on_silence(
                audio,
                min_silence_len=500,
                silence_thresh=silence_thresh,
                keep_silence=100
            )
            
            if not chunks:
                return audio
            
            # 청크 합치기
            result = chunks[0]
            for chunk in chunks[1:]:
                result += chunk
            
            return result
        except Exception:
            return audio
    
    def prepare_audio(self, audio_path: Path) -> Tuple[bytes, int, Dict]:
        """
        오디오 전처리 파이프라인
        순서: 로드 → 볼륨 정규화 → denoise → VAD → 16kHz mono 변환
        """
        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub not available")
        
        preprocess_meta = {
            "volume_normalized": False,
            "denoised": False,
            "vad_applied": False,
            "original_duration_ms": 0,
            "processed_duration_ms": 0
        }
        
        # Load audio
        audio = AudioSegment.from_file(str(audio_path))
        preprocess_meta["original_duration_ms"] = len(audio)
        
        # 1. 볼륨 정규화
        if self.enable_volume_normalize:
            audio = self.apply_volume_normalization(audio)
            preprocess_meta["volume_normalized"] = True
        
        # 2. Denoise (파일 단위)
        if self.enable_denoise:
            audio = self.apply_denoise(audio)
            preprocess_meta["denoised"] = NOISEREDUCE_AVAILABLE
        
        # 3. VAD (무음 구간 제거)
        if self.enable_vad:
            audio = self.apply_vad(audio)
            preprocess_meta["vad_applied"] = True
        
        # 4. 16kHz mono로 변환
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        preprocess_meta["processed_duration_ms"] = len(audio)
        
        return audio.raw_data, len(audio), preprocess_meta
    
    def audio_generator(self, raw_bytes: bytes) -> Generator[bytes, None, None]:
        """Generate audio chunks with optional pacing"""
        bytes_per_ms = 16000 * 2 // 1000  # 32 bytes per ms
        chunk_bytes = self.chunk_ms * bytes_per_ms
        
        offset = 0
        while offset < len(raw_bytes):
            chunk = raw_bytes[offset:offset + chunk_bytes]
            yield chunk
            offset += chunk_bytes
            
            if self.pacing and offset < len(raw_bytes):
                time.sleep(self.chunk_ms / 1000)
    
    def run_streaming_stt(self, audio_path: Path) -> Dict:
        """Run streaming STT on a single file"""
        
        start_time = time.time()
        first_interim_time = None
        final_time = None
        
        interim_results = []
        final_text = ""
        confidence = 0.0
        error = None
        chunk_count = 0
        preprocess_meta = {}
        
        try:
            # Prepare audio with preprocessing
            raw_bytes, duration_ms, preprocess_meta = self.prepare_audio(audio_path)
            
            # Streaming config
            streaming_config = speech.StreamingRecognitionConfig(
                config=speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="ko-KR",
                    enable_automatic_punctuation=True,
                    model="default",
                ),
                interim_results=True,
                single_utterance=False,
            )
            
            # Create request generator
            def request_generator():
                nonlocal chunk_count
                for chunk in self.audio_generator(raw_bytes):
                    chunk_count += 1
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            
            # Call streaming_recognize
            responses = self.client.streaming_recognize(
                config=streaming_config,
                requests=request_generator()
            )
            
            # Process responses
            for response in responses:
                if not response.results:
                    continue
                
                result = response.results[0]
                transcript = result.alternatives[0].transcript if result.alternatives else ""
                
                if result.is_final:
                    final_time = time.time()
                    final_text = transcript
                    confidence = result.alternatives[0].confidence if result.alternatives else 0.0
                else:
                    if first_interim_time is None:
                        first_interim_time = time.time()
                    interim_results.append({
                        "text": transcript,
                        "time_ms": int((time.time() - start_time) * 1000)
                    })
        
        except Exception as e:
            error = str(e)
        
        # Calculate latencies
        first_interim_latency_ms = None
        final_latency_ms = None
        
        if first_interim_time:
            first_interim_latency_ms = int((first_interim_time - start_time) * 1000)
        if final_time:
            final_latency_ms = int((final_time - start_time) * 1000)
        
        total_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "final_text": final_text,
            "confidence": confidence,
            "interim_count": len(interim_results),
            "interim_results": interim_results[:5],
            "first_interim_latency_ms": first_interim_latency_ms,
            "final_latency_ms": final_latency_ms,
            "total_time_ms": total_time_ms,
            "chunk_count": chunk_count,
            "duration_ms": duration_ms if 'duration_ms' in dir() else None,
            "preprocess_meta": preprocess_meta,
            "error": error
        }
    
    def _check_keyword_hit(self, stt_text: str, gt_keywords: List[str]) -> Tuple[bool, List[str]]:
        """Check if any gt_keyword is in stt_text (OR logic)"""
        if not stt_text or not gt_keywords:
            return False, []
        
        normalized_stt = normalize_ko(stt_text)
        matched = []
        
        for kw in gt_keywords:
            normalized_kw = normalize_ko(kw)
            if normalized_kw and normalized_kw in normalized_stt:
                matched.append(kw)
        
        return len(matched) > 0, matched
    
    def _calculate_cer(self, expected: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        ref = normalize_ko(expected)
        hyp = normalize_ko(hypothesis)
        
        if not ref:
            return 0.0 if not hyp else 1.0
        
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        
        return round(d[len(ref)][len(hyp)] / len(ref), 4)
    
    def run_single_test(self, audio_path: Path, test_id: str, manifest_entry: Optional[Dict]) -> Dict:
        """Run single streaming test"""
        
        timestamp = datetime.now().isoformat()
        relative_path = str(audio_path.relative_to(self.input_dir.parent.parent)) if self.input_dir.parent.parent in audio_path.parents else str(audio_path)
        
        # Get manifest data
        expected_text = manifest_entry.get("expected_text") if manifest_entry else None
        gt_keywords = manifest_entry.get("gt_keywords", []) if manifest_entry else []
        utterance_type = manifest_entry.get("utterance_type", "unknown") if manifest_entry else "unknown"
        
        # Run streaming STT
        stt_result = self.run_streaming_stt(audio_path)
        
        # Check failure
        is_failure = (
            stt_result["error"] is not None or
            not stt_result["final_text"] or
            stt_result["final_text"].strip() == ""
        )
        
        # 후처리 적용
        final_text_raw = stt_result["final_text"]
        final_text_processed = final_text_raw
        postprocess_applied = False
        
        if not is_failure and self.postprocessor and final_text_raw:
            try:
                final_text_processed = self.postprocessor.postprocess(final_text_raw)
                postprocess_applied = True
            except Exception:
                final_text_processed = final_text_raw
        
        # Calculate metrics (후처리된 텍스트 기준)
        cer = None
        keyword_hit = None
        matched_keywords = []
        
        if not is_failure and expected_text:
            cer = self._calculate_cer(expected_text, final_text_processed)
        
        if not is_failure and gt_keywords:
            if final_text_processed and final_text_processed.strip():
                keyword_hit, matched_keywords = self._check_keyword_hit(final_text_processed, gt_keywords)
            else:
                keyword_hit = False
        
        # Build result
        result = {
            "test_id": test_id,
            "timestamp": timestamp,
            "audio_path": relative_path,
            "file_name": audio_path.name,
            "utterance_type": utterance_type,
            
            "expected_text": expected_text,
            "gt_keywords": gt_keywords,
            
            "provider": "google_streaming",
            "pacing": self.pacing,
            "chunk_ms": self.chunk_ms,
            
            # STT 결과 (원본 + 후처리)
            "stt_text": final_text_raw,
            "stt_text_processed": final_text_processed,
            "confidence": stt_result["confidence"],
            
            "is_failure": is_failure,
            "cer": cer,
            "keyword_hit": keyword_hit,
            "matched_keywords": matched_keywords,
            
            # Latency
            "first_interim_latency_ms": stt_result["first_interim_latency_ms"],
            "final_latency_ms": stt_result["final_latency_ms"],
            "total_time_ms": stt_result["total_time_ms"],
            "interim_count": stt_result["interim_count"],
            "chunk_count": stt_result["chunk_count"],
            
            # 전처리/후처리 메타
            "postprocess_applied": postprocess_applied,
            "preprocess_meta": stt_result.get("preprocess_meta", {}),
            
            "error": stt_result["error"]
        }
        
        return result
    
    def run_batch(self) -> str:
        """Run batch test on all audio files"""
        
        if not self.client:
            print("[FAIL] Google Speech client not available")
            return None
        
        audio_files = self.find_audio_files()
        
        if not audio_files:
            print(f"[FAIL] No audio files found in {self.input_dir}")
            return None
        
        print(f"\n[FILES] Found {len(audio_files)} audio files")
        print(f"[RUN] Google Streaming STT (pacing={self.pacing}, chunk_ms={self.chunk_ms})")
        print(f"[RUN] Options: postprocess={self.enable_postprocess}, vol_norm={self.enable_volume_normalize}, vad={self.enable_vad}, denoise={self.enable_denoise}\n")
        
        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        options_str = f"paced{'_vad' if self.enable_vad else ''}{'_denoise' if self.enable_denoise else ''}"
        output_file = self.output_dir / f"streaming_results_{timestamp}_{options_str}.jsonl"
        report_file = self.output_dir / f"streaming_report_{timestamp}_{options_str}.txt"
        
        all_results = []
        
        with open(output_file, "w", encoding="utf-8") as f:
            for file_idx, audio_file in enumerate(audio_files):
                test_id = f"stream_{timestamp}_{file_idx:04d}"
                
                # Find manifest entry
                relative_path = audio_file.relative_to(self.input_dir.parent).as_posix()
                manifest_entry = self.manifest_data.get(relative_path)
                
                print(f"Processing {file_idx + 1}/{len(audio_files)}: {audio_file.name}")
                
                result = self.run_single_test(audio_file, test_id, manifest_entry)
                all_results.append(result)
                
                # Write JSONL
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                
                # Print result summary
                if not result.get("is_failure"):
                    text = result.get("stt_text_processed", "")[:25] + "..." if len(result.get("stt_text_processed", "") or "") > 25 else result.get("stt_text_processed", "")
                    cer = result.get("cer")
                    cer_str = f"CER={cer:.2%}" if cer is not None else ""
                    kw_hit = "[KW]" if result.get("keyword_hit") else ""
                    final_ms = result.get("final_latency_ms") or 0
                    vad_str = "[VAD]" if result.get("preprocess_meta", {}).get("vad_applied") else ""
                    print(f"  [OK] \"{text}\" {cer_str} {kw_hit} {vad_str} final={final_ms}ms")
                else:
                    print(f"  [FAIL] {result.get('error', 'No transcription')}")
        
        # Generate report
        report = self._generate_report(all_results)
        
        # Save report
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n{'='*60}")
        print(report)
        print(f"{'='*60}")
        print(f"\n[SAVED] Results: {output_file}")
        print(f"[SAVED] Report: {report_file}")
        
        return str(output_file)
    
    def _generate_report(self, results: List[Dict]) -> str:
        """Generate summary report with preprocessing options comparison"""
        
        total = len(results)
        if total == 0:
            return "No results"
        
        # Filter valid results
        valid = [r for r in results if not r.get("is_failure")]
        failures = [r for r in results if r.get("is_failure")]
        
        # Calculate metrics
        with_cer = [r for r in valid if r.get("cer") is not None]
        with_kw = [r for r in valid if r.get("keyword_hit") is not None]
        
        avg_cer = sum(r["cer"] for r in with_cer) / len(with_cer) if with_cer else 0
        kw_hits = sum(1 for r in with_kw if r.get("keyword_hit"))
        kw_rate = kw_hits / len(with_kw) if with_kw else 0
        
        # Latency
        with_final = [r for r in valid if r.get("final_latency_ms")]
        avg_final = sum(r["final_latency_ms"] for r in with_final) / len(with_final) if with_final else 0
        
        # Duration reduction from VAD
        vad_results = [r for r in valid if r.get("preprocess_meta", {}).get("vad_applied")]
        if vad_results:
            orig_dur = sum(r["preprocess_meta"]["original_duration_ms"] for r in vad_results)
            proc_dur = sum(r["preprocess_meta"]["processed_duration_ms"] for r in vad_results)
            vad_reduction = (1 - proc_dur / orig_dur) * 100 if orig_dur > 0 else 0
        else:
            vad_reduction = 0
        
        report = f"""
======================================================================
[REPORT] Google Streaming STT Batch Test Results (with Preprocessing)
======================================================================
Test Settings:
  - pacing: {self.pacing}
  - chunk_ms: {self.chunk_ms}
  - input_dir: {self.input_dir}
  
Preprocessing Options:
  - volume_normalize: {self.enable_volume_normalize}
  - vad (silence removal): {self.enable_vad}
  - denoise: {self.enable_denoise}
  - postprocess: {self.enable_postprocess}
  
Summary:
  - Total files: {total}
  - Success: {len(valid)} ({len(valid)/total*100:.1f}%)
  - Failures: {len(failures)} ({len(failures)/total*100:.1f}%)

Accuracy (after postprocessing):
  - CER mean: {avg_cer:.2%} (n={len(with_cer)})
  - Keyword Hit: {kw_rate:.2%} ({kw_hits}/{len(with_kw)})

Latency:
  - Final result avg: {avg_final:.0f}ms (n={len(with_final)})
  
VAD Impact:
  - VAD applied: {len(vad_results)} files
  - Audio duration reduced: {vad_reduction:.1f}%

======================================================================
"""
        return report


def main():
    parser = argparse.ArgumentParser(description="Google Streaming STT Batch Test (with Preprocessing)")
    parser.add_argument("--input-dir", default="data/test_audio/01_general", help="Input audio directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--manifest", default="data/test_manifest.tsv", help="Manifest TSV file")
    parser.add_argument("--credentials", default="backend/daisoproject-sst.json", help="Google credentials path")
    parser.add_argument("--pacing", type=lambda x: x.lower() == 'true', default=True, help="Enable pacing (default: True)")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size in ms (default: 100)")
    
    # 전처리/후처리 옵션 (PoC 배치 전용)
    parser.add_argument("--postprocess", type=lambda x: x.lower() == 'true', default=True, help="Enable postprocessing (default: True)")
    parser.add_argument("--volume-norm", type=lambda x: x.lower() == 'true', default=True, help="Enable volume normalization (default: True)")
    parser.add_argument("--vad", type=lambda x: x.lower() == 'true', default=False, help="Enable VAD silence removal (default: False)")
    parser.add_argument("--denoise", type=lambda x: x.lower() == 'true', default=False, help="Enable noise reduction (default: False)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("[START] 2nd PoC: Google Streaming STT Batch Test (with Preprocessing)")
    print(f"{'='*60}")
    
    runner = StreamingBatchTestRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        credentials_path=args.credentials,
        pacing=args.pacing,
        chunk_ms=args.chunk_ms,
        enable_postprocess=args.postprocess,
        enable_volume_normalize=args.volume_norm,
        enable_vad=args.vad,
        enable_denoise=args.denoise
    )
    
    runner.run_batch()


if __name__ == "__main__":
    main()
