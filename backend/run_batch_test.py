# backend/run_batch_test.py
"""
STT Batch Comparison Test Runner (v2)
1차 테스트: 기본 음성 인식 정확도 검증

핵심 지표:
- CER (mean, p90)
- Keyword Hit Rate
- Failure Rate
- Latency (mean, p90)
+ 유형별 분해 (general/short/ambiguous/dialect/tts)
"""

import os
import sys
import json
import time
import math
import re
import argparse
import unicodedata
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Fix Windows console encoding for emoji/unicode
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from stt import get_adapter, QualityGate, PolicyGate, AudioConverter
from stt import TextPostprocessor, AudioPreprocessor
from stt.types import STTResult


# def normalize_ko(text: str) -> str:
#     """Normalize Korean text for comparison"""
#     if not text:
#         return ""
#     # Unicode normalization
#     text = unicodedata.normalize('NFC', text)
#     # Remove punctuation and special chars
#     text = re.sub(r'[^\w\s가-힣]', '', text)
#     # Lowercase and strip
#     text = text.lower().strip()
#     # Collapse multiple spaces
#     text = re.sub(r'\s+', ' ', text)
#     return text

def normalize_ko(text: str) -> str:
    """Normalize Korean text for comparison (evaluation-friendly)"""
    if not text:
        return ""

    # 1) Unicode normalize (compat)
    text = unicodedata.normalize("NFKC", text)

    # 2) lowercase
    text = text.lower()

    # 3) remove bracket tags like [펫], [행사], etc.
    text = re.sub(r"\[[^\]]+\]", "", text)

    # 4) remove all whitespace (important!)
    text = re.sub(r"\s+", "", text)

    # 5) keep only korean/english/number
    text = re.sub(r"[^0-9a-z가-힣]", "", text)

    return text


class CostCalculator:
    """Calculate STT API costs"""
    
    GOOGLE_RATES = {
        "default": 0.006,    # per 15 seconds
        "enhanced": 0.009,
        "premium": 0.012
    }
    
    @staticmethod
    def calculate(duration_sec: float, provider: str, model: str = "default") -> Dict:
        if provider == "whisper":
            return {
                "provider": "whisper",
                "calculated_cost_usd": 0.0,
                "notes": "Whisper is free (self-hosted)"
            }
        
        elif provider == "google":
            billing_unit = 15
            billable_units = max(1, math.ceil(duration_sec / billing_unit))
            rate = CostCalculator.GOOGLE_RATES.get(model, CostCalculator.GOOGLE_RATES["default"])
            cost = billable_units * rate
            
            return {
                "provider": "google",
                "calculated_cost_usd": round(cost, 6),
                "notes": f"{billable_units} units x ${rate}"
            }
        
        return {"provider": provider, "calculated_cost_usd": 0.0}


class BatchTestRunner:
    """Run batch STT comparison tests with manifest support"""
    
    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    
    def __init__(
        self,
        input_dir: str = "data/test_audio",
        output_dir: str = "outputs",
        config_path: str = "backend/config.yaml",
        manifest_path: Optional[str] = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_path
        self.manifest_data = {}
        
        # Initialize components
        self.converter = AudioConverter(output_dir=str(self.output_dir / "normalized"))
        
        # Load config
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize adapters
        print("[INIT] Initializing STT adapters...")
        self.whisper_adapter = get_adapter("whisper", **self.config["stt"]["whisper"])
        
        google_config = self.config["stt"].get("google", {})
        google_config["credentials_path"] = "backend/daisoproject-sst.json"
        self.google_adapter = get_adapter("google", **google_config)
        
        # Initialize gates
        self.quality_gate = QualityGate(**self.config["quality_gate"])
        
        # Initialize preprocessor/postprocessor (v2 고도화)
        self.preprocessor = AudioPreprocessor(
            output_dir=str(self.output_dir / "preprocessed")
        )
        self.postprocessor = TextPostprocessor(
            config=self.config.get("postprocessing", {})
        )
        
        print("[OK] All components initialized")
        
        # Load manifest if provided
        if manifest_path:
            self.load_manifest(manifest_path)
    
    def load_manifest(self, manifest_path: str):
        """Load test manifest TSV file"""
        print(f"[MANIFEST] Loading manifest: {manifest_path}")
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                audio_path = row.get('audio_path', '').strip()
                if audio_path:
                    # Normalize path to POSIX format (forward slashes)
                    normalized_path = Path(audio_path).as_posix()
                    self.manifest_data[normalized_path] = {
                        'expected_text': row.get('expected_text', '').strip(),
                        'gt_keywords': self._parse_keywords(row.get('gt_keywords', '')),
                        'utterance_type': row.get('utterance_type', 'unknown').strip()
                    }
        
        print(f"[OK] Loaded {len(self.manifest_data)} entries from manifest")
    
    # def _parse_keywords(self, keywords_str: str) -> List[str]:
    #     """Parse gt_keywords: comma split, strip, remove empty, filter by length >= 3"""
    #     if not keywords_str:
    #         return []
    #     keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
    #     # Only keep keywords with length >= 3 (excluding special chars)
    #     return [k for k in keywords if len(k) >= 2]  # Will check length >= 3 in matching
    
    def _parse_keywords(self, keywords_str: str) -> List[str]:
        if not keywords_str:
            return []
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
        # keep keywords that still have enough signal after normalization
        return [k for k in keywords if len(normalize_ko(k)) >= 3]
    
    def _check_keyword_hit(self, stt_text: str, gt_keywords: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if STT result contains any keywords (substring matching)
        Returns: (hit: bool, matched_keywords: list)
        """
        if not stt_text or not gt_keywords:
            return False, []
        
        stt_normalized = normalize_ko(stt_text)
        matched = []
        
        for kw in gt_keywords:
            kw_normalized = normalize_ko(kw)
            # Only check keywords with length >= 3
            if len(kw_normalized) >= 3 and kw_normalized in stt_normalized:
                matched.append(kw)
        
        return len(matched) > 0, matched
    
    def find_audio_files_from_manifest(self) -> List[Path]:
        """Find audio files based on manifest entries"""
        files = []
        for audio_path in self.manifest_data.keys():
            full_path = self.input_dir / audio_path
            if full_path.exists():
                files.append(full_path)
            else:
                print(f"⚠️ File not found: {full_path}")
        return sorted(files)
    
    def find_audio_files(self) -> List[Path]:
        """Find all supported audio files in input directory"""
        if self.manifest_data:
            return self.find_audio_files_from_manifest()
        
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_dir.rglob(f"*{ext}"))
            files.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        ref = normalize_ko(reference)
        hyp = normalize_ko(hypothesis)
        
        if len(ref) == 0:
            return 0.0 if len(hyp) == 0 else 1.0
        
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i-1] == hyp[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return round(d[len(ref)][len(hyp)] / len(ref), 4)
    
    def run_single_test(
        self,
        audio_path: Path,
        provider: str,
        test_id: str,
        manifest_entry: Optional[Dict] = None
    ) -> Dict:
        """Run single STT test and return detailed result"""
        
        timestamp = datetime.now().isoformat()
        relative_path = str(audio_path.relative_to(self.input_dir))
        
        # Get manifest data
        expected_text = manifest_entry.get('expected_text') if manifest_entry else None
        gt_keywords = manifest_entry.get('gt_keywords', []) if manifest_entry else []
        utterance_type = manifest_entry.get('utterance_type', 'unknown') if manifest_entry else 'unknown'
        
        try:
            # 0. 전처리 (v2 고도화: 볼륨 정규화 + 옵션 노이즈 제거)
            preprocess_config = self.config.get("preprocessing", {})
            try:
                preprocessed_path, preprocess_meta = self.preprocessor.preprocess(
                    str(audio_path),
                    preprocess_config,
                    test_id,
                    provider
                )
            except Exception:
                preprocessed_path = str(audio_path)
                preprocess_meta = {
                    "preprocessing_executed": False,
                    "preprocessing_time_ms": 0
                }
            
            # 1. Normalize audio (conversion time excluded from latency)
            conversion_result = self.converter.normalize(preprocessed_path)
            normalized_path = conversion_result["normalized_path"]
            audio_metadata = conversion_result["audio_metadata"]
            duration_sec = audio_metadata["normalized"]["duration_sec"]
            
            # 2. Select adapter
            adapter = self.whisper_adapter if provider == "whisper" else self.google_adapter
            model = self.config["stt"]["whisper"]["model_size"] if provider == "whisper" else "default"
            
            # 3. Run STT (latency measured inside adapter)
            stt_result = adapter.transcribe(normalized_path)
            
            # 4. Check failure: error/None/empty
            is_failure = (
                stt_result.error is not None or 
                stt_result.text_raw is None or 
                stt_result.text_raw.strip() == ""
            )
            
            # 5. 후처리 (v2 고도화: text_raw 보존, text_processed 생성)
            text_raw = stt_result.text_raw
            text_processed = None
            postprocess_executed = False
            postprocess_time_ms = 0
            
            if not is_failure and text_raw:
                import time as time_module
                post_start = time_module.time()
                try:
                    postprocess_config = self.config.get("postprocessing", {})
                    text_processed = self.postprocessor.postprocess(text_raw, postprocess_config)
                    postprocess_executed = True
                except Exception:
                    text_processed = text_raw  # fallback
                postprocess_time_ms = int((time_module.time() - post_start) * 1000)
            
            # 6. Calculate metrics
            cer = None
            keyword_hit = None
            matched_keywords = []
            
            # CER은 text_raw 기준 (원문 비교)
            if not is_failure and expected_text:
                cer = self._calculate_cer(expected_text, text_raw)
            
            # Keyword Hit은 text_processed 기준 (후처리 적용)
            if not is_failure and gt_keywords:
                # text_processed가 None/empty면 keyword_hit=False
                if text_processed and text_processed.strip():
                    keyword_hit, matched_keywords = self._check_keyword_hit(text_processed, gt_keywords)
                else:
                    keyword_hit = False
                    matched_keywords = []
            
            # 7. Calculate cost
            cost_info = CostCalculator.calculate(duration_sec, provider, model)
            
            # Build result
            result = {
                "test_id": test_id,
                "timestamp": timestamp,
                "audio_path": relative_path,
                "file_name": audio_path.name,
                "utterance_type": utterance_type,
                
                "expected_text": expected_text,
                "gt_keywords": gt_keywords,
                
                "provider": provider,
                "model": model,
                
                "stt_text": text_raw,
                "stt_text_processed": text_processed,  # v2 신규
                "confidence": stt_result.confidence,
                "latency_ms": stt_result.latency_ms,
                "error": stt_result.error,
                
                "is_failure": is_failure,
                "cer": cer,
                "keyword_hit": keyword_hit,
                "matched_keywords": matched_keywords,
                
                "audio_duration_sec": duration_sec,
                "cost_usd": cost_info["calculated_cost_usd"],
                
                # v2 고도화 메타데이터
                "preprocessing_executed": preprocess_meta.get("preprocessing_executed", False),
                "preprocessing_time_ms": preprocess_meta.get("preprocessing_time_ms", 0),
                "postprocessing_executed": postprocess_executed,
                "postprocessing_time_ms": postprocess_time_ms
            }
            
            return result
            
        except Exception as e:
            return {
                "test_id": test_id,
                "timestamp": timestamp,
                "audio_path": relative_path,
                "file_name": audio_path.name,
                "utterance_type": utterance_type,
                "provider": provider,
                "is_failure": True,
                "error": str(e),
                "cer": None,
                "keyword_hit": None,
                "latency_ms": None,
                "stt_text_processed": None,
                "preprocessing_executed": False,
                "postprocessing_executed": False
            }
    
    def run_batch(self, providers: List[str] = None) -> str:
        """Run batch test on all audio files"""
        
        if providers is None:
            providers = ["whisper", "google"]
        
        audio_files = self.find_audio_files()
        
        if not audio_files:
            print(f"[FAIL] No audio files found in {self.input_dir}")
            return None
        
        print(f"\n[FILES] Found {len(audio_files)} audio files")
        print(f"[RUN] Testing with providers: {providers}\n")
        
        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"batch_results_{timestamp}.jsonl"
        report_file = self.output_dir / f"batch_report_{timestamp}.txt"
        
        all_results = []
        total_tests = len(audio_files) * len(providers)
        completed = 0
        
        with open(output_file, "w", encoding="utf-8") as f:
            for file_idx, audio_file in enumerate(audio_files):
                # Normalize path to POSIX format for manifest matching
                relative_path = audio_file.relative_to(self.input_dir).as_posix()
                manifest_entry = self.manifest_data.get(relative_path)
                
                # Debug: Log unmatched files
                if manifest_entry is None and file_idx < 5:
                    print(f"[DEBUG] No manifest for: {relative_path}")
                
                for provider in providers:
                    test_id = f"test_{timestamp}_{file_idx:04d}_{provider}"
                    
                    print(f"Processing {completed + 1}/{total_tests}: {audio_file.name} [{provider}]")
                    
                    result = self.run_single_test(audio_file, provider, test_id, manifest_entry)
                    all_results.append(result)
                    
                    # Write JSONL
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    
                    # Print result summary
                    if not result.get("is_failure"):
                        text = result.get("stt_text", "")[:25] + "..." if len(result.get("stt_text", "")) > 25 else result.get("stt_text", "")
                        cer = result.get("cer")
                        cer_str = f"CER={cer:.2%}" if cer is not None else ""
                        kw_hit = "[KW]" if result.get("keyword_hit") else ""
                        # v2: PREP/POST 상태 표시
                        prep_status = "[PREP]" if result.get("preprocessing_executed") else ""
                        post_status = "[POST]" if result.get("postprocessing_executed") else ""
                        print(f"  [OK] \"{text}\" {cer_str} {kw_hit} {prep_status}{post_status}")
                    else:
                        print(f"  [FAIL] {result.get('error', 'No transcription')}")
                    
                    completed += 1
                
                # Checkpoint
                if (file_idx + 1) % 20 == 0:
                    print(f"\n[CHECKPOINT] {file_idx + 1}/{len(audio_files)} files\n")
        
        # Generate report
        report = self._generate_report(all_results, providers)
        
        # Save report
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        # Print report
        print("\n" + report)
        print(f"\n[RESULTS] {output_file}")
        print(f"[REPORT] {report_file}")
        
        return str(output_file)
    
    def _generate_report(self, results: List[Dict], providers: List[str]) -> str:
        """Generate detailed report with metrics by provider and utterance type"""
        
        lines = []
        lines.append("=" * 70)
        lines.append("[REPORT] 1st Batch STT Accuracy Test Results")
        lines.append("=" * 70)
        lines.append(f"테스트 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"총 파일 수: {len(results) // len(providers)}")
        lines.append(f"총 테스트 수: {len(results)}")
        lines.append("")
        
        # Group results by provider
        by_provider = defaultdict(list)
        for r in results:
            by_provider[r.get("provider", "unknown")].append(r)
        
        # Overall metrics by provider
        lines.append("-" * 70)
        lines.append("[METRICS] Overall Metrics by Provider")
        lines.append("-" * 70)
        lines.append(f"{'Provider':<12} {'CER mean':>10} {'CER p90':>10} {'KW Hit':>10} {'Fail':>10} {'Latency':>12} {'Cost':>10}")
        lines.append("-" * 70)
        
        for provider in providers:
            provider_results = by_provider[provider]
            metrics = self._calculate_metrics(provider_results)
            
            lines.append(
                f"{provider:<12} "
                f"{metrics['cer_mean']:>9.2%} "
                f"{metrics['cer_p90']:>9.2%} "
                f"{metrics['kw_hit_rate']:>9.2%} "
                f"{metrics['failure_rate']:>9.2%} "
                f"{metrics['latency_mean']:>8.0f}ms "
                f"${metrics['total_cost']:>8.4f}"
            )
        
        lines.append("-" * 70)
        lines.append("")
        
        # Metrics by utterance type
        utterance_types = ["general", "short", "ambiguous", "dialect", "tts"]
        
        lines.append("-" * 70)
        lines.append("[CER] CER by Utterance Type")
        lines.append("-" * 70)
        lines.append(f"{'Type':<15} " + " ".join(f"{p:>12}" for p in providers))
        lines.append("-" * 70)
        
        for utype in utterance_types:
            row = f"{utype:<15} "
            for provider in providers:
                type_results = [r for r in by_provider[provider] if r.get("utterance_type") == utype]
                if type_results:
                    metrics = self._calculate_metrics(type_results)
                    row += f"{metrics['cer_mean']:>11.2%} "
                else:
                    row += f"{'N/A':>12} "
            lines.append(row)
        
        lines.append("-" * 70)
        lines.append("")
        
        # Keyword Hit by type
        lines.append("-" * 70)
        lines.append("[KEYWORD] Keyword Hit Rate by Type")
        lines.append("-" * 70)
        lines.append(f"{'Type':<15} " + " ".join(f"{p:>12}" for p in providers))
        lines.append("-" * 70)
        
        for utype in utterance_types:
            row = f"{utype:<15} "
            for provider in providers:
                type_results = [r for r in by_provider[provider] if r.get("utterance_type") == utype]
                if type_results:
                    metrics = self._calculate_metrics(type_results)
                    row += f"{metrics['kw_hit_rate']:>11.2%} "
                else:
                    row += f"{'N/A':>12} "
            lines.append(row)
        
        lines.append("-" * 70)
        lines.append("")
        
        # Failure cases
        lines.append("-" * 70)
        lines.append("[FAILURES] Failure Cases")
        lines.append("-" * 70)
        
        for provider in providers:
            failures = [r for r in by_provider[provider] if r.get("is_failure")]
            lines.append(f"\n[{provider}] 실패: {len(failures)}건")
            for f in failures[:10]:  # Show max 10
                lines.append(f"  - {f.get('file_name')}: {f.get('error', 'No result')}")
            if len(failures) > 10:
                lines.append(f"  ... and {len(failures) - 10} more")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("[CONCLUSION]")
        lines.append("=" * 70)
        
        # Determine winner
        whisper_metrics = self._calculate_metrics(by_provider["whisper"]) if by_provider["whisper"] else {}
        google_metrics = self._calculate_metrics(by_provider["google"]) if by_provider["google"] else {}
        
        if whisper_metrics and google_metrics:
            cer_winner = "Whisper" if whisper_metrics["cer_mean"] < google_metrics["cer_mean"] else "Google"
            kw_winner = "Whisper" if whisper_metrics["kw_hit_rate"] > google_metrics["kw_hit_rate"] else "Google"
            latency_winner = "Google" if whisper_metrics["latency_mean"] > google_metrics["latency_mean"] else "Whisper"
            
            lines.append(f"- CER 최저: {cer_winner}")
            lines.append(f"- Keyword Hit 최고: {kw_winner}")
            lines.append(f"- Latency 최저: {latency_winner}")
            lines.append(f"- 비용: Whisper $0 / Google ${google_metrics['total_cost']:.4f}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics for a set of results"""
        
        if not results:
            return {
                "cer_mean": 0.0, "cer_p90": 0.0,
                "kw_hit_rate": 0.0, "failure_rate": 0.0,
                "latency_mean": 0.0, "latency_p90": 0.0,
                "total_cost": 0.0
            }
        
        # Filter out None values for each metric
        cer_values = [r["cer"] for r in results if r.get("cer") is not None]
        kw_hits = [r.get("keyword_hit") for r in results if r.get("keyword_hit") is not None]
        latency_values = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
        failures = [r for r in results if r.get("is_failure")]
        costs = [r.get("cost_usd", 0) for r in results if r.get("cost_usd") is not None]
        
        return {
            "cer_mean": np.mean(cer_values) if cer_values else 0.0,
            "cer_p90": np.percentile(cer_values, 90) if cer_values else 0.0,
            "kw_hit_rate": sum(1 for k in kw_hits if k) / len(kw_hits) if kw_hits else 0.0,
            "failure_rate": len(failures) / len(results) if results else 0.0,
            "latency_mean": np.mean(latency_values) if latency_values else 0.0,
            "latency_p90": np.percentile(latency_values, 90) if latency_values else 0.0,
            "total_cost": sum(costs)
        }


def main():
    parser = argparse.ArgumentParser(description="STT Batch Comparison Test (v2)")
    parser.add_argument("--input", "-i", default="data/test_audio",
                        help="Input directory with audio files")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Output directory for results")
    parser.add_argument("--manifest", "-m", default=None,
                        help="Manifest TSV file with expected_text and gt_keywords")
    parser.add_argument("--providers", "-p", nargs="+", default=["whisper", "google"],
                        choices=["whisper", "google"],
                        help="STT providers to test")
    
    args = parser.parse_args()
    
    print("[START] STT Batch Comparison Test (v2)")
    print("=" * 50)
    print("[METRICS] CER, Keyword Hit, Failure Rate, Latency")
    print("=" * 50)
    
    runner = BatchTestRunner(
        input_dir=args.input,
        output_dir=args.output,
        manifest_path=args.manifest
    )
    
    result_file = runner.run_batch(providers=args.providers)
    
    if result_file:
        print(f"\n[DONE] Test completed! Results: {result_file}")


if __name__ == "__main__":
    main()
