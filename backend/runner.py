# backend/runner.py
"""
CLI Runner for STT Pipeline
Processes text simulation TSV (no audio files needed for PoC Phase 1)
"""

import sys
import json
import csv
import time
from pathlib import Path
from typing import List, Dict
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from stt import QualityGate, PolicyGate
from stt.types import STTResult, QualityGateResult, PolicyIntent


def load_config(config_path: str = "backend/config.yaml") -> dict:
    """Load configuration from YAML"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate_stt(text_raw: str) -> STTResult:
    """Simulate STT result from text (for testing without audio)"""
    # Simulate confidence based on text length (heuristic)
    confidence = None
    if text_raw and len(text_raw) > 0:
        confidence = min(0.95, 0.5 + len(text_raw) * 0.05)
    
    return STTResult(
        text_raw=text_raw,
        confidence=confidence,
        lang="ko",
        latency_ms=100,  # Simulated
        error=None
    )


def process_test_cases(
    tsv_path: str,
    output_path: str,
    config: dict
):
    """Process test cases from TSV and save results as JSON"""
    
    # Initialize gates
    quality_gate = QualityGate(**config["quality_gate"])
    policy_gate = PolicyGate(
        fixed_locations=config["policy_gate"]["fixed_locations"],
        unsupported_patterns=config["policy_gate"]["unsupported_patterns"]
    )
    
    # Load test cases
    test_cases = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        test_cases = list(reader)
    
    # Process each case
    results = []
    for case in test_cases:
        case_id = case["case_id"]
        text_raw = case["text_raw"]
        expected_intent = case.get("expected_intent", "")
        
        print(f"Processing {case_id}: '{text_raw}'...")
        
        # Simulate STT
        stt_result = simulate_stt(text_raw)
        
        # Quality Gate (attempt=1)
        quality_result = quality_gate.evaluate(stt_result, attempt=1)
        
        # Policy Gate (only if OK)
        policy_intent = None
        final_response = ""
        
        if quality_result.status == "OK":
            policy_intent = policy_gate.classify(text_raw)
            
            if policy_intent.intent_type == "FIXED_LOCATION":
                for loc in config["policy_gate"]["fixed_locations"]:
                    if loc["target"] == policy_intent.location_target:
                        final_response = loc["response"]
                        break
            elif policy_intent.intent_type == "UNSUPPORTED":
                final_response = config["policy_gate"]["fallback_message"]
            else:  # PRODUCT_SEARCH
                final_response = f"[PRODUCT_SEARCH] '{text_raw}' 검색 예정"
        elif quality_result.status == "RETRY":
            final_response = config["policy_gate"]["retry_message"]
        else:
            final_response = "음성 인식 실패 (FAIL)"
        
        # Collect result
        result = {
            "case_id": case_id,
            "input_text": text_raw,
            "expected_intent": expected_intent,
            "stt": {
                "text_raw": stt_result.text_raw,
                "confidence": stt_result.confidence,
                "error": stt_result.error
            },
            "quality_gate": {
                "status": quality_result.status,
                "is_usable": quality_result.is_usable,
                "reason": quality_result.reason
            },
            "policy_intent": {
                "intent_type": policy_intent.intent_type if policy_intent else None,
                "location_target": policy_intent.location_target if policy_intent else None,
                "confidence": policy_intent.confidence if policy_intent else None,
                "reason": policy_intent.reason if policy_intent else None
            } if policy_intent else None,
            "final_response": final_response
        }
        results.append(result)
        
        print(f"  → Quality: {quality_result.status} ({quality_result.reason})")
        if policy_intent:
            print(f"  → Intent: {policy_intent.intent_type}")
        print(f"  → Response: {final_response}\n")
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    quality_stats = {}
    intent_stats = {}
    
    for r in results:
        q_status = r["quality_gate"]["status"]
        quality_stats[q_status] = quality_stats.get(q_status, 0) + 1
        
        if r["policy_intent"]:
            intent_type = r["policy_intent"]["intent_type"]
            intent_stats[intent_type] = intent_stats.get(intent_type, 0) + 1
    
    print(f"Total cases: {len(results)}")
    print(f"\nQuality Gate:")
    for status, count in quality_stats.items():
        print(f"  {status}: {count}")
    
    print(f"\nPolicy Intent (OK cases only):")
    for intent, count in intent_stats.items():
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    config = load_config()
    
    tsv_path = "data/test_cases.tsv"
    output_path = "outputs/test_results.json"
    
    if not Path(tsv_path).exists():
        print(f"❌ Test cases file not found: {tsv_path}")
        sys.exit(1)
    
    process_test_cases(tsv_path, output_path, config)
