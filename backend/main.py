# backend/main.py
"""
FastAPI Server for STT Pipeline
PoC Phase 1: Batch audio processing with Whisper + Google comparison
"""

import yaml
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import time
import uuid

import sys
sys.path.append(str(Path(__file__).parent))

from stt import get_adapter, QualityGate, PolicyGate, AudioConverter
from stt.types import (
    PipelineResult, STTResult, QualityGateResult, PolicyIntent,
    ProviderResult, ComparisonPipelineResult
)

# Audio converter for normalizing audio to WAV/LINEAR16/16kHz/mono
audio_converter = AudioConverter(output_dir="outputs/normalized")


# Load configuration
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Initialize components
print("🔄 Initializing STT adapters...")

whisper_adapter = get_adapter(
    "whisper",
    **config["stt"]["whisper"]
)

# Initialize Google adapter
google_config = config["stt"].get("google", {})
google_config["credentials_path"] = "backend/daisoproject-sst.json"
google_adapter = get_adapter("google", **google_config)

quality_gate = QualityGate(
    **config["quality_gate"]
)

policy_gate = PolicyGate(
    fixed_locations=config["policy_gate"]["fixed_locations"],
    unsupported_patterns=config["policy_gate"]["unsupported_patterns"]
)

print("✅ All adapters initialized")

# FastAPI app
app = FastAPI(
    title="Daiso STT Pipeline API",
    description="Speech-to-Text pipeline with quality & policy gates - Whisper + Google comparison",
    version="1.1.0-poc"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import WebSocket handler
from fastapi import WebSocket
from ws_stt import handle_streaming_stt

@app.websocket("/ws/stt")
async def websocket_stt_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming STT"""
    # Accept the connection first (bypass origin check for dev)
    await handle_streaming_stt(websocket)


@app.get("/")
def root():
    return {
        "service": "Daiso STT Pipeline",
        "version": "PoC Phase 1.1",
        "status": "running",
        "providers": ["whisper", "google"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "whisper_model": whisper_adapter.model_size,
        "google_ready": google_adapter.client is not None
    }


def run_single_provider(audio_path: str, provider: str, attempt: int = 1):
    """Run STT pipeline for a single provider"""
    adapter = whisper_adapter if provider == "whisper" else google_adapter
    model = config["stt"]["whisper"]["model_size"] if provider == "whisper" else "default"
    
    # Convert audio to WAV/LINEAR16/16kHz/mono for STT
    try:
        conversion_result = audio_converter.normalize(audio_path)
        normalized_path = conversion_result["normalized_path"]
        print(f"🔄 Audio normalized: {audio_path} → {normalized_path}")
    except Exception as e:
        print(f"⚠️ Audio conversion failed, using original: {e}")
        normalized_path = audio_path
    
    # STT (use normalized path)
    stt_result = adapter.transcribe(normalized_path)
    
    # Quality Gate
    quality_result = quality_gate.evaluate(stt_result, attempt)
    
    # Policy Gate (only if OK)
    policy_intent = None
    if quality_result.status == "OK" and stt_result.text_raw:
        policy_intent = policy_gate.classify(stt_result.text_raw)
    
    return ProviderResult(
        provider=provider,
        model=model,
        stt=stt_result,
        quality_gate=quality_result,
        policy_intent=policy_intent
    )


def generate_final_response(provider_result: ProviderResult) -> str:
    """Generate final response based on provider result"""
    if provider_result.quality_gate.status == "OK":
        if provider_result.policy_intent:
            if provider_result.policy_intent.intent_type == "FIXED_LOCATION":
                for loc in config["policy_gate"]["fixed_locations"]:
                    if loc["target"] == provider_result.policy_intent.location_target:
                        return loc["response"]
            elif provider_result.policy_intent.intent_type == "UNSUPPORTED":
                return config["policy_gate"]["fallback_message"]
            else:  # PRODUCT_SEARCH
                return f"[PRODUCT_SEARCH] '{provider_result.stt.text_raw}' 검색 예정"
    elif provider_result.quality_gate.status == "RETRY":
        return config["policy_gate"]["retry_message"]
    
    return "죄송합니다. 음성을 인식할 수 없었습니다."


@app.post("/stt/compare", response_model=ComparisonPipelineResult)
async def compare_audio(
    audio: UploadFile = File(...),
    attempt: int = 1
):
    """
    Process audio through both Whisper and Google STT for comparison
    
    Returns results from both providers for performance comparison
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    Path("outputs").mkdir(exist_ok=True)
    
    # Use original filename if available, otherwise generate
    original_filename = audio.filename or f"recording_{request_id}.wav"
    temp_audio_path = f"outputs/temp_{request_id}_{original_filename}"
    
    print(f"📁 Saving file: {temp_audio_path}")
    
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        file_size = Path(temp_audio_path).stat().st_size
        print(f"📁 File saved: {file_size} bytes")
        
        # Run both providers
        print("🔄 Running Whisper STT...")
        whisper_result = run_single_provider(temp_audio_path, "whisper", attempt)
        print(f"✅ Whisper: {whisper_result.stt.text_raw}")
        
        print("🔄 Running Google STT...")
        google_result = run_single_provider(temp_audio_path, "google", attempt)
        print(f"✅ Google: {google_result.stt.text_raw}")
        
        # Generate final response (using whisper as primary by default)
        final_response = generate_final_response(whisper_result)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return ComparisonPipelineResult(
            request_id=request_id,
            file_name=original_filename,
            saved_path=temp_audio_path,
            whisper=whisper_result,
            google=google_result,
            primary_provider="whisper",
            final_response=final_response,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Keep original endpoint for backward compatibility
@app.post("/stt/process", response_model=PipelineResult)
async def process_audio(
    audio: UploadFile = File(...),
    attempt: int = 1
):
    """
    Process audio through Whisper STT pipeline (original endpoint)
    For comparison, use /stt/compare instead
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    Path("outputs").mkdir(exist_ok=True)
    temp_audio_path = f"outputs/temp_{request_id}.wav"
    
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Step 1: STT
        stt_result = whisper_adapter.transcribe(temp_audio_path)
        
        # Step 2: Quality Gate
        quality_result = quality_gate.evaluate(stt_result, attempt)
        
        # Step 3 & 4: Policy Gate + Response Generation
        policy_intent = None
        final_response = ""
        
        if quality_result.status == "OK":
            policy_intent = policy_gate.classify(stt_result.text_raw)
            
            if policy_intent.intent_type == "FIXED_LOCATION":
                for loc in config["policy_gate"]["fixed_locations"]:
                    if loc["target"] == policy_intent.location_target:
                        final_response = loc["response"]
                        break
            elif policy_intent.intent_type == "UNSUPPORTED":
                final_response = config["policy_gate"]["fallback_message"]
            else:  # PRODUCT_SEARCH
                final_response = f"[PRODUCT_SEARCH] '{stt_result.text_raw}' 검색 예정"
                
        elif quality_result.status == "RETRY":
            final_response = config["policy_gate"]["retry_message"]
        else:  # FAIL
            final_response = "죄송합니다. 음성을 인식할 수 없었습니다."
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return PipelineResult(
            request_id=request_id,
            stt=stt_result,
            quality_gate=quality_result,
            policy_intent=policy_intent,
            normalized_text=stt_result.text_raw,
            final_response=final_response,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
