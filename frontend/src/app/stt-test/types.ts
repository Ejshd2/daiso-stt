// frontend/src/app/stt-test/types.ts
// STT 테스트 페이지 공통 타입 정의

export interface STTResult {
    text_raw: string | null;
    confidence: number | null;
    lang: string;
    latency_ms: number;
    error: string | null;
}

export interface QualityGateResult {
    status: 'OK' | 'RETRY' | 'FAIL';
    is_usable: boolean;
    reason: string;
}

export interface PolicyIntent {
    intent_type: 'PRODUCT_SEARCH' | 'FIXED_LOCATION' | 'UNSUPPORTED';
    location_target: string | null;
    confidence: number;
    reason: string;
}

// Single provider result (for comparison)
export interface ProviderResult {
    provider: string;
    model: string;
    stt: STTResult;
    quality_gate: QualityGateResult;
    policy_intent: PolicyIntent | null;
}

// Comparison result (both Whisper and Google)
export interface ComparisonPipelineResult {
    request_id: string;
    file_name: string;
    saved_path: string;
    whisper: ProviderResult;
    google: ProviderResult;
    primary_provider: string;
    final_response: string;
    processing_time_ms: number;
}

// Legacy single provider response (backward compatibility)
export interface STTResponse {
    request_id: string;
    stt: STTResult;
    quality_gate: QualityGateResult;
    policy_intent: PolicyIntent | null;
    final_response: string;
    processing_time_ms: number;
}

// 스트리밍용 타입 (Phase 2)
export interface StreamingMessage {
    type: 'start' | 'audio' | 'stop' | 'interim' | 'final' | 'meta' | 'error';
    text?: string;
    is_final?: boolean;
    seq?: number;
    pcm_b64?: string;
    config?: StreamingConfig;
    latency_ms?: number;
    first_interim_ms?: number;
    message?: string;
}

export interface StreamingConfig {
    sample_rate: number;
    encoding: string;
    language: string;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected';
