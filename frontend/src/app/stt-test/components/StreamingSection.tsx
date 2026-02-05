'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected';

interface StreamingMeta {
    confidence: number;
    latency_ms: number;
    first_interim_ms: number | null;
    duration_sec: number;
}

interface FinalResult {
    text: string;
    status: string;
    meta: StreamingMeta;
    timestamp: Date;
    runId: string;
    testId: string;
}

export default function StreamingSection() {
    // Connection state
    const [status, setStatus] = useState<ConnectionStatus>('disconnected');
    const [error, setError] = useState<string | null>(null);

    // Metadata state (for PoC)
    const [runId, setRunId] = useState<string>(`RUN_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`);
    const [testId, setTestId] = useState<string>('');
    const [utteranceType, setUtteranceType] = useState<string>('general');
    const [expectedText, setExpectedText] = useState<string>('');
    const [saveAudio, setSaveAudio] = useState<boolean>(false);

    // STT results
    const [interimText, setInterimText] = useState<string>('');
    const [finalResults, setFinalResults] = useState<FinalResult[]>([]);

    // Refs
    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const seqRef = useRef<number>(0);

    // Generate new Test ID automatically
    useEffect(() => {
        setTestId(`TEST_${Math.floor(Math.random() * 10000)}`);
    }, []);

    // Cleanup function
    const cleanup = useCallback(() => {
        // Close WebSocket
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        // Stop audio
        if (workletNodeRef.current) {
            workletNodeRef.current.disconnect();
            workletNodeRef.current = null;
        }

        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        seqRef.current = 0;
        setStatus('disconnected');
    }, []);

    // Start streaming
    const startStreaming = async () => {
        setError(null);
        setInterimText('');
        setStatus('connecting');

        try {
            // 1. Get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            streamRef.current = stream;

            // 2. Create AudioContext
            const audioContext = new AudioContext({ sampleRate: 16000 });
            audioContextRef.current = audioContext;

            // 3. Load AudioWorklet
            // Note: In Next.js/React, ensure worklet file is continuously served from public folder
            try {
                await audioContext.audioWorklet.addModule('/worklet-processor.js');
            } catch (e) {
                console.warn("Worklet addModule failed (maybe already loaded?)", e);
            }

            // 4. Create worklet node
            const workletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
            workletNodeRef.current = workletNode;

            // 5. Connect WebSocket
            const ws = new WebSocket('ws://localhost:8000/ws/stt');
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');
                // Send start message with metadata
                ws.send(JSON.stringify({
                    type: 'start',
                    config: {
                        sample_rate: 16000,
                        language: 'ko-KR'
                    },
                    meta: {
                        run_id: runId,
                        test_id: testId,
                        utterance_type: utteranceType,
                        spoken_text: expectedText,
                        save_audio: saveAudio
                    }
                }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                switch (data.type) {
                    case 'started':
                        setStatus('connected');
                        break;

                    case 'interim':
                        setInterimText(data.text);
                        break;

                    case 'final':
                        setInterimText('');
                        setFinalResults(prev => [{
                            text: data.text,
                            status: data.status,
                            meta: data.meta,
                            timestamp: new Date(),
                            runId: runId,
                            testId: testId
                        }, ...prev]);

                        // Regenerate Test ID for next run
                        setTestId(`TEST_${Math.floor(Math.random() * 10000)}`);

                        // Session ends on final
                        cleanup();
                        break;

                    case 'error':
                        setError(data.message);
                        cleanup();
                        break;
                }
            };

            ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                setError('WebSocket 연결 오류');
                cleanup();
            };

            ws.onclose = () => {
                console.log('WebSocket closed');
                if (status === 'connected') {
                    cleanup();
                }
            };

            // 6. Connect audio pipeline
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(workletNode);

            // 7. Handle audio data from worklet
            workletNode.port.onmessage = (event) => {
                if (event.data.type === 'audio' && ws.readyState === WebSocket.OPEN) {
                    // Convert ArrayBuffer to base64
                    const uint8Array = new Uint8Array(event.data.buffer);
                    const base64 = btoa(Array.from(uint8Array).map(b => String.fromCharCode(b)).join(''));

                    // Send to server
                    ws.send(JSON.stringify({
                        type: 'audio',
                        seq: seqRef.current++,
                        pcm_b64: base64
                    }));
                }
            };

        } catch (err: any) {
            console.error('Start streaming error:', err);
            setError(err.message || '스트리밍 시작 실패');
            cleanup();
        }
    };

    // Stop streaming
    const stopStreaming = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'stop' }));
        }
        cleanup();
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => cleanup();
    }, [cleanup]);

    // Get status color
    const getStatusColor = () => {
        switch (status) {
            case 'connected': return 'bg-green-500';
            case 'connecting': return 'bg-yellow-500 animate-pulse';
            default: return 'bg-gray-400';
        }
    };

    const getStatusText = () => {
        switch (status) {
            case 'connected': return '🔴 연결됨 (녹음 중)';
            case 'connecting': return '⏳ 연결 중...';
            default: return '⚪ 연결 안됨';
        }
    };

    return (
        <div className="space-y-6">
            {/* 섹션 헤더 */}
            <div className="border-b pb-4">
                <h2 className="text-2xl font-bold text-gray-800">
                    🎙️ 실시간 스트리밍 STT PoC
                </h2>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                    <p>Run ID, Test ID를 설정하고 테스트를 진행하세요.</p>
                    <p>결과는 서버의 `outputs/streaming_poc_results.csv`에 자동 저장됩니다.</p>
                </div>
            </div>

            {/* 메타데이터 컨트롤 */}
            <div className="bg-gray-50 p-4 rounded-lg border space-y-4">
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-semibold text-gray-500 mb-1">RUN ID</label>
                        <input
                            type="text"
                            value={runId}
                            onChange={(e) => setRunId(e.target.value)}
                            className="w-full text-sm p-2 border rounded"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-semibold text-gray-500 mb-1">TEST ID (Auto)</label>
                        <input
                            type="text"
                            value={testId}
                            onChange={(e) => setTestId(e.target.value)}
                            className="w-full text-sm p-2 border rounded bg-gray-100"
                            readOnly
                        />
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-semibold text-gray-500 mb-1">Utterance Type</label>
                        <select
                            value={utteranceType}
                            onChange={(e) => setUtteranceType(e.target.value)}
                            className="w-full text-sm p-2 border rounded"
                        >
                            <option value="general">일반 (General)</option>
                            <option value="long">긴 발화 (Long)</option>
                            <option value="ambiguous">모호한 발화 (Ambiguous)</option>
                            <option value="dialect">방언/사투리 (Dialect)</option>
                            <option value="noise">소음 환경 (Noise)</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-semibold text-gray-500 mb-1">Expected Text (Ref)</label>
                        <input
                            type="text"
                            value={expectedText}
                            onChange={(e) => setExpectedText(e.target.value)}
                            placeholder="(옵션) 예상 발화 텍스트"
                            className="w-full text-sm p-2 border rounded"
                        />
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="saveAudio"
                        checked={saveAudio}
                        onChange={(e) => setSaveAudio(e.target.checked)}
                        className="w-4 h-4"
                    />
                    <label htmlFor="saveAudio" className="text-sm text-gray-700 font-medium">
                        서버에 오디오 저장 (WAV)
                    </label>
                </div>
            </div>

            {/* 연결 상태 + 버튼 */}
            <div className="flex items-center gap-4">
                {/* 상태 표시 */}
                <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
                    <span className="text-sm text-gray-600">{getStatusText()}</span>
                </div>

                {/* 버튼 */}
                {status === 'disconnected' ? (
                    <button
                        onClick={startStreaming}
                        className="px-6 py-3 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors"
                    >
                        ▶️ PoC 테스트 시작
                    </button>
                ) : (
                    <button
                        onClick={stopStreaming}
                        disabled={status === 'connecting'}
                        className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
                    >
                        ⏹️ 중지 (결과 저장)
                    </button>
                )}
            </div>

            {/* 에러 표시 */}
            {error && (
                <div className="p-4 bg-red-50 border-l-4 border-red-500 rounded">
                    <p className="text-red-700">❌ {error}</p>
                </div>
            )}

            {/* Interim 결과 */}
            <div className="p-6 bg-blue-50 rounded-xl min-h-[80px]">
                <h3 className="text-sm font-medium text-gray-500 mb-2">
                    💬 실시간 인식 중...
                </h3>
                <p className="text-xl text-gray-800 font-medium">
                    {interimText || (status === 'connected' ? '말씀해 주세요...' : '대기 중')}
                </p>
            </div>

            {/* Final 결과 리스트 */}
            {finalResults.length > 0 && (
                <div className="space-y-3">
                    <h3 className="text-lg font-semibold text-gray-700">
                        📝 테스트 결과 ({finalResults.length}건)
                    </h3>
                    <div className="space-y-2 max-h-[300px] overflow-y-auto">
                        {finalResults.map((result, idx) => (
                            <div
                                key={idx}
                                className={`p-4 rounded-lg border ${result.status === 'OK'
                                    ? 'bg-green-50 border-green-200'
                                    : 'bg-yellow-50 border-yellow-200'
                                    }`}
                            >
                                <div className="flex justify-between items-start">
                                    <div className="flex-1">
                                        <div className="flex gap-2 mb-1">
                                            <span className="text-xs font-mono bg-gray-200 px-1 rounded">{result.testId}</span>
                                            <span className={`text-xs px-2 py-0.5 rounded ${result.status === 'OK' ? 'bg-green-200 text-green-800' : 'bg-yellow-200 text-yellow-800'}`}>
                                                {result.status}
                                            </span>
                                        </div>
                                        <p className="text-lg font-medium text-gray-800">
                                            {result.text || '(인식된 텍스트 없음)'}
                                        </p>
                                    </div>
                                </div>
                                <div className="mt-2 text-sm text-gray-500 flex gap-4">
                                    <span>신뢰도: {(result.meta.confidence * 100).toFixed(1)}%</span>
                                    <span>지연: {result.meta.latency_ms}ms</span>
                                    <span>길이: {result.meta.duration_sec}s</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
