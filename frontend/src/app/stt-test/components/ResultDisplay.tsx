'use client';

import { ComparisonPipelineResult, ProviderResult } from '../types';

interface ResultDisplayProps {
    result: ComparisonPipelineResult | null;
    loading: boolean;
    error: string | null;
}

// Single provider result card
function ProviderCard({ provider, data }: { provider: string; data: ProviderResult }) {
    const isWhisper = provider === 'whisper';
    const bgColor = isWhisper ? 'from-blue-50 to-blue-100' : 'from-green-50 to-green-100';
    const borderColor = isWhisper ? 'border-blue-200' : 'border-green-200';
    const iconColor = isWhisper ? 'text-blue-600' : 'text-green-600';

    return (
        <div className={`p-5 bg-gradient-to-br ${bgColor} rounded-xl border ${borderColor}`}>
            <div className="flex items-center gap-2 mb-4">
                <span className={`text-2xl ${iconColor}`}>
                    {isWhisper ? '🔊' : '☁️'}
                </span>
                <h4 className="text-lg font-bold text-gray-800">
                    {isWhisper ? 'Whisper' : 'Google STT'}
                </h4>
                <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded">
                    {data.model}
                </span>
            </div>

            {/* STT 결과 */}
            <div className="space-y-2 text-gray-700">
                <p>
                    <span className="font-medium">인식 결과:</span>{' '}
                    <span className="text-lg font-semibold">
                        {data.stt.text_raw || '(인식 실패)'}
                    </span>
                </p>
                <div className="flex gap-4 text-sm">
                    <span>
                        신뢰도: {data.stt.confidence
                            ? `${(data.stt.confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                    </span>
                    <span>
                        속도: <strong>{data.stt.latency_ms}ms</strong>
                    </span>
                </div>

                {/* Quality Gate */}
                <div className="mt-3 pt-3 border-t border-gray-200">
                    <span className="text-sm font-medium">품질: </span>
                    <span className={`text-sm px-2 py-0.5 rounded ${data.quality_gate.status === 'OK'
                            ? 'bg-green-200 text-green-800'
                            : data.quality_gate.status === 'RETRY'
                                ? 'bg-yellow-200 text-yellow-800'
                                : 'bg-red-200 text-red-800'
                        }`}>
                        {data.quality_gate.status}
                    </span>
                </div>

                {/* Policy Intent */}
                {data.policy_intent && (
                    <div className="text-sm">
                        <span className="font-medium">의도: </span>
                        <span className="px-2 py-0.5 bg-indigo-100 text-indigo-800 rounded">
                            {data.policy_intent.intent_type}
                        </span>
                    </div>
                )}

                {/* Error */}
                {data.stt.error && (
                    <p className="text-red-600 text-sm">
                        ⚠️ {data.stt.error}
                    </p>
                )}
            </div>
        </div>
    );
}

export default function ResultDisplay({ result, loading, error }: ResultDisplayProps) {
    return (
        <>
            {/* 에러 표시 */}
            {error && (
                <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded">
                    <p className="text-red-700 font-medium">❌ {error}</p>
                </div>
            )}

            {/* 로딩 표시 */}
            {loading && (
                <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                    <p className="text-blue-700 font-medium">
                        ⏳ Whisper + Google STT 처리 중...
                    </p>
                </div>
            )}

            {/* 결과 표시 */}
            {result && (
                <div className="space-y-6">
                    {/* 파일 정보 */}
                    <div className="p-4 bg-gray-100 rounded-lg">
                        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                            <span>📁 <strong>{result.file_name}</strong></span>
                            <span>🆔 {result.request_id}</span>
                            <span>⏱️ 총 {result.processing_time_ms}ms</span>
                        </div>
                    </div>

                    {/* 비교 표 */}
                    <div className="grid md:grid-cols-2 gap-4">
                        <ProviderCard provider="whisper" data={result.whisper} />
                        <ProviderCard provider="google" data={result.google} />
                    </div>

                    {/* 비교 요약 */}
                    <div className="p-4 bg-yellow-50 rounded-xl border border-yellow-200">
                        <h4 className="font-bold text-gray-800 mb-2">📊 비교 요약</h4>
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b">
                                    <th className="text-left py-2">항목</th>
                                    <th className="text-center py-2">Whisper</th>
                                    <th className="text-center py-2">Google</th>
                                    <th className="text-center py-2">차이</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-b">
                                    <td className="py-2">신뢰도</td>
                                    <td className="text-center">
                                        {result.whisper.stt.confidence
                                            ? `${(result.whisper.stt.confidence * 100).toFixed(1)}%`
                                            : 'N/A'}
                                    </td>
                                    <td className="text-center">
                                        {result.google.stt.confidence
                                            ? `${(result.google.stt.confidence * 100).toFixed(1)}%`
                                            : 'N/A'}
                                    </td>
                                    <td className="text-center font-medium">
                                        {result.whisper.stt.confidence && result.google.stt.confidence
                                            ? `${((result.google.stt.confidence - result.whisper.stt.confidence) * 100).toFixed(1)}%`
                                            : '-'}
                                    </td>
                                </tr>
                                <tr className="border-b">
                                    <td className="py-2">속도</td>
                                    <td className="text-center">{result.whisper.stt.latency_ms}ms</td>
                                    <td className="text-center">{result.google.stt.latency_ms}ms</td>
                                    <td className="text-center font-medium">
                                        {result.google.stt.latency_ms < result.whisper.stt.latency_ms
                                            ? `Google ${((1 - result.google.stt.latency_ms / result.whisper.stt.latency_ms) * 100).toFixed(0)}% 빠름`
                                            : `Whisper ${((1 - result.whisper.stt.latency_ms / result.google.stt.latency_ms) * 100).toFixed(0)}% 빠름`}
                                    </td>
                                </tr>
                                <tr>
                                    <td className="py-2">결과 일치</td>
                                    <td colSpan={3} className="text-center">
                                        {result.whisper.stt.text_raw === result.google.stt.text_raw
                                            ? <span className="text-green-600 font-medium">✅ 완전 일치</span>
                                            : <span className="text-yellow-600 font-medium">⚠️ 다름</span>}
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    {/* 최종 응답 */}
                    <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border-2 border-green-200">
                        <div className="flex items-center gap-2 mb-2">
                            <h3 className="text-lg font-semibold text-gray-700">
                                💬 최종 응답
                            </h3>
                            <span className="text-xs bg-gray-200 px-2 py-0.5 rounded">
                                기준: {result.primary_provider}
                            </span>
                        </div>
                        <p className="text-xl text-gray-800 font-medium">
                            {result.final_response}
                        </p>
                    </div>
                </div>
            )}
        </>
    );
}
