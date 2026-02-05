'use client';

import { useState } from 'react';
import { ComparisonPipelineResult } from './types';
import FileUploadSection from './components/FileUploadSection';
import ResultDisplay from './components/ResultDisplay';
import StreamingSection from './components/StreamingSection';

export default function STTTestPage() {
    // 공통 상태 (모든 섹션에서 공유)
    const [result, setResult] = useState<ComparisonPipelineResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // 결과 핸들러
    const handleResult = (newResult: ComparisonPipelineResult) => {
        setResult(newResult);
        setError(null);
    };

    // 에러 핸들러
    const handleError = (newError: string) => {
        setError(newError || null);
        if (newError) {
            setResult(null);
        }
    };

    // 로딩 상태 핸들러
    const handleLoadingChange = (isLoading: boolean) => {
        setLoading(isLoading);
        if (isLoading) {
            setResult(null);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
            <div className="max-w-5xl mx-auto">
                {/* 페이지 헤더 */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <h1 className="text-3xl font-bold text-gray-800 mb-2">
                        🎤 STT 파이프라인 테스트
                    </h1>
                    <p className="text-gray-600">
                        음성 파일을 업로드하면 <strong>Whisper</strong>와 <strong>Google STT</strong> 결과를 동시에 비교합니다
                    </p>
                    <div className="mt-4 flex gap-2">
                        <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                            🔊 Whisper (로컬)
                        </span>
                        <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                            ☁️ Google Cloud STT
                        </span>
                    </div>
                </div>

                {/* 파일 업로드 섹션 */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <FileUploadSection
                        onResult={handleResult}
                        onError={handleError}
                        onLoadingChange={handleLoadingChange}
                    />
                </div>

                {/* 실시간 스트리밍 섹션 */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <StreamingSection />
                </div>

                {/* 결과 표시 영역 */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <div className="border-b pb-4 mb-6">
                        <h2 className="text-2xl font-bold text-gray-800">
                            📊 비교 결과
                        </h2>
                        <p className="text-gray-600 text-sm">
                            Whisper와 Google STT 결과를 나란히 비교합니다
                        </p>
                    </div>
                    <ResultDisplay
                        result={result}
                        loading={loading}
                        error={error}
                    />
                    {!result && !loading && !error && (
                        <div className="text-center text-gray-400 py-8">
                            음성을 녹음하거나 파일을 업로드한 후 "STT 비교 시작" 버튼을 눌러주세요
                        </div>
                    )}
                </div>

                {/* 사용 안내 */}
                <div className="bg-white rounded-xl shadow-md p-6">
                    <h3 className="text-lg font-semibold text-gray-700 mb-3">
                        📖 사용 방법
                    </h3>
                    <ol className="list-decimal list-inside space-y-2 text-gray-600">
                        <li>
                            <strong>녹음:</strong> "녹음 시작" 버튼을 눌러 음성을 녹음하거나
                        </li>
                        <li>
                            <strong>업로드:</strong> WAV/MP3 파일을 직접 업로드하세요
                        </li>
                        <li>
                            <strong>비교:</strong> "STT 비교 시작" 버튼을 클릭하면 Whisper와 Google 결과가 동시에 표시됩니다
                        </li>
                        <li>
                            <strong>분석:</strong> 신뢰도, 속도, 결과 일치 여부를 비교하세요
                        </li>
                    </ol>
                    <p className="mt-4 text-sm text-gray-500">
                        ⚠️ 백엔드 서버가 http://localhost:8000 에서 실행 중이어야 합니다
                    </p>
                </div>
            </div>
        </div>
    );
}
