'use client';

import { useState, useRef } from 'react';
import { ComparisonPipelineResult } from '../types';

interface FileUploadSectionProps {
    onResult: (result: ComparisonPipelineResult) => void;
    onError: (error: string) => void;
    onLoadingChange: (loading: boolean) => void;
}

export default function FileUploadSection({
    onResult,
    onError,
    onLoadingChange
}: FileUploadSectionProps) {
    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [localLoading, setLocalLoading] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    // 녹음 시작
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
                setAudioBlob(blob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
            onError('');
        } catch (err) {
            onError('마이크 권한을 허용해주세요.');
            console.error(err);
        }
    };

    // 녹음 중지
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    // 파일 업로드
    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setAudioBlob(file);
            onError('');
        }
    };

    // STT 비교 API 호출 (Whisper + Google)
    const processAudio = async (attempt: number = 1) => {
        if (!audioBlob) {
            onError('먼저 녹음하거나 파일을 업로드해주세요.');
            return;
        }

        setLocalLoading(true);
        onLoadingChange(true);
        onError('');

        try {
            const formData = new FormData();

            // Use original filename if it's a File, otherwise use default
            const fileName = audioBlob instanceof File ? audioBlob.name : 'recording.wav';
            formData.append('audio', audioBlob, fileName);
            formData.append('attempt', attempt.toString());

            // Call comparison endpoint
            const response = await fetch('http://localhost:8000/stt/compare', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'STT 처리 실패');
            }

            const data: ComparisonPipelineResult = await response.json();
            onResult(data);
        } catch (err: any) {
            onError(err.message || '서버 오류가 발생했습니다.');
        } finally {
            setLocalLoading(false);
            onLoadingChange(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* 섹션 헤더 */}
            <div className="border-b pb-4">
                <h2 className="text-2xl font-bold text-gray-800">
                    📁 파일 업로드 STT (Whisper + Google 비교)
                </h2>
                <p className="text-gray-600 mt-1">
                    음성을 녹음하거나 파일을 업로드하면 Whisper와 Google STT 결과를 비교합니다
                </p>
            </div>

            {/* 녹음 섹션 */}
            <div className="p-6 bg-blue-50 rounded-xl">
                <h3 className="text-xl font-semibold text-gray-700 mb-4">
                    🎙️ 음성 녹음
                </h3>
                <div className="flex gap-4">
                    <button
                        onClick={isRecording ? stopRecording : startRecording}
                        className={`px-6 py-3 rounded-lg font-medium transition-colors ${isRecording
                            ? 'bg-red-500 hover:bg-red-600 text-white'
                            : 'bg-blue-500 hover:bg-blue-600 text-white'
                            }`}
                    >
                        {isRecording ? '🔴 녹음 중지' : '⏺️ 녹음 시작'}
                    </button>
                    {audioBlob && !isRecording && (
                        <span className="flex items-center text-green-600 font-medium">
                            ✅ 녹음 완료
                        </span>
                    )}
                </div>
            </div>

            {/* 파일 업로드 섹션 */}
            <div className="p-6 bg-purple-50 rounded-xl">
                <h3 className="text-xl font-semibold text-gray-700 mb-4">
                    📁 음성 파일 업로드
                </h3>
                <input
                    type="file"
                    accept="audio/*"
                    onChange={handleFileUpload}
                    className="block w-full text-sm text-gray-600
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-lg file:border-0
                        file:text-sm file:font-semibold
                        file:bg-purple-500 file:text-white
                        hover:file:bg-purple-600
                        cursor-pointer"
                />
            </div>

            {/* 처리 버튼 */}
            <div>
                <button
                    onClick={() => processAudio(1)}
                    disabled={!audioBlob || localLoading}
                    className="w-full px-6 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 
                        text-white rounded-xl font-semibold text-lg
                        hover:from-indigo-600 hover:to-purple-700
                        disabled:opacity-50 disabled:cursor-not-allowed
                        transition-all shadow-lg hover:shadow-xl"
                >
                    {localLoading ? '처리 중... (Whisper + Google)' : '🚀 STT 비교 시작 (Whisper + Google)'}
                </button>
            </div>
        </div>
    );
}
