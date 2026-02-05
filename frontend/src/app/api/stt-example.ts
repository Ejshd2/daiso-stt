// frontend/src/app/api/stt-example.ts
/**
 * Frontend Example: STT API Integration
 * Demonstrates how to call the FastAPI backend from Next.js
 */

// Type definitions (matching backend PipelineResult)
interface STTAPIResponse {
    request_id: string;
    stt: {
        text_raw: string | null;
        confidence: number | null;
        lang: string;
        latency_ms: number;
        error: string | null;
    };
    quality_gate: {
        status: "OK" | "RETRY" | "FAIL";
        is_usable: boolean;
        reason: string;
    };
    policy_intent: {
        intent_type: "PRODUCT_SEARCH" | "FIXED_LOCATION" | "UNSUPPORTED";
        location_target: string | null;
        confidence: number;
        reason: string;
    } | null;
    normalized_text: string | null;
    final_response: string;
    processing_time_ms: number;
}

/**
 * Send audio file to STT pipeline
 * 
 * @param audioBlob - Recorded audio as Blob
 * @param attempt - Attempt number (1 or 2 for retry)
 * @returns Pipeline processing result
 */
export async function processAudio(
    audioBlob: Blob,
    attempt: number = 1
): Promise<STTAPIResponse> {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    formData.append("attempt", attempt.toString());

    const response = await fetch("http://localhost:8000/stt/process", {
        method: "POST",
        body: formData,
        // Note: Do NOT set Content-Type header, browser will set it automatically with boundary
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "STT processing failed");
    }

    return response.json();
}

/**
 * Example usage in a React component:
 * 
 * ```tsx
 * import { processAudio } from './api/stt-example';
 * 
 * function VoiceSearchComponent() {
 *   const [result, setResult] = useState<STTAPIResponse | null>(null);
 *   const [isRecording, setIsRecording] = useState(false);
 *   
 *   const handleVoiceSearch = async (audioBlob: Blob) => {
 *     try {
 *       const response = await processAudio(audioBlob, 1);
 *       setResult(response);
 *       
 *       // Handle different statuses
 *       if (response.quality_gate.status === "RETRY") {
 *         // Show retry message to user
 *         alert(response.final_response);
 *         // Optionally trigger re-recording
 *       } else if (response.quality_gate.status === "FAIL") {
 *         // Show failure message
 *         alert(response.final_response);
 *       } else {
 *         // OK - Show final response
 *         // If PRODUCT_SEARCH, route to search results
 *         // If FIXED_LOCATION, show location on map
 *         alert(response.final_response);
 *       }
 *     } catch (error) {
 *       console.error("STT error:", error);
 *     }
 *   };
 *   
 *   return (
 *     <div>
 *       <button onClick={() => setIsRecording(!isRecording)}>
 *         {isRecording ? "Stop Recording" : "Start Voice Search"}
 *       </button>
 *       {result && (
 *         <div>
 *           <p>Recognized: {result.stt.text_raw}</p>
 *           <p>Response: {result.final_response}</p>
 *         </div>
 *       )}
 *     </div>
 *   );
 * }
 * ```
 */
