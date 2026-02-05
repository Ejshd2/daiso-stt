// frontend/public/worklet-processor.js
// AudioWorklet processor for real-time PCM audio capture
// Converts audio to 16kHz mono PCM int16 format for STT

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 1600; // 100ms at 16kHz = 1600 samples
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        this.sampleRateRatio = 1; // Will be set based on actual sample rate
    }

    static get parameterDescriptors() {
        return [];
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            return true;
        }

        const inputChannel = input[0];

        // Process each sample
        for (let i = 0; i < inputChannel.length; i++) {
            // Simple downsampling (if needed)
            this.buffer[this.bufferIndex++] = inputChannel[i];

            // When buffer is full, send to main thread
            if (this.bufferIndex >= this.bufferSize) {
                // Convert Float32 to Int16
                const int16Buffer = this.float32ToInt16(this.buffer);

                // Post to main thread
                this.port.postMessage({
                    type: 'audio',
                    buffer: int16Buffer.buffer
                }, [int16Buffer.buffer]);

                // Reset buffer
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            }
        }

        return true;
    }

    float32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Clamp and convert
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
