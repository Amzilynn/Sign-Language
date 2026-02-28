import * as ort from 'onnxruntime-web';

export class SignLanguageInference {
    private session: ort.InferenceSession | null = null;
    private classes: string[] = [];

    constructor() { }

    async init() {
        try {
            // Load classes
            const resp = await fetch('/classes.json');
            this.classes = await resp.json();

            // Load ONNX model
            this.session = await ort.InferenceSession.create('/model.onnx', {
                executionProviders: ['webgl'], // Use GPU if available
                graphOptimizationLevel: 'all'
            });
            console.log('ONNX Model Loaded successfully');
        } catch (e) {
            console.error('Failed to initialize ONNX session:', e);
        }
    }

    async predict(landmarks: number[][][], width: number, height: number): Promise<{ label: string; confidence: number } | null> {
        if (!this.session || landmarks.length < 60) return null;

        try {
            // Input shape: [1, 3, 60, 21]
            // landmarks is [60, 21, 3] -> (T, V, C)
            // We need to permute to (1, C, T, V)

            const inputData = new Float32Array(1 * 3 * 60 * 21);

            // Landmarks from MediaPipe are normalized [0, 1]
            // We need to structure them as (Channels, Time, Vertices)
            for (let t = 0; t < 60; t++) {
                for (let v = 0; v < 21; v++) {
                    const x = landmarks[t][v][0];
                    const y = landmarks[t][v][1];
                    const z = landmarks[t][v][2];

                    // Indexing: c * (T*V) + t * V + v
                    inputData[0 * (60 * 21) + t * 21 + v] = x;
                    inputData[1 * (60 * 21) + t * 21 + v] = y;
                    inputData[2 * (60 * 21) + t * 21 + v] = z;
                }
            }

            const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 60, 21]);
            const feeds: Record<string, ort.Tensor> = {};
            feeds[this.session.inputNames[0]] = inputTensor;

            const outputMap = await this.session.run(feeds);
            const output = outputMap[this.session.outputNames[0]];
            const data = output.data as Float32Array;

            // Softmax/Argmax
            let maxIdx = 0;
            let maxVal = -Infinity;
            const expValues = data.map(v => Math.exp(v));
            const sumExp = expValues.reduce((a, b) => a + b, 0);
            const softmax = expValues.map(v => v / sumExp);

            for (let i = 0; i < softmax.length; i++) {
                if (softmax[i] > maxVal) {
                    maxVal = softmax[i];
                    maxIdx = i;
                }
            }

            return {
                label: this.classes[maxIdx],
                confidence: maxVal
            };
        } catch (e) {
            console.error('Inference error:', e);
            return null;
        }
    }
}
