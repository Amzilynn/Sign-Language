import { useEffect, useRef, useState, useCallback } from 'react';
import { SignLanguageInference } from '../utils/inference';

export const useSignLanguage = (videoElement: HTMLVideoElement | null, canvasElement: HTMLCanvasElement | null) => {
    const [prediction, setPrediction] = useState<{ label: string; confidence: number } | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [isCapturing, setIsCapturing] = useState(false);
    const [status, setStatus] = useState('Initializing AI...');

    const handsRef = useRef<any>(null);
    const cameraRef = useRef<any>(null);
    const inferenceRef = useRef<SignLanguageInference | null>(null);
    const frameBufferRef = useRef<number[][][]>([]);

    const onResults = useCallback(async (results: any) => {
        if (!canvasElement || !videoElement) return;

        const ctx = canvasElement.getContext('2d');
        if (!ctx) return;

        ctx.save();
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        ctx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            const landmarkArray = landmarks.map((l: any) => [l.x, l.y, l.z]);

            frameBufferRef.current.push(landmarkArray);
            if (frameBufferRef.current.length > 60) {
                frameBufferRef.current.shift();
            }

            ctx.fillStyle = '#3b82f6';
            for (const l of landmarks) {
                ctx.beginPath();
                ctx.arc(l.x * canvasElement.width, l.y * canvasElement.height, 4, 0, 2 * Math.PI);
                ctx.fill();
            }

            if (frameBufferRef.current.length === 60) {
                const result = await inferenceRef.current?.predict(
                    frameBufferRef.current,
                    canvasElement.width,
                    canvasElement.height
                );
                if (result) setPrediction(result);
            }
        }
        ctx.restore();
    }, [canvasElement, videoElement]);

    useEffect(() => {
        if (!videoElement || !canvasElement || typeof window === 'undefined') return;

        const init = async () => {
            try {
                // 1. Init Inference
                inferenceRef.current = new SignLanguageInference();
                await inferenceRef.current.init();

                // 2. Init Hands (from CDN)
                const HandsObj = (window as any).Hands;
                if (!HandsObj) throw new Error("MediaPipe Hands script not loaded");

                const hands = new HandsObj({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
                });

                hands.setOptions({
                    maxNumHands: 1,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                hands.onResults(onResults);
                handsRef.current = hands;

                // 3. Init Camera (from CDN)
                const CameraObj = (window as any).Camera;
                if (!CameraObj) throw new Error("MediaPipe Camera script not loaded");

                const camera = new CameraObj(videoElement, {
                    onFrame: async () => {
                        await hands.send({ image: videoElement });
                    },
                    width: 1280,
                    height: 720,
                });
                cameraRef.current = camera;

                setIsLoaded(true);
                setStatus('Neural Core Ready');
            } catch (err) {
                console.error("Initialization error:", err);
                setStatus('System Offline (Check Console)');
            }
        };

        if ((window as any).Hands && (window as any).Camera) {
            init();
        } else {
            // Wait for scripts to load
            const interval = setInterval(() => {
                if ((window as any).Hands && (window as any).Camera) {
                    clearInterval(interval);
                    init();
                }
            }, 500);
            return () => clearInterval(interval);
        }

        return () => {
            cameraRef.current?.stop();
            handsRef.current?.close();
        };
    }, [videoElement, canvasElement, onResults]);

    const start = () => {
        cameraRef.current?.start();
        setIsCapturing(true);
        setStatus('Detecting Gestures...');
    };

    const stop = () => {
        cameraRef.current?.stop();
        setIsCapturing(false);
        setStatus('Camera Stopped');
    };

    return { prediction, isLoaded, isCapturing, status, start, stop };
};
