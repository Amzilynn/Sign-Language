"use client";

import { useEffect, useRef, useState } from "react";
import { useSignLanguage } from "@/hooks/useSignLanguage";
import { Camera, CameraOff, Volume2, Info, Activity, Layers } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [lastSpoken, setLastSpoken] = useState<string>("");

  const { prediction, isLoaded, isCapturing, status, start, stop } = useSignLanguage(
    videoRef.current,
    canvasRef.current
  );

  // Audio Feedback + History
  useEffect(() => {
    if (prediction && prediction.confidence > 0.4) {
      if (prediction.label !== lastSpoken) {
        setLastSpoken(prediction.label);
        setHistory(prev => [prediction.label.toUpperCase(), ...prev].slice(0, 10));

        // Speak using Web Speech API
        const utterance = new SpeechSynthesisUtterance(prediction.label);
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
      }
    }
  }, [prediction, lastSpoken]);

  return (
    <main className="min-h-screen bg-[#050505] text-white flex flex-col p-6 items-center selection:bg-blue-500/30">
      {/* Background Decor */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 blur-[120px] rounded-full" />
      </div>

      {/* Header */}
      <header className="w-full max-w-6xl flex justify-between items-center mb-8 relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-600/20">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">NEURAL TRANSLATOR</h1>
            <p className="text-xs text-zinc-500 font-mono">VERSION 3.0.1 ALPHA (WEB-SOTA)</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className={cn(
            "px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border",
            isLoaded ? "bg-green-500/10 border-green-500/50 text-green-400" : "bg-yellow-500/10 border-yellow-500/50 text-yellow-400"
          )}>
            {isLoaded ? "Neural Core Online" : "Loading Weights..."}
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">

        {/* Cam & Overlay Section */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          <div className="relative aspect-video bg-zinc-900 rounded-3xl overflow-hidden border border-white/5 shadow-2xl group">
            {/* Hidden Video for MP */}
            <video ref={videoRef} className="hidden" />

            {/* Main Display Canvas */}
            <canvas
              ref={canvasRef}
              width={1280}
              height={720}
              className="w-full h-full object-cover mirror"
              style={{ transform: 'scaleX(-1)' }} // Local mirror fix
            />

            {!isCapturing && (
              <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center">
                <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mb-6 border border-white/10">
                  <CameraOff className="w-10 h-10 text-zinc-400" />
                </div>
                <h2 className="text-2xl font-bold mb-2">Initialize Interpretation Engine</h2>
                <p className="text-zinc-400 max-w-sm mb-8">Grant camera permissions to start the zero-latency neural translation session.</p>
                <button
                  onClick={start}
                  disabled={!isLoaded}
                  className="px-8 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all rounded-2xl font-bold flex items-center gap-3 shadow-xl shadow-blue-600/20"
                >
                  <Camera className="w-5 h-5" />
                  Launch Scanner
                </button>
              </div>
            )}

            {/* In-feed Hud */}
            {isCapturing && (
              <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end pointer-events-none">
                <div className="p-4 rounded-2xl bg-black/40 backdrop-blur-md border border-white/10 max-w-[200px]">
                  <p className="text-[10px] text-zinc-500 uppercase font-bold mb-1">Engine Status</p>
                  <p className="text-sm font-medium tracking-wide flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    {status}
                  </p>
                </div>
                <button
                  onClick={stop}
                  className="p-4 rounded-2xl bg-red-600/80 hover:bg-red-500 backdrop-blur-md transition-all pointer-events-auto shadow-xl shadow-red-600/10"
                >
                  <CameraOff className="w-6 h-6" />
                </button>
              </div>
            )}
          </div>

          {/* Current Prediction Display */}
          <div className="h-32 bg-blue-600 rounded-3xl p-6 flex items-center justify-between shadow-2xl shadow-blue-600/20 relative overflow-hidden group">
            <div className="absolute right-[-20px] top-[-20px] opacity-10 group-hover:opacity-20 transition-opacity">
              <Layers className="w-48 h-48 rotate-12" />
            </div>

            <div className="relative z-10">
              <p className="text-[10px] text-blue-100 font-bold uppercase tracking-widest mb-1">Live Interpretation</p>
              <h2 className="text-5xl font-black italic tracking-tighter uppercase whitespace-nowrap">
                {prediction && prediction.confidence > 0.4 ? prediction.label : "---"}
              </h2>
            </div>

            {prediction && (
              <div className="relative z-10 text-right bg-white/10 backdrop-blur-md p-3 rounded-2xl border border-white/20">
                <p className="text-[10px] text-blue-100 font-bold mb-1">Confidence</p>
                <p className="text-2xl font-mono font-bold">{(prediction.confidence * 100).toFixed(1)}%</p>
              </div>
            )}
          </div>
        </div>

        {/* Analytics & History Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-white/5 rounded-3xl p-6 border border-white/10 backdrop-blur-xl flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-bold flex items-center gap-2">
                <Volume2 className="w-5 h-5 text-blue-400" />
                History Stream
              </h3>
              <Info className="w-4 h-4 text-zinc-600 cursor-help" />
            </div>

            <div className="flex-1 space-y-3 overflow-y-auto max-h-[400px] pr-2 custom-scrollbar">
              {history.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-zinc-600 p-8 text-center">
                  <p className="text-xs italic">Waiting for neural events...</p>
                </div>
              ) : (
                history.map((text, i) => (
                  <div
                    key={i}
                    className={cn(
                      "p-4 rounded-2xl border transition-all animate-in slide-in-from-right-4 duration-500",
                      i === 0 ? "bg-white/10 border-white/20" : "bg-white/5 border-white/2 border-transparent opacity-50"
                    )}
                  >
                    <p className={cn("font-bold text-lg", i === 0 ? "text-blue-400" : "text-zinc-400")}>
                      {text}
                    </p>
                    <p className="text-[10px] text-zinc-600 uppercase font-mono mt-1">
                      {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-zinc-900 rounded-3xl p-6 border border-white/5">
            <p className="text-[10px] text-zinc-500 font-bold uppercase mb-3">Model Details</p>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-400">Architecture</span>
                <span className="font-mono text-zinc-200">Personalized ST-GCN</span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-400">Framework</span>
                <span className="font-mono text-zinc-200">ONNX Runtime Web</span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-400">Tracking Engine</span>
                <span className="font-mono text-zinc-200">MediaPipe GPU Edge</span>
              </div>
            </div>
          </div>
        </div>

      </div>

      <style jsx global>{`
        .mirror {
          transform: scaleX(-1);
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
      `}</style>
    </main>
  );
}
