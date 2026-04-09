"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTenant } from "./TenantProvider";
import { RefreshCw, Camera, ChevronRight, Shield } from "lucide-react";
import type { FaceLandmarker, FaceLandmarkerResult } from "@mediapipe/tasks-vision";

type CameraFacing = "user" | "environment";
interface FaceGuidance { ready: boolean; message: string }

// ─── MediaPipe loader ─────────────────────────────────────────────────────────
let _landmarker: FaceLandmarker | null = null;
let _loading = false;
let _cbs: Array<(lm: FaceLandmarker) => void> = [];

async function getLandmarker(): Promise<FaceLandmarker> {
  if (_landmarker) return _landmarker;
  if (_loading) return new Promise(r => _cbs.push(r));
  _loading = true;
  const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  _landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
  _cbs.forEach(cb => cb(_landmarker!));
  _cbs = [];
  return _landmarker;
}

function evaluateFace(result: FaceLandmarkerResult): FaceGuidance {
  if (!result.faceLandmarks?.length)
    return { ready: false, message: "Position your face in the frame" };
  const lm = result.faceLandmarks[0];
  const nose = lm[1], leftEye = lm[33], rightEye = lm[263];
  const chin = lm[152], forehead = lm[10];
  const faceH = Math.abs(chin.y - forehead.y);
  const faceX = nose.x, faceY = (chin.y + forehead.y) / 2;
  if (Math.abs(faceX - 0.5) > 0.18) return { ready: false, message: faceX < 0.5 ? "Move right" : "Move left" };
  if (Math.abs(faceY - 0.5) > 0.18) return { ready: false, message: faceY < 0.5 ? "Move down" : "Move up" };
  if (faceH < 0.25) return { ready: false, message: "Move closer" };
  if (faceH > 0.75) return { ready: false, message: "Move farther back" };
  if (Math.abs(leftEye.y - rightEye.y) > 0.06) return { ready: false, message: "Keep your head straight" };
  if (Math.abs(nose.x - (leftEye.x + rightEye.x) / 2) > 0.05) return { ready: false, message: "Face the camera directly" };
  return { ready: true, message: "Perfect — hold still" };
}

function measureSharpness(canvas: HTMLCanvasElement): number {
  const ctx = canvas.getContext("2d");
  if (!ctx) return 999;
  const { width: w, height: h } = canvas;
  const data = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++)
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  let sumSq = 0, sum = 0, count = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const lap = -4 * gray[y * w + x] + gray[(y-1)*w+x] + gray[(y+1)*w+x] + gray[y*w+x-1] + gray[y*w+x+1];
    sum += lap; sumSq += lap * lap; count++;
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

export default function CameraCapture() {
  const tenant = useTenant();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef(-1);
  const [facing, setFacing] = useState<CameraFacing>("user");
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [landmarkerReady, setLandmarkerReady] = useState(false);
  const [guidance, setGuidance] = useState<FaceGuidance>({ ready: false, message: "Loading camera AI…" });
  const [captured, setCaptured] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getLandmarker().then(() => { setLandmarkerReady(true); setGuidance({ ready: false, message: "Position your face in the frame" }); });
  }, []);

  const startCamera = useCallback(async (f: CameraFacing) => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: f, width: { ideal: 1280 }, height: { ideal: 720 } } });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setHasPermission(true);
    } catch { setHasPermission(false); }
  }, []);

  useEffect(() => {
    if (!captured) startCamera(facing);
    return () => { cancelAnimationFrame(rafRef.current); streamRef.current?.getTracks().forEach(t => t.stop()); };
  }, [facing, captured, startCamera]);

  useEffect(() => {
    if (!landmarkerReady || captured) return;
    let stopped = false;
    async function loop() {
      const lm = await getLandmarker();
      const video = videoRef.current;
      if (!video || stopped) return;
      function tick() {
        if (stopped || !video || video.readyState < 2) { rafRef.current = requestAnimationFrame(tick); return; }
        if (video.currentTime !== lastTimeRef.current) {
          lastTimeRef.current = video.currentTime;
          try { setGuidance(evaluateFace(lm.detectForVideo(video, performance.now()))); } catch {}
        }
        rafRef.current = requestAnimationFrame(tick);
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    loop();
    return () => { stopped = true; cancelAnimationFrame(rafRef.current); };
  }, [landmarkerReady, captured]);

  function capture() {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d")!;
    if (facing === "user") { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(video, 0, 0);
    if (facing === "user") ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (measureSharpness(canvas) < 60) { setError("Image is blurry — hold still and try again."); return; }
    setCaptured(canvas.toDataURL("image/jpeg", 0.88));
    setError(null);
  }

  async function submit() {
    if (!captured) return;
    setAnalyzing(true); setError(null);
    try {
      const res = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: captured }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Analysis failed");
      if (data.image_quality === "poor") { setError(data.quality_issues?.[0] ?? "Poor image quality — please retake."); setAnalyzing(false); setCaptured(null); return; }
      sessionStorage.setItem("skinAnalysis", JSON.stringify(data));
      router.push("/results");
    } catch (err) { setError(err instanceof Error ? err.message : "Something went wrong"); setAnalyzing(false); }
  }

  if (analyzing) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-white px-6 gap-8">
        <div className="relative w-20 h-20">
          <div className="absolute inset-0 rounded-full border-2 border-stone-100" />
          <div className="absolute inset-0 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: `${tenant.primaryColor} transparent transparent transparent` }} />
        </div>
        <div className="text-center space-y-2">
          <p className="text-lg font-semibold text-stone-800">Analysing your skin</p>
          <p className="text-sm text-stone-400 max-w-xs leading-relaxed">Running segmentation, CV analysis, and generating your personalised report.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-black">
      {/* Camera viewport — fills screen */}
      <div className="relative flex-1 overflow-hidden">
        {captured ? (
          <img src={captured} alt="Captured" className="w-full h-full object-cover" />
        ) : (
          <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover ${facing === "user" ? "scale-x-[-1]" : ""}`} />
        )}

        {/* Dark vignette overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-transparent to-black/60 pointer-events-none" />

        {/* Face oval */}
        {!captured && hasPermission && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <svg viewBox="0 0 300 400" className="h-[65vh] w-auto max-w-[85vw] max-h-[600px]">
              <defs>
                <mask id="oval-mask">
                  <rect width="300" height="400" fill="white" />
                  <ellipse cx="150" cy="200" rx="110" ry="145" fill="black" />
                </mask>
              </defs>
              <rect width="300" height="400" fill="rgba(0,0,0,0.45)" mask="url(#oval-mask)" />
              <ellipse cx="150" cy="200" rx="110" ry="145" fill="none" strokeWidth="2"
                stroke={guidance.ready ? "#4ade80" : "rgba(255,255,255,0.6)"}
                strokeDasharray={guidance.ready ? "0" : "8 4"} />
            </svg>
          </div>
        )}

        {/* Top bar */}
        <div className="absolute top-0 left-0 right-0 px-5 pt-12 pb-4 flex items-center justify-between">
          <div>
            <p className="text-white/50 text-xs font-medium tracking-widest uppercase">{tenant.name}</p>
            <p className="text-white font-semibold text-base mt-0.5">Skin Analysis</p>
          </div>
          {!captured && hasPermission && (
            <button onClick={() => setFacing(f => f === "user" ? "environment" : "user")}
              className="w-10 h-10 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center border border-white/20">
              <RefreshCw size={18} className="text-white" />
            </button>
          )}
        </div>

        {/* Guidance pill */}
        {!captured && hasPermission && (
          <div className="absolute bottom-40 left-0 right-0 flex justify-center">
            <div className={`px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm border transition-all ${
              guidance.ready ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" : "bg-white/10 text-white/80 border-white/20"
            }`}>
              {guidance.ready ? "✓ " : ""}{guidance.message}
            </div>
          </div>
        )}
      </div>

      {/* Bottom panel */}
      <div className="bg-white px-5 pt-5 pb-10 space-y-4">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-2xl px-4 py-3">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}
        {hasPermission === false && (
          <p className="text-sm text-stone-500 text-center">Allow camera access in your browser settings, then refresh.</p>
        )}

        <canvas ref={canvasRef} className="hidden" />

        {captured ? (
          <div className="flex gap-3">
            <button onClick={() => { setCaptured(null); setError(null); }}
              className="flex-1 h-14 rounded-2xl border border-stone-200 text-stone-600 font-medium text-sm hover:bg-stone-50 transition-colors">
              Retake
            </button>
            <button onClick={submit}
              className="flex-1 h-14 rounded-2xl text-white font-semibold text-sm flex items-center justify-center gap-2 transition-opacity hover:opacity-90"
              style={{ backgroundColor: tenant.primaryColor }}>
              Analyse My Skin
              <ChevronRight size={18} />
            </button>
          </div>
        ) : hasPermission && (
          <button onClick={capture} disabled={!guidance.ready}
            className="w-full h-14 rounded-2xl text-white font-semibold text-base flex items-center justify-center gap-2 transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90"
            style={{ backgroundColor: guidance.ready ? tenant.primaryColor : "#94a3b8" }}>
            <Camera size={20} />
            {guidance.ready ? "Capture Photo" : "Align Your Face"}
          </button>
        )}

        <div className="flex items-center justify-center gap-1.5">
          <Shield size={12} className="text-stone-300" />
          <p className="text-xs text-stone-400">Your photo is analysed privately and never stored</p>
        </div>
      </div>
    </div>
  );
}
