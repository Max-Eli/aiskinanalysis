"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTenant } from "./TenantProvider";
import type {
  FaceLandmarker,
  FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";

type CameraFacing = "user" | "environment";

interface FaceGuidance {
  ready: boolean;
  message: string;
}

// ─── MediaPipe loader (lazy, cached) ─────────────────────────────────────────

let _landmarker: FaceLandmarker | null = null;
let _landmarkerLoading = false;
let _landmarkerCallbacks: Array<(lm: FaceLandmarker) => void> = [];

async function getLandmarker(): Promise<FaceLandmarker> {
  if (_landmarker) return _landmarker;
  if (_landmarkerLoading) {
    return new Promise((resolve) => _landmarkerCallbacks.push(resolve));
  }
  _landmarkerLoading = true;

  const { FaceLandmarker, FilesetResolver } = await import(
    "@mediapipe/tasks-vision"
  );
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  _landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });

  _landmarkerCallbacks.forEach((cb) => cb(_landmarker!));
  _landmarkerCallbacks = [];
  return _landmarker;
}

// ─── Face guidance logic ──────────────────────────────────────────────────────

/**
 * Checks face alignment from landmarks:
 * - Face detected at all
 * - Nose tip is roughly centered horizontally
 * - Face bounding box is large enough (not too far)
 * - Eyes are roughly level (not tilted)
 */
function evaluateFace(
  result: FaceLandmarkerResult,
  videoW: number,
  videoH: number
): FaceGuidance {
  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    return { ready: false, message: "Position your face in the oval" };
  }

  const lm = result.faceLandmarks[0];

  // Landmarks are normalised [0,1].
  // Nose tip: landmark 1
  // Left eye outer: 33 | Right eye outer: 263
  // Chin: 152 | Forehead: 10
  const noseTip = lm[1];
  const leftEye = lm[33];
  const rightEye = lm[263];
  const chin = lm[152];
  const forehead = lm[10];

  // ── Centering ──
  const centerX = 0.5;
  const centerY = 0.5;
  const faceX = noseTip.x;
  const faceY = (chin.y + forehead.y) / 2;

  if (Math.abs(faceX - centerX) > 0.18) {
    return {
      ready: false,
      message: faceX < centerX ? "Move right" : "Move left",
    };
  }
  if (Math.abs(faceY - centerY) > 0.18) {
    return {
      ready: false,
      message: faceY < centerY ? "Move down" : "Move up",
    };
  }

  // ── Distance (face height) ──
  const faceHeight = Math.abs(chin.y - forehead.y);
  if (faceHeight < 0.25) {
    return { ready: false, message: "Move closer" };
  }
  if (faceHeight > 0.75) {
    return { ready: false, message: "Move farther away" };
  }

  // ── Tilt (eyes should be level) ──
  const eyeTilt = Math.abs(leftEye.y - rightEye.y);
  if (eyeTilt > 0.06) {
    return { ready: false, message: "Tilt your head straight" };
  }

  // ── Yaw (face should be forward — nose should be between eyes) ──
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  if (Math.abs(noseTip.x - eyeMidX) > 0.05) {
    return { ready: false, message: "Face the camera directly" };
  }

  return { ready: true, message: "Hold still…" };
}

// ─── Blur detection (JS canvas Laplacian) ─────────────────────────────────────

function measureSharpness(canvas: HTMLCanvasElement): number {
  const ctx = canvas.getContext("2d");
  if (!ctx) return 999;
  const { width: w, height: h } = canvas;
  const imageData = ctx.getImageData(0, 0, w, h);
  const pixels = imageData.data;

  // Greyscale + Laplacian variance
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    gray[i] = 0.299 * pixels[i * 4] + 0.587 * pixels[i * 4 + 1] + 0.114 * pixels[i * 4 + 2];
  }

  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const lap =
        -4 * gray[y * w + x] +
        gray[(y - 1) * w + x] +
        gray[(y + 1) * w + x] +
        gray[y * w + (x - 1)] +
        gray[y * w + (x + 1)];
      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;   // variance
}

const BLUR_THRESHOLD = 60;   // below this → blurry

// ─── Component ────────────────────────────────────────────────────────────────

export default function CameraCapture() {
  const tenant = useTenant();
  const router = useRouter();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const lastVideoTimeRef = useRef(-1);

  const [facing, setFacing] = useState<CameraFacing>("user");
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [landmarkerReady, setLandmarkerReady] = useState(false);
  const [guidance, setGuidance] = useState<FaceGuidance>({
    ready: false,
    message: "Loading face detector…",
  });
  const [captured, setCaptured] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load MediaPipe once
  useEffect(() => {
    getLandmarker().then(() => {
      setLandmarkerReady(true);
      setGuidance({ ready: false, message: "Position your face in the oval" });
    });
  }, []);

  // Start camera
  const startCamera = useCallback(async (facingMode: CameraFacing) => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setHasPermission(true);
    } catch {
      setHasPermission(false);
    }
  }, []);

  useEffect(() => {
    if (!captured) startCamera(facing);
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [facing, captured, startCamera]);

  // Real-time inference loop
  useEffect(() => {
    if (!landmarkerReady || captured) return;

    let stopped = false;

    async function loop() {
      const lm = await getLandmarker();
      const video = videoRef.current;
      if (!video || stopped) return;

      function tick() {
        if (stopped || !video || video.readyState < 2) {
          rafRef.current = requestAnimationFrame(tick);
          return;
        }
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          try {
            const result = lm.detectForVideo(video, performance.now());
            const g = evaluateFace(result, video.videoWidth, video.videoHeight);
            setGuidance(g);
          } catch { /* model not ready yet */ }
        }
        rafRef.current = requestAnimationFrame(tick);
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    loop();

    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [landmarkerReady, captured]);

  function capture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d")!;

    if (facing === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0);
    if (facing === "user") ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Client-side blur check
    const sharpness = measureSharpness(canvas);
    if (sharpness < BLUR_THRESHOLD) {
      setError("Photo is blurry — please hold still and try again.");
      return;
    }

    const dataUrl = canvas.toDataURL("image/jpeg", 0.88);
    setCaptured(dataUrl);
    setError(null);
  }

  function retake() {
    setCaptured(null);
    setError(null);
  }

  async function submit() {
    if (!captured) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: captured }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Analysis failed");
      if (data.image_quality === "poor") {
        setError(data.quality_issues?.[0] ?? "Poor image quality. Please retake.");
        setAnalyzing(false);
        setCaptured(null);
        return;
      }
      sessionStorage.setItem("skinAnalysis", JSON.stringify(data));
      router.push("/results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setAnalyzing(false);
    }
  }

  // ── Analysing screen ──
  if (analyzing) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-6 px-4">
        <div
          className="w-20 h-20 rounded-full border-4 border-t-transparent animate-spin"
          style={{ borderColor: `${tenant.primaryColor} transparent transparent transparent` }}
        />
        <p className="text-lg font-semibold text-stone-700">Analyzing your skin…</p>
        <p className="text-sm text-stone-500 text-center max-w-xs">
          Running segmentation and skin analysis. This takes a few seconds.
        </p>
      </div>
    );
  }

  const guidanceColor = guidance.ready
    ? "#4ade80"
    : hasPermission
    ? "#fbbf24"
    : "#e5e7eb";

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-stone-50 px-4 py-8">
      <div className="w-full max-w-sm space-y-4">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-2xl font-semibold text-stone-800">{tenant.name}</h1>
          <p className="text-sm text-stone-500 mt-1">
            {captured
              ? "Looks good — tap Analyze to continue"
              : "Center your face in the oval"}
          </p>
        </div>

        {/* Camera / preview */}
        <div className="relative rounded-3xl overflow-hidden bg-black shadow-xl aspect-[3/4]">
          {captured ? (
            <img src={captured} alt="Captured" className="w-full h-full object-cover" />
          ) : (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${facing === "user" ? "scale-x-[-1]" : ""}`}
            />
          )}

          {/* Face oval guide */}
          {!captured && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div
                className="transition-all duration-300"
                style={{
                  width: "55%",
                  paddingBottom: "72%",
                  borderRadius: "50%",
                  border: `3px solid ${guidanceColor}`,
                  boxShadow: `0 0 0 9999px rgba(0,0,0,0.35), 0 0 20px ${guidanceColor}44`,
                }}
              />
            </div>
          )}

          {/* Guidance pill */}
          {!captured && hasPermission && (
            <div className="absolute bottom-4 left-0 right-0 flex justify-center">
              <span
                className="px-4 py-1.5 rounded-full text-sm font-medium text-white backdrop-blur-sm"
                style={{ backgroundColor: guidance.ready ? "#15803d99" : "#78350f99" }}
              >
                {guidance.message}
              </span>
            </div>
          )}

          {/* Flip camera */}
          {!captured && hasPermission && (
            <button
              onClick={() => setFacing((f) => (f === "user" ? "environment" : "user"))}
              className="absolute top-3 right-3 bg-black/40 text-white rounded-full p-2 backdrop-blur-sm"
              aria-label="Flip camera"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          )}
        </div>

        <canvas ref={canvasRef} className="hidden" />

        {/* Error */}
        {error && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-200 px-4 py-2.5 rounded-xl text-center">
            {error}
          </p>
        )}

        {/* Permission denied */}
        {hasPermission === false && (
          <p className="text-sm text-stone-500 text-center">
            Camera access denied. Allow camera permissions and refresh.
          </p>
        )}

        {/* Actions */}
        {captured ? (
          <div className="flex gap-3">
            <button
              onClick={retake}
              className="flex-1 py-3 rounded-xl border border-stone-300 text-stone-700 font-medium text-sm hover:bg-stone-100 transition-colors"
            >
              Retake
            </button>
            <button
              onClick={submit}
              className="flex-1 py-3 rounded-xl text-white font-semibold text-sm hover:opacity-90 transition-opacity"
              style={{ backgroundColor: tenant.primaryColor }}
            >
              Analyze My Skin
            </button>
          </div>
        ) : (
          hasPermission && (
            <button
              onClick={capture}
              disabled={!guidance.ready}
              className="w-full py-4 rounded-xl text-white font-semibold text-base transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 shadow-md"
              style={{ backgroundColor: tenant.primaryColor }}
            >
              {guidance.ready ? "Capture Photo" : "Align Your Face First"}
            </button>
          )
        )}

        <p className="text-xs text-stone-400 text-center">
          Your photo is analyzed privately and never stored.
        </p>
      </div>
    </div>
  );
}
