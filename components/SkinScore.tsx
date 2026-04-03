"use client";

import { useTenant } from "./TenantProvider";

interface Props {
  score: number;
  skinType: string;
}

export default function SkinScore({ score, skinType }: Props) {
  const tenant = useTenant();

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 100) * circumference;

  function scoreLabel(s: number) {
    if (s >= 80) return "Excellent";
    if (s >= 65) return "Good";
    if (s >= 50) return "Fair";
    return "Needs Care";
  }

  function scoreColor(s: number) {
    if (s >= 80) return "#4ade80";
    if (s >= 65) return "#86efac";
    if (s >= 50) return "#fbbf24";
    return "#f87171";
  }

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Circular gauge */}
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 128 128">
          {/* Track */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke="#e7e5e4"
            strokeWidth="10"
          />
          {/* Progress */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke={scoreColor(score)}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            style={{ transition: "stroke-dashoffset 1s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-stone-800">{score}</span>
          <span className="text-xs text-stone-500 font-medium">/ 100</span>
        </div>
      </div>

      {/* Label + skin type */}
      <div className="text-center">
        <p className="font-semibold text-stone-700 text-base">
          {scoreLabel(score)}
        </p>
        <p className="text-xs text-stone-400 capitalize mt-0.5">
          {skinType} skin
        </p>
      </div>
    </div>
  );
}
