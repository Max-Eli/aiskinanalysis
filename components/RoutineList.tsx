"use client";

import { useState } from "react";
import { useTenant } from "./TenantProvider";

interface Props {
  morningRoutine: string[];
  eveningRoutine: string[];
}

export default function RoutineList({ morningRoutine, eveningRoutine }: Props) {
  const tenant = useTenant();
  const [active, setActive] = useState<"morning" | "evening">("morning");

  const steps = active === "morning" ? morningRoutine : eveningRoutine;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider">
        Skincare Routine
      </h3>

      {/* Toggle */}
      <div className="flex rounded-xl overflow-hidden border border-stone-200 bg-stone-100 p-1 gap-1">
        {(["morning", "evening"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            className={`flex-1 py-1.5 rounded-lg text-sm font-medium capitalize transition-all ${
              active === tab
                ? "bg-white text-stone-800 shadow-sm"
                : "text-stone-500 hover:text-stone-700"
            }`}
          >
            {tab === "morning" ? "☀️" : "🌙"} {tab}
          </button>
        ))}
      </div>

      {/* Steps */}
      <ol className="space-y-2">
        {steps.map((step, i) => (
          <li
            key={i}
            className="flex items-start gap-3 bg-white rounded-xl px-3 py-2.5 border border-stone-100 shadow-sm"
          >
            <span
              className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white"
              style={{ backgroundColor: tenant.accentColor }}
            >
              {i + 1}
            </span>
            <span className="text-sm text-stone-700 leading-snug">{step}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
