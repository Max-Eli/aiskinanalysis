"use client";

import { useTenant } from "./TenantProvider";

interface Props {
  treatments: string[];
}

const TREATMENT_ICONS: Record<string, string> = {
  hydrafacial: "💧",
  "chemical peel": "✨",
  microneedling: "🔬",
  "botox": "💉",
  "botox / neurotoxin": "💉",
  "laser resurfacing": "🔆",
  "prp facial": "🩸",
  dermaplaning: "🪒",
  "ipl photofacial": "🌟",
  "led light therapy": "💡",
};

function getIcon(name: string): string {
  return TREATMENT_ICONS[name.toLowerCase()] ?? "🌿";
}

export default function TreatmentCards({ treatments }: Props) {
  const tenant = useTenant();

  if (treatments.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider">
        Recommended Treatments
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {treatments.map((t) => (
          <div
            key={t}
            className="rounded-xl p-3 flex items-center gap-2 border border-stone-100 bg-white shadow-sm"
          >
            <span className="text-xl">{getIcon(t)}</span>
            <span className="text-sm font-medium text-stone-700 leading-tight">
              {t}
            </span>
          </div>
        ))}
      </div>

      {tenant.bookingUrl && (
        <a
          href={tenant.bookingUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center w-full mt-3 py-3 rounded-xl text-white font-semibold text-sm transition-opacity hover:opacity-90"
          style={{ backgroundColor: tenant.primaryColor }}
        >
          Book a Consultation →
        </a>
      )}
    </div>
  );
}
