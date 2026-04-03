"use client";

interface Props {
  conditions: string[];
}

const COLORS = [
  "bg-rose-50 text-rose-700 border-rose-200",
  "bg-amber-50 text-amber-700 border-amber-200",
  "bg-sky-50 text-sky-700 border-sky-200",
  "bg-violet-50 text-violet-700 border-violet-200",
  "bg-teal-50 text-teal-700 border-teal-200",
];

export default function ConditionBadges({ conditions }: Props) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider">
        Detected Conditions
      </h3>
      <div className="flex flex-wrap gap-2">
        {conditions.map((c, i) => (
          <span
            key={c}
            className={`px-3 py-1 rounded-full border text-sm font-medium capitalize ${
              COLORS[i % COLORS.length]
            }`}
          >
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}
