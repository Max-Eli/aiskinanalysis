"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useTenant } from "@/components/TenantProvider";
import SkinScore from "@/components/SkinScore";
import TreatmentCards from "@/components/TreatmentCards";
import RoutineList from "@/components/RoutineList";

interface Concern  { score: number; severity: string }
interface Positive { score: number; label: string }
interface Zone     { dominant_concern: string; score: number }

interface AnalysisResult {
  image_quality: string;
  confidence: number;
  analysis_method: string;
  skin_type: string;
  overall_score: number;
  concerns: Record<string, Concern>;
  positives: Record<string, Positive>;
  zone_analysis: Record<string, Zone>;
  notes: string[];
  headline: string;
  overview: string;
  concern_details: Record<string, string>;
  positive_details: Record<string, string>;
  zone_summary: Record<string, string>;
  morning_routine: string[];
  evening_routine: string[];
  spa_treatments: string[];
  key_ingredients: string[];
  lifestyle_tips: string[];
  disclaimer: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function severityColor(s: string) {
  return s === "none"     ? "text-emerald-600 bg-emerald-50 border-emerald-200"
    : s === "mild"        ? "text-amber-600 bg-amber-50 border-amber-200"
    : s === "moderate"    ? "text-orange-600 bg-orange-50 border-orange-200"
    :                       "text-red-600 bg-red-50 border-red-200";
}

function barColor(score: number) {
  if (score < 0.25) return "#4ade80";
  if (score < 0.50) return "#fbbf24";
  if (score < 0.72) return "#f97316";
  return "#ef4444";
}

function positiveBarColor(score: number) {
  if (score >= 0.70) return "#4ade80";
  if (score >= 0.45) return "#86efac";
  return "#fcd34d";
}

function formatLabel(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

const ZONE_ICONS: Record<string, string> = {
  forehead: "👆", left_cheek: "◀", right_cheek: "▶", nose: "▼", chin: "⬇",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-stone-100">
        <h2 className="text-xs font-bold text-stone-400 uppercase tracking-widest">{title}</h2>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function MetricBar({ label, score, colorFn, badge }: {
  label: string; score: number; colorFn: (s: number) => string; badge?: string;
}) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="w-36 text-sm text-stone-600 flex-shrink-0">{label}</span>
      <div className="flex-1 bg-stone-100 rounded-full h-2 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.round(score * 100)}%`, backgroundColor: colorFn(score) }}
        />
      </div>
      <span className="text-xs text-stone-400 w-8 text-right flex-shrink-0">
        {Math.round(score * 100)}%
      </span>
      {badge && (
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border capitalize w-20 text-center flex-shrink-0 ${severityColor(badge)}`}>
          {badge}
        </span>
      )}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function ResultsPage() {
  const router = useRouter();
  const tenant = useTenant();
  const [result, setResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("skinAnalysis");
    if (!raw) { router.replace("/"); return; }
    try { setResult(JSON.parse(raw) as AnalysisResult); }
    catch { router.replace("/"); }
  }, [router]);

  if (!result) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="w-10 h-10 rounded-full border-4 border-t-transparent animate-spin"
          style={{ borderColor: `${tenant.primaryColor} transparent transparent transparent` }} />
      </div>
    );
  }

  const detectedConcerns = result.concerns ? Object.entries(result.concerns).filter(([, d]) => d.severity !== "none") : [];
  const allConcerns = result.concerns ? Object.entries(result.concerns) : [];

  return (
    <div className="min-h-screen bg-stone-50 pb-16">

      {/* ── Hero header ── */}
      <div className="text-white px-4 pt-10 pb-16"
        style={{ background: `linear-gradient(135deg, ${tenant.primaryColor}, ${tenant.accentColor})` }}>
        <div className="max-w-sm mx-auto space-y-2">
          <p className="text-white/60 text-xs font-bold uppercase tracking-widest text-center">{tenant.name}</p>
          <h1 className="text-2xl font-bold text-center">Skin Analysis Report</h1>
          {result.headline && (
            <p className="text-white/90 text-sm text-center leading-relaxed font-medium">{result.headline}</p>
          )}
          <div className="flex justify-center gap-4 pt-1 text-xs text-white/60">
            <span className="capitalize">{result.skin_type} skin</span>
            <span>·</span>
            <span>{Math.round(result.confidence * 100)}% confidence</span>
            <span>·</span>
            <span>CV {result.analysis_method}</span>
          </div>
        </div>
      </div>

      <div className="max-w-sm mx-auto px-4 space-y-4 -mt-10">

        {/* ── Overall score card ── */}
        <div className="bg-white rounded-2xl shadow-lg p-6 flex flex-col items-center">
          <SkinScore score={result.overall_score} skinType={result.skin_type} />
          {result.overview && (
            <p className="text-sm text-stone-600 text-center leading-relaxed mt-4 border-t border-stone-100 pt-4">
              {result.overview}
            </p>
          )}
        </div>

        {/* ── Skin strengths ── */}
        {result.positives && Object.keys(result.positives).length > 0 && (
          <Section title="What's Looking Great">
            <div className="space-y-1">
              {Object.entries(result.positives)
                .sort((a, b) => b[1].score - a[1].score)
                .map(([name, data]) => (
                  <div key={name}>
                    <MetricBar
                      label={formatLabel(name)}
                      score={data.score}
                      colorFn={positiveBarColor}
                      badge={data.label}
                    />
                    {result.positive_details?.[name] && (
                      <p className="text-xs text-stone-400 ml-36 pl-3 pb-1 leading-relaxed">
                        {result.positive_details[name]}
                      </p>
                    )}
                  </div>
                ))}
            </div>
          </Section>
        )}

        {/* ── Concerns ── */}
        <Section title="Areas to Address">
          {allConcerns.length === 0 ? (
            <p className="text-sm text-stone-500">No concerns detected.</p>
          ) : (
            <div className="space-y-1">
              {allConcerns
                .sort((a, b) => b[1].score - a[1].score)
                .map(([name, data]) => (
                  <div key={name}>
                    <MetricBar
                      label={formatLabel(name)}
                      score={data.score}
                      colorFn={barColor}
                      badge={data.severity}
                    />
                    {data.severity !== "none" && result.concern_details?.[name] && (
                      <p className="text-xs text-stone-400 ml-36 pl-3 pb-1 leading-relaxed">
                        {result.concern_details[name]}
                      </p>
                    )}
                  </div>
                ))}
            </div>
          )}
        </Section>

        {/* ── Zone analysis ── */}
        {result.zone_analysis && Object.keys(result.zone_analysis).length > 0 && (
          <Section title="Face Zone Analysis">
            <div className="space-y-3">
              {Object.entries(result.zone_analysis).map(([zone]) => (
                <div key={zone} className="flex items-start gap-3">
                  <span className="text-base mt-0.5 flex-shrink-0">{ZONE_ICONS[zone] ?? "•"}</span>
                  <div>
                    <p className="text-sm font-semibold text-stone-700 capitalize">{formatLabel(zone)}</p>
                    <p className="text-xs text-stone-500 leading-relaxed mt-0.5">
                      {result.zone_summary?.[zone] ?? `Primary finding: ${result.zone_analysis[zone].dominant_concern.replace(/_/g, " ")}`}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* ── Recommended treatments ── */}
        {result.spa_treatments?.length > 0 && (
          <Section title="Recommended Treatments">
            <TreatmentCards treatments={result.spa_treatments.map(t => t.split(" — ")[0])} />
            {result.spa_treatments.some(t => t.includes(" — ")) && (
              <div className="mt-3 space-y-1">
                {result.spa_treatments.filter(t => t.includes(" — ")).map((t, i) => {
                  const [name, reason] = t.split(" — ");
                  return (
                    <p key={i} className="text-xs text-stone-500">
                      <span className="font-medium text-stone-700">{name}</span> — {reason}
                    </p>
                  );
                })}
              </div>
            )}
          </Section>
        )}

        {/* ── AM / PM Routine ── */}
        <Section title="Your Personalised Routine">
          <RoutineList
            morningRoutine={result.morning_routine}
            eveningRoutine={result.evening_routine}
          />
        </Section>

        {/* ── Key ingredients ── */}
        {result.key_ingredients?.length > 0 && (
          <Section title="Key Ingredients to Look For">
            <div className="space-y-2">
              {result.key_ingredients.map((ing, i) => {
                const [name, reason] = ing.includes(" — ") ? ing.split(" — ") : [ing, ""];
                return (
                  <div key={i} className="flex items-start gap-2">
                    <span className="w-2 h-2 rounded-full bg-stone-300 flex-shrink-0 mt-1.5" />
                    <p className="text-sm text-stone-700">
                      <span className="font-semibold">{name}</span>
                      {reason && <span className="text-stone-500 font-normal"> — {reason}</span>}
                    </p>
                  </div>
                );
              })}
            </div>
          </Section>
        )}

        {/* ── Lifestyle tips ── */}
        {result.lifestyle_tips?.length > 0 && (
          <Section title="Lifestyle Recommendations">
            <div className="space-y-2">
              {result.lifestyle_tips.map((tip, i) => (
                <div key={i} className="flex items-start gap-3">
                  <span className="text-sm font-bold text-stone-400 flex-shrink-0 w-5">{i + 1}.</span>
                  <p className="text-sm text-stone-600 leading-relaxed">{tip}</p>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* ── Clinical notes ── */}
        {result.notes?.length > 0 && (
          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 space-y-1">
            <p className="text-xs font-bold text-amber-700 uppercase tracking-wider">Clinical Observations</p>
            {result.notes.map((n, i) => <p key={i} className="text-sm text-amber-800">{n}</p>)}
          </div>
        )}

        {/* ── Disclaimer ── */}
        <p className="text-xs text-stone-400 text-center leading-relaxed px-2">
          {result.disclaimer}
        </p>

        <button
          onClick={() => { sessionStorage.removeItem("skinAnalysis"); router.push("/"); }}
          className="w-full py-3 rounded-xl border border-stone-200 text-stone-600 font-medium text-sm hover:bg-stone-100 transition-colors"
        >
          New Scan
        </button>
      </div>
    </div>
  );
}
