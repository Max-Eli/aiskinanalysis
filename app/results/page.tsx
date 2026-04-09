"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useTenant } from "@/components/TenantProvider";
import { CheckCircle2, AlertCircle, ChevronRight, Sun, Moon, Leaf, MapPin, Sparkles, RotateCcw, Info } from "lucide-react";

interface Concern  { score: number; severity: string }
interface Positive { score: number; label: string }
interface Zone     { dominant_concern: string; score: number; all_scores?: Record<string,number> }

interface AnalysisResult {
  overall_score: number;
  skin_type: string;
  confidence: number;
  analysis_method: string;
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

function fmt(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function scoreRing(score: number) {
  if (score >= 80) return { color: "#4ade80", label: "Excellent", bg: "bg-emerald-50", text: "text-emerald-700" };
  if (score >= 65) return { color: "#86efac", label: "Good",      bg: "bg-green-50",   text: "text-green-700" };
  if (score >= 50) return { color: "#fbbf24", label: "Fair",      bg: "bg-amber-50",   text: "text-amber-700" };
  return                  { color: "#f87171", label: "Needs Care",bg: "bg-red-50",     text: "text-red-700" };
}

function concernColor(severity: string) {
  if (severity === "none")     return { bar: "#d1fae5", badge: "bg-emerald-50 text-emerald-700 border-emerald-100" };
  if (severity === "mild")     return { bar: "#fcd34d", badge: "bg-amber-50 text-amber-700 border-amber-100" };
  if (severity === "moderate") return { bar: "#fb923c", badge: "bg-orange-50 text-orange-700 border-orange-100" };
  return                              { bar: "#f87171", badge: "bg-red-50 text-red-700 border-red-100" };
}

function positiveColor(label: string) {
  if (label === "excellent") return "#4ade80";
  if (label === "good")      return "#86efac";
  if (label === "fair")      return "#fcd34d";
  return "#fca5a5";
}

// ─── Components ───────────────────────────────────────────────────────────────

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white rounded-3xl overflow-hidden ${className}`}>{children}</div>;
}

function SectionHeader({ title, icon }: { title: string; icon?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      {icon && <span className="text-stone-400">{icon}</span>}
      <h2 className="text-xs font-bold text-stone-400 uppercase tracking-[0.12em]">{title}</h2>
    </div>
  );
}

function ScoreRing({ score }: { score: number }) {
  const r = 52, c = 2 * Math.PI * r;
  const { color, label, bg, text } = scoreRing(score);
  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r={r} fill="none" stroke="#f5f5f4" strokeWidth="8" />
          <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="8"
            strokeLinecap="round" strokeDasharray={c}
            strokeDashoffset={c - (score / 100) * c}
            style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)" }} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold text-stone-800 leading-none">{score}</span>
          <span className="text-xs text-stone-400 mt-1">/ 100</span>
        </div>
      </div>
      <span className={`text-sm font-semibold px-4 py-1.5 rounded-full ${bg} ${text}`}>{label}</span>
    </div>
  );
}

function ConcernRow({ name, data, detail }: { name: string; data: Concern; detail?: string }) {
  const { bar, badge } = concernColor(data.severity);
  const pct = Math.round(data.score * 100);
  return (
    <div className="py-3 border-b border-stone-50 last:border-0">
      <div className="flex items-center gap-3 mb-1.5">
        <span className="flex-1 text-sm font-medium text-stone-700">{fmt(name)}</span>
        <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border capitalize ${badge}`}>{data.severity}</span>
        <span className="text-xs text-stone-400 w-8 text-right">{pct}%</span>
      </div>
      <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: bar }} />
      </div>
      {detail && data.severity !== "none" && (
        <p className="text-xs text-stone-400 mt-1.5 leading-relaxed">{detail}</p>
      )}
    </div>
  );
}

function PositiveRow({ name, data, detail }: { name: string; data: Positive; detail?: string }) {
  const pct = Math.round(data.score * 100);
  return (
    <div className="py-3 border-b border-stone-50 last:border-0">
      <div className="flex items-center gap-3 mb-1.5">
        <CheckCircle2 size={14} className="text-emerald-400 flex-shrink-0" />
        <span className="flex-1 text-sm font-medium text-stone-700">{fmt(name)}</span>
        <span className="text-xs font-semibold text-emerald-600 capitalize">{data.label}</span>
        <span className="text-xs text-stone-400 w-8 text-right">{pct}%</span>
      </div>
      <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden ml-5">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: positiveColor(data.label) }} />
      </div>
      {detail && <p className="text-xs text-stone-400 mt-1.5 ml-5 leading-relaxed">{detail}</p>}
    </div>
  );
}

function RoutineTab({ steps, type }: { steps: string[]; type: "morning" | "evening" }) {
  return (
    <div className="space-y-2.5">
      {steps.map((step, i) => {
        const [label, ...rest] = step.split(": ");
        const hasLabel = rest.length > 0;
        return (
          <div key={i} className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 text-xs font-bold text-white"
              style={{ backgroundColor: type === "morning" ? "#f59e0b" : "#6366f1" }}>
              {i + 1}
            </div>
            <p className="text-sm text-stone-600 leading-relaxed">
              {hasLabel ? <><span className="font-semibold text-stone-800">{label}</span>: {rest.join(": ")}</> : step}
            </p>
          </div>
        );
      })}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function ResultsPage() {
  const router = useRouter();
  const tenant = useTenant();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [routineTab, setRoutineTab] = useState<"morning" | "evening">("morning");

  useEffect(() => {
    const raw = sessionStorage.getItem("skinAnalysis");
    if (!raw) { router.replace("/"); return; }
    try { setResult(JSON.parse(raw)); }
    catch { router.replace("/"); }
  }, [router]);

  if (!result) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-stone-50">
        <div className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin"
          style={{ borderColor: `${tenant.primaryColor} transparent transparent transparent` }} />
      </div>
    );
  }

  const concerns = result.concerns ?? {};
  const positives = result.positives ?? {};
  const zones = result.zone_analysis ?? {};
  const detectedCount = Object.values(concerns).filter(d => d.severity !== "none").length;

  return (
    <div className="min-h-screen bg-stone-50">

      {/* ── Hero ── */}
      <div className="relative text-white px-5 pt-14 pb-24"
        style={{ background: `linear-gradient(150deg, ${tenant.primaryColor} 0%, ${tenant.accentColor} 100%)` }}>
        <div className="max-w-sm mx-auto">
          <p className="text-white/50 text-xs font-semibold tracking-[0.15em] uppercase mb-1">{tenant.name}</p>
          <h1 className="text-2xl font-bold tracking-tight">Skin Analysis Report</h1>
          {result.headline && (
            <p className="text-white/80 text-sm leading-relaxed mt-2">{result.headline}</p>
          )}
          <div className="flex items-center gap-3 mt-3">
            <span className="text-xs text-white/50 capitalize bg-white/10 px-2.5 py-1 rounded-full">
              {result.skin_type} skin
            </span>
            <span className="text-xs text-white/50 bg-white/10 px-2.5 py-1 rounded-full">
              {Math.round(result.confidence * 100)}% confidence
            </span>
          </div>
        </div>
      </div>

      <div className="max-w-sm mx-auto px-4 -mt-14 pb-16 space-y-4">

        {/* ── Score + Overview ── */}
        <Card className="shadow-xl shadow-stone-200/60">
          <div className="p-6 flex flex-col items-center border-b border-stone-50">
            <ScoreRing score={result.overall_score} />
          </div>
          {result.overview && (
            <div className="px-5 py-4">
              <p className="text-sm text-stone-600 leading-relaxed">{result.overview}</p>
            </div>
          )}
          <div className="px-5 pb-4 flex gap-3">
            <div className="flex-1 bg-stone-50 rounded-2xl p-3 text-center">
              <p className="text-xs text-stone-400 mb-1">Concerns</p>
              <p className="text-xl font-bold text-stone-800">{detectedCount}</p>
            </div>
            <div className="flex-1 bg-stone-50 rounded-2xl p-3 text-center">
              <p className="text-xs text-stone-400 mb-1">Strengths</p>
              <p className="text-xl font-bold text-stone-800">{Object.values(positives).filter(p => p.label === "good" || p.label === "excellent").length}</p>
            </div>
            <div className="flex-1 bg-stone-50 rounded-2xl p-3 text-center">
              <p className="text-xs text-stone-400 mb-1">Skin Type</p>
              <p className="text-sm font-bold text-stone-800 capitalize">{result.skin_type}</p>
            </div>
          </div>
        </Card>

        {/* ── Strengths ── */}
        {Object.keys(positives).length > 0 && (
          <Card>
            <div className="px-5 pt-5 pb-1">
              <SectionHeader title="Skin Strengths" icon={<CheckCircle2 size={14} />} />
              {Object.entries(positives)
                .sort((a, b) => b[1].score - a[1].score)
                .map(([name, data]) => (
                  <PositiveRow key={name} name={name} data={data} detail={result.positive_details?.[name]} />
                ))}
            </div>
            <div className="h-4" />
          </Card>
        )}

        {/* ── Concerns ── */}
        {(() => {
          const detected = Object.entries(concerns).filter(([, d]) => d.severity !== "none").sort((a, b) => b[1].score - a[1].score);
          const low = Object.entries(concerns).filter(([, d]) => d.severity === "none").sort((a, b) => b[1].score - a[1].score);
          return (
            <Card>
              <div className="px-5 pt-5 pb-1">
                <SectionHeader title="Detailed Analysis" icon={<AlertCircle size={14} />} />
                {detected.length === 0 ? (
                  <p className="text-sm text-stone-500 py-3">No significant concerns detected. Your skin is in good health.</p>
                ) : (
                  detected.map(([name, data]) => (
                    <ConcernRow key={name} name={name} data={data} detail={result.concern_details?.[name]} />
                  ))
                )}
                {low.length > 0 && (
                  <details className="mt-3 mb-2">
                    <summary className="text-xs text-stone-400 cursor-pointer select-none hover:text-stone-500">
                      {low.length} areas within normal range
                    </summary>
                    <div className="mt-2">
                      {low.map(([name, data]) => (
                        <ConcernRow key={name} name={name} data={data} />
                      ))}
                    </div>
                  </details>
                )}
              </div>
              <div className="h-4" />
            </Card>
          );
        })()}

        {/* ── Zone Analysis ── */}
        {Object.keys(zones).length > 0 && (
          <Card>
            <div className="px-5 pt-5 pb-5">
              <SectionHeader title="Face Zone Breakdown" icon={<MapPin size={14} />} />
              <div className="space-y-3">
                {Object.entries(zones).map(([zone]) => (
                  <div key={zone} className="bg-stone-50 rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-semibold text-stone-800">{fmt(zone)}</p>
                      {zones[zone].score > 0 && (
                        <span className={`text-xs px-2 py-0.5 rounded-full border ${concernColor(
                          zones[zone].score < 0.25 ? "none" : zones[zone].score < 0.50 ? "mild" : zones[zone].score < 0.70 ? "moderate" : "severe"
                        ).badge}`}>
                          {fmt(zones[zone].dominant_concern)}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-stone-500 leading-relaxed">
                      {result.zone_summary?.[zone] ?? `Primary finding: ${fmt(zones[zone].dominant_concern)}`}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        )}

        {/* ── Recommended Treatments ── */}
        {result.spa_treatments?.length > 0 && (
          <Card>
            <div className="px-5 pt-5 pb-5">
              <SectionHeader title="Recommended Treatments" icon={<Sparkles size={14} />} />
              <div className="space-y-3">
                {result.spa_treatments.map((t, i) => {
                  const [name, reason] = t.includes(" — ") ? t.split(" — ") : [t, ""];
                  return (
                    <div key={i} className="flex items-start gap-3 p-3 bg-stone-50 rounded-2xl">
                      <div className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: `${tenant.primaryColor}18` }}>
                        <span className="text-sm">✦</span>
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-stone-800">{name}</p>
                        {reason && <p className="text-xs text-stone-500 mt-0.5 leading-relaxed">{reason}</p>}
                      </div>
                    </div>
                  );
                })}
              </div>
              {tenant.bookingUrl && (
                <a href={tenant.bookingUrl} target="_blank" rel="noopener noreferrer"
                  className="mt-4 w-full h-12 rounded-2xl text-white font-semibold text-sm flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
                  style={{ backgroundColor: tenant.primaryColor }}>
                  Book a Consultation <ChevronRight size={16} />
                </a>
              )}
            </div>
          </Card>
        )}

        {/* ── Routine ── */}
        {(result.morning_routine?.length > 0 || result.evening_routine?.length > 0) && (
          <Card>
            <div className="px-5 pt-5 pb-5">
              <SectionHeader title="Your Skincare Routine" />
              {/* Tabs */}
              <div className="flex bg-stone-100 rounded-2xl p-1 mb-5 gap-1">
                {(["morning", "evening"] as const).map(tab => (
                  <button key={tab} onClick={() => setRoutineTab(tab)}
                    className={`flex-1 h-9 rounded-xl text-sm font-semibold flex items-center justify-center gap-1.5 transition-all ${
                      routineTab === tab ? "bg-white text-stone-800 shadow-sm" : "text-stone-400"
                    }`}>
                    {tab === "morning" ? <Sun size={14} /> : <Moon size={14} />}
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>
              <RoutineTab steps={routineTab === "morning" ? result.morning_routine : result.evening_routine} type={routineTab} />
            </div>
          </Card>
        )}

        {/* ── Key Ingredients ── */}
        {result.key_ingredients?.length > 0 && (
          <Card>
            <div className="px-5 pt-5 pb-5">
              <SectionHeader title="Key Ingredients" icon={<Leaf size={14} />} />
              <div className="space-y-3">
                {result.key_ingredients.map((ing, i) => {
                  const [name, reason] = ing.includes(" — ") ? ing.split(" — ") : [ing, ""];
                  return (
                    <div key={i} className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-stone-300 flex-shrink-0 mt-2" />
                      <div>
                        <span className="text-sm font-semibold text-stone-800">{name}</span>
                        {reason && <span className="text-sm text-stone-500"> — {reason}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        )}

        {/* ── Lifestyle Tips ── */}
        {result.lifestyle_tips?.length > 0 && (
          <Card>
            <div className="px-5 pt-5 pb-5">
              <SectionHeader title="Lifestyle Recommendations" />
              <div className="space-y-3">
                {result.lifestyle_tips.map((tip, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="text-xs font-bold text-stone-300 w-5 flex-shrink-0 mt-0.5">{String(i + 1).padStart(2, "0")}</span>
                    <p className="text-sm text-stone-600 leading-relaxed">{tip}</p>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        )}

        {/* ── Clinical Notes ── */}
        {result.notes?.length > 0 && (
          <div className="bg-amber-50 border border-amber-200/60 rounded-3xl px-5 py-4 space-y-2">
            <div className="flex items-center gap-2">
              <Info size={14} className="text-amber-600" />
              <p className="text-xs font-bold text-amber-700 uppercase tracking-wider">Analysis Notes</p>
            </div>
            {result.notes.map((n, i) => <p key={i} className="text-sm text-amber-800 leading-relaxed">{n}</p>)}
          </div>
        )}

        {/* ── Disclaimer ── */}
        <p className="text-xs text-stone-400 text-center leading-relaxed px-4">
          {result.disclaimer}
        </p>

        {/* ── New scan ── */}
        <button onClick={() => { sessionStorage.removeItem("skinAnalysis"); router.push("/"); }}
          className="w-full h-12 rounded-2xl border border-stone-200 text-stone-500 font-medium text-sm flex items-center justify-center gap-2 hover:bg-stone-100 transition-colors">
          <RotateCcw size={15} />
          New Scan
        </button>
      </div>
    </div>
  );
}
