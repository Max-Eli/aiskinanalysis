import { NextRequest, NextResponse } from "next/server";
import { generateReport } from "@/lib/llm";
import { getCurrentTenant } from "@/lib/tenant";

const CV_SERVICE_URL = process.env.CV_SERVICE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image } = body as { image?: string };

    if (!image || !image.startsWith("data:image/")) {
      return NextResponse.json({ error: "No valid image provided" }, { status: 400 });
    }

    const tenant = await getCurrentTenant();

    // ── Step 1: CV pipeline ───────────────────────────────────────────────────
    const cvRes = await fetch(`${CV_SERVICE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });

    if (!cvRes.ok) {
      const err = await cvRes.text();
      console.error("[cv-service]", err);
      return NextResponse.json({ error: "CV analysis service error" }, { status: 502 });
    }

    const cv = await cvRes.json();

    // Log raw CV scores for debugging
    console.log("[cv-scores]", JSON.stringify({
      score: cv.overall_score,
      skin_type: cv.skin_type,
      confidence: cv.confidence,
      concerns: cv.concerns,
    }, null, 2));

    // Quality gate
    if (cv.image_quality === "poor") {
      return NextResponse.json({
        image_quality: "poor",
        quality_issues: cv.quality_issues ?? cv.notes,
      });
    }

    if (cv.confidence < 0.15) {
      return NextResponse.json({
        image_quality: "poor",
        quality_issues: ["Could not detect enough skin — ensure your face is well-lit and centred."],
      });
    }

    // ── Step 2: LLM generates professional narrative ──────────────────────────
    const report = await generateReport(cv, tenant.name, tenant.services);

    // ── Step 3: Assemble full response ────────────────────────────────────────
    return NextResponse.json({
      image_quality: cv.image_quality,
      confidence: cv.confidence,
      analysis_method: cv.analysis_method,
      skin_type: cv.skin_type,
      overall_score: cv.overall_score,
      concerns: cv.concerns,
      positives: cv.positives,
      zone_analysis: cv.zone_analysis,
      notes: cv.notes,
      // LLM narrative
      headline: report.headline,
      overview: report.overview,
      concern_details: report.concern_details,
      positive_details: report.positive_details,
      zone_summary: report.zone_summary,
      morning_routine: report.morning_routine,
      evening_routine: report.evening_routine,
      spa_treatments: report.treatment_suggestions,
      key_ingredients: report.key_ingredients,
      lifestyle_tips: report.lifestyle_tips,
      disclaimer: report.disclaimer,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Analysis failed";
    console.error("[analyze]", err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
