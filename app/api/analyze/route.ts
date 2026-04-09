import { NextRequest, NextResponse } from "next/server";
import { generateReport } from "@/lib/llm";
import { scoreSkin } from "@/lib/gemini";
import { getCurrentTenant } from "@/lib/tenant";

export const maxDuration = 60; // Vercel max function timeout

const CV_SERVICE_URL = process.env.CV_SERVICE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image } = body as { image?: string };

    if (!image || !image.startsWith("data:image/")) {
      return NextResponse.json({ error: "No valid image provided" }, { status: 400 });
    }

    const tenant = await getCurrentTenant();

    // ── Step 1: CV service — face segmentation + crop ─────────────────────────
    console.log("[cv-service] URL:", CV_SERVICE_URL);
    let cvRes: Response;
    try {
      cvRes = await fetch(`${CV_SERVICE_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
    } catch (fetchErr) {
      console.error("[cv-service] fetch failed:", fetchErr);
      return NextResponse.json({ error: `Cannot reach CV service: ${fetchErr}` }, { status: 502 });
    }

    if (!cvRes.ok) {
      const err = await cvRes.text();
      console.error("[cv-service] bad status:", cvRes.status, err);
      return NextResponse.json({ error: `CV service returned ${cvRes.status}: ${err}` }, { status: 502 });
    }

    const cv = await cvRes.json();

    // Quality gate from CV service
    if (cv.image_quality === "poor") {
      return NextResponse.json({
        image_quality: "poor",
        quality_issues: cv.quality_issues,
      });
    }

    const croppedFace: string = cv.cropped_face;
    if (!croppedFace) {
      return NextResponse.json({
        image_quality: "poor",
        quality_issues: ["No face detected. Ensure your face is clearly visible and well-lit."],
      });
    }

    // ── Step 2: Gemini Vision — skin scoring ─────────────────────────────────
    console.log("[gemini] scoring skin…");
    let scores;
    try {
      scores = await scoreSkin(croppedFace);
    } catch (geminiErr) {
      console.error("[gemini] failed:", geminiErr);
      return NextResponse.json({
        error: "Skin analysis failed — could not score the image. Please try again with better lighting.",
      }, { status: 500 });
    }

    // ── Step 3: LLM generates professional narrative ──────────────────────────
    const cvForReport = {
      image_quality: cv.image_quality,
      confidence: scores.confidence,
      analysis_method: "gemini-vision",
      skin_type: scores.skin_type,
      overall_score: scores.overall_score,
      concerns: scores.concerns,
      positives: scores.positives,
      zone_analysis: scores.zone_analysis,
      notes: [] as string[],
    };

    const report = await generateReport(cvForReport, tenant.name, tenant.services);

    // ── Step 4: Assemble full response ────────────────────────────────────────
    return NextResponse.json({
      image_quality: cv.image_quality,
      confidence: scores.confidence,
      analysis_method: "gemini-vision",
      skin_type: scores.skin_type,
      overall_score: scores.overall_score,
      concerns: scores.concerns,
      positives: scores.positives,
      zone_analysis: scores.zone_analysis,
      notes: [],
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
