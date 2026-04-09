/**
 * Gemini 2.0 Flash Vision — skin scoring from a face image.
 * Uses responseMimeType:"application/json" to guarantee structured output.
 * Throws on failure — caller must handle and surface the error.
 */

import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";

// ─── Response schema (forces Gemini to always return valid JSON) ──────────────

const CONCERN_SCHEMA = {
  type: SchemaType.OBJECT,
  properties: {
    score:    { type: SchemaType.NUMBER },
    severity: { type: SchemaType.STRING },
  },
  required: ["score", "severity"],
};

const POSITIVE_SCHEMA = {
  type: SchemaType.OBJECT,
  properties: {
    score: { type: SchemaType.NUMBER },
    label: { type: SchemaType.STRING },
  },
  required: ["score", "label"],
};

const ZONE_SCHEMA = {
  type: SchemaType.OBJECT,
  properties: {
    dominant_concern: { type: SchemaType.STRING },
    score:            { type: SchemaType.NUMBER },
  },
  required: ["dominant_concern", "score"],
};

const RESPONSE_SCHEMA = {
  type: SchemaType.OBJECT,
  properties: {
    skin_type:     { type: SchemaType.STRING },
    overall_score: { type: SchemaType.INTEGER },
    confidence:    { type: SchemaType.NUMBER },
    concerns: {
      type: SchemaType.OBJECT,
      properties: {
        acne:             CONCERN_SCHEMA,
        hyperpigmentation: CONCERN_SCHEMA,
        melasma:          CONCERN_SCHEMA,
        redness:          CONCERN_SCHEMA,
        wrinkles:         CONCERN_SCHEMA,
        fine_lines:       CONCERN_SCHEMA,
        dryness:          CONCERN_SCHEMA,
        pore_visibility:  CONCERN_SCHEMA,
        oiliness:         CONCERN_SCHEMA,
        dark_circles:     CONCERN_SCHEMA,
        uneven_texture:   CONCERN_SCHEMA,
      },
    },
    positives: {
      type: SchemaType.OBJECT,
      properties: {
        hydration:  POSITIVE_SCHEMA,
        evenness:   POSITIVE_SCHEMA,
        luminosity: POSITIVE_SCHEMA,
        firmness:   POSITIVE_SCHEMA,
      },
    },
    zone_analysis: {
      type: SchemaType.OBJECT,
      properties: {
        forehead:    ZONE_SCHEMA,
        left_cheek:  ZONE_SCHEMA,
        right_cheek: ZONE_SCHEMA,
        nose:        ZONE_SCHEMA,
        chin:        ZONE_SCHEMA,
      },
    },
  },
  required: ["skin_type", "overall_score", "confidence", "concerns", "positives", "zone_analysis"],
};

// ─── Prompt ───────────────────────────────────────────────────────────────────

const PROMPT = `You are a clinical dermatology AI performing a professional skin assessment for a medical spa. Study this face image carefully before scoring.

STEP 1 — OBSERVE: Examine each facial zone (forehead, cheeks, nose, chin, under-eyes) and note what you actually see.

STEP 2 — SCORE: Based on your observations, assign scores for each concern.

SCORING GUIDE (be honest and precise — clients rely on this for treatment decisions):
• acne: Count active lesions, blackheads, whiteheads, papules, pustules, post-acne marks. Even 1-2 small pimples = 0.22+
• hyperpigmentation: Look for dark spots, sun spots, uneven tone patches. Minor unevenness = 0.20+
• melasma: Larger bilateral brown/grey patches on cheeks, forehead, upper lip. Absent unless clearly visible = 0.0
• redness: Flushing, visible capillaries, rosy/red areas. Slight flushing = 0.20+
• wrinkles: Deep-set lines — forehead furrows, nasolabial folds, crow's feet. Fine = 0.20+, obvious = 0.45+
• fine_lines: Subtle surface lines, especially under-eye, forehead, perioral. Very common in adults = 0.20+
• dryness: Dull finish, flakiness, tight-looking skin, lack of natural sheen. Slightly dry = 0.20+
• pore_visibility: Enlarged pores on nose, T-zone. Visible pores are normal and common = 0.20+
• oiliness: Shine or specular highlights especially T-zone. Slight shine = 0.15+
• dark_circles: Under-eye discolouration, darkness, shadows. Mild = 0.20+
• uneven_texture: Surface irregularities, rough patches. Minor = 0.20+

SEVERITY MAPPING (MUST match score):
• 0.00–0.19 → "none"
• 0.20–0.41 → "mild"
• 0.42–0.64 → "moderate"
• 0.65–1.00 → "severe"

POSITIVE ATTRIBUTES:
• hydration: Does the skin look plump and moisturised? (dull/flaky = low, dewy/plump = high)
• evenness: Is the skin tone uniform? (spots/patches = low, uniform = high)
• luminosity: Does the skin have a healthy natural glow? (dull = low, radiant = high)
• firmness: Does the skin look firm and elastic? (sagging/lined = low, taut = high)

POSITIVE LABELS: 0.0–0.39 = "needs improvement", 0.40–0.59 = "fair", 0.60–0.79 = "good", 0.80–1.0 = "excellent"

OVERALL SCORE: 100 = flawless clinical skin, 75 = healthy average adult skin, 50 = moderate concerns, 25 = significant issues. Most adults score 55–80.

dominant_concern per zone must be one of: acne, hyperpigmentation, melasma, redness, wrinkles, fine_lines, dryness, pore_visibility, oiliness, dark_circles, uneven_texture`;

// ─── Types ────────────────────────────────────────────────────────────────────

export interface GeminiSkinScores {
  skin_type: string;
  overall_score: number;
  confidence: number;
  concerns: Record<string, { score: number; severity: string }>;
  positives: Record<string, { score: number; label: string }>;
  zone_analysis: Record<string, { dominant_concern: string; score: number }>;
}

// ─── CV measurement formatter ─────────────────────────────────────────────────

function formatMeasurements(m: Record<string, unknown>): string {
  if (!m || Object.keys(m).length === 0) return "Not available.";

  const zr = (m.zone_redness as Record<string, number>) ?? {};
  const ed = (m.edge_density as Record<string, number>) ?? {};

  const lines = [
    `• Dark spots detected: ${m.dark_spot_count ?? 0} spots covering ${m.dark_spot_area_pct ?? 0}% of skin`,
    `  (>5 spots = notable hyperpigmentation; >15 spots = significant)`,
    `• Red/inflamed spots: ${m.red_spot_count ?? 0} covering ${m.red_spot_area_pct ?? 0}% of skin`,
    `  (>2 red spots = acne present; >8 = moderate acne)`,
    `• Overall redness index: ${m.redness_index ?? 0} (>0.05=slight, >0.12=moderate, >0.20=significant)`,
    `• Zone redness: forehead=${zr.forehead?.toFixed(3) ?? "n/a"}, left cheek=${zr.left_cheek?.toFixed(3) ?? "n/a"}, right cheek=${zr.right_cheek?.toFixed(3) ?? "n/a"}, nose=${zr.nose?.toFixed(3) ?? "n/a"}`,
    `• Pigmentation std dev: ${m.pigmentation_std ?? 0} L* (<8=even, 8–14=minor, >14=notable, >20=significant)`,
    `• Dark circle luminance delta: ${m.dark_circle_delta ?? 0} L* (>4=mild, >8=moderate, >12=significant)`,
    `• Oiliness (specular highlights): ${m.oiliness_pct ?? 0}% of skin area (>1.5%=slight, >4%=oily)`,
    `• Mean skin saturation: ${m.mean_saturation ?? 0} (>0.30=hydrated, <0.15=dry/dull)`,
    `• Texture roughness: ${m.texture_roughness ?? 0} (<0.010=smooth, 0.010–0.020=moderate, >0.020=rough)`,
    `• Pore density (nose): ${m.pore_density ?? 0} (<0.008=fine, 0.008–0.018=visible, >0.018=enlarged)`,
    `• Edge density (lines/wrinkles): forehead=${ed.forehead?.toFixed(4) ?? "n/a"}, left cheek=${ed.left_cheek?.toFixed(4) ?? "n/a"}, chin=${ed.chin?.toFixed(4) ?? "n/a"}`,
    `  (<0.04=smooth, 0.04–0.08=some lines, >0.08=notable lines)`,
  ];
  return lines.join("\n");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export async function scoreSkin(
  croppedFaceDataUrl: string,
  cvMeasurements: Record<string, unknown> = {}
): Promise<GeminiSkinScores> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.0-flash",
    generationConfig: {
      temperature: 0.2,
      responseMimeType: "application/json",
      responseSchema: RESPONSE_SCHEMA,
    },
  });

  const [, b64] = croppedFaceDataUrl.split(",");
  const imagePart = { inlineData: { data: b64, mimeType: "image/jpeg" as const } };

  // Inject CV measurements into the prompt as grounding data
  const measurementBlock = `\n\nOBJECTIVE CV MEASUREMENTS (pixel-level analysis — use these to calibrate your scores):\n${formatMeasurements(cvMeasurements)}\n\nThese measurements are computed from the actual pixel data. Your scores MUST be consistent with them. If dark_spot_count is 12, hyperpigmentation cannot be "none". If red_spot_count is 5, acne cannot be "none".`;

  const result = await model.generateContent([PROMPT + measurementBlock, imagePart]);
  const text = result.response.text().trim();
  console.log("[gemini] raw response (first 300):", text.slice(0, 300));

  const raw = JSON.parse(text) as GeminiSkinScores;

  console.log("[gemini] scores — overall:", raw.overall_score, "skin_type:", raw.skin_type,
    "top concerns:", Object.entries(raw.concerns ?? {})
      .filter(([, v]) => v.score > 0.1)
      .sort((a, b) => b[1].score - a[1].score)
      .slice(0, 4)
      .map(([k, v]) => `${k}:${v.score.toFixed(2)}`)
      .join(", ")
  );

  return {
    skin_type:     raw.skin_type    ?? "normal",
    overall_score: Number(raw.overall_score ?? 70),
    confidence:    Number(raw.confidence    ?? 0.75),
    concerns:      raw.concerns     ?? {},
    positives:     raw.positives    ?? {},
    zone_analysis: raw.zone_analysis ?? {},
  };
}
