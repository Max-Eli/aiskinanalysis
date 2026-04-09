/**
 * Gemini 2.0 Flash Vision — skin scoring from a face image.
 * Called server-side from /api/analyze.
 */

import { GoogleGenerativeAI, GenerationConfig } from "@google/generative-ai";

const CONCERNS = [
  "acne", "hyperpigmentation", "melasma", "redness",
  "wrinkles", "fine_lines", "dryness", "pore_visibility",
  "oiliness", "dark_circles", "uneven_texture",
] as const;

const PROMPT = `You are a professional dermatology AI trained to assess skin from photographs.
Analyze this face image and return a JSON skin assessment.

Return ONLY valid JSON — no markdown, no explanation — with this exact structure:
{
  "skin_type": "<oily|dry|combination|normal|sensitive>",
  "overall_score": <integer 0-100, higher = healthier>,
  "confidence": <float 0.0-1.0, how clearly visible the skin is>,
  "concerns": {
    "acne": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "hyperpigmentation": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "melasma": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "redness": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "wrinkles": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "fine_lines": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "dryness": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "pore_visibility": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "oiliness": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "dark_circles": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "uneven_texture": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"}
  },
  "positives": {
    "hydration": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"},
    "evenness": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"},
    "luminosity": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"},
    "firmness": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"}
  },
  "zone_analysis": {
    "forehead": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "left_cheek": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "right_cheek": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "nose": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "chin": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>}
  }
}

Scoring rules:
- score 0.00–0.19 = none severity
- score 0.20–0.41 = mild severity
- score 0.42–0.64 = moderate severity
- score 0.65–1.00 = severe severity
- Severity MUST match the score range.
- Be objective. If skin looks healthy, reflect that — do not invent concerns.
- overall_score: 100 = perfect skin, 0 = severe issues.
- dominant_concern must be one of the 11 concern keys.`;

export interface GeminiSkinScores {
  skin_type: string;
  overall_score: number;
  confidence: number;
  concerns: Record<string, { score: number; severity: string }>;
  positives: Record<string, { score: number; label: string }>;
  zone_analysis: Record<string, { dominant_concern: string; score: number }>;
}

function fallback(): GeminiSkinScores {
  return {
    skin_type: "normal",
    overall_score: 70,
    confidence: 0.5,
    concerns: Object.fromEntries(CONCERNS.map(c => [c, { score: 0, severity: "none" }])),
    positives: {
      hydration: { score: 0.6, label: "good" },
      evenness: { score: 0.6, label: "good" },
      luminosity: { score: 0.6, label: "good" },
      firmness: { score: 0.6, label: "good" },
    },
    zone_analysis: {},
  };
}

export async function scoreSkin(croppedFaceDataUrl: string): Promise<GeminiSkinScores> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("[gemini] GEMINI_API_KEY not set");
    return fallback();
  }

  try {
    const genAI = new GoogleGenerativeAI(apiKey);
    const config: GenerationConfig = { temperature: 0.1 };
    const model = genAI.getGenerativeModel({
      model: "gemini-2.0-flash",
      generationConfig: config,
    });

    // Convert data URL to inline image part
    const [, b64] = croppedFaceDataUrl.split(",");
    const imagePart = {
      inlineData: { data: b64, mimeType: "image/jpeg" as const },
    };

    const result = await model.generateContent([PROMPT, imagePart]);
    const text = result.response.text().trim();

    // Strip markdown code fences if present
    const clean = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
    const match = clean.match(/\{[\s\S]*\}/);
    const raw = JSON.parse(match ? match[0] : clean) as GeminiSkinScores;

    return {
      skin_type: raw.skin_type ?? "normal",
      overall_score: Number(raw.overall_score ?? 70),
      confidence: Number(raw.confidence ?? 0.75),
      concerns: raw.concerns ?? fallback().concerns,
      positives: raw.positives ?? fallback().positives,
      zone_analysis: raw.zone_analysis ?? {},
    };
  } catch (err) {
    console.error("[gemini] vision scoring failed:", err);
    return fallback();
  }
}
