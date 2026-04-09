/**
 * LLM integration — generates a professional skin report from CV measurements.
 * The LLM writes the narrative; all scores come from the CV pipeline.
 */

import { Ollama } from "ollama";

export interface CVAnalysisResult {
  image_quality: string;
  confidence: number;
  analysis_method: string;
  skin_type: string;
  overall_score: number;
  concerns: Record<string, { score: number; severity: string }>;
  positives: Record<string, { score: number; label: string }>;
  zone_analysis: Record<string, { dominant_concern: string; score: number }>;
  notes: string[];
}

export interface LLMReport {
  headline: string;
  overview: string;
  concern_details: Record<string, string>;
  positive_details: Record<string, string>;
  zone_summary: Record<string, string>;
  morning_routine: string[];
  evening_routine: string[];
  treatment_suggestions: string[];
  key_ingredients: string[];
  lifestyle_tips: string[];
  disclaimer: string;
}

function getClient(): Ollama {
  const host = process.env.LLM_BASE_URL ?? "https://api.ollama.com";
  const apiKey = process.env.LLM_API_KEY;
  return new Ollama({
    host,
    ...(apiKey ? { headers: { Authorization: `Bearer ${apiKey}` } } : {}),
  });
}

function buildPrompt(cv: CVAnalysisResult, spaName: string, spaServices: string[]): string {
  const detected = Object.entries(cv.concerns)
    .filter(([, d]) => d.severity !== "none")
    .sort((a, b) => b[1].score - a[1].score)
    .map(([n, d]) => `  • ${n.replace(/_/g, " ")}: ${d.severity} (${Math.round(d.score * 100)}%)`)
    .join("\n") || "  • No significant concerns detected";

  const healthy = Object.entries(cv.positives)
    .sort((a, b) => b[1].score - a[1].score)
    .map(([n, d]) => `  • ${n}: ${d.label} (${Math.round(d.score * 100)}%)`)
    .join("\n");

  const zones = Object.entries(cv.zone_analysis)
    .map(([z, d]) => `  • ${z.replace(/_/g, " ")}: dominant concern is ${d.dominant_concern.replace(/_/g, " ")} (${Math.round(d.score * 100)}%)`)
    .join("\n");

  const services = spaServices.map(s => `  • ${s}`).join("\n");

  // Find dominant concern and best positive for narrative anchoring
  const topConcern = Object.entries(cv.concerns)
    .filter(([, d]) => d.severity !== "none")
    .sort((a, b) => b[1].score - a[1].score)[0];
  const topPositive = Object.entries(cv.positives)
    .sort((a, b) => b[1].score - a[1].score)[0];

  return `You are a senior dermatology-trained skin analysis AI producing a professional skin health report for a client of ${spaName}.

═══════════════════════════════════════
CV MEASUREMENT DATA (authoritative — do not alter these numbers)
═══════════════════════════════════════

OVERALL SKIN HEALTH SCORE: ${cv.overall_score}/100
SKIN TYPE: ${cv.skin_type}
ANALYSIS CONFIDENCE: ${Math.round(cv.confidence * 100)}%

CONCERNS DETECTED (ordered by severity):
${detected}

SKIN STRENGTHS (what is healthy):
${healthy}

ZONE-BY-ZONE BREAKDOWN:
${zones}

CV CLINICAL NOTES: ${cv.notes.length > 0 ? cv.notes.join("; ") : "none"}

SPA TREATMENTS AVAILABLE AT ${spaName.toUpperCase()}:
${services}

═══════════════════════════════════════
REPORT REQUIREMENTS
═══════════════════════════════════════

Write a professional, empathetic skin analysis report. Follow these rules strictly:

1. HEADLINE: One punchy sentence that captures this specific person's skin profile (mention skin type and dominant concern: "${topConcern?.[0] ?? "healthy skin"}").

2. OVERVIEW: 3-4 sentences. Lead with the score (${cv.overall_score}/100), name the top strengths (especially "${topPositive?.[0]}"), then address the primary concern ("${topConcern?.[0] ?? "none"}"). Be specific — no generic filler.

3. CONCERN_DETAILS: For each detected concern ONLY (skip severity "none"), write 1-2 sentences explaining what this measurement means for the client's skin and why it matters. Use plain language a client would understand.

4. POSITIVE_DETAILS: For each positive attribute, write 1 sentence explaining why this is a strength worth protecting.

5. ZONE_SUMMARY: For each face zone, write 1 sentence describing what was found there and what it means.

6. MORNING_ROUTINE: 5 specific steps tailored to this person's skin type ("${cv.skin_type}") and top concerns. Include product types AND key active ingredients for each step.

7. EVENING_ROUTINE: 5 specific steps. Evening routine should address repair and treatment of the detected concerns, especially "${topConcern?.[0] ?? "general skin health"}".

8. TREATMENT_SUGGESTIONS: 2-4 treatments from the spa's menu that directly address the top concerns. Explain briefly why each is recommended for this person's specific findings.

9. KEY_INGREDIENTS: 5-7 specific cosmeceutical ingredients that target the detected concerns. For each, make the array entry: "Ingredient Name — why it helps this person's skin".

10. LIFESTYLE_TIPS: 3-4 personalised tips (diet, sleep, sun protection, etc.) based on the detected concerns.

Return ONLY valid JSON — no markdown, no extra text:
{
  "headline": "...",
  "overview": "...",
  "concern_details": {
    "<concern_name>": "<explanation>"
  },
  "positive_details": {
    "<positive_name>": "<explanation>"
  },
  "zone_summary": {
    "<zone_name>": "<what was found>"
  },
  "morning_routine": ["Step 1: ...", "Step 2: ..."],
  "evening_routine": ["Step 1: ...", "Step 2: ..."],
  "treatment_suggestions": ["Treatment — reason"],
  "key_ingredients": ["Ingredient — why it helps"],
  "lifestyle_tips": ["Tip 1", "Tip 2"],
  "disclaimer": "This analysis is generated by an AI-powered computer-vision system for cosmetic informational purposes only. It does not constitute a medical diagnosis. Please consult a licensed dermatologist for medical skin concerns."
}`;
}

export async function generateReport(
  cv: CVAnalysisResult,
  spaName: string,
  spaServices: string[]
): Promise<LLMReport> {
  const client = getClient();
  const model = process.env.LLM_MODEL ?? "glm-5";

  const prompt = buildPrompt(cv, spaName, spaServices);

  const response = await client.chat({
    model,
    messages: [{ role: "user", content: prompt }],
    stream: false,
    options: { temperature: 0.2 },
  });

  const raw = response.message.content.trim();
  console.log("[llm] raw response (first 200):", raw.slice(0, 200));

  // Extract JSON block — model sometimes wraps in markdown or prefixes with text
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) {
    console.error("[llm] no JSON found in response:", raw.slice(0, 500));
    throw new Error("LLM did not return valid JSON. Response: " + raw.slice(0, 100));
  }

  try {
    return JSON.parse(match[0]) as LLMReport;
  } catch (e) {
    console.error("[llm] JSON parse failed:", e, "\nText:", match[0].slice(0, 500));
    throw new Error("Failed to parse LLM response as JSON.");
  }
}
