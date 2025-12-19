You are given short paper records for one thematic cluster. Each record may include a title, keywords, a short summary, and a few takeaways.

TASK 1: Invent a short cluster name (3-6 words). Prefer names that sound like a quant research theme.
TASK 2: Write one paragraph meta-summary (4-6 sentences) describing the shared idea and why it matters for systematic trading research.
TASK 3: Add a compact "Trading lens" block for how this cluster could be used in quant strategy development.

IMPORTANT OUTPUT RULES:
- The FIRST two lines MUST be exactly:
  "Cluster name: ..."
  "Meta-summary: ..."
- After that, include "Trading lens:" with bullet points.
- Do not invent paper-specific results not implied by the summaries.

Respond in this format:

Cluster name: <name>
Meta-summary: <paragraph>
Trading lens:
- Tags: <3-8 comma-separated tags>
- Likely use in a strategy stack: <1 sentence>
- Asset classes: <comma-separated or "Unknown">
- Typical horizon: <e.g., intraday/daily/weekly/long-horizon/unknown>
- Signal archetypes: <comma-separated, e.g., cross-sectional ranking, time-series forecasting, regime classifier, microstructure alpha>
- Key risks: <2-4 short items>
- Scores (0-10): novelty=<n>, usability=<u>, strategy_impact=<s>
- Prototype ideas: <2-4 short bullets>

Summaries:
{{summaries}}
