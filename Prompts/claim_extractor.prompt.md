Extract atomic, testable claims from the provided paper text.

Text:
{{text}}

Return a JSON array. Each element must have:
- "id": short string (e.g., "c1")
- "claim": one sentence, specific and falsifiable
- "type": one of ["background","problem","method","result","limitation","assumption"]
- "evidence": a short supporting snippet (<=25 words) copied from the text when possible
- "confidence": number from 0.0 to 1.0 (based only on clarity in the text)

Rules:
- Use ONLY what is supported by the text.
- If evidence is not explicitly present, still extract the claim but set evidence to "" and lower confidence.
- Do not include more than 12 claims.
Return ONLY valid JSON.
