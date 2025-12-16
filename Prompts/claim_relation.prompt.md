Determine the relationship between two claims using ONLY the provided claim texts/snippets.

Claim A:
{{claim_a}}
Evidence A:
{{evidence_a}}

Claim B:
{{claim_b}}
Evidence B:
{{evidence_b}}

Return a JSON object with:
- "relation": one of ["supports","contradicts","extends","equivalent","independent","unclear"]
- "confidence": 0.0 to 1.0
- "rationale": 1–2 sentences grounded in the claim texts

Rules:
- Do not introduce external facts.
- If not enough information, use relation "unclear".
Return ONLY valid JSON.
