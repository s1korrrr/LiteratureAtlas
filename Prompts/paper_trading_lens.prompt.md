Title: {{title}}
Keywords: {{keywords}}

Technical summary:
{{summary}}

Takeaways:
{{takeaways}}

Return VALID JSON ONLY (no extra text) with keys:
title,
trading_tags, asset_classes, horizons, signal_archetypes,
where_it_fits { pipeline_stage, primary_use },
alpha_hypotheses [{ hypothesis, features, target, horizon }],
data_requirements { must_have, nice_to_have },
evaluation_notes { recommended_metrics, must_check },
risk_flags,
scores { novelty, usability, strategy_impact, confidence },
one_line_verdict.

Rules:
- If not specified, use "Unknown" or [].
- alpha_hypotheses: 1-3 items max.
- risk_flags: 0-4 items max.
