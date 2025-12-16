Simulate a structured debate between two research ideas using ONLY the provided context.

Side A: {{left_title}}
Summary A: {{left_summary}}
Exemplars A:
{{left_snippets}}

Side B: {{right_title}}
Summary B: {{right_summary}}
Exemplars B:
{{right_snippets}}

Output format:
- Start with: Rounds: {{rounds}} (steps: {{steps}})
- Then produce {{rounds}} rounds. Each round has:
  - A: 2-3 sentences
  - B: 2-3 sentences
- End with bullet lists for: Agreements, Disagreements, Next experiments.

Keep it grounded: do not invent citations or facts not implied by the summaries.
