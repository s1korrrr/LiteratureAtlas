#!/usr/bin/env python3
"""
Audit LiteratureAtlas generated artifacts under Output/.

This is a lightweight sanity check intended to catch common regressions:
- Takeaways placeholders ("Not specified"/"Unknown")
- Legacy markdown bold takeaways bug ("Problem**:" missing leading "**")
- Old Obsidian markdown exports missing obsidian_format_version: 2 / callouts

Usage:
  python3 scripts/audit_output_artifacts.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


OUTPUT_ROOT = "Output"


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    detail: str


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_placeholder(text: str) -> bool:
    low = (text or "").strip().lower()
    return ("not specified" in low) or ("unknown" in low)


def audit_paper_json() -> list[Finding]:
    findings: list[Finding] = []
    paths = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "papers", "*.paper.json")))
    for path in paths:
        data = _read_json(path)
        takeaways = data.get("takeaways") or []

        if not isinstance(takeaways, list):
            findings.append(Finding("paper_json.takeaways_type", path, f"Expected list, got {type(takeaways).__name__}"))
            continue

        # Expected by prompt: 3–5 takeaways.
        if not (3 <= len(takeaways) <= 5):
            findings.append(Finding("paper_json.takeaways_len", path, f"Expected 3–5, got {len(takeaways)}"))

        # Placeholders reduce usefulness (prompts allow them, but takeaways should stay crisp).
        if any(isinstance(t, str) and _has_placeholder(t) for t in takeaways):
            findings.append(Finding("paper_json.takeaways_placeholder", path, "Contains 'not specified'/'unknown'"))

        # Legacy bug: leading '**' stripped from '**Label**:' leaving 'Label**:'.
        for t in takeaways:
            if not isinstance(t, str):
                continue
            if "**:" in t and not t.strip().startswith("**"):
                findings.append(Finding("paper_json.takeaways_legacy_bold_bug", path, f"Suspect: {t.strip()[:90]}"))
                break

    return findings


def audit_paper_obsidian_linkage() -> list[Finding]:
    findings: list[Finding] = []

    paper_paths = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "papers", "*.paper.json")))
    note_paths = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "obsidian", "papers", "*.md")))

    note_ids: set[str] = set()
    note_id_re = re.compile(r"\[([0-9A-Fa-f-]{36})\]\.md$")
    for path in note_paths:
        m = note_id_re.search(path)
        if m:
            note_ids.add(m.group(1).upper())

    for path in paper_paths:
        data = _read_json(path)
        raw_id = data.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            findings.append(Finding("paper_json.missing_id", path, "Missing paper id"))
            continue

        paper_id = raw_id.strip().upper()
        if paper_id not in note_ids:
            findings.append(Finding("paper_json.missing_obsidian_note", path, f"No Obsidian note with id [{paper_id}].md"))

    return findings


def audit_trading_lens_json() -> list[Finding]:
    findings: list[Finding] = []
    paths = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "papers", "*.paper.json")))
    for path in paths:
        data = _read_json(path)
        lens = data.get("trading_lens")
        if lens is None:
            continue
        if not isinstance(lens, dict):
            findings.append(Finding("paper_json.trading_lens_type", path, f"Expected object, got {type(lens).__name__}"))
            continue

        # Prompt limits (soft sanity checks).
        alpha = lens.get("alpha_hypotheses")
        if isinstance(alpha, list) and len(alpha) > 3:
            findings.append(Finding("trading_lens.alpha_hypotheses_len", path, f"Expected <=3, got {len(alpha)}"))

        risk = lens.get("risk_flags")
        if isinstance(risk, list) and len(risk) > 4:
            findings.append(Finding("trading_lens.risk_flags_len", path, f"Expected <=4, got {len(risk)}"))

        # Score ranges.
        scores = lens.get("scores")
        if isinstance(scores, dict):
            for k in ("novelty", "usability", "strategy_impact"):
                v = scores.get(k)
                if isinstance(v, (int, float)) and not (0 <= float(v) <= 10):
                    findings.append(Finding("trading_lens.score_range", path, f"{k} out of range: {v}"))
            v = scores.get("confidence")
            if isinstance(v, (int, float)) and not (0 <= float(v) <= 1):
                findings.append(Finding("trading_lens.confidence_range", path, f"confidence out of range: {v}"))

    return findings


def _obsidian_note_checks(path: str, required_version: int = 2) -> list[Finding]:
    findings: list[Finding] = []
    text = _read_text(path)

    if "<!-- atlas:begin -->" not in text or "<!-- atlas:end -->" not in text:
        findings.append(Finding("obsidian.md.missing_markers", path, "Missing <!-- atlas:begin/end --> markers"))

    if f"obsidian_format_version: {required_version}" not in text:
        findings.append(Finding("obsidian.md.missing_format_version", path, f"Missing obsidian_format_version: {required_version}"))

    return findings


def audit_obsidian_notes() -> list[Finding]:
    findings: list[Finding] = []

    # Papers
    for path in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "obsidian", "papers", "*.md"))):
        findings.extend(_obsidian_note_checks(path))
        text = _read_text(path)
        if "> [!info] Meta" not in text:
            findings.append(Finding("obsidian.paper.missing_meta_callout", path, "Expected Meta callout block"))

    # Clusters
    for path in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "obsidian", "clusters", "*.md"))):
        findings.extend(_obsidian_note_checks(path))

    # Atlas
    atlas_path = os.path.join(OUTPUT_ROOT, "obsidian", "Atlas.md")
    if os.path.exists(atlas_path):
        findings.extend(_obsidian_note_checks(atlas_path))
    else:
        findings.append(Finding("obsidian.atlas.missing", atlas_path, "Atlas.md not found"))

    # Setup note (optional but expected if exporter ran)
    setup_path = os.path.join(OUTPUT_ROOT, "obsidian", "Obsidian Setup.md")
    if os.path.exists(setup_path):
        findings.extend(_obsidian_note_checks(setup_path))
    else:
        findings.append(Finding("obsidian.setup.missing", setup_path, "Obsidian Setup.md not found"))

    # Snippet file (optional but expected if exporter ran)
    snippet_path = os.path.join(OUTPUT_ROOT, "obsidian", ".obsidian", "snippets", "literature-atlas.css")
    if not os.path.exists(snippet_path):
        findings.append(Finding("obsidian.snippet.missing", snippet_path, "CSS snippet not found"))

    return findings


def _summarize(findings: Iterable[Finding]) -> str:
    by_kind: dict[str, int] = {}
    total = 0
    for f in findings:
        total += 1
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1

    lines = []
    lines.append(f"Findings: {total}")
    for kind, count in sorted(by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {kind}: {count}")
    return "\n".join(lines)


def _write_report(findings: list[Finding]) -> str | None:
    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(report_dir, f"output_audit_{stamp}.md")

    lines = []
    lines.append("# LiteratureAtlas Output Audit")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{stamp}`")
    lines.append(f"- Total findings: `{len(findings)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```")
    lines.append(_summarize(findings))
    lines.append("```")
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for f in findings[:250]:
        lines.append(f"- **{f.kind}** — `{f.path}` — {f.detail}")
    if len(findings) > 250:
        lines.append(f"- _(truncated; showing first 250 of {len(findings)})_")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return path


def main() -> int:
    if not os.path.isdir(OUTPUT_ROOT):
        print(f"Missing `{OUTPUT_ROOT}/` folder; nothing to audit.", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    findings.extend(audit_paper_json())
    findings.extend(audit_paper_obsidian_linkage())
    findings.extend(audit_trading_lens_json())
    findings.extend(audit_obsidian_notes())

    print(_summarize(findings))
    report = _write_report(findings)
    if report:
        print(f"Report written: {report}")

    # Non-zero exit if any issues found.
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
