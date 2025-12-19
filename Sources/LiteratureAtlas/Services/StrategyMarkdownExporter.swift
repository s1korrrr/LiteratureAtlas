import Foundation

enum StrategyMarkdownExporter {
    static let obsidianFormatVersion = 2

    static func write(project: StrategyProject, outputRoot: URL, paperTitlesByID: [UUID: String]) throws -> URL {
        let folder = outputRoot
            .appendingPathComponent("obsidian", isDirectory: true)
            .appendingPathComponent("strategies", isDirectory: true)

        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let fileName = markdownFileName(for: project)
        let destination = folder.appendingPathComponent(fileName)

        try reconcileExistingFile(strategyID: project.id, folder: folder, destination: destination)

        let markdown: String
        if FileManager.default.fileExists(atPath: destination.path),
           let existing = try? String(contentsOf: destination, encoding: .utf8) {
            markdown = updateMarkdown(existing: existing, project: project, paperTitlesByID: paperTitlesByID)
        } else {
            markdown = renderNewMarkdown(for: project, paperTitlesByID: paperTitlesByID)
        }

        try Data(markdown.utf8).write(to: destination, options: .atomic)
        return destination
    }

    static func markdownFileName(for project: StrategyProject) -> String {
        let baseName = project.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Strategy" : project.title
        let sanitized = sanitizeFileName(baseName, maxLength: 140)
        return "\(sanitized) [\(project.id.uuidString)].md"
    }

    private static func reconcileExistingFile(strategyID: UUID, folder: URL, destination: URL) throws {
        let fm = FileManager.default
        let idToken = strategyID.uuidString

        let candidates = (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])) ?? []
        guard let existing = candidates.first(where: { url in
            url.pathExtension.lowercased() == "md" && url.lastPathComponent.contains(idToken)
        }) else {
            return
        }

        guard existing.standardizedFileURL != destination.standardizedFileURL else { return }

        if fm.fileExists(atPath: destination.path) {
            try? fm.removeItem(at: destination)
        }
        try? fm.moveItem(at: existing, to: destination)
    }

    private static let managedBlockBegin = "<!-- atlas:begin -->"
    private static let managedBlockEnd = "<!-- atlas:end -->"

    private static func renderNewMarkdown(for project: StrategyProject, paperTitlesByID: [UUID: String]) -> String {
        let frontmatter = renderFrontmatter(for: project, preservingFrom: nil)
        let managed = renderManagedBlock(for: project, paperTitlesByID: paperTitlesByID)
        return [
            frontmatter,
            "",
            managed,
            "",
            "## Notes",
            "",
            ""
        ].joined(separator: "\n")
    }

    private static func updateMarkdown(existing: String, project: StrategyProject, paperTitlesByID: [UUID: String]) -> String {
        let (existingFrontmatter, existingBody) = splitFrontmatter(from: existing)
        let preservedTail = preserveTail(fromBody: existingBody)

        let frontmatter = renderFrontmatter(for: project, preservingFrom: existingFrontmatter)
        let managed = renderManagedBlock(for: project, paperTitlesByID: paperTitlesByID)

        return [
            frontmatter,
            "",
            managed,
            preservedTail
        ].joined(separator: "\n")
    }

    private static func renderFrontmatter(for project: StrategyProject, preservingFrom existingFrontmatter: String?) -> String {
        var managed: [String] = []
        managed.append("type: strategy")
        managed.append("obsidian_format_version: \(obsidianFormatVersion)")
        managed.append("id: \(yamlString(project.id.uuidString))")
        managed.append("title: \(yamlString(project.title.isEmpty ? "Strategy" : project.title))")
        managed.append("cssclass: atlas-strategy")
        let aliases = [ObsidianIDs.strategyAlias(project.id), project.title].filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        managed.append("aliases: \(yamlStringArray(aliases))")
        managed.append("created_at: \(yamlString(iso8601(project.createdAt)))")
        managed.append("updated_at: \(yamlString(iso8601(project.updatedAt)))")
        if let archived = project.archived { managed.append("archived: \(archived)") }
        if let tags = project.tags, !tags.isEmpty { managed.append("tags: \(yamlStringArray(tags))") }
        if !project.paperIDs.isEmpty {
            managed.append("paper_ids: \(yamlStringArray(project.paperIDs.map(\.uuidString)))")
        }

        let preserved = preserveFrontmatterLines(existing: existingFrontmatter, excludingKeys: managedFrontmatterKeys)
        return ([ "---" ] + managed + preserved + [ "---" ]).joined(separator: "\n")
    }

    private static var managedFrontmatterKeys: [String] {
        [
            "type",
            "obsidian_format_version",
            "id",
            "title",
            "cssclass",
            "aliases",
            "created_at",
            "updated_at",
            "archived",
            "tags",
            "paper_ids"
        ]
    }

    private static func renderManagedBlock(for project: StrategyProject, paperTitlesByID: [UUID: String]) -> String {
        var lines: [String] = []
        lines.append(managedBlockBegin)
        lines.append("# \(project.title.isEmpty ? "Strategy" : project.title)")
        lines.append("")

        if !project.paperIDs.isEmpty {
            lines.append("## Papers")
            for pid in project.paperIDs {
                let title = paperTitlesByID[pid] ?? "Paper"
                lines.append("- \(title) (\(pid.uuidString))")
            }
            lines.append("")
        }

        if let idea = project.idea?.text, !idea.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("## Idea")
            lines.append(idea.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }

        if !project.features.isEmpty {
            lines.append("## Features")
            for feature in project.features {
                let name = feature.name.trimmingCharacters(in: .whitespacesAndNewlines)
                if name.isEmpty { continue }
                if let desc = feature.description?.trimmingCharacters(in: .whitespacesAndNewlines), !desc.isEmpty {
                    lines.append("- **\(name)**: \(desc)")
                } else {
                    lines.append("- **\(name)**")
                }
            }
            lines.append("")
        }

        if let model = project.model {
            let name = model.name.trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty {
                lines.append("## Model")
                lines.append("- Name: \(name)")
                if let desc = model.description?.trimmingCharacters(in: .whitespacesAndNewlines), !desc.isEmpty {
                    lines.append("- Notes: \(desc)")
                }
                lines.append("")
            }
        }

        if let trade = project.tradePlan {
            let hasAny = [
                trade.universe,
                trade.horizon,
                trade.signalDefinition,
                trade.portfolioConstruction,
                trade.costsAndSlippage,
                trade.constraints,
                trade.executionNotes
            ]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .contains(where: { !$0.isEmpty })
            if hasAny {
                lines.append("## Trade Plan")
                if let v = trade.universe?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Universe: \(v)") }
                if let v = trade.horizon?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Horizon: \(v)") }
                if let v = trade.signalDefinition?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Signal: \(v)") }
                if let v = trade.portfolioConstruction?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Portfolio: \(v)") }
                if let v = trade.costsAndSlippage?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Costs: \(v)") }
                if let v = trade.constraints?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Constraints: \(v)") }
                if let v = trade.executionNotes?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { lines.append("- Execution: \(v)") }
                lines.append("")
            }
        }

        if !project.decisions.isEmpty {
            lines.append("## Decisions")
            for d in project.decisions.sorted(by: { $0.madeAt > $1.madeAt }) {
                let rationale = d.rationale.trimmingCharacters(in: .whitespacesAndNewlines)
                if rationale.isEmpty {
                    lines.append("- \(d.kind.label) (\(iso8601(d.madeAt)))")
                } else {
                    lines.append("- \(d.kind.label) (\(iso8601(d.madeAt))): \(rationale)")
                }
            }
            lines.append("")
        }

        if !project.outcomes.isEmpty {
            lines.append("## Outcomes")
            for o in project.outcomes.sorted(by: { $0.measuredAt > $1.measuredAt }) {
                var parts: [String] = []
                parts.append(o.kind.label)
                parts.append(iso8601(o.measuredAt))
                if let sharpe = o.metrics?.sharpe { parts.append("Sharpe \(String(format: "%.3f", sharpe))") }
                if let pnl = o.metrics?.pnl { parts.append("PnL \(String(format: "%.3f", pnl))") }
                if let dd = o.metrics?.maxDrawdown { parts.append("MaxDD \(String(format: "%.3f", dd))") }
                lines.append("- \(parts.joined(separator: " · "))")
                if let notes = o.notes?.trimmingCharacters(in: .whitespacesAndNewlines), !notes.isEmpty {
                    lines.append("  - \(notes)")
                }
            }
            lines.append("")
        }

        if !project.feedback.isEmpty {
            lines.append("## Feedback")
            for f in project.feedback.sorted(by: { $0.at > $1.at }) {
                let text = f.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                lines.append("- \(iso8601(f.at)): \(text)")
            }
            lines.append("")
        }

        lines.append(managedBlockEnd)
        return lines.joined(separator: "\n")
    }

    // MARK: - Small helpers (local to exporter)

    private static func sanitizeFileName(_ name: String, maxLength: Int) -> String {
        let invalid = CharacterSet(charactersIn: "/\\?%*:|\"<>")
        let cleaned = name.components(separatedBy: invalid).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        let collapsed = cleaned.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        if collapsed.count <= maxLength { return collapsed }
        let idx = collapsed.index(collapsed.startIndex, offsetBy: maxLength)
        return String(collapsed[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func yamlString(_ value: String) -> String {
        let escaped = value.replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }

    private static func yamlStringArray(_ values: [String]) -> String {
        let cleaned = values
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let quoted = cleaned.map { yamlString($0) }.joined(separator: ", ")
        return "[\(quoted)]"
    }

    private static func iso8601(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    private static func splitFrontmatter(from markdown: String) -> (frontmatter: String?, body: String) {
        let lines = markdown.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).map(String.init)
        guard lines.first?.trimmingCharacters(in: .whitespacesAndNewlines) == "---" else { return (nil, markdown) }
        guard let endIndex = lines.dropFirst().firstIndex(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines) == "---" }) else { return (nil, markdown) }
        let frontmatterLines = Array(lines[0...endIndex])
        let restLines = Array(lines[(endIndex + 1)...])
        return (frontmatterLines.joined(separator: "\n"), restLines.joined(separator: "\n"))
    }

    private static func preserveFrontmatterLines(existing: String?, excludingKeys: [String]) -> [String] {
        guard let existing else { return [] }
        let rawLines = existing.split(whereSeparator: \.isNewline).map(String.init)
        let trimmed = rawLines.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard trimmed.first == "---" else { return [] }
        guard let end = trimmed.dropFirst().firstIndex(of: "---") else { return [] }
        let body = rawLines[1..<end]

        var preserved: [String] = []
        for line in body {
            let t = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let key = t.split(separator: ":", maxSplits: 1).first.map(String.init) else { continue }
            if excludingKeys.contains(key) { continue }
            preserved.append(line)
        }
        return preserved
    }

    private static func preserveTail(fromBody body: String) -> String {
        guard let endRange = body.range(of: managedBlockEnd) else {
            let existing = body.trimmingCharacters(in: .whitespacesAndNewlines)
            if existing.isEmpty {
                return ["", "## Notes", "", ""].joined(separator: "\n")
            }
            return ["", "## Notes", "", existing, ""].joined(separator: "\n")
        }

        let after = body[endRange.upperBound...]
        let tail = after.trimmingCharacters(in: .whitespacesAndNewlines)
        if tail.isEmpty {
            return ["", "## Notes", "", ""].joined(separator: "\n")
        }
        return "\n" + tail + "\n"
    }
}
