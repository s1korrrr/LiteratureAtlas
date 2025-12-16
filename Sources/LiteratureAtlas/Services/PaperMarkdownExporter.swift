import Foundation

enum PaperMarkdownExporter {
    static func write(paper: Paper, outputRoot: URL) throws -> URL {
        let folder = outputRoot
            .appendingPathComponent("obsidian", isDirectory: true)
            .appendingPathComponent("papers", isDirectory: true)

        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let fileName = markdownFileName(for: paper)
        let destination = folder.appendingPathComponent(fileName)

        try reconcileExistingFile(paperID: paper.id, folder: folder, destination: destination)

        let markdown: String
        if FileManager.default.fileExists(atPath: destination.path),
           let existing = try? String(contentsOf: destination, encoding: .utf8) {
            markdown = updateMarkdown(existing: existing, paper: paper)
        } else {
            markdown = renderNewMarkdown(for: paper)
        }
        let data = Data(markdown.utf8)
        try data.write(to: destination, options: .atomic)
        return destination
    }

    static func markdownFileName(for paper: Paper) -> String {
        let baseName = paper.title.isEmpty ? paper.originalFilename : paper.title
        let sanitized = sanitizeFileName(baseName, maxLength: 140)
        return "\(sanitized) [\(paper.id.uuidString)].md"
    }

    private static func reconcileExistingFile(paperID: UUID, folder: URL, destination: URL) throws {
        let fm = FileManager.default
        let idToken = paperID.uuidString

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

    private static func renderNewMarkdown(for paper: Paper) -> String {
        let frontmatter = renderFrontmatter(for: paper, preservingFrom: nil)
        let managed = renderManagedBlock(for: paper)
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

    private static func updateMarkdown(existing: String, paper: Paper) -> String {
        let (existingFrontmatter, existingBody) = splitFrontmatter(from: existing)
        let preservedTail = preserveTail(fromBody: existingBody)

        let frontmatter = renderFrontmatter(for: paper, preservingFrom: existingFrontmatter)
        let managed = renderManagedBlock(for: paper)

        return [
            frontmatter,
            "",
            managed,
            preservedTail
        ].joined(separator: "\n")
    }

    private static func renderFrontmatter(for paper: Paper, preservingFrom existingFrontmatter: String?) -> String {
        var managed: [String] = []
        managed.append("type: paper")
        managed.append("id: \(yamlString(paper.id.uuidString))")
        managed.append("title: \(yamlString(paper.title.isEmpty ? paper.originalFilename : paper.title))")
        if let year = paper.year { managed.append("year: \(year)") }
        if let pageCount = paper.pageCount { managed.append("page_count: \(pageCount)") }
        if let clusterID = paper.clusterIndex { managed.append("cluster_id: \(clusterID)") }
        if let ingestedAt = paper.ingestedAt { managed.append("ingested_at: \(yamlString(iso8601(ingestedAt)))") }
        if let firstReadAt = paper.firstReadAt { managed.append("first_read_at: \(yamlString(iso8601(firstReadAt)))") }
        if let status = paper.readingStatus?.rawValue { managed.append("reading_status: \(status)") }
        if let important = paper.isImportant { managed.append("important: \(important)") }
        if let tags = paper.userTags, !tags.isEmpty { managed.append("tags: \(yamlStringArray(tags))") }
        if let keywords = paper.keywords, !keywords.isEmpty { managed.append("keywords: \(yamlStringArray(keywords))") }
        managed.append("source_pdf: \(yamlString(paper.filePath))")
        managed.append("original_filename: \(yamlString(paper.originalFilename))")

        let preserved = preserveFrontmatterLines(existing: existingFrontmatter, excludingKeys: managedFrontmatterKeys)
        return ([ "---" ] + managed + preserved + [ "---" ]).joined(separator: "\n")
    }

    private static var managedFrontmatterKeys: [String] {
        [
            "type",
            "id",
            "title",
            "year",
            "page_count",
            "cluster_id",
            "ingested_at",
            "first_read_at",
            "reading_status",
            "important",
            "tags",
            "keywords",
            "source_pdf",
            "original_filename"
        ]
    }

    private static func preserveFrontmatterLines(existing: String?, excludingKeys keys: [String]) -> [String] {
        guard let existing, !existing.isEmpty else { return [] }
        let keySet = Set(keys.map { $0.lowercased() })
        return existing
            .split(whereSeparator: \.isNewline)
            .map { String($0) }
            .filter { line in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return true }
                guard let keyEnd = trimmed.firstIndex(of: ":") else { return true }
                let key = trimmed[..<keyEnd].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                return !keySet.contains(key)
            }
    }

    private static func splitFrontmatter(from markdown: String) -> (frontmatter: String?, body: String) {
        let trimmed = markdown.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("---") else { return (nil, markdown) }
        let lines = markdown.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).map(String.init)
        guard lines.first == "---" else { return (nil, markdown) }
        if let endIndex = lines.dropFirst().firstIndex(of: "---") {
            let frontmatterLines = Array(lines[1..<endIndex])
            let restLines = Array(lines[(endIndex + 1)...])
            return (frontmatterLines.joined(separator: "\n"), restLines.joined(separator: "\n"))
        }
        return (nil, markdown)
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

    private static func renderManagedBlock(for paper: Paper) -> String {
        var lines: [String] = []

        lines.append(managedBlockBegin)
        lines.append("# \(paper.title.isEmpty ? paper.originalFilename : paper.title)")
        lines.append("")

        if let intro = paper.introSummary, !intro.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("## Introduction")
            lines.append(intro.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }

        if !paper.summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("## Summary")
            lines.append(paper.summary.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }

        if let method = paper.methodSummary, !method.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("## Methods")
            lines.append(method.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }

        if let results = paper.resultsSummary, !results.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("## Results")
            lines.append(results.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }

        if let takeaways = paper.takeaways, !takeaways.isEmpty {
            lines.append("## Takeaways")
            lines.append(markdownBullets(takeaways))
            lines.append("")
        }

        if let claims = paper.claims, !claims.isEmpty {
            lines.append("## Claims")
            for claim in claims {
                let statement = claim.statement.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !statement.isEmpty else { continue }
                let strength = String(format: "%.2f", claim.strength)
                lines.append("- \(statement) _(strength \(strength))_")
            }
            lines.append("")
        }

        if let assumptions = paper.assumptions, !assumptions.isEmpty {
            lines.append("## Assumptions (Heuristic)")
            lines.append(markdownBullets(assumptions))
            lines.append("")
        }

        if let evaluation = paper.evaluationContext {
            let dataset = evaluation.dataset?.trimmingCharacters(in: .whitespacesAndNewlines)
            let period = evaluation.period?.trimmingCharacters(in: .whitespacesAndNewlines)
            let metrics = evaluation.metrics

            if (dataset != nil && !(dataset ?? "").isEmpty) || (period != nil && !(period ?? "").isEmpty) || !metrics.isEmpty {
                lines.append("## Evaluation Context (Heuristic)")
                if let dataset, !dataset.isEmpty { lines.append("- Dataset: \(dataset)") }
                if let period, !period.isEmpty { lines.append("- Period: \(period)") }
                if !metrics.isEmpty { lines.append("- Metrics: \(metrics.joined(separator: ", "))") }
                lines.append("")
            }
        }

        if let pipeline = paper.methodPipeline, !pipeline.steps.isEmpty {
            lines.append("## Method Pipeline (Heuristic)")
            for step in pipeline.steps {
                let detail = step.detail?.trimmingCharacters(in: .whitespacesAndNewlines)
                if let detail, !detail.isEmpty {
                    lines.append("- [\(step.stage.rawValue)] \(step.label) — \(detail)")
                } else {
                    lines.append("- [\(step.stage.rawValue)] \(step.label)")
                }
            }
            lines.append("")
        }

        if let notes = paper.userNotes, !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("## Atlas Notes")
            lines.append(notes.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }

        if let questions = paper.userQuestions, !questions.isEmpty {
            lines.append("## Study Questions")
            lines.append(markdownBullets(questions))
            lines.append("")
        }

        if let flashcards = paper.flashcards, !flashcards.isEmpty {
            lines.append("## Flashcards")
            for card in flashcards {
                let q = card.question.trimmingCharacters(in: .whitespacesAndNewlines)
                let a = card.answer.trimmingCharacters(in: .whitespacesAndNewlines)
                if q.isEmpty && a.isEmpty { continue }
                lines.append("### Q: \(q.isEmpty ? "(empty)" : q)")
                lines.append(a.isEmpty ? "(empty)" : a)
                lines.append("")
            }
        }

        lines.append(managedBlockEnd)
        if lines.last != "" { lines.append("") }
        return lines.joined(separator: "\n")
    }

    private static func markdownBullets(_ items: [String]) -> String {
        items
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { "- \($0)" }
            .joined(separator: "\n")
    }

    private static func sanitizeFileName(_ raw: String, maxLength: Int) -> String {
        let trimmed = raw
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let forbidden = CharacterSet(charactersIn: "/\\\\:?%*|\"<>")
        var string = ""
        string.reserveCapacity(trimmed.count)
        for scalar in trimmed.unicodeScalars {
            if forbidden.contains(scalar) {
                string.append("-")
            } else {
                string.unicodeScalars.append(scalar)
            }
        }
        string = string.replacingOccurrences(of: "  ", with: " ")
        while string.contains("  ") { string = string.replacingOccurrences(of: "  ", with: " ") }
        string = string.trimmingCharacters(in: .whitespacesAndNewlines)

        if string.isEmpty { string = "Untitled" }
        if string.count > maxLength {
            string = String(string.prefix(maxLength)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return string
    }

    private static func iso8601(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    private static func yamlString(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }

    private static func yamlStringArray(_ values: [String]) -> String {
        "[" + values.map { yamlString($0) }.joined(separator: ", ") + "]"
    }
}
