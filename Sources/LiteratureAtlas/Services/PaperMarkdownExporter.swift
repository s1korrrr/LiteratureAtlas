import Foundation

enum PaperMarkdownExporter {
    static let obsidianFormatVersion = 2

    struct Context {
        let allPapers: [Paper]
        let clustersByID: [Int: Cluster]
        let clusterNameSources: [Int: ClusterNameSource]

        init(allPapers: [Paper],
             clustersByID: [Int: Cluster] = [:],
             clusterNameSources: [Int: ClusterNameSource] = [:]) {
            self.allPapers = allPapers
            self.clustersByID = clustersByID
            self.clusterNameSources = clusterNameSources
        }

        init(allPapers: [Paper],
             clusters: [Cluster],
             clusterNameSources: [Int: ClusterNameSource] = [:]) {
            self.allPapers = allPapers
            self.clustersByID = Dictionary(uniqueKeysWithValues: clusters.map { ($0.id, $0) })
            self.clusterNameSources = clusterNameSources
        }
    }

    static func write(paper: Paper, outputRoot: URL, context: Context? = nil) throws -> URL {
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
            markdown = updateMarkdown(existing: existing, paper: paper, context: context)
        } else {
            markdown = renderNewMarkdown(for: paper, context: context)
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

    private static func renderNewMarkdown(for paper: Paper, context: Context?) -> String {
        let frontmatter = renderFrontmatter(for: paper, preservingFrom: nil, context: context)
        let managed = renderManagedBlock(for: paper, context: context)
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

    private static func updateMarkdown(existing: String, paper: Paper, context: Context?) -> String {
        let (existingFrontmatter, existingBody) = splitFrontmatter(from: existing)
        let preservedTail = preserveTail(fromBody: existingBody)

        let frontmatter = renderFrontmatter(for: paper, preservingFrom: existingFrontmatter, context: context)
        let managed = renderManagedBlock(for: paper, context: context)

        return [
            frontmatter,
            "",
            managed,
            preservedTail
        ].joined(separator: "\n")
    }

    private static func renderFrontmatter(for paper: Paper, preservingFrom existingFrontmatter: String?, context: Context?) -> String {
        var managed: [String] = []
        managed.append("type: paper")
        managed.append("obsidian_format_version: \(obsidianFormatVersion)")
        managed.append("id: \(yamlString(paper.id.uuidString))")
        managed.append("title: \(yamlString(paper.title.isEmpty ? paper.originalFilename : paper.title))")
        managed.append("cssclass: atlas-paper")

        var aliases: [String] = []
        aliases.append(ObsidianIDs.paperAlias(paper.id))
        let rawTitle = paper.title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !rawTitle.isEmpty { aliases.append(rawTitle) }
        let originalBase = URL(fileURLWithPath: paper.originalFilename).deletingPathExtension().lastPathComponent
        if !originalBase.isEmpty { aliases.append(originalBase) }
        aliases = Array(NSOrderedSet(array: aliases)).compactMap { $0 as? String }
        managed.append("aliases: \(yamlStringArray(aliases))")

        if let year = paper.year { managed.append("year: \(year)") }
        if let pageCount = paper.pageCount { managed.append("page_count: \(pageCount)") }
        if let clusterID = paper.clusterIndex {
            managed.append("cluster_id: \(clusterID)")
            if let clusterName = context?.clustersByID[clusterID]?.name.trimmingCharacters(in: .whitespacesAndNewlines),
               !clusterName.isEmpty {
                managed.append("cluster_name: \(yamlString(clusterName))")
            }
        }
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
            "obsidian_format_version",
            "id",
            "title",
            "cssclass",
            "aliases",
            "year",
            "page_count",
            "cluster_id",
            "cluster_name",
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

    private static func renderManagedBlock(for paper: Paper, context: Context?) -> String {
        var lines: [String] = []

        lines.append(managedBlockBegin)
        lines.append("# \(paper.title.isEmpty ? paper.originalFilename : paper.title)")
        lines.append("")

        // Meta / quick links
        var metaLines: [String] = []
        metaLines.append("- Atlas: [[Atlas]]")
        metaLines.append("- PDF: [Open](\(paper.fileURL.absoluteString))")
        if let year = paper.year { metaLines.append("- Year: \(year)") }
        if let pages = paper.pageCount { metaLines.append("- Pages: \(pages)") }
        if let ingestedAt = paper.ingestedAt { metaLines.append("- Ingested: \(iso8601(ingestedAt))") }
        if let firstReadAt = paper.firstReadAt { metaLines.append("- First read: \(iso8601(firstReadAt))") }
        if let status = paper.readingStatus?.label { metaLines.append("- Status: \(status)") }
        if let clusterID = paper.clusterIndex, let cluster = context?.clustersByID[clusterID] {
            let name = cluster.name.trimmingCharacters(in: .whitespacesAndNewlines)
            let source = context?.clusterNameSources[clusterID]?.label ?? "Unknown"
            metaLines.append("- Cluster: [[\(ObsidianIDs.clusterAlias(clusterID))|\(name.isEmpty ? "Cluster \(clusterID)" : name)]] _(name source: \(source))_")
        } else if let clusterID = paper.clusterIndex {
            metaLines.append("- Cluster: \(clusterID)")
        }
        if let tags = paper.userTags, !tags.isEmpty {
            let rendered = tags.prefix(10).map { "`\($0)`" }.joined(separator: ", ")
            metaLines.append("- Tags: \(rendered)")
        }
        if let keywords = paper.keywords, !keywords.isEmpty {
            let rendered = keywords.prefix(12).map { "`\($0)`" }.joined(separator: ", ")
            metaLines.append("- Keywords: \(rendered)")
        }
        lines.append(contentsOf: callout(type: "info", title: "Meta", body: metaLines.joined(separator: "\n")))
        lines.append("")

        // Summary / takeaways / related
        let summary = sanitizeLLMSection(paper.summary)
        if !summary.isEmpty {
            lines.append(contentsOf: callout(type: "summary", title: "Summary", body: summary))
            lines.append("")
        }

        if let takeaways = paper.takeaways, !takeaways.isEmpty {
            func isPlaceholderTakeaway(_ text: String) -> Bool {
                let lower = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                if lower.isEmpty { return true }
                if lower.contains("not specified") || lower.contains("unknown") { return true }
                return false
            }

            let cleanedAll = takeaways
                .map(cleanBulletText)
                .filter { !$0.isEmpty }

            let cleaned = cleanedAll.filter { !isPlaceholderTakeaway($0) }

            let chosen = cleaned.isEmpty ? cleanedAll : cleaned
            let takeawaysBody = chosen
                .prefix(5)
                .map { "- \($0)" }
                .joined(separator: "\n")
            if !takeawaysBody.isEmpty {
                lines.append(contentsOf: callout(type: "abstract", title: "Takeaways", body: takeawaysBody))
                lines.append("")
            }
        }

        if let context, let related = relatedPapers(for: paper, in: context.allPapers, topK: 5), !related.isEmpty {
            let body = related.map { entry in
                let link = "[[\(ObsidianIDs.paperAlias(entry.paper.id))|\(entry.paper.title.isEmpty ? entry.paper.originalFilename : entry.paper.title)]]"
                return String(format: "- %@ _(sim %.2f)_", link, entry.score)
            }.joined(separator: "\n")
            lines.append(contentsOf: callout(type: "example", title: "Related Papers", body: body, collapsedByDefault: true))
            lines.append("")
        }

        if let lens = paper.tradingLens {
            let lensText = renderTradingLensBlock(lens)
            if !lensText.isEmpty {
                lines.append(contentsOf: callout(type: "tip", title: "Trading Lens", body: lensText))
                lines.append("")
            }
        }

        if let intro = paper.introSummary {
            let cleaned = sanitizeLLMSection(intro)
            if !cleaned.isEmpty {
                lines.append(contentsOf: callout(type: "note", title: "Introduction", body: cleaned, collapsedByDefault: true))
                lines.append("")
            }
        }

        if let method = paper.methodSummary {
            let cleaned = sanitizeLLMSection(method)
            if !cleaned.isEmpty {
                lines.append(contentsOf: callout(type: "note", title: "Methods", body: cleaned, collapsedByDefault: true))
                lines.append("")
            }
        }

        if let results = paper.resultsSummary {
            let cleaned = sanitizeLLMSection(results)
            if !cleaned.isEmpty {
                lines.append(contentsOf: callout(type: "note", title: "Results", body: cleaned, collapsedByDefault: true))
                lines.append("")
            }
        }

        if let blueprint = paper.strategyBlueprint, !blueprint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append(contentsOf: callout(
                type: "todo",
                title: "Strategy Blueprint",
                body: demoteMarkdownHeadings(blueprint.trimmingCharacters(in: .whitespacesAndNewlines), by: 2),
                collapsedByDefault: true
            ))
            lines.append("")
        }

        if let audit = paper.backtestAudit, !audit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append(contentsOf: callout(
                type: "warning",
                title: "Backtest Audit",
                body: demoteMarkdownHeadings(audit.trimmingCharacters(in: .whitespacesAndNewlines), by: 2),
                collapsedByDefault: true
            ))
            lines.append("")
        }

        if let claims = paper.claims, !claims.isEmpty {
            var claimLines: [String] = []
            for claim in claims {
                let statement = claim.statement.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !statement.isEmpty else { continue }
                let strength = String(format: "%.2f", claim.strength)
                claimLines.append("- \(statement) _(strength \(strength))_")
            }
            if !claimLines.isEmpty {
                lines.append(contentsOf: callout(type: "quote", title: "Claims", body: claimLines.joined(separator: "\n"), collapsedByDefault: true))
                lines.append("")
            }
        }

        if let assumptions = paper.assumptions, !assumptions.isEmpty {
            let body = markdownBullets(assumptions)
            if !body.isEmpty {
                lines.append(contentsOf: callout(type: "warning", title: "Assumptions (Heuristic)", body: body, collapsedByDefault: true))
                lines.append("")
            }
        }

        if let evaluation = paper.evaluationContext {
            let dataset = evaluation.dataset?.trimmingCharacters(in: .whitespacesAndNewlines)
            let period = evaluation.period?.trimmingCharacters(in: .whitespacesAndNewlines)
            let metrics = evaluation.metrics

            if (dataset != nil && !(dataset ?? "").isEmpty) || (period != nil && !(period ?? "").isEmpty) || !metrics.isEmpty {
                var bodyLines: [String] = []
                if let dataset, !dataset.isEmpty { bodyLines.append("- Dataset: \(dataset)") }
                if let period, !period.isEmpty { bodyLines.append("- Period: \(period)") }
                if !metrics.isEmpty { bodyLines.append("- Metrics: \(metrics.joined(separator: ", "))") }
                lines.append(contentsOf: callout(type: "info", title: "Evaluation Context (Heuristic)", body: bodyLines.joined(separator: "\n"), collapsedByDefault: true))
                lines.append("")
            }
        }

        if let pipeline = paper.methodPipeline, !pipeline.steps.isEmpty {
            var bodyLines: [String] = []
            for step in pipeline.steps {
                let detail = step.detail?.trimmingCharacters(in: .whitespacesAndNewlines)
                if let detail, !detail.isEmpty {
                    bodyLines.append("- [\(step.stage.rawValue)] \(step.label) — \(detail)")
                } else {
                    bodyLines.append("- [\(step.stage.rawValue)] \(step.label)")
                }
            }
            if !bodyLines.isEmpty {
                lines.append(contentsOf: callout(type: "info", title: "Method Pipeline (Heuristic)", body: bodyLines.joined(separator: "\n"), collapsedByDefault: true))
                lines.append("")
            }
        }

        if let notes = paper.userNotes, !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append(contentsOf: callout(type: "note", title: "Atlas Notes", body: notes.trimmingCharacters(in: .whitespacesAndNewlines), collapsedByDefault: true))
            lines.append("")
        }

        if let questions = paper.userQuestions, !questions.isEmpty {
            let body = markdownBullets(questions)
            if !body.isEmpty {
                lines.append(contentsOf: callout(type: "question", title: "Study Questions", body: body, collapsedByDefault: true))
                lines.append("")
            }
        }

        if let flashcards = paper.flashcards, !flashcards.isEmpty {
            var cardLines: [String] = []
            for card in flashcards {
                let q = card.question.trimmingCharacters(in: .whitespacesAndNewlines)
                let a = card.answer.trimmingCharacters(in: .whitespacesAndNewlines)
                if q.isEmpty && a.isEmpty { continue }
                cardLines.append("**Q:** \(q.isEmpty ? "(empty)" : q)")
                cardLines.append("")
                cardLines.append(a.isEmpty ? "(empty)" : a)
                cardLines.append("")
            }
            let body = cardLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !body.isEmpty {
                lines.append(contentsOf: callout(type: "flashcard", title: "Flashcards", body: body, collapsedByDefault: true))
                lines.append("")
            }
        }

        lines.append(managedBlockEnd)
        if lines.last != "" { lines.append("") }
        return lines.joined(separator: "\n")
    }

    private static func markdownBullets(_ items: [String]) -> String {
        items
            .map(cleanBulletText)
            .filter { !$0.isEmpty }
            .map { "- \($0)" }
            .joined(separator: "\n")
    }

    private static func cleanBulletText(_ raw: String) -> String {
        var t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return "" }

        t = t
            .replacingOccurrences(of: "•", with: "-")
            .replacingOccurrences(of: "–", with: "-")

        let bulletChars: Set<Character> = ["-", "*"]
        func stripOneBullet(_ text: inout String) -> Bool {
            guard let first = text.first, bulletChars.contains(first) else { return false }
            guard text.count >= 2 else { return false }
            let second = text[text.index(after: text.startIndex)]
            guard second.isWhitespace else { return false }
            text.removeFirst()
            while let next = text.first, next.isWhitespace { text.removeFirst() }
            return true
        }
        while stripOneBullet(&t) {}

        let digits = t.prefix(while: { $0.isNumber })
        if !digits.isEmpty {
            let idx = t.index(t.startIndex, offsetBy: digits.count, limitedBy: t.endIndex) ?? t.endIndex
            if idx < t.endIndex, (t[idx] == "." || t[idx] == ")") {
                t = String(t[t.index(after: idx)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // Legacy takeaways parsing bug stripped the leading "**" of markdown bold labels like "**Problem**:".
        // Repair it here so exported notes render correctly even if cached JSON is old.
        if t.contains("**:"), !t.hasPrefix("**") {
            t = "**" + t
        }

        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func callout(type: String, title: String, body: String, collapsedByDefault: Bool? = nil) -> [String] {
        let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedBody.isEmpty else { return [] }

        let suffix: String
        if let collapsedByDefault {
            suffix = collapsedByDefault ? "-" : "+"
        } else {
            suffix = ""
        }

        var lines: [String] = []
        lines.append("> [!\(type)]\(suffix) \(title)")

        let bodyLines = trimmedBody.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).map(String.init)
        for line in bodyLines {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                lines.append(">")
            } else {
                lines.append("> \(line)")
            }
        }
        return lines
    }

    private static func sanitizeLLMSection(_ raw: String) -> String {
        let lines = raw.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).map(String.init)
        var cleaned = lines.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        // Trim empty lines.
        while let first = cleaned.first, first.isEmpty { cleaned.removeFirst() }
        while let last = cleaned.last, last.isEmpty { cleaned.removeLast() }

        func isPreamble(_ line: String) -> Bool {
            let lower = line.lowercased()
            if lower.hasPrefix("sure") { return true }
            if lower.hasPrefix("of course") { return true }
            if lower.hasPrefix("certainly") { return true }
            if lower.hasPrefix("here is") { return true }
            if lower.hasPrefix("here's") { return true }
            if lower.hasPrefix("summary of") { return true }
            if lower.hasPrefix("this section") && lower.contains("summary") { return true }
            return false
        }

        func isOutro(_ line: String) -> Bool {
            let lower = line.lowercased()
            if lower.contains("let me know") { return true }
            if lower.contains("i hope this helps") { return true }
            if lower.contains("hope this helps") { return true }
            if lower.contains("happy to") { return true }
            if lower.contains("if you'd like") { return true }
            if lower.contains("feel free") { return true }
            return false
        }

        func looksLikeContent(_ line: String) -> Bool {
            if line.hasPrefix("#") { return true }
            if line.hasPrefix(">") { return true }
            if line.hasPrefix("```") { return true }
            if line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("• ") { return true }
            return false
        }

        while let first = cleaned.first, !first.isEmpty, !looksLikeContent(first), isPreamble(first) {
            cleaned.removeFirst()
            while let next = cleaned.first, next.isEmpty { cleaned.removeFirst() }
        }
        while let last = cleaned.last, !last.isEmpty, isOutro(last) {
            cleaned.removeLast()
            while let prev = cleaned.last, prev.isEmpty { cleaned.removeLast() }
        }

        return cleaned.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func relatedPapers(for paper: Paper, in allPapers: [Paper], topK: Int) -> [(paper: Paper, score: Double)]? {
        guard !paper.embedding.isEmpty else { return nil }
        let target = paper.embedding
        let dim = target.count

        var scored: [(paper: Paper, score: Double)] = []
        scored.reserveCapacity(min(topK * 6, allPapers.count))

        for other in allPapers {
            guard other.id != paper.id else { continue }
            let v = other.embedding
            guard v.count == dim, !v.isEmpty else { continue }
            let sim = cosineSimilarity(target, v)
            if sim.isNaN { continue }
            scored.append((paper: other, score: sim))
        }

        guard !scored.isEmpty else { return [] }
        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(max(0, topK)))
    }

    private static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        for i in 0..<n {
            let x = Double(a[i])
            let y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        let denom = (na.squareRoot() * nb.squareRoot())
        if denom == 0 { return 0 }
        return dot / denom
    }

    private static func renderTradingLensBlock(_ lens: PaperTradingLens) -> String {
        func cell(_ raw: String?) -> String {
            let t = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if t.isEmpty { return "—" }
            return t.replacingOccurrences(of: "|", with: "\\|")
        }

        func listCell(_ items: [String]?, max: Int = 8) -> String {
            let cleaned = (items ?? [])
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            guard !cleaned.isEmpty else { return "—" }
            return cleaned.prefix(max).joined(separator: ", ").replacingOccurrences(of: "|", with: "\\|")
        }

        var scoreParts: [String] = []
        if let scores = lens.scores {
            if let novelty = scores.novelty { scoreParts.append(String(format: "novelty=%.1f", novelty)) }
            if let usability = scores.usability { scoreParts.append(String(format: "usability=%.1f", usability)) }
            if let impact = scores.strategyImpact { scoreParts.append(String(format: "impact=%.1f", impact)) }
            if let conf = scores.confidence { scoreParts.append(String(format: "confidence=%.2f", conf)) }
        }

        var rows: [(String, String)] = []
        rows.append(("Verdict", cell(lens.oneLineVerdict)))
        rows.append(("Scores", scoreParts.isEmpty ? "—" : scoreParts.joined(separator: ", ")))
        rows.append(("Trading tags", listCell(lens.tradingTags)))
        rows.append(("Asset classes", listCell(lens.assetClasses)))
        rows.append(("Horizons", listCell(lens.horizons)))
        rows.append(("Signal archetypes", listCell(lens.signalArchetypes)))
        rows.append(("Primary use", cell(lens.whereItFits?.primaryUse)))
        rows.append(("Pipeline stage", listCell(lens.whereItFits?.pipelineStage)))
        rows.append(("Must-have data", listCell(lens.dataRequirements?.mustHave)))
        rows.append(("Nice-to-have data", listCell(lens.dataRequirements?.niceToHave)))
        rows.append(("Recommended metrics", listCell(lens.evaluationNotes?.recommendedMetrics)))
        rows.append(("Must-check", listCell(lens.evaluationNotes?.mustCheck)))
        rows.append(("Risk flags", listCell(lens.riskFlags)))

        var out: [String] = []
        out.append("| Field | Value |")
        out.append("| --- | --- |")
        for (k, v) in rows {
            out.append("| \(k) | \(v) |")
        }

        if let hyps = lens.alphaHypotheses, !hyps.isEmpty {
            out.append("")
            out.append("**Alpha hypotheses**")
            for h in hyps.prefix(3) {
                let hypothesis = h.hypothesis?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !hypothesis.isEmpty else { continue }
                var bits: [String] = []
                if let target = h.target?.trimmingCharacters(in: .whitespacesAndNewlines), !target.isEmpty { bits.append("target: \(target)") }
                if let horizon = h.horizon?.trimmingCharacters(in: .whitespacesAndNewlines), !horizon.isEmpty { bits.append("horizon: \(horizon)") }
                if let features = h.features, !features.isEmpty { bits.append("features: \(features.prefix(6).joined(separator: ", "))") }
                let suffix = bits.isEmpty ? "" : " — " + bits.joined(separator: "; ")
                out.append("- \(hypothesis)\(suffix)")
            }
        }

        return out.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func demoteMarkdownHeadings(_ markdown: String, by levels: Int) -> String {
        guard levels > 0 else { return markdown }
        var inFence = false
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false)
        let adjusted = lines.map { raw -> String in
            let line = String(raw)
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("```") {
                inFence.toggle()
                return line
            }
            guard !inFence else { return line }
            guard line.hasPrefix("#") else { return line }
            let hashCount = line.prefix(while: { $0 == "#" }).count
            guard hashCount > 0 else { return line }
            let newCount = min(6, hashCount + levels)
            let rest = line.dropFirst(hashCount)
            return String(repeating: "#", count: newCount) + rest
        }
        return adjusted.joined(separator: "\n")
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
