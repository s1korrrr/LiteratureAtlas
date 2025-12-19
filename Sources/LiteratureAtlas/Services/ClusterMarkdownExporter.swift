import Foundation

enum ClusterMarkdownExporter {
    static let obsidianFormatVersion = 2

    static func write(cluster: Cluster, outputRoot: URL, papersByID: [UUID: Paper], nameSource: ClusterNameSource? = nil) throws -> URL {
        let folder = outputRoot
            .appendingPathComponent("obsidian", isDirectory: true)
            .appendingPathComponent("clusters", isDirectory: true)

        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let fileName = markdownFileName(for: cluster)
        let destination = folder.appendingPathComponent(fileName)

        try reconcileExistingFile(clusterID: cluster.id, folder: folder, destination: destination)

        let markdown: String
        if FileManager.default.fileExists(atPath: destination.path),
           let existing = try? String(contentsOf: destination, encoding: .utf8) {
            markdown = updateMarkdown(existing: existing, cluster: cluster, papersByID: papersByID, nameSource: nameSource)
        } else {
            markdown = renderNewMarkdown(for: cluster, papersByID: papersByID, nameSource: nameSource)
        }

        try Data(markdown.utf8).write(to: destination, options: .atomic)
        return destination
    }

    static func markdownFileName(for cluster: Cluster) -> String {
        let baseName = cluster.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Cluster \(cluster.id)" : cluster.name
        let sanitized = sanitizeFileName(baseName, maxLength: 140)
        return "\(sanitized) [\(ObsidianIDs.clusterAlias(cluster.id))].md"
    }

    private static let managedBlockBegin = "<!-- atlas:begin -->"
    private static let managedBlockEnd = "<!-- atlas:end -->"

    private static func reconcileExistingFile(clusterID: Int, folder: URL, destination: URL) throws {
        let fm = FileManager.default
        let token = ObsidianIDs.clusterAlias(clusterID)

        let candidates = (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])) ?? []
        guard let existing = candidates.first(where: { url in
            url.pathExtension.lowercased() == "md" && url.lastPathComponent.contains(token)
        }) else {
            return
        }

        guard existing.standardizedFileURL != destination.standardizedFileURL else { return }

        if fm.fileExists(atPath: destination.path) {
            try? fm.removeItem(at: destination)
        }
        try? fm.moveItem(at: existing, to: destination)
    }

    private static func renderNewMarkdown(for cluster: Cluster, papersByID: [UUID: Paper], nameSource: ClusterNameSource?) -> String {
        let frontmatter = renderFrontmatter(for: cluster, preservingFrom: nil, papersByID: papersByID, nameSource: nameSource)
        let managed = renderManagedBlock(for: cluster, papersByID: papersByID, nameSource: nameSource)
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

    private static func updateMarkdown(existing: String, cluster: Cluster, papersByID: [UUID: Paper], nameSource: ClusterNameSource?) -> String {
        let (existingFrontmatter, existingBody) = splitFrontmatter(from: existing)
        let preservedTail = preserveTail(fromBody: existingBody)

        let frontmatter = renderFrontmatter(for: cluster, preservingFrom: existingFrontmatter, papersByID: papersByID, nameSource: nameSource)
        let managed = renderManagedBlock(for: cluster, papersByID: papersByID, nameSource: nameSource)

        return [
            frontmatter,
            "",
            managed,
            preservedTail
        ].joined(separator: "\n")
    }

    private static func renderFrontmatter(for cluster: Cluster, preservingFrom existingFrontmatter: String?, papersByID: [UUID: Paper], nameSource: ClusterNameSource?) -> String {
        var managed: [String] = []
        managed.append("type: cluster")
        managed.append("obsidian_format_version: \(obsidianFormatVersion)")
        managed.append("cluster_id: \(cluster.id)")
        managed.append("name: \(yamlString(cluster.name.trimmingCharacters(in: .whitespacesAndNewlines)))")
        managed.append("cssclass: atlas-cluster")

        var aliases: [String] = []
        aliases.append(ObsidianIDs.clusterAlias(cluster.id))
        aliases.append(cluster.name)
        aliases.append("Cluster \(cluster.id)")
        aliases = Array(NSOrderedSet(array: aliases)).compactMap { $0 as? String }.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        managed.append("aliases: \(yamlStringArray(aliases))")

        if let v = cluster.corpusVersion?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { managed.append("corpus_version: \(yamlString(v))") }
        if let k = cluster.resolutionK { managed.append("resolution_k: \(k)") }
        managed.append("paper_count: \(cluster.memberPaperIDs.count)")
        if let src = nameSource { managed.append("name_source: \(src.rawValue)") }

        // Helpful for Dataview / debugging. Keep it bounded.
        let memberIDs = cluster.memberPaperIDs.prefix(120).map(\.uuidString)
        if !memberIDs.isEmpty {
            managed.append("member_paper_ids: \(yamlStringArray(memberIDs))")
        }

        let preserved = preserveFrontmatterLines(existing: existingFrontmatter, excludingKeys: managedFrontmatterKeys)
        return ([ "---" ] + managed + preserved + [ "---" ]).joined(separator: "\n")
    }

    private static var managedFrontmatterKeys: [String] {
        [
            "type",
            "obsidian_format_version",
            "cluster_id",
            "name",
            "cssclass",
            "aliases",
            "corpus_version",
            "resolution_k",
            "paper_count",
            "name_source",
            "member_paper_ids"
        ]
    }

    private static func renderManagedBlock(for cluster: Cluster, papersByID: [UUID: Paper], nameSource: ClusterNameSource?) -> String {
        var lines: [String] = []

        lines.append(managedBlockBegin)
        lines.append("# \(cluster.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Cluster \(cluster.id)" : cluster.name)")
        lines.append("")

        var meta: [String] = []
        meta.append("- Atlas: [[Atlas]]")
        meta.append("- ID: \(cluster.id)")
        meta.append("- Papers: \(cluster.memberPaperIDs.count)")
        if let v = cluster.corpusVersion?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty { meta.append("- Corpus version: `\(v)`") }
        if let k = cluster.resolutionK { meta.append("- Resolution k: \(k)") }
        if let nameSource { meta.append("- Name source: \(nameSource.label)") }

        if let yearRange = yearDomain(for: cluster, papersByID: papersByID) {
            meta.append("- Year range: \(yearRange.lowerBound)–\(yearRange.upperBound)")
        }

        lines.append(contentsOf: callout(type: "info", title: "Meta", body: meta.joined(separator: "\n")))
        lines.append("")

        let metaSummary = cluster.metaSummary.trimmingCharacters(in: .whitespacesAndNewlines)
        if !metaSummary.isEmpty {
            lines.append(contentsOf: callout(type: "summary", title: "Meta-summary", body: metaSummary))
            lines.append("")
        }

        if let lens = cluster.tradingLens?.trimmingCharacters(in: .whitespacesAndNewlines), !lens.isEmpty {
            lines.append(contentsOf: callout(type: "tip", title: "Trading Lens", body: lens, collapsedByDefault: true))
            lines.append("")
        }

        let keywords = topKeywords(for: cluster, papersByID: papersByID, limit: 14)
        if !keywords.isEmpty {
            let rendered = keywords.map { "`\($0)`" }.joined(separator: ", ")
            lines.append(contentsOf: callout(type: "info", title: "Keywords", body: rendered))
            lines.append("")
        }

        // Static list for quick scan + optional Dataview block.
        let members = cluster.memberPaperIDs.compactMap { papersByID[$0] }.sorted { l, r in
            let ly = l.year ?? -10_000
            let ry = r.year ?? -10_000
            if ly != ry { return ly > ry }
            return l.title < r.title
        }

        if !members.isEmpty {
            let list = members.prefix(40).map { p in
                let title = p.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? p.originalFilename : p.title
                if let year = p.year {
                    return "- [[\(ObsidianIDs.paperAlias(p.id))|\(title)]] (\(year))"
                }
                return "- [[\(ObsidianIDs.paperAlias(p.id))|\(title)]]"
            }.joined(separator: "\n")
            lines.append(contentsOf: callout(type: "example", title: "Papers (top \(min(40, members.count)))", body: list, collapsedByDefault: true))
            lines.append("")
        }

        lines.append("```dataview")
        lines.append("TABLE year, file.link AS Paper")
        lines.append("FROM \"papers\"")
        lines.append("WHERE cluster_id = this.cluster_id")
        lines.append("SORT year desc")
        lines.append("```")
        lines.append("")

        lines.append(managedBlockEnd)
        if lines.last != "" { lines.append("") }
        return lines.joined(separator: "\n")
    }

    private static func topKeywords(for cluster: Cluster, papersByID: [UUID: Paper], limit: Int) -> [String] {
        var counts: [String: Int] = [:]
        for pid in cluster.memberPaperIDs {
            guard let paper = papersByID[pid] else { continue }
            for kw in paper.keywords ?? [] {
                let cleaned = kw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                guard cleaned.count >= 3 else { continue }
                counts[cleaned, default: 0] += 1
            }
        }
        return counts
            .sorted { l, r in l.value == r.value ? l.key < r.key : l.value > r.value }
            .prefix(limit)
            .map(\.key)
    }

    private static func yearDomain(for cluster: Cluster, papersByID: [UUID: Paper]) -> ClosedRange<Int>? {
        let years = cluster.memberPaperIDs.compactMap { papersByID[$0]?.year }
        guard let minY = years.min(), let maxY = years.max() else { return nil }
        return minY...maxY
    }

    // MARK: - Small helpers (local to exporter)

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

        if string.isEmpty { string = "Cluster" }
        if string.count > maxLength {
            string = String(string.prefix(maxLength)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return string
    }

    private static func yamlString(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }

    private static func yamlStringArray(_ values: [String]) -> String {
        let cleaned = values
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return "[" + cleaned.map { yamlString($0) }.joined(separator: ", ") + "]"
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

