import Foundation

enum AtlasMarkdownExporter {
    static let obsidianFormatVersion = 2

    static func write(outputRoot: URL,
                      corpusVersion: String?,
                      papers: [Paper],
                      clusters: [Cluster],
                      megaClusters: [Cluster],
                      clusterNameSources: [Int: ClusterNameSource],
                      pinnedClusterIDs: Set<Int>,
                      strategyProjects: [StrategyProject]) throws -> URL {
        let folder = outputRoot.appendingPathComponent("obsidian", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let destination = folder.appendingPathComponent("Atlas.md")

        let markdown: String
        if FileManager.default.fileExists(atPath: destination.path),
           let existing = try? String(contentsOf: destination, encoding: .utf8) {
            markdown = updateMarkdown(
                existing: existing,
                corpusVersion: corpusVersion,
                papers: papers,
                clusters: clusters,
                megaClusters: megaClusters,
                clusterNameSources: clusterNameSources,
                pinnedClusterIDs: pinnedClusterIDs,
                strategyProjects: strategyProjects
            )
        } else {
            markdown = renderNewMarkdown(
                corpusVersion: corpusVersion,
                papers: papers,
                clusters: clusters,
                megaClusters: megaClusters,
                clusterNameSources: clusterNameSources,
                pinnedClusterIDs: pinnedClusterIDs,
                strategyProjects: strategyProjects
            )
        }

        try Data(markdown.utf8).write(to: destination, options: .atomic)
        return destination
    }

    private static let managedBlockBegin = "<!-- atlas:begin -->"
    private static let managedBlockEnd = "<!-- atlas:end -->"

    private static func renderNewMarkdown(corpusVersion: String?,
                                         papers: [Paper],
                                         clusters: [Cluster],
                                         megaClusters: [Cluster],
                                         clusterNameSources: [Int: ClusterNameSource],
                                         pinnedClusterIDs: Set<Int>,
                                         strategyProjects: [StrategyProject]) -> String {
        let frontmatter = renderFrontmatter(preservingFrom: nil, corpusVersion: corpusVersion, papers: papers, clusters: clusters, strategyProjects: strategyProjects)
        let managed = renderManagedBlock(corpusVersion: corpusVersion, papers: papers, clusters: clusters, megaClusters: megaClusters, clusterNameSources: clusterNameSources, pinnedClusterIDs: pinnedClusterIDs, strategyProjects: strategyProjects)
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

    private static func updateMarkdown(existing: String,
                                      corpusVersion: String?,
                                      papers: [Paper],
                                      clusters: [Cluster],
                                      megaClusters: [Cluster],
                                      clusterNameSources: [Int: ClusterNameSource],
                                      pinnedClusterIDs: Set<Int>,
                                      strategyProjects: [StrategyProject]) -> String {
        let (existingFrontmatter, existingBody) = splitFrontmatter(from: existing)
        let preservedTail = preserveTail(fromBody: existingBody)

        let frontmatter = renderFrontmatter(preservingFrom: existingFrontmatter, corpusVersion: corpusVersion, papers: papers, clusters: clusters, strategyProjects: strategyProjects)
        let managed = renderManagedBlock(corpusVersion: corpusVersion, papers: papers, clusters: clusters, megaClusters: megaClusters, clusterNameSources: clusterNameSources, pinnedClusterIDs: pinnedClusterIDs, strategyProjects: strategyProjects)

        return [
            frontmatter,
            "",
            managed,
            preservedTail
        ].joined(separator: "\n")
    }

    private static func renderFrontmatter(preservingFrom existingFrontmatter: String?, corpusVersion: String?, papers: [Paper], clusters: [Cluster], strategyProjects: [StrategyProject]) -> String {
        var managed: [String] = []
        managed.append("type: atlas")
        managed.append("obsidian_format_version: \(obsidianFormatVersion)")
        managed.append("title: \"LiteratureAtlas\"")
        managed.append("cssclass: atlas-dashboard")
        managed.append("aliases: \(yamlStringArray(["Atlas", "LiteratureAtlas"]))")
        if let corpusVersion { managed.append("corpus_version: \(yamlString(corpusVersion))") }
        managed.append("updated_at: \(yamlString(iso8601(Date())))")
        managed.append("paper_count: \(papers.count)")
        managed.append("cluster_count: \(clusters.count)")
        managed.append("strategy_count: \(strategyProjects.count)")

        let preserved = preserveFrontmatterLines(existing: existingFrontmatter, excludingKeys: managedFrontmatterKeys)
        return ([ "---" ] + managed + preserved + [ "---" ]).joined(separator: "\n")
    }

    private static var managedFrontmatterKeys: [String] {
        [
            "type",
            "obsidian_format_version",
            "title",
            "cssclass",
            "aliases",
            "corpus_version",
            "updated_at",
            "paper_count",
            "cluster_count",
            "strategy_count"
        ]
    }

    private static func renderManagedBlock(corpusVersion: String?,
                                          papers: [Paper],
                                          clusters: [Cluster],
                                          megaClusters: [Cluster],
                                          clusterNameSources: [Int: ClusterNameSource],
                                          pinnedClusterIDs: Set<Int>,
                                          strategyProjects: [StrategyProject]) -> String {
        var lines: [String] = []
        lines.append(managedBlockBegin)
        lines.append("# LiteratureAtlas")
        lines.append("")

        var meta: [String] = []
        if let corpusVersion { meta.append("- Corpus version: `\(corpusVersion)`") }
        meta.append("- Papers: \(papers.count)")
        meta.append("- Clusters: \(clusters.count)")
        meta.append("- Strategies: \(strategyProjects.count)")
        meta.append("- Updated: \(iso8601(Date()))")
        lines.append(contentsOf: callout(type: "info", title: "Status", body: meta.joined(separator: "\n")))
        lines.append("")

        lines.append(contentsOf: callout(type: "tip", title: "Setup", body: "- [[Obsidian Setup]] _(Dataview + optional CSS snippet)_"))
        lines.append("")

        lines.append("## Inbox")
        lines.append("- [ ] Review newest papers")
        lines.append("- [ ] Pin 2–3 clusters to focus this week")
        lines.append("- [ ] Convert 1 paper into a strategy project")
        lines.append("")

        // Recent papers (static)
        let recent = papers.sorted { l, r in
            (l.ingestedAt ?? .distantPast) > (r.ingestedAt ?? .distantPast)
        }
        if !recent.isEmpty {
            let body = recent.prefix(15).map { p in
                let title = p.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? p.originalFilename : p.title
                if let year = p.year {
                    return "- [[\(ObsidianIDs.paperAlias(p.id))|\(title)]] (\(year))"
                }
                return "- [[\(ObsidianIDs.paperAlias(p.id))|\(title)]]"
            }.joined(separator: "\n")
            lines.append(contentsOf: callout(type: "summary", title: "Recent Papers", body: body))
            lines.append("")
        }

        // Clusters (static)
        let allClusters = flattenClusters(megaClusters: megaClusters, subclusters: clusters)
        let clusterList = allClusters.sorted { l, r in
            let lp = pinnedClusterIDs.contains(l.id)
            let rp = pinnedClusterIDs.contains(r.id)
            if lp != rp { return lp && !rp }
            if l.memberPaperIDs.count != r.memberPaperIDs.count { return l.memberPaperIDs.count > r.memberPaperIDs.count }
            return l.id < r.id
        }

        if !clusterList.isEmpty {
            let body = clusterList.prefix(20).map { c in
                let pinned = pinnedClusterIDs.contains(c.id) ? "[PINNED] " : ""
                let source = clusterNameSources[c.id]?.label ?? "Unknown"
                let name = c.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Cluster \(c.id)" : c.name
                return "- \(pinned)[[\(ObsidianIDs.clusterAlias(c.id))|\(name)]] (\(c.memberPaperIDs.count) papers; \(source))"
            }.joined(separator: "\n")
            lines.append(contentsOf: callout(type: "example", title: "Clusters", body: body, collapsedByDefault: true))
            lines.append("")
        }

        if !strategyProjects.isEmpty {
            let body = strategyProjects
                .sorted { $0.updatedAt > $1.updatedAt }
                .prefix(10)
                .map { p in
                    let title = p.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Strategy" : p.title
                    return "- [[\(ObsidianIDs.strategyAlias(p.id))|\(title)]]"
                }
                .joined(separator: "\n")
            lines.append(contentsOf: callout(type: "tip", title: "Strategy Projects", body: body, collapsedByDefault: true))
            lines.append("")
        }

        lines.append("## Dataview (optional)")
        lines.append("```dataview")
        lines.append("TABLE year, ingested_at, file.link AS Paper")
        lines.append("FROM \"papers\"")
        lines.append("SORT ingested_at desc")
        lines.append("LIMIT 25")
        lines.append("```")
        lines.append("")
        lines.append("```dataview")
        lines.append("TABLE paper_count, name, file.link AS Cluster")
        lines.append("FROM \"clusters\"")
        lines.append("SORT paper_count desc")
        lines.append("LIMIT 30")
        lines.append("```")
        lines.append("")

        lines.append(managedBlockEnd)
        if lines.last != "" { lines.append("") }
        return lines.joined(separator: "\n")
    }

    private static func flattenClusters(megaClusters: [Cluster], subclusters: [Cluster]) -> [Cluster] {
        var byID: [Int: Cluster] = [:]
        func add(_ c: Cluster) {
            byID[c.id] = c
            for sub in c.subclusters ?? [] { add(sub) }
        }
        for m in megaClusters { add(m) }
        for c in subclusters { add(c) }
        return Array(byID.values)
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
