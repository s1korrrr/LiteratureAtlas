import Foundation

enum ObsidianVaultAssetsExporter {
    static let obsidianFormatVersion = 2

    private static let managedBlockBegin = "<!-- atlas:begin -->"
    private static let managedBlockEnd = "<!-- atlas:end -->"

    static func write(outputRoot: URL) throws {
        let obsidianRoot = outputRoot.appendingPathComponent("obsidian", isDirectory: true)
        try FileManager.default.createDirectory(at: obsidianRoot, withIntermediateDirectories: true)

        try writeCSSSnippet(obsidianRoot: obsidianRoot)
        try writeSetupNote(obsidianRoot: obsidianRoot)
    }

    private static func writeCSSSnippet(obsidianRoot: URL) throws {
        let snippetsFolder = obsidianRoot
            .appendingPathComponent(".obsidian", isDirectory: true)
            .appendingPathComponent("snippets", isDirectory: true)
        try FileManager.default.createDirectory(at: snippetsFolder, withIntermediateDirectories: true)

        let url = snippetsFolder.appendingPathComponent("literature-atlas.css")
        let css = """
        /* LiteratureAtlas — Obsidian CSS snippet
           Enable via: Settings → Appearance → CSS snippets → literature-atlas
           Scope: Notes exported by LiteratureAtlas (cssclass: atlas-*)
        */

        /* ------------------------------
           Base: dashboards + cards
           ------------------------------ */

        .atlas-dashboard,
        .atlas-paper,
        .atlas-cluster,
        .atlas-strategy,
        .atlas-setup {
          --atlas-radius: 12px;
          --atlas-border: var(--background-modifier-border);
        }

        .atlas-dashboard .callout,
        .atlas-paper .callout,
        .atlas-cluster .callout,
        .atlas-strategy .callout,
        .atlas-setup .callout {
          border-left-width: 4px;
          border-radius: var(--atlas-radius);
          border: 1px solid var(--atlas-border);
          background-color: var(--background-secondary);
        }

        .atlas-dashboard .callout-title,
        .atlas-paper .callout-title,
        .atlas-cluster .callout-title,
        .atlas-strategy .callout-title,
        .atlas-setup .callout-title {
          font-weight: 700;
        }

        /* Slightly denser tables for Trading Lens blocks. */
        .atlas-dashboard table,
        .atlas-paper table,
        .atlas-strategy table {
          width: 100%;
          border-collapse: collapse;
          border-radius: var(--atlas-radius);
          overflow: hidden;
        }
        .atlas-dashboard table th,
        .atlas-paper table th,
        .atlas-strategy table th {
          text-align: left;
          font-weight: 700;
          padding: 8px 10px;
          background-color: var(--background-secondary-alt);
          border-bottom: 1px solid var(--atlas-border);
        }
        .atlas-dashboard table td,
        .atlas-paper table td,
        .atlas-strategy table td {
          padding: 7px 10px;
          vertical-align: top;
          border-bottom: 1px solid var(--atlas-border);
          font-variant-numeric: tabular-nums;
        }
        .atlas-dashboard table tr:nth-child(even) td,
        .atlas-paper table tr:nth-child(even) td,
        .atlas-strategy table tr:nth-child(even) td {
          background-color: var(--background-primary-alt);
        }

        /* Keep managed headers compact. */
        .atlas-dashboard h1,
        .atlas-paper h1,
        .atlas-cluster h1,
        .atlas-strategy h1,
        .atlas-setup h1 {
          margin-block-end: 0.6em;
        }

        .atlas-dashboard code,
        .atlas-paper code,
        .atlas-cluster code,
        .atlas-strategy code,
        .atlas-setup code {
          border-radius: 6px;
          padding: 0.12em 0.35em;
        }
        """
        try Data(css.utf8).write(to: url, options: .atomic)
    }

    private static func writeSetupNote(obsidianRoot: URL) throws {
        let url = obsidianRoot.appendingPathComponent("Obsidian Setup.md")

        let existingText = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        let (existingFrontmatter, existingBody) = splitFrontmatter(from: existingText)
        let preservedTail = preserveTail(fromBody: existingBody)

        let frontmatter = renderFrontmatter(preservingFrom: existingFrontmatter)
        let managed = renderManagedBlock()

        let markdown: String
        if existingText.isEmpty {
            markdown = [frontmatter, "", managed, "", "## Notes", "", ""].joined(separator: "\n")
        } else {
            markdown = [frontmatter, "", managed, preservedTail].joined(separator: "\n")
        }

        try Data(markdown.utf8).write(to: url, options: .atomic)
    }

    private static func renderFrontmatter(preservingFrom existingFrontmatter: String?) -> String {
        var managed: [String] = []
        managed.append("type: atlas_setup")
        managed.append("obsidian_format_version: \(obsidianFormatVersion)")
        managed.append("title: \"Obsidian Setup\"")
        managed.append("cssclass: atlas-setup")
        managed.append("aliases: [\"Obsidian Setup\", \"Atlas Setup\"]")

        let preserved = preserveFrontmatterLines(existing: existingFrontmatter, excludingKeys: managedFrontmatterKeys)
        return ([ "---" ] + managed + preserved + [ "---" ]).joined(separator: "\n")
    }

    private static var managedFrontmatterKeys: [String] {
        [
            "type",
            "obsidian_format_version",
            "title",
            "cssclass",
            "aliases"
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

    private static func renderManagedBlock() -> String {
        var lines: [String] = []
        lines.append(managedBlockBegin)
        lines.append("# Obsidian Setup")
        lines.append("")

        lines.append("> [!info] Recommended Plugins")
        lines.append("> - **Dataview** (enables the `TABLE` blocks in Atlas/Cluster notes)")
        lines.append("> - **Style Settings** (optional, if you use a theme like Minimal)")
        lines.append("")

        lines.append("> [!tip] Enable CSS Snippet")
        lines.append("> - Obsidian → Settings → Appearance → CSS snippets → enable `literature-atlas`")
        lines.append("> - Snippet file: `.obsidian/snippets/literature-atlas.css`")
        lines.append("")

        lines.append("> [!example] Suggested Workflow")
        lines.append("> - Start in [[Atlas]] and pin 2–3 clusters to focus this week")
        lines.append("> - Read a paper, add 2–3 `## Notes`, then turn it into a strategy project")
        lines.append("> - Use the Related Papers callout to snowball into nearby work")
        lines.append("")

        lines.append(managedBlockEnd)
        if lines.last != "" { lines.append("") }
        return lines.joined(separator: "\n")
    }
}
