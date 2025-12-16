import Foundation

enum PromptStore {
    /// Loads a prompt template from `Prompts/` (repo root) or from `LITERATURE_ATLAS_PROMPTS_DIR` if set.
    /// Returns `fallback` if the file is missing/unreadable.
    static func loadText(_ relativePath: String, fallback: String) -> String {
        guard let url = resolveURL(relativePath) else { return fallback }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return fallback }
        return text
    }

    static func render(template: String, variables: [String: String]) -> String {
        var rendered = template
        for (key, value) in variables {
            rendered = rendered.replacingOccurrences(of: "{{\(key)}}", with: value)
        }
        return rendered
    }

    private static func resolveURL(_ relativePath: String) -> URL? {
        let fm = FileManager.default

        if let override = ProcessInfo.processInfo.environment["LITERATURE_ATLAS_PROMPTS_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let base = URL(fileURLWithPath: override, isDirectory: true)
            let url = base.appendingPathComponent(relativePath)
            if fm.fileExists(atPath: url.path) { return url }
        }

        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        let defaultBase = cwd.appendingPathComponent("Prompts", isDirectory: true)
        let url = defaultBase.appendingPathComponent(relativePath)
        if fm.fileExists(atPath: url.path) { return url }

        return nil
    }
}
