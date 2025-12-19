import Foundation

enum LLMText {
    static func clip(_ text: String, maxChars: Int, suffix: String = "…") -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard maxChars > 0 else { return "" }
        guard trimmed.count > maxChars else { return trimmed }
        if maxChars == 1 { return suffix }
        let idx = trimmed.index(trimmed.startIndex, offsetBy: maxChars - 1)
        return String(trimmed[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines) + suffix
    }

    static func collapseWhitespace(_ text: String) -> String {
        let parts = text.split(whereSeparator: { $0.isWhitespace })
        return parts.joined(separator: " ")
    }

    static func isContextLimitError(_ error: Error) -> Bool {
        let msg = error.localizedDescription.lowercased()
        if msg.contains("out of context") { return true }
        if msg.contains("context") && (msg.contains("window") || msg.contains("limit") || msg.contains("length") || msg.contains("token")) { return true }
        if msg.contains("maximum") && msg.contains("context") { return true }
        if msg.contains("too long") && (msg.contains("context") || msg.contains("prompt") || msg.contains("input")) { return true }
        if msg.contains("too many") && msg.contains("token") { return true }
        return false
    }
}

