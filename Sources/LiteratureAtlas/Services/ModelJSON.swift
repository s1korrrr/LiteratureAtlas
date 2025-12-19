import Foundation

enum ModelJSON {
    /// Best-effort extraction of the first JSON object/array embedded in `text`.
    static func extractFirstJSON(from text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let fenced = extractFirstCodeFencePayload(from: trimmed), !fenced.isEmpty {
            return extractFirstJSON(from: fenced) ?? normalize(json: fenced)
        }

        guard let range = firstJSONRange(in: trimmed) else { return nil }
        let candidate = String(trimmed[range])
        return normalize(json: candidate)
    }

    static func decodeFirstJSON<T: Decodable>(_ type: T.Type, from text: String, decoder: JSONDecoder = JSONDecoder()) throws -> T {
        guard let json = extractFirstJSON(from: text) else {
            throw NSError(domain: "ModelJSON", code: 1, userInfo: [NSLocalizedDescriptionKey: "No JSON object found in model output."])
        }
        guard let data = json.data(using: .utf8) else {
            throw NSError(domain: "ModelJSON", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to encode extracted JSON as UTF-8."])
        }
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw NSError(domain: "ModelJSON", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Failed to decode extracted JSON: \(error.localizedDescription)"
            ])
        }
    }

    // MARK: - Internals

    private static func extractFirstCodeFencePayload(from text: String) -> String? {
        guard let open = text.range(of: "```") else { return nil }
        let afterOpen = text[open.upperBound...]
        guard let newline = afterOpen.firstIndex(of: "\n") else { return nil }
        let afterHeader = afterOpen[afterOpen.index(after: newline)...]
        guard let close = afterHeader.range(of: "```") else { return nil }
        let payload = afterHeader[..<close.lowerBound]
        return payload.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func firstJSONRange(in text: String) -> Range<String.Index>? {
        guard let start = firstJSONStartIndex(in: text) else { return nil }
        let openChar = text[start]
        let closeChar: Character = (openChar == "{") ? "}" : "]"

        var depth = 0
        var inString = false
        var isEscaped = false

        var idx = start
        while idx < text.endIndex {
            let ch = text[idx]
            if inString {
                if isEscaped {
                    isEscaped = false
                } else if ch == "\\" {
                    isEscaped = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                if ch == "\"" {
                    inString = true
                } else if ch == openChar {
                    depth += 1
                } else if ch == closeChar {
                    depth -= 1
                    if depth == 0 {
                        let end = text.index(after: idx)
                        return start..<end
                    }
                }
            }
            idx = text.index(after: idx)
        }

        return nil
    }

    private static func firstJSONStartIndex(in text: String) -> String.Index? {
        let obj = text.firstIndex(of: "{")
        let arr = text.firstIndex(of: "[")
        switch (obj, arr) {
        case (nil, nil): return nil
        case (let i?, nil): return i
        case (nil, let j?): return j
        case (let i?, let j?): return min(i, j)
        }
    }

    private static func normalize(json: String) -> String {
        var s = json.trimmingCharacters(in: .whitespacesAndNewlines)

        // Replace common “smart quotes” that break JSON parsing.
        s = s
            .replacingOccurrences(of: "“", with: "\"")
            .replacingOccurrences(of: "”", with: "\"")
            .replacingOccurrences(of: "’", with: "'")

        // Remove trailing commas before closing } or ] (common model mistake).
        s = s.replacingOccurrences(of: ",\\s*([}\\]])", with: "$1", options: .regularExpression)

        return s
    }
}
