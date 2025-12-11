import Foundation
import FoundationModels

@available(macOS 26, iOS 26, *)
actor PaperSummarizerActor {
    private let instructions: String
    private let sectionInstructions: String

    struct SummaryOutput {
        let summary: String
        let chunksUsed: Int
        let maxChunkCharsUsed: Int
    }

    init() {
        instructions = """
        You are an expert research assistant. Given partial text from an academic paper, write a concise technical summary in 3-6 bullet points. Focus on: the problem, main method, and key results.
        Keep it tight; avoid long quotes.
        """
        sectionInstructions = """
        Summarize the given section of an academic paper in 2-3 bullet points for a technical reader. Focus on the section's purpose and key ideas.
        """
    }

    func generateTakeaways(title: String, text: String) async throws -> [String] {
        let prompt = """
        Title: \(title)

        Summary or text:
        \(text.prefix(2000))

        List 3-5 crisp bullet takeaways highlighting the core idea, method, and result. Keep each under 18 words.
        """
        let response = try await makeSession().respond(to: prompt)
        let lines = response.content
            .split(whereSeparator: \.isNewline)
            .map { $0.replacingOccurrences(of: "•", with: "").trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return lines
    }

    func summarize(title: String, text: String) async throws -> SummaryOutput {
        // Chunk the paper body before sending to the model.
        let chunks = chunk(text: text, maxChars: 600, overlap: 100, maxChunks: 12)
        guard !chunks.isEmpty else {
            return SummaryOutput(summary: fallbackSummary(title: title, text: text), chunksUsed: 0, maxChunkCharsUsed: 0)
        }

        var chunkSummaries: [String] = []
        var maxUsed = 0
        for (idx, chunk) in chunks.enumerated() {
            do {
                let (summary, usedChars) = try await summarizeSingleChunk(title: title, chunk: chunk, chunkIndex: idx, totalChunks: chunks.count)
                maxUsed = max(maxUsed, usedChars)
                chunkSummaries.append(summary)
            } catch {
                // If a chunk fails repeatedly, drop it and continue.
                chunkSummaries.append("Chunk \(idx + 1) summary unavailable due to context limit.")
            }
        }

        // Consolidate into final summary.
        let consolidatePrompt = """
        Title: \(title)
        Merge the following chunk bullets into a single 3-6 bullet technical summary covering:
        - core problem
        - main method/approach
        - key results or contributions (if present)
        Keep total under 110 words.

        Chunk bullets:
        \(chunkSummaries.enumerated().map { "Chunk \($0 + 1): \($1)" }.joined(separator: "\n"))
        """

        do {
            let consolidated = try await makeSession().respond(to: consolidatePrompt)
            return SummaryOutput(summary: consolidated.content, chunksUsed: chunks.count, maxChunkCharsUsed: maxUsed)
        } catch {
            // Fallback: join chunk bullets if consolidation fails.
            let joined = chunkSummaries.joined(separator: "\n")
            return SummaryOutput(summary: joined, chunksUsed: chunks.count, maxChunkCharsUsed: maxUsed)
        }
    }

    private func summarizeSingleChunk(title: String, chunk: String, chunkIndex: Int, totalChunks: Int) async throws -> (String, Int) {
        var limit = min(chunk.count, 600)
        var lastError: Error?

        while limit >= 250 {
            let snippet = String(chunk.prefix(limit))
            let prompt = """
            Title: \(title)
            Chunk \(chunkIndex + 1) of \(totalChunks):
            \(snippet)

            Write exactly 2 concise bullet points: (1) problem, (2) method / key finding.
            Keep under 45 words total.
            """
            do {
                let response = try await makeSession().respond(to: prompt)
                return (response.content, snippet.count)
            } catch {
                lastError = error
                // shrink aggressively to stay under model context window
                limit = Int(Double(limit) * 0.5)
            }
        }
        throw lastError ?? NSError(domain: "Summarizer", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to summarize chunk within limits."])
    }

    private func fallbackSummary(title: String, text: String) -> String {
        let snippet = String(text.prefix(400))
        return "Preliminary summary for \(title): \(snippet)"
    }

    private func makeSession() -> LanguageModelSession {
        LanguageModelSession(instructions: instructions)
    }

    func summarizeSection(title: String, sectionName: String, text: String) async throws -> String {
        let snippet = String(text.prefix(1200))
        let prompt = """
        Title: \(title)
        Section: \(sectionName)

        Text:
        \(snippet)

        Write 2-3 bullet points focusing on what this section covers.
        """
        let session = LanguageModelSession(instructions: sectionInstructions)
        let response = try await session.respond(to: prompt)
        return response.content
    }

    private func chunk(text: String, maxChars: Int, overlap: Int, maxChunks: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, maxChars > 0 else { return [] }

        var chunks: [String] = []
        var start = trimmed.startIndex

        while start < trimmed.endIndex && chunks.count < maxChunks {
            let hardEnd = trimmed.index(start, offsetBy: maxChars, limitedBy: trimmed.endIndex) ?? trimmed.endIndex
            var end = hardEnd

            // Try to break at whitespace within the last 120 chars to avoid mid-word cuts.
            if hardEnd != trimmed.endIndex {
                let backWindowStart = trimmed.index(hardEnd, offsetBy: -min(120, maxChars), limitedBy: trimmed.startIndex) ?? trimmed.startIndex
                if let lastSpace = trimmed[backWindowStart..<hardEnd].lastIndex(where: { $0.isWhitespace }) {
                    end = lastSpace
                }
            }

            let slice = trimmed[start..<end]
            chunks.append(String(slice))

            if end == trimmed.endIndex { break }

            // Move start forward with overlap.
            let advance = trimmed.index(end, offsetBy: -overlap, limitedBy: trimmed.startIndex) ?? trimmed.startIndex
            start = advance
        }

        return chunks
    }
}

@available(macOS 26, iOS 26, *)
actor ClusterSummarizerActor {
    private let session: LanguageModelSession

    init() {
        let instructions = """
        You group research papers into thematic clusters and create short, descriptive names and meta-summaries for each cluster.
        """
        session = LanguageModelSession(instructions: instructions)
    }

    func summarizeCluster(index: Int, summaries: [String]) async throws -> ClusterSummary {
        let limited = summaries.prefix(20)
        var list = ""
        for (i, summary) in limited.enumerated() {
            list += "Paper \(i + 1): \(summary)\n\n"
        }

        let prompt = """
        You are given short summaries of research papers that belong to one cluster.

        TASK 1: Invent a short cluster name (3-6 words).
        TASK 2: Write one paragraph meta-summary (4-6 sentences).

        Respond EXACTLY in this format:

        Cluster name: <name>
        Meta-summary: <paragraph>

        Summaries:
        \(list)
        """

        let response = try await session.respond(to: prompt)
        let content = response.content

        var name = "Cluster \(index + 1)"
        var meta = "Contains \(summaries.count) papers."

        if let nameRange = content.range(of: "Cluster name:") {
            let rest = content[nameRange.upperBound...]
            if let eol = rest.firstIndex(of: "\n") {
                let line = rest[..<eol].trimmingCharacters(in: .whitespacesAndNewlines)
                if !line.isEmpty { name = line }
            } else {
                let line = rest.trimmingCharacters(in: .whitespacesAndNewlines)
                if !line.isEmpty { name = line }
            }
        }

        if let metaRange = content.range(of: "Meta-summary:") {
            let rest = content[metaRange.upperBound...]
            let text = rest.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { meta = text }
        }

        return ClusterSummary(name: name, metaSummary: meta)
    }
}

@available(macOS 26, iOS 26, *)
actor QuestionAnswerActor {
    private let session: LanguageModelSession

    init() {
        let instructions = """
        You are a helpful research assistant. You write short literature survey-style answers using the provided paper summaries, aimed at a technically savvy reader.
        """
        session = LanguageModelSession(instructions: instructions)
    }

    func answer(question: String, topPapers: [ScoredPaper]) async throws -> String {
        var context = ""
        for (i, scored) in topPapers.enumerated() {
            let paper = scored.paper
            let scoreText = String(format: "%.2f", scored.score)
            context += "Paper \(i + 1): \(paper.title)\nSummary: \(paper.summary)\nRelevance score: \(scoreText)\n\n"
        }

        let prompt = """
        Use the following papers to answer the research question.

        Question:
        \(question)

        Papers:
        \(context)

        Write a concise survey-style answer in 3-6 paragraphs. Group similar approaches, mention specific papers by title, and explain the main ideas. If something is unclear from the summaries, be honest about the uncertainty.
        """

        let response = try await session.respond(to: prompt)
        return response.content
    }

    func answer(question: String, evidence: [ChunkEvidence]) async throws -> String {
        let limited = evidence.prefix(12)
        let context = limited.enumerated().map { idx, ev in
            let snippet = ev.chunk.text.prefix(1200)
            return """
            Evidence \(idx + 1) — \(ev.paperTitle) (score \(String(format: "%.3f", ev.score))):
            \(snippet)
            """
        }.joined(separator: "\n\n")

        let prompt = """
        You are a cautious research assistant. Using ONLY the evidence snippets, answer the user's question.

        Question: \(question)

        Evidence:
        \(context)

        Write 3-5 concise paragraphs. Cite papers by title when appropriate. If the evidence is insufficient, say so explicitly.
        """

        let response = try await session.respond(to: prompt)
        return response.content
    }
}
