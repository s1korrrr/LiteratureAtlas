import Foundation
import FoundationModels

@available(macOS 26, iOS 26, *)
actor PaperSummarizerActor {
    private let instructions: String
    private let sectionInstructions: String
    private let takeawaysTemplate: String
    private let consolidateTemplate: String
    private let chunkTemplate: String
    private let sectionTemplate: String

    struct SummaryOutput {
        let summary: String
        let chunksUsed: Int
        let maxChunkCharsUsed: Int
    }

    init() {
        let fallbackInstructions = """
        You are an expert research assistant. Given partial text from an academic paper, write a concise technical summary in 3-6 bullet points. Focus on: the problem, main method, and key results.
        Keep it tight; avoid long quotes.
        """
        let fallbackSectionInstructions = """
        Summarize the given section of an academic paper in 2-3 bullet points for a technical reader. Focus on the section's purpose and key ideas.
        """
        let fallbackTakeawaysTemplate = """
        Title: {{title}}

        Summary or text:
        {{text}}

        List 3-5 crisp bullet takeaways highlighting the core idea, method, and result. Keep each under 18 words.
        """
        let fallbackConsolidateTemplate = """
        Title: {{title}}
        Merge the following chunk bullets into a single 3-6 bullet technical summary covering:
        - core problem
        - main method/approach
        - key results or contributions (if present)
        Keep total under 110 words.

        Chunk bullets:
        {{chunk_bullets}}
        """
        let fallbackChunkTemplate = """
        Title: {{title}}
        Chunk {{chunk_index}} of {{total_chunks}}:
        {{snippet}}

        Write exactly 2 concise bullet points: (1) problem, (2) method / key finding.
        Keep under 45 words total.
        """
        let fallbackSectionTemplate = """
        Title: {{title}}
        Section: {{section_name}}

        Text:
        {{text}}

        Write 2-3 bullet points focusing on what this section covers.
        """

        instructions = PromptStore.loadText("paper_summarizer.instructions.md", fallback: fallbackInstructions)
        sectionInstructions = PromptStore.loadText("paper_summarizer.section_instructions.md", fallback: fallbackSectionInstructions)
        takeawaysTemplate = PromptStore.loadText("paper_summarizer.takeaways.prompt.md", fallback: fallbackTakeawaysTemplate)
        consolidateTemplate = PromptStore.loadText("paper_summarizer.consolidate.prompt.md", fallback: fallbackConsolidateTemplate)
        chunkTemplate = PromptStore.loadText("paper_summarizer.chunk.prompt.md", fallback: fallbackChunkTemplate)
        sectionTemplate = PromptStore.loadText("paper_summarizer.section.prompt.md", fallback: fallbackSectionTemplate)
    }

    func generateTakeaways(title: String, text: String) async throws -> [String] {
        let prompt = PromptStore.render(template: takeawaysTemplate, variables: [
            "title": title,
            "text": String(text.prefix(2000))
        ])
        let response = try await makeSession().respond(to: prompt)
        let rawLines = response.content
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        func looksLikeBullet(_ line: String) -> Bool {
            let t = line.trimmingCharacters(in: .whitespacesAndNewlines)
            let bulletChars: Set<Character> = ["-", "*", "•", "–"]
            if let first = t.first,
               bulletChars.contains(first),
               t.count >= 2 {
                let second = t[t.index(after: t.startIndex)]
                if second.isWhitespace { return true }
            }
            let digits = t.prefix(while: { $0.isNumber })
            guard !digits.isEmpty else { return false }
            let idx = t.index(t.startIndex, offsetBy: digits.count, limitedBy: t.endIndex) ?? t.endIndex
            guard idx < t.endIndex else { return false }
            let ch = t[idx]
            return ch == "." || ch == ")"
        }

        let hasBullets = rawLines.contains(where: looksLikeBullet)
        let candidates = hasBullets ? rawLines.filter(looksLikeBullet) : rawLines

        func stripBulletPrefix(_ line: String) -> String {
            var t = line.trimmingCharacters(in: .whitespacesAndNewlines)
            let bulletChars: Set<Character> = ["-", "*", "•", "–"]
            func stripOneBullet(_ text: inout String) -> Bool {
                guard let first = text.first, bulletChars.contains(first) else { return false }
                guard text.count >= 2 else { return false }
                let second = text[text.index(after: text.startIndex)]
                guard second.isWhitespace else { return false }

                // Drop the marker, then any whitespace after it.
                text.removeFirst()
                while let next = text.first, next.isWhitespace { text.removeFirst() }
                return true
            }

            // Strip repeated bullet markers like "- - foo" (but don't destroy Markdown like "**Bold**").
            while stripOneBullet(&t) {}

            // Strip numbered bullets like "1. foo" / "2) foo".
            let digits = t.prefix(while: { $0.isNumber })
            if !digits.isEmpty {
                let idx = t.index(t.startIndex, offsetBy: digits.count, limitedBy: t.endIndex) ?? t.endIndex
                if idx < t.endIndex, (t[idx] == "." || t[idx] == ")") {
                    t = String(t[t.index(after: idx)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    while stripOneBullet(&t) {}
                }
            }

            return t.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        func isPlaceholderTakeaway(_ text: String) -> Bool {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let lower = trimmed.lowercased()
            if lower.isEmpty { return true }
            // The summarizer prompts encourage "Not specified" placeholders; keep takeaways crisp by dropping them.
            if lower == "not specified" || lower == "unknown" { return true }
            if lower.hasPrefix("not specified") || lower.hasPrefix("unknown") { return true }
            if lower.contains("not specified") || lower.contains("unknown") { return true }
            return false
        }

        let cleanedAll = candidates
            .map(stripBulletPrefix)
            .filter { !$0.isEmpty }

        let cleaned = cleanedAll
            .filter { !isPlaceholderTakeaway($0) }

        // If everything was filtered out (model returned only placeholders), keep the first few anyway.
        let chosen = cleaned.isEmpty ? cleanedAll : cleaned
        return Array(chosen.prefix(5))
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
        do {
            let consolidated = try await consolidateSummary(title: title, chunkSummaries: chunkSummaries)
            return SummaryOutput(summary: consolidated, chunksUsed: chunks.count, maxChunkCharsUsed: maxUsed)
        } catch {
            // Fallback: join chunk bullets if consolidation fails.
            let joined = chunkSummaries.joined(separator: "\n")
            return SummaryOutput(summary: joined, chunksUsed: chunks.count, maxChunkCharsUsed: maxUsed)
        }
    }

    private func consolidateSummary(title: String, chunkSummaries: [String]) async throws -> String {
        let budgets = [3200, 2400, 1800, 1300, 900]
        var lastError: Error? = nil

        for maxChunkBulletsChars in budgets {
            let chunkBullets = buildChunkBullets(chunkSummaries, maxChars: maxChunkBulletsChars)
            let prompt = PromptStore.render(template: consolidateTemplate, variables: [
                "title": title,
                "chunk_bullets": chunkBullets
            ])

            do {
                let consolidated = try await makeSession().respond(to: prompt)
                return consolidated.content
            } catch {
                lastError = error
                if LLMText.isContextLimitError(error) { continue }
                throw error
            }
        }

        throw lastError ?? NSError(domain: "Summarizer", code: 4, userInfo: [
            NSLocalizedDescriptionKey: "Failed to consolidate summary within limits."
        ])
    }

    private func buildChunkBullets(_ chunkSummaries: [String], maxChars: Int) -> String {
        guard !chunkSummaries.isEmpty else { return "" }
        let n = chunkSummaries.count
        var perChunkMax = max(80, min(480, (maxChars / max(1, n)) - 16))

        func render(indices: [Int], perChunkMax: Int) -> String {
            indices.map { idx in
                let cleaned = LLMText.collapseWhitespace(chunkSummaries[idx])
                let clipped = LLMText.clip(cleaned, maxChars: perChunkMax)
                return "Chunk \(idx + 1): \(clipped)"
            }.joined(separator: "\n")
        }

        let all = Array(chunkSummaries.indices)
        var out = render(indices: all, perChunkMax: perChunkMax)
        while out.count > maxChars && perChunkMax > 60 {
            perChunkMax = max(60, Int(Double(perChunkMax) * 0.85))
            out = render(indices: all, perChunkMax: perChunkMax)
        }
        if out.count <= maxChars { return out }

        // Still too long: select a subset, prioritizing coverage from start/end.
        var order: [Int] = []
        var left = 0
        var right = n - 1
        while left <= right {
            order.append(left)
            if right != left { order.append(right) }
            left += 1
            right -= 1
        }

        var selected: [Int] = []
        var best = ""
        for idx in order {
            let candidate = (selected + [idx]).sorted()
            let attempt = render(indices: candidate, perChunkMax: perChunkMax)
            if attempt.count <= maxChars {
                selected = candidate
                best = attempt
            }
        }

        return best.isEmpty ? LLMText.clip(out, maxChars: maxChars) : best
    }

    private func summarizeSingleChunk(title: String, chunk: String, chunkIndex: Int, totalChunks: Int) async throws -> (String, Int) {
        var limit = min(chunk.count, 600)
        var lastError: Error?

        while limit >= 250 {
            let snippet = String(chunk.prefix(limit))
            let prompt = PromptStore.render(template: chunkTemplate, variables: [
                "title": title,
                "chunk_index": String(chunkIndex + 1),
                "total_chunks": String(totalChunks),
                "snippet": snippet
            ])
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
        let prompt = PromptStore.render(template: sectionTemplate, variables: [
            "title": title,
            "section_name": sectionName,
            "text": snippet
        ])
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
actor PaperTradingLensActor {
    private let instructions: String
    private let promptTemplate: String

    init() {
        let fallbackInstructions = """
        You are a quant research assistant. Your job is to convert a paper summary into a trading applicability scorecard.
        OUTPUT MUST BE VALID JSON ONLY (no Markdown fences, no extra text).
        Be grounded in the provided context. If missing, use null, empty lists, or "Unknown".
        """
        let fallbackPromptTemplate = """
        Title: {{title}}
        Keywords: {{keywords}}

        Technical summary:
        {{summary}}

        Takeaways:
        {{takeaways}}

        Return VALID JSON ONLY (no extra text) with keys:
        title,
        trading_tags, asset_classes, horizons, signal_archetypes,
        where_it_fits { pipeline_stage, primary_use },
        alpha_hypotheses [{ hypothesis, features, target, horizon }],
        data_requirements { must_have, nice_to_have },
        evaluation_notes { recommended_metrics, must_check },
        risk_flags,
        scores { novelty, usability, strategy_impact, confidence },
        one_line_verdict.

        Rules:
        - If not specified, use "Unknown" or [].
        - alpha_hypotheses: 1-3 items max.
        - risk_flags: 0-4 items max.
        """

        instructions = PromptStore.loadText("paper_trading_lens.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("paper_trading_lens.prompt.md", fallback: fallbackPromptTemplate)
    }

    private func isContextLimitError(_ error: Error) -> Bool {
        LLMText.isContextLimitError(error)
    }

    func scorecard(title: String, keywords: [String]?, summary: String, takeaways: [String]?) async throws -> PaperTradingLens {
        let cleanedKeywords = (keywords ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let keywordText = LLMText.clip(cleanedKeywords.prefix(20).joined(separator: ", "), maxChars: 220)

        let cleanedSummary = LLMText.collapseWhitespace(summary)
        let cleanedTakeaways = (takeaways ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let takeawaysText = LLMText.collapseWhitespace(cleanedTakeaways.prefix(8).joined(separator: "\n"))

        let attempts: [(summaryLimit: Int, takeawaysLimit: Int)] = [
            (summaryLimit: 1800, takeawaysLimit: 900),
            (summaryLimit: 1300, takeawaysLimit: 650),
            (summaryLimit: 900, takeawaysLimit: 450),
            (summaryLimit: 650, takeawaysLimit: 300),
            (summaryLimit: 450, takeawaysLimit: 220),
            (summaryLimit: 320, takeawaysLimit: 160),
            (summaryLimit: 220, takeawaysLimit: 120),
            (summaryLimit: 160, takeawaysLimit: 80)
        ]

        var lastError: Error?
        for attempt in attempts {
            let prompt = PromptStore.render(template: promptTemplate, variables: [
                "title": LLMText.clip(title, maxChars: 140),
                "keywords": keywordText,
                "summary": LLMText.clip(cleanedSummary, maxChars: attempt.summaryLimit),
                "takeaways": LLMText.clip(takeawaysText, maxChars: attempt.takeawaysLimit)
            ])
            do {
                // A LanguageModelSession can retain conversation context; use a fresh session per attempt to avoid growth
                // across many scorecards (e.g., trading-lens backfills).
                let session = LanguageModelSession(instructions: instructions)
                let response = try await session.respond(to: prompt)
                return try ModelJSON.decodeFirstJSON(PaperTradingLens.self, from: response.content)
            } catch {
                lastError = error
                if isContextLimitError(error) { continue }
                throw error
            }
        }

        throw lastError ?? NSError(domain: "PaperTradingLens", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to generate trading lens scorecard."])
    }
}

@available(macOS 26, iOS 26, *)
actor ClusterSummarizerActor {
    private let instructions: String
    private let promptTemplate: String

    init() {
        let fallbackInstructions = """
        You group research papers into thematic clusters and create short, descriptive names and meta-summaries for each cluster.
        """
        let fallbackPromptTemplate = """
        You are given short paper records for one thematic cluster. Each record may include a title, keywords, a short summary, and a few takeaways.

        TASK 1: Invent a short cluster name (3-6 words).
        TASK 2: Write one paragraph meta-summary (4-6 sentences).

        Respond EXACTLY in this format:

        Cluster name: <name>
        Meta-summary: <paragraph>

        Summaries:
        {{summaries}}
        """
        instructions = PromptStore.loadText("cluster_summarizer.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("cluster_summarizer.prompt.md", fallback: fallbackPromptTemplate)
    }

    func summarizeCluster(index: Int, summaries: [String]) async throws -> ClusterSummary {
        func buildList(maxChars: Int) -> String {
            let limited = summaries.prefix(20)
            var list = ""
            for (i, summary) in limited.enumerated() {
                let cleaned = LLMText.collapseWhitespace(summary)
                let clipped = LLMText.clip(cleaned, maxChars: 700)
                let entry = "Paper \(i + 1): \(clipped)\n\n"

                if !list.isEmpty, list.count + entry.count > maxChars { break }
                list += entry
                if list.count >= maxChars { break }
            }
            return list.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let budgets = [5200, 3800, 2800, 2000, 1400]
        var content: String? = nil
        var lastError: Error? = nil
        for budget in budgets {
            let list = buildList(maxChars: budget)
            let prompt = PromptStore.render(template: promptTemplate, variables: [
                "summaries": list
            ])

            do {
                let session = LanguageModelSession(instructions: instructions)
                let response = try await session.respond(to: prompt)
                content = response.content
                break
            } catch {
                lastError = error
                if LLMText.isContextLimitError(error) { continue }
                throw error
            }
        }

        guard let content else {
            throw lastError ?? NSError(domain: "ClusterSummarizer", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to summarize cluster."
            ])
        }

        var name = "Cluster \(index + 1)"
        var meta = "Contains \(summaries.count) papers."
        var lensBlock: String? = nil

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
            if let lens = text.range(of: "Trading lens:") {
                let onlyMeta = text[..<lens.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                if !onlyMeta.isEmpty { meta = onlyMeta }
                let lensText = text[lens.lowerBound...].trimmingCharacters(in: .whitespacesAndNewlines)
                if !lensText.isEmpty { lensBlock = lensText }
            } else if !text.isEmpty {
                meta = text
            }
        }

        return ClusterSummary(name: name, metaSummary: meta, tradingLens: lensBlock)
    }
}

@available(macOS 26, iOS 26, *)
actor QuestionAnswerActor {
    private let instructions: String
    private let topPapersTemplate: String
    private let evidenceTemplate: String

    init() {
        let fallbackInstructions = """
        You are a helpful research assistant. You write short literature survey-style answers using the provided paper summaries, aimed at a technically savvy reader.
        """
        let fallbackTopPapersTemplate = """
        Use the following papers to answer the research question.

        Question:
        {{question}}

        Papers:
        {{papers_context}}

        Write a concise survey-style answer in 3-6 paragraphs. Group similar approaches, mention specific papers by title, and explain the main ideas. If something is unclear from the summaries, be honest about the uncertainty.
        """
        let fallbackEvidenceTemplate = """
        You are a cautious research assistant. Using ONLY the evidence snippets, answer the user's question.

        Question: {{question}}

        Evidence:
        {{evidence_context}}

        Write 3-5 concise paragraphs. Cite papers by title when appropriate. If the evidence is insufficient, say so explicitly.
        """
        instructions = PromptStore.loadText("question_answerer.instructions.md", fallback: fallbackInstructions)
        topPapersTemplate = PromptStore.loadText("question_answerer.top_papers.prompt.md", fallback: fallbackTopPapersTemplate)
        evidenceTemplate = PromptStore.loadText("question_answerer.evidence.prompt.md", fallback: fallbackEvidenceTemplate)
    }

    func answer(question: String, topPapers: [ScoredPaper]) async throws -> String {
        func buildContext(maxChars: Int, perSummaryChars: Int) -> String {
            var context = ""
            for (i, scored) in topPapers.prefix(10).enumerated() {
                let paper = scored.paper
                let scoreText = String(format: "%.2f", scored.score)
                let summary = LLMText.clip(LLMText.collapseWhitespace(paper.summary), maxChars: perSummaryChars)
                let entry = "Paper \(i + 1): \(paper.title)\nSummary: \(summary)\nRelevance score: \(scoreText)\n\n"
                if !context.isEmpty, context.count + entry.count > maxChars { break }
                context += entry
                if context.count >= maxChars { break }
            }
            return context.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let budgets: [(maxChars: Int, perSummaryChars: Int)] = [
            (maxChars: 6500, perSummaryChars: 900),
            (maxChars: 4800, perSummaryChars: 650),
            (maxChars: 3400, perSummaryChars: 450)
        ]

        var lastError: Error? = nil
        for budget in budgets {
            let prompt = PromptStore.render(template: topPapersTemplate, variables: [
                "question": LLMText.clip(question, maxChars: 700),
                "papers_context": buildContext(maxChars: budget.maxChars, perSummaryChars: budget.perSummaryChars)
            ])

            do {
                let session = LanguageModelSession(instructions: instructions)
                let response = try await session.respond(to: prompt)
                return response.content
            } catch {
                lastError = error
                if LLMText.isContextLimitError(error) { continue }
                throw error
            }
        }

        throw lastError ?? NSError(domain: "QuestionAnswer", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Failed to answer question within context limits."
        ])
    }

    func answer(question: String, evidence: [ChunkEvidence]) async throws -> String {
        func buildEvidenceContext(maxChars: Int, perSnippetChars: Int) -> String {
            var context = ""
            for (idx, ev) in evidence.prefix(12).enumerated() {
                let snippet = LLMText.clip(LLMText.collapseWhitespace(String(ev.chunk.text.prefix(perSnippetChars * 2))), maxChars: perSnippetChars)
                let entry = """
                Evidence \(idx + 1) — \(ev.paperTitle) (score \(String(format: "%.3f", ev.score))):
                \(snippet)
                """
                let chunk = entry + "\n\n"
                if !context.isEmpty, context.count + chunk.count > maxChars { break }
                context += chunk
                if context.count >= maxChars { break }
            }
            return context.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let budgets: [(maxChars: Int, perSnippetChars: Int)] = [
            (maxChars: 6500, perSnippetChars: 900),
            (maxChars: 4800, perSnippetChars: 650),
            (maxChars: 3400, perSnippetChars: 450)
        ]

        var lastError: Error? = nil
        for budget in budgets {
            let prompt = PromptStore.render(template: evidenceTemplate, variables: [
                "question": LLMText.clip(question, maxChars: 700),
                "evidence_context": buildEvidenceContext(maxChars: budget.maxChars, perSnippetChars: budget.perSnippetChars)
            ])

            do {
                let session = LanguageModelSession(instructions: instructions)
                let response = try await session.respond(to: prompt)
                return response.content
            } catch {
                lastError = error
                if LLMText.isContextLimitError(error) { continue }
                throw error
            }
        }

        throw lastError ?? NSError(domain: "QuestionAnswer", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "Failed to answer with evidence within context limits."
        ])
    }
}

@available(macOS 26, iOS 26, *)
actor CorpusBriefingActor {
    private let instructions: String
    private let promptTemplate: String

    init() {
        let fallbackInstructions = """
        You are a principal research librarian. You read a topic hierarchy and condensed metadata about a large research corpus and produce an executive briefing that helps a researcher quickly understand the landscape.
        Output must be structured Markdown with short, information-dense sections.
        """
        let fallbackPromptTemplate = """
        You are given a topic hierarchy (mega-topics and subtopics) with sample paper titles and keyword themes.
        The goal is to summarize the corpus at a high level for a technically savvy reader.

        Write a Markdown briefing with these sections (use these exact headings):

        # Executive Summary
        # Theme Map
        # Methods & Data
        # Key Debates / Tensions
        # Frontiers & Open Questions
        # Suggested Reading Path

        Rules:
        - Be concise but specific; prefer concrete phrases over vague claims.
        - Use the provided topic names, keywords, and sample titles.
        - If something is not supported by the context, say "Not enough evidence in the provided metadata."
        - Keep the whole response under ~900 words.

        Context:
        {{context}}
        """
        instructions = PromptStore.loadText("corpus_briefing.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("corpus_briefing.prompt.md", fallback: fallbackPromptTemplate)
    }

    func brief(context: String) async throws -> String {
        let prompt = PromptStore.render(template: promptTemplate, variables: [
            "context": LLMText.clip(context, maxChars: 8_000)
        ])

        let session = LanguageModelSession(instructions: instructions)
        let response = try await session.respond(to: prompt)
        return response.content
    }
}

@available(macOS 26, iOS 26, *)
actor TopicDossierActor {
    private let instructions: String
    private let promptTemplate: String

    init() {
        let fallbackInstructions = """
        You are a senior research assistant. You write compact, structured topic dossiers that help a reader understand a research topic, its subareas, and representative papers.
        Output must be structured Markdown and should stay concrete (use the provided titles/keywords).
        """
        let fallbackPromptTemplate = """
        Write a topic dossier for: {{topic_name}}

        Use these exact headings:

        # Topic Summary
        # Subareas / Facets
        # Representative Papers
        # What’s Mature vs. What’s Emerging
        # Open Questions
        # Suggested Next Reads

        Rules:
        - Base claims only on the provided context; if uncertain, say so explicitly.
        - Prefer short bullet lists over long paragraphs.
        - Mention papers by title when listing representative work.
        - Keep the whole response under ~700 words.

        Context:
        {{context}}
        """
        instructions = PromptStore.loadText("topic_dossier.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("topic_dossier.prompt.md", fallback: fallbackPromptTemplate)
    }

    func dossier(topicName: String, context: String) async throws -> String {
        let prompt = PromptStore.render(template: promptTemplate, variables: [
            "topic_name": topicName,
            "context": LLMText.clip(context, maxChars: 8_000)
        ])

        let session = LanguageModelSession(instructions: instructions)
        let response = try await session.respond(to: prompt)
        return response.content
    }
}
