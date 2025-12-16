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
        let chunkBullets = chunkSummaries.enumerated().map { "Chunk \($0 + 1): \($1)" }.joined(separator: "\n")
        let consolidatePrompt = PromptStore.render(template: consolidateTemplate, variables: [
            "title": title,
            "chunk_bullets": chunkBullets
        ])

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
actor ClusterSummarizerActor {
    private let session: LanguageModelSession
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
        let instructions = PromptStore.loadText("cluster_summarizer.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("cluster_summarizer.prompt.md", fallback: fallbackPromptTemplate)
        session = LanguageModelSession(instructions: instructions)
    }

    func summarizeCluster(index: Int, summaries: [String]) async throws -> ClusterSummary {
        let limited = summaries.prefix(20)
        var list = ""
        for (i, summary) in limited.enumerated() {
            list += "Paper \(i + 1): \(summary)\n\n"
        }

        let prompt = PromptStore.render(template: promptTemplate, variables: [
            "summaries": list
        ])

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
        let instructions = PromptStore.loadText("question_answerer.instructions.md", fallback: fallbackInstructions)
        topPapersTemplate = PromptStore.loadText("question_answerer.top_papers.prompt.md", fallback: fallbackTopPapersTemplate)
        evidenceTemplate = PromptStore.loadText("question_answerer.evidence.prompt.md", fallback: fallbackEvidenceTemplate)
        session = LanguageModelSession(instructions: instructions)
    }

    func answer(question: String, topPapers: [ScoredPaper]) async throws -> String {
        var context = ""
        for (i, scored) in topPapers.enumerated() {
            let paper = scored.paper
            let scoreText = String(format: "%.2f", scored.score)
            context += "Paper \(i + 1): \(paper.title)\nSummary: \(paper.summary)\nRelevance score: \(scoreText)\n\n"
        }

        let prompt = PromptStore.render(template: topPapersTemplate, variables: [
            "question": question,
            "papers_context": context
        ])

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

        let prompt = PromptStore.render(template: evidenceTemplate, variables: [
            "question": question,
            "evidence_context": context
        ])

        let response = try await session.respond(to: prompt)
        return response.content
    }
}

@available(macOS 26, iOS 26, *)
actor CorpusBriefingActor {
    private let session: LanguageModelSession
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
        let instructions = PromptStore.loadText("corpus_briefing.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("corpus_briefing.prompt.md", fallback: fallbackPromptTemplate)
        session = LanguageModelSession(instructions: instructions)
    }

    func brief(context: String) async throws -> String {
        let prompt = PromptStore.render(template: promptTemplate, variables: [
            "context": context
        ])

        let response = try await session.respond(to: prompt)
        return response.content
    }
}

@available(macOS 26, iOS 26, *)
actor TopicDossierActor {
    private let session: LanguageModelSession
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
        let instructions = PromptStore.loadText("topic_dossier.instructions.md", fallback: fallbackInstructions)
        promptTemplate = PromptStore.loadText("topic_dossier.prompt.md", fallback: fallbackPromptTemplate)
        session = LanguageModelSession(instructions: instructions)
    }

    func dossier(topicName: String, context: String) async throws -> String {
        let prompt = PromptStore.render(template: promptTemplate, variables: [
            "topic_name": topicName,
            "context": context
        ])

        let response = try await session.respond(to: prompt)
        return response.content
    }
}
