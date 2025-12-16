import Foundation
import OSLog
import SwiftUI
import Accelerate
import FoundationModels
import Darwin

enum MapLens: String, CaseIterable, Identifiable {
    case standard
    case time
    case method
    case dataRegime
    case interest

    var id: String { rawValue }

    var label: String {
        switch self {
        case .standard: return "Default"
        case .time: return "Time"
        case .method: return "Method"
        case .dataRegime: return "Data regime"
        case .interest: return "My interest"
        }
    }
}

@available(macOS 26, iOS 26, *)
@MainActor
final class AppModel: ObservableObject {
    // Data
    @Published var papers: [Paper] = [] {
        didSet {
            // Avoid O(n²) recomputes during batch ingestion/clustering loops.
            guard !isIngesting, !isClustering else { return }
            recomputeCorpusYearDomain()
        }
    }
    @Published var megaClusters: [Cluster] = []
    @Published var clusters: [Cluster] = []
    @Published var paperChunks: [PaperChunk] = []
    @Published var questionEvidence: [ChunkEvidence] = []
    @Published var analyticsSummary: AnalyticsSummary? = nil
    @Published var analyticsLoadError: String? = nil
    @Published var analyticsRebuildInFlight: Bool = false
    @Published var analyticsRebuildMessage: String? = nil
    @Published var analyticsRebuildOutput: String? = nil
    @Published var analyticsDepsInstallInFlight: Bool = false
    @Published var analyticsDepsInstallMessage: String? = nil
    @Published var analyticsDepsInstallOutput: String? = nil
    @Published var recommendationFeedback: [UUID: Bool] = [:]
    @Published var corpusBriefing: String = ""
    @Published var corpusBriefingError: String? = nil
    @Published var isGeneratingBriefing: Bool = false

    // UI state
    @Published var selectedFolder: URL?
    @Published var ingestionLog: String = ""
    @Published var isIngesting: Bool = false
    @Published var ingestionProgress: Double = 0
    @Published var ingestionCurrentFile: String = ""
    @Published var ingestionCompletedCount: Int = 0
    @Published var ingestionTotalCount: Int = 0

    // Data roots (all confined to repo Output to avoid writing outside the app directory)
    private let outputRoot: URL
    private let legacyOutputRoot: URL
    private var dataRoots: [URL] { Array(Set([outputRoot, legacyOutputRoot])) }

    @Published var isClustering: Bool = false
    @Published var clusteringProgress: Double = 0
    @Published var selectedClusterIDs: Set<Int> = []
    @Published var pinnedClusterIDs: Set<Int> = []
    @Published private(set) var clusterNameSources: [Int: ClusterNameSource] = [:]

    @Published var questionAnswer: String = ""
    @Published var questionTopPapers: [ScoredPaper] = []
    @Published var isAnswering: Bool = false
    @Published var readingProfileVector: [Float]? = nil

    // Global year filter (cross-view)
    @Published var yearFilterEnabled: Bool = false {
        didSet { recomputeExplorationCache() }
    }
    @Published var yearFilterStart: Int = 0 {
        didSet { recomputeExplorationCache() }
    }
    @Published var yearFilterEnd: Int = 0 {
        didSet { recomputeExplorationCache() }
    }
    @Published private(set) var corpusYearDomain: ClosedRange<Int>? = nil
    @Published private(set) var explorationPapers: [Paper] = []
    @Published private(set) var explorationPaperIDs: Set<UUID> = []

    // Services
    private let pdfProcessor = PDFProcessor()
    private let summarizer = PaperSummarizerActor()
    private let clusterSummarizer = ClusterSummarizerActor()
    private let questionAnswerer = QuestionAnswerActor()
    private let corpusBriefingActor = CorpusBriefingActor()
    private let topicDossierActor = TopicDossierActor()
    private let embedder = SentenceEmbedder(language: .english)
    private let logger = Logger(subsystem: "LiteratureAtlas", category: "AppModel")
    private var ingestionTask: Task<Void, Never>?
    private var canonicalEmbeddingDim: Int?
    private var clusterCache: [ClusterCacheKey: [Cluster]] = [:]
    private var subclusterCache: [Int: [Cluster]] = [:]
    private var paperANNCache: PaperANNCache?
    private var chunksByPaperID: [UUID: [PaperChunk]] = [:]
    private var chunksByPaperIDBuiltCount: Int = 0
    private var questionHistory: [String] = []
    private var questionEmbeddings: [[Float]] = []

    init(skipInitialLoad: Bool = false, customOutputRoot: URL? = nil) {
        if let custom = customOutputRoot {
            outputRoot = custom
            legacyOutputRoot = custom
            AppModel.prepareOutputRoot(custom)
        } else {
            outputRoot = AppModel.makePrimaryOutputRoot()
            legacyOutputRoot = AppModel.makeLegacyOutputRoot()
        }
        if canonicalEmbeddingDim == nil {
            let d = embedder.dimension
            if d > 0 { canonicalEmbeddingDim = d }
        }
        if !skipInitialLoad {
            Task {
                await loadSavedPapersIfNeeded()
                await loadSavedChunksIfNeeded()
                await loadSavedClustersIfNeeded()
                await loadSavedGalaxyIfPresent()
                await loadSavedCorpusBriefingIfPresent()
                await loadAnalyticsSummaryIfPresent()
            }
        }
    }

    private static func makePrimaryOutputRoot() -> URL {
        // Constrain all persisted data to the app directory (repo Output folder) so nothing leaks into ~/Documents.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let root = cwd.appendingPathComponent("Output", isDirectory: true)
        prepareOutputRoot(root)
        return root
    }

    private static func prepareOutputRoot(_ root: URL) {
        let logger = Logger(subsystem: "LiteratureAtlas", category: "IO")
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create Output root at \(root.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
        for folder in ["papers", "qa", "clusters", "chunks", "analytics", "reports", "obsidian/papers"] {
            let url = root.appendingPathComponent(folder, isDirectory: true)
            do {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            } catch {
                logger.error("Failed to create Output subfolder \(url.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    private static func makeLegacyOutputRoot() -> URL {
        // Legacy path now matches primary to keep everything local to the app directory.
        return makePrimaryOutputRoot()
    }

    private func fallbackEmbedding(for text: String, dimension: Int = 256) -> [Float] {
        guard !text.isEmpty else { return [] }
        let dim = canonicalEmbeddingDim ?? dimension
        var vec = [Float](repeating: 0, count: dim)
        for token in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
            let h = abs(token.hashValue) % dim
            vec[h] += 1
        }
        let norm = sqrt(vDSP.dot(vec, vec))
        if norm > 0 {
            vDSP.divide(vec, norm, result: &vec)
        }
        return vec
    }

    private func extractKeywords(title: String, summary: String, limit: Int = 12) -> [String] {
        let text = (title + " " + summary).lowercased()
        let tokens = text.split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        var freq: [String: Int] = [:]
        for t in tokens where t.count > 3 {
            freq[String(t), default: 0] += 1
        }
        return Array(freq.sorted { $0.value > $1.value }.prefix(limit).map { $0.key })
    }

    private func normalizeEmbedding(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }
        if canonicalEmbeddingDim == nil {
            canonicalEmbeddingDim = vector.count
            return vector
        }
        guard let dim = canonicalEmbeddingDim else { return vector }
        if vector.count == dim { return vector }
        if vector.count > dim { return Array(vector.prefix(dim)) }
        var v = vector
        v.append(contentsOf: repeatElement(0, count: dim - vector.count))
        return v
    }

    // MARK: - Global year filter

    var effectiveYearRange: ClosedRange<Int>? {
        guard yearFilterEnabled, let domain = corpusYearDomain else { return nil }
        let startCandidate = yearFilterStart == 0 ? domain.lowerBound : yearFilterStart
        let endCandidate = yearFilterEnd == 0 ? domain.upperBound : yearFilterEnd
        let start = min(max(startCandidate, domain.lowerBound), domain.upperBound)
        let end = min(max(endCandidate, domain.lowerBound), domain.upperBound)
        return min(start, end)...max(start, end)
    }

    func resetYearFilter() {
        yearFilterEnabled = false
        guard let domain = corpusYearDomain else { return }
        yearFilterStart = domain.lowerBound
        yearFilterEnd = domain.upperBound
    }

    func paperPassesYearFilter(_ paper: Paper) -> Bool {
        guard let range = effectiveYearRange else { return true }
        guard let year = paper.year else { return false }
        return range.contains(year)
    }

    func paperCount(for cluster: Cluster) -> Int {
        guard yearFilterEnabled else { return cluster.memberPaperIDs.count }
        guard !explorationPaperIDs.isEmpty else { return 0 }
        return cluster.memberPaperIDs.reduce(0) { partial, paperID in
            partial + (explorationPaperIDs.contains(paperID) ? 1 : 0)
        }
    }

    private func recomputeCorpusYearDomain() {
        let currentYear = Calendar.current.component(.year, from: Date())
        let years = papers.compactMap(\.year).filter { $0 >= 1900 && $0 <= currentYear + 1 }
        guard let minYear = years.min(), let maxYear = years.max(), minYear <= maxYear else {
            corpusYearDomain = nil
            recomputeExplorationCache()
            return
        }

        corpusYearDomain = minYear...maxYear
        if yearFilterStart == 0 { yearFilterStart = minYear }
        if yearFilterEnd == 0 { yearFilterEnd = maxYear }
        if yearFilterStart < minYear { yearFilterStart = minYear }
        if yearFilterStart > maxYear { yearFilterStart = maxYear }
        if yearFilterEnd < minYear { yearFilterEnd = minYear }
        if yearFilterEnd > maxYear { yearFilterEnd = maxYear }

        recomputeExplorationCache()
    }

    private func recomputeExplorationCache() {
        guard let range = effectiveYearRange else {
            explorationPapers = papers
            explorationPaperIDs = Set(papers.map(\.id))
            return
        }

        let filtered = papers.filter { paper in
            guard let year = paper.year else { return false }
            return range.contains(year)
        }
        explorationPapers = filtered
        explorationPaperIDs = Set(filtered.map(\.id))
    }

    // MARK: - Ingestion

    func ingestFolder(url: URL) {
        selectedFolder = url
        ingestionLog = "Selected folder: \(url.lastPathComponent)"
        logger.info("[Ingest] Selected folder: \(url.path, privacy: .public)")
        ingestionTask?.cancel()
        ingestionTask = Task { await runIngestion(folderURL: url) }
    }

    func cancelIngestion() {
        ingestionTask?.cancel()
    }

    private func runIngestion(folderURL: URL) async {
        defer {
            isIngesting = false
            ingestionTask = nil
        }
        isIngesting = true
        clusters.removeAll()
        selectedClusterIDs.removeAll()
        ingestionProgress = 0
        ingestionCompletedCount = 0
        ingestionCurrentFile = ""
        ingestionTotalCount = 0

        await loadSavedPapersIfNeeded()
        await loadSavedChunksIfNeeded()
        var knownPaths = Set(papers.map { $0.filePath })

        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
            ingestionLog += "\nFailed to read folder."
            isIngesting = false
            return
        }

        let pdfs = items.filter { $0.pathExtension.lowercased() == "pdf" }
        guard !pdfs.isEmpty else {
            ingestionLog += "\nNo PDF files found in folder."
            logger.info("[Ingest] No PDFs found in folder \(folderURL.path, privacy: .public)")
            isIngesting = false
            return
        }

        ingestionTotalCount = pdfs.count

        for (index, pdfURL) in pdfs.enumerated() {
            #if os(iOS) || os(macOS)
            let needsAccess = pdfURL.startAccessingSecurityScopedResource()
            defer { if needsAccess { pdfURL.stopAccessingSecurityScopedResource() } }
            #endif

            ingestionCurrentFile = pdfURL.lastPathComponent
            ingestionLog += "\n\n[>] Processing \(pdfURL.lastPathComponent)..."
            logger.info("[Ingest] Processing: \(pdfURL.lastPathComponent, privacy: .public)")

            if Task.isCancelled { break }

            let baseName = pdfURL.deletingPathExtension().lastPathComponent
            if shouldSkipPDF(pdfURL: pdfURL, baseName: baseName, knownPaths: knownPaths) {
                ingestionLog += "\n  Skipped: up to date."
                ingestionProgress = Double(index + 1) / Double(max(pdfs.count, 1))
                ingestionCompletedCount = index + 1
                continue
            }

            do {
                if Task.isCancelled { break }
                let text = try pdfProcessor.extractFirstPagesText(from: pdfURL, maxPages: 3)
                ingestionLog += "\n  Extracted \(text.count) characters."

                if Task.isCancelled { break }
                let title = pdfProcessor.inferTitle(for: pdfURL, text: text)
                var year = pdfProcessor.inferYear(from: pdfURL)

                ingestionLog += "\n  Summarizing with on-device model (chunked)..."
                let summaryOutput = try await summarizer.summarize(title: title, text: text)
                let summary = summaryOutput.summary
                ingestionLog += "\n  Summary length: \(summary.count). Chunks: \(summaryOutput.chunksUsed) (max chars used: \(summaryOutput.maxChunkCharsUsed))."

                // Section summaries (rough slicing)
                let thirds = max(1, text.count / 3)
                let introText = String(text.prefix(thirds))
                let methodText = text.count > thirds ? String(text.dropFirst(thirds).prefix(thirds)) : introText
                let resultsText = text.count > 2 * thirds ? String(text.dropFirst(2 * thirds)) : methodText

                let introSummary = try? await summarizer.summarizeSection(title: title, sectionName: "Introduction", text: introText)
                let methodSummary = try? await summarizer.summarizeSection(title: title, sectionName: "Methods", text: methodText)
                let resultsSummary = try? await summarizer.summarizeSection(title: title, sectionName: "Results", text: resultsText)
                let takeaways = try? await summarizer.generateTakeaways(title: title, text: summary)
                // Heuristic claim/assumption extraction for later evidence graph.
                let preliminaryPaperID = UUID()
                let claimExtraction = ClaimExtractor.heuristicExtraction(summary: summary, paperID: preliminaryPaperID, year: year)
                // Derive a lightweight method pipeline from the method summary if available.
                let pipelineSource = methodSummary ?? summary
                let pipeline = MethodPipelineExtractor.extract(from: pipelineSource)

                if year == nil {
                    year = pdfProcessor.inferYear(fromText: text)
                }
                let pageCount = pdfProcessor.pageCount(for: pdfURL)
                if let y = year {
                    ingestionLog += "\n  Inferred year: \(y)"
                } else {
                    ingestionLog += "\n  Year unknown."
                }
                if let pages = pageCount {
                    ingestionLog += "\n  Pages: \(pages)"
                }

                let fingerprint = "\(title)\n\(summary)"
                var embedding = normalizeEmbedding(await embedder.encode(for: fingerprint) ?? [])
                if embedding.isEmpty {
                    embedding = normalizeEmbedding(fallbackEmbedding(for: fingerprint))
                    ingestionLog += "\n  Embedding unavailable; using fallback hashing embedding (dim=\(embedding.count))."
                } else {
                    ingestionLog += "\n  Embedding dimension: \(embedding.count)."
                }

                if embedding.isEmpty {
                    ingestionLog += "\n  Skipped: embedding unavailable."
                } else {
                    // Use the pre-generated UUID so claim extraction references match.
                    let paperID = preliminaryPaperID
                    let paper = Paper(
                        version: 1,
                        filePath: pdfURL.path,
                        id: paperID,
                        originalFilename: pdfURL.lastPathComponent,
                        title: title,
                        introSummary: introSummary,
                        summary: summary,
                        methodSummary: methodSummary,
                        resultsSummary: resultsSummary,
                        takeaways: takeaways,
                        keywords: extractKeywords(title: title, summary: summary),
                        userNotes: nil,
                        userTags: nil,
                        readingStatus: nil,
                        noteEmbedding: nil,
                        userQuestions: nil,
                        flashcards: nil,
                        year: year,
                        embedding: embedding,
                        clusterIndex: nil,
                        claims: claimExtraction.claims,
                        assumptions: claimExtraction.assumptions,
                        evaluationContext: claimExtraction.evaluation,
                        methodPipeline: pipeline,
                        firstReadAt: nil,
                        ingestedAt: Date(),
                        pageCount: pageCount
                    )
                    // Chunk-level embeddings for RAG.
                    let chunks = await buildChunks(for: text, paperID: paperID)
                    if !chunks.isEmpty {
                        paperChunks.removeAll(where: { $0.paperID == paperID })
                        paperChunks.append(contentsOf: chunks)
                        invalidateChunkLookupCache()
                    }

                    upsertPaper(paper)

                    let jsonURL = try savePaperJSON(paper)
                    ingestionLog += "\n  Saved JSON: \(jsonURL.lastPathComponent)"
                    if let mdURL = try? PaperMarkdownExporter.write(paper: paper, outputRoot: outputRoot) {
                        ingestionLog += "\n  Saved MD: \(mdURL.lastPathComponent)"
                    }
                    saveChunkIndex()
                    knownPaths.insert(pdfURL.path)
                }
            } catch {
                ingestionLog += "\n  Error: \(error.localizedDescription)"
                logger.error("[Ingest] Error: \(error.localizedDescription, privacy: .public)")
            }

            ingestionProgress = Double(index + 1) / Double(max(pdfs.count, 1))
            ingestionCompletedCount = index + 1
        }

        ingestionCurrentFile = ""
        if Task.isCancelled {
            ingestionLog += "\nIngestion cancelled after \(ingestionCompletedCount) files."
        } else {
            ingestionLog += "\n\nIngestion complete. Processed \(papers.count) papers."
            logger.info("[Ingest] Completed. Processed \(self.papers.count, privacy: .public) papers.")
            recomputeReadingProfile(extra: nil)
            if papers.count >= 3 {
                Task { await buildMultiScaleGalaxy() }
            } else {
                ingestionLog += "\nNot enough papers to auto-cluster (need >=3)."
            }
            savePaperIndex()
        }
        saveChunkIndex()
        recomputeCorpusYearDomain()
        isIngesting = false
    }

    private func savePaperJSON(_ paper: Paper) throws -> URL {
        let baseName = paper.title.isEmpty ? paper.originalFilename : paper.title
        let safeName = baseName.replacingOccurrences(of: "/", with: "-")
        let url = outputRoot.appendingPathComponent("papers", isDirectory: true).appendingPathComponent(safeName + ".paper.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        let data = try encoder.encode(paper)
        try data.write(to: url, options: .atomic)
        return url
    }

    private func shouldSkipPDF(pdfURL: URL, baseName: String, knownPaths: Set<String>) -> Bool {
        let fm = FileManager.default
        guard knownPaths.contains(pdfURL.path) else { return false }
        guard let attr = try? fm.attributesOfItem(atPath: pdfURL.path),
              let modDate = attr[.modificationDate] as? Date else { return false }

        for root in dataRoots {
            let jsonURL = root.appendingPathComponent("papers", isDirectory: true).appendingPathComponent(baseName + ".paper.json")
            if fm.fileExists(atPath: jsonURL.path),
               let jAttr = try? fm.attributesOfItem(atPath: jsonURL.path),
               let jDate = jAttr[.modificationDate] as? Date,
               jDate >= modDate,
               validatedCachedPaper(baseName: baseName) != nil {
                ingestionLog += "\n  Validated existing record; skipping."
                return true
            }
        }
        return false
    }

    private func validatedCachedPaper(baseName: String) -> Paper? {
        let decoder = JSONDecoder()
        for root in dataRoots {
            let jsonURL = root.appendingPathComponent("papers", isDirectory: true).appendingPathComponent(baseName + ".paper.json")
            if let data = try? Data(contentsOf: jsonURL),
               let paper = try? decoder.decode(Paper.self, from: data) {
                if !paper.summary.isEmpty, !paper.embedding.isEmpty {
                    return paper
                }
            }
        }
        return nil
    }

    private func buildChunks(for text: String, paperID: UUID) async -> [PaperChunk] {
        let segments = chunk(text: text, maxChars: 1800, overlap: 200, maxChunks: 20)
        var results: [PaperChunk] = []
        for (idx, segment) in segments.enumerated() {
            var vec = normalizeEmbedding(await embedder.encode(for: segment) ?? [])
            if vec.isEmpty {
                vec = normalizeEmbedding(fallbackEmbedding(for: segment))
            }
            guard !vec.isEmpty else { continue }
            let chunk = PaperChunk(id: UUID(), paperID: paperID, text: segment, embedding: vec, order: idx, pageHint: nil)
            results.append(chunk)
        }
        return results
    }

    private func chunk(text: String, maxChars: Int, overlap: Int, maxChunks: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, maxChars > 0 else { return [] }

        var chunks: [String] = []
        var start = trimmed.startIndex

        while start < trimmed.endIndex && chunks.count < maxChunks {
            let hardEnd = trimmed.index(start, offsetBy: maxChars, limitedBy: trimmed.endIndex) ?? trimmed.endIndex
            var end = hardEnd

            if hardEnd != trimmed.endIndex {
                let backWindowStart = trimmed.index(hardEnd, offsetBy: -min(120, maxChars), limitedBy: trimmed.startIndex) ?? trimmed.startIndex
                if let lastSpace = trimmed[backWindowStart..<hardEnd].lastIndex(where: { $0.isWhitespace }) {
                    end = lastSpace
                }
            }

            let slice = trimmed[start..<end]
            chunks.append(String(slice))

            if end == trimmed.endIndex { break }

            let advance = trimmed.index(end, offsetBy: -overlap, limitedBy: trimmed.startIndex) ?? trimmed.startIndex
            start = advance
        }

        return chunks
    }

#if DEBUG
    // Test hooks
    func testUpsert(_ paper: Paper) {
        upsertPaper(paper)
    }

    func testRunClustering(k: Int) async {
        await runClustering(k: k)
    }

    func testChunkSegments(text: String, maxChars: Int, overlap: Int, maxChunks: Int) -> [String] {
        chunk(text: text, maxChars: maxChars, overlap: overlap, maxChunks: maxChunks)
    }
#endif

    private func upsertPaper(_ paper: Paper) {
        if let idx = papers.firstIndex(where: { $0.filePath == paper.filePath }) {
            papers[idx] = paper
        } else {
            papers.append(paper)
        }
    }

    private func loadSavedPapersIfNeeded() async {
        let decoder = JSONDecoder()
        var newestByPath: [String: (paper: Paper, date: Date?)] = [:]

        let candidates = listPaperJSONCandidates()
        guard !candidates.isEmpty else { return }

        for url in candidates {
            guard let data = try? Data(contentsOf: url),
                  var paper = try? decoder.decode(Paper.self, from: data) else { continue }
            paper.embedding = normalizeEmbedding(paper.embedding)
            paper.noteEmbedding = normalizeEmbedding(paper.noteEmbedding ?? [])
            if let year = paper.year {
                let currentYear = Calendar.current.component(.year, from: Date())
                if year < 1900 || year > currentYear + 1 {
                    paper.year = nil
                }
            }
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            let modDate = values?.contentModificationDate

            if let existing = newestByPath[paper.filePath] {
                // Prefer the most recently modified JSON if duplicates exist.
                if let modDate = modDate, let existingDate = existing.date, modDate < existingDate { continue }
            }
            newestByPath[paper.filePath] = (paper, modDate)
        }

        let loaded = newestByPath.values.map { $0.paper }
        if !loaded.isEmpty {
            papers = loaded
            ingestionLog += "\nLoaded \(loaded.count) papers from disk (Output folder)."
            savePaperIndex()

            // One-time Obsidian export for existing papers (only if missing).
            let root = outputRoot
            Task.detached(priority: .utility) {
                let fm = FileManager.default
                let folder = root
                    .appendingPathComponent("obsidian", isDirectory: true)
                    .appendingPathComponent("papers", isDirectory: true)
                let existing = (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
                let existingNames = existing.map(\.lastPathComponent)
                for paper in loaded {
                    let token = paper.id.uuidString
                    guard !existingNames.contains(where: { $0.contains(token) }) else { continue }
                    _ = try? PaperMarkdownExporter.write(paper: paper, outputRoot: root)
                }
            }
        }
    }

    private func listPaperJSONCandidates() -> [URL] {
        var urls: Set<URL> = []
        let fm = FileManager.default

        for root in dataRoots {
            let folder = root.appendingPathComponent("papers", isDirectory: true)
            if let files = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                for file in files where file.lastPathComponent.hasSuffix(".paper.json") {
                    urls.insert(file)
                }
            }
        }

        return Array(urls)
    }

    private func loadSavedChunksIfNeeded() async {
        let decoder = JSONDecoder()
        var newest: (chunks: [PaperChunk], date: Date?)?
        for root in dataRoots {
            let url = root.appendingPathComponent("chunks", isDirectory: true).appendingPathComponent("chunks.json")
            guard FileManager.default.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url),
                  let decoded = try? decoder.decode([PaperChunk].self, from: data) else { continue }
            let mod = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            if let existingDate = newest?.date, let mod = mod, mod < existingDate { continue }
            let normalized = decoded.map { chunk -> PaperChunk in
                var c = chunk
                c.embedding = normalizeEmbedding(chunk.embedding)
                return c
            }
            newest = (normalized, mod)
        }
        if let loaded = newest?.chunks {
            paperChunks = loaded
            invalidateChunkLookupCache()
        }
    }

    func reloadAnalyticsSummary() {
        let url = outputRoot.appendingPathComponent("analytics", isDirectory: true).appendingPathComponent("analytics.json")
        do {
            analyticsSummary = try AnalyticsStore.loadSummary(from: url)
            analyticsLoadError = analyticsSummary == nil ? "analytics.json not found. Run the Python rebuild." : nil
        } catch {
            analyticsSummary = nil
            analyticsLoadError = error.localizedDescription
        }
    }

    private func loadAnalyticsSummaryIfPresent() async {
        await MainActor.run {
            reloadAnalyticsSummary()
        }
    }

#if os(macOS)
    /// Runs the Python analytics rebuild script (analytics/rebuild_analytics.py) from the repo root.
    func rebuildAnalyticsViaPython() {
        guard !analyticsRebuildInFlight else { return }
        analyticsRebuildInFlight = true
        analyticsRebuildMessage = "Running analytics rebuild…"
        analyticsRebuildOutput = nil

        let repoRoot = outputRoot.deletingLastPathComponent()
        let script = repoRoot.appendingPathComponent("analytics/rebuild_analytics.py")
        persistClaimGraphEdgesSnapshot()

        Task.detached { [weak self] in
            let args = ["--base", repoRoot.path]
            let result: Result<(Int32, String, String), Error> = AppModel.runPython(arguments: [script.path] + args, cwd: repoRoot)
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.analyticsRebuildInFlight = false
                switch result {
                case .success(let (status, output, interpreter)):
                    let tail = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.analyticsRebuildOutput = String(tail.suffix(8_000))
                    if status == 0 {
                        self.analyticsRebuildMessage = "Analytics rebuilt (\(interpreter))."
                    } else if output.contains("ModuleNotFoundError") || output.contains("No module named") {
                        self.analyticsRebuildMessage = "Missing Python packages (click “Install Python deps”). (\(interpreter))"
                    } else {
                        self.analyticsRebuildMessage = "Rebuild failed (exit \(status), \(interpreter))."
                    }
                    self.ingestionLog += "\n[analytics] \(output)"
                    self.reloadAnalyticsSummary()
                case .failure(let error):
                    self.analyticsRebuildMessage = "Failed to run analytics script: \(error.localizedDescription)"
                }
            }
        }
    }

    func rebuildAnalyticsWithCutoffs(_ cutoffs: [Int]) {
        guard !analyticsRebuildInFlight else { return }
        analyticsRebuildInFlight = true
        analyticsRebuildMessage = "Recomputing analytics…"
        analyticsRebuildOutput = nil

        let repoRoot = outputRoot.deletingLastPathComponent()
        let script = repoRoot.appendingPathComponent("analytics/rebuild_analytics.py")
        let args = ["--base", repoRoot.path, "--counterfactual-cutoffs"] + cutoffs.map(String.init)
        persistClaimGraphEdgesSnapshot()

        Task.detached { [weak self] in
            let result: Result<(Int32, String, String), Error> = AppModel.runPython(arguments: [script.path] + args, cwd: repoRoot)
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.analyticsRebuildInFlight = false
                switch result {
                case .success(let (status, output, interpreter)):
                    let tail = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.analyticsRebuildOutput = String(tail.suffix(8_000))
                    if status == 0 {
                        self.analyticsRebuildMessage = "Analytics recomputed (\(interpreter))."
                    } else if output.contains("ModuleNotFoundError") || output.contains("No module named") {
                        self.analyticsRebuildMessage = "Missing Python packages (click “Install Python deps”). (\(interpreter))"
                    } else {
                        self.analyticsRebuildMessage = "Rebuild failed (exit \(status), \(interpreter))."
                    }
                    self.ingestionLog += "\n[analytics] \(output)"
                    self.reloadAnalyticsSummary()
                case .failure(let error):
                    self.analyticsRebuildMessage = "Failed to run analytics script: \(error.localizedDescription)"
                }
            }
        }
    }

    func installAnalyticsPythonDependencies() {
        guard !analyticsDepsInstallInFlight else { return }
        analyticsDepsInstallInFlight = true
        analyticsDepsInstallMessage = "Installing Python packages…"
        analyticsDepsInstallOutput = nil

        let repoRoot = outputRoot.deletingLastPathComponent()
        let requirements = repoRoot.appendingPathComponent("analytics/requirements.txt")

        Task.detached { [weak self] in
            let result: Result<(Int32, String, String), Error> = AppModel.runPython(arguments: ["-m", "pip", "install", "-r", requirements.path], cwd: repoRoot)
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.analyticsDepsInstallInFlight = false
                switch result {
                case .success(let (status, output, interpreter)):
                    let tail = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.analyticsDepsInstallOutput = String(tail.suffix(8_000))
                    self.analyticsDepsInstallMessage = status == 0 ? "Python deps installed (\(interpreter))." : "Install failed (exit \(status), \(interpreter))."
                case .failure(let error):
                    self.analyticsDepsInstallMessage = "Install failed: \(error.localizedDescription)"
                }
            }
        }
    }

    nonisolated private static func runPython(arguments: [String], cwd: URL) -> Result<(Int32, String, String), Error> {
        enum CandidateKind {
            case direct(URL)
            case env(String)

            var label: String {
                switch self {
                case .direct(let url): return url.path
                case .env(let cmd): return cmd
                }
            }
        }

        let fm = FileManager.default
        var candidates: [CandidateKind] = []
        var seen: Set<String> = []
        func add(_ candidate: CandidateKind) {
            let key = candidate.label
            guard !seen.contains(key) else { return }
            seen.insert(key)
            candidates.append(candidate)
        }

        if let override = ProcessInfo.processInfo.environment["LITERATURE_ATLAS_PYTHON"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if override.contains("/") {
                let resolved: URL = override.hasPrefix("/") ? URL(fileURLWithPath: override) : cwd.appendingPathComponent(override)
                if fm.fileExists(atPath: resolved.path) {
                    add(.direct(resolved))
                }
            } else {
                add(.env(override))
            }
        }

        // Prefer repo-local virtual envs (but still allow fallbacks).
        for rel in [".venv/bin/python3", ".venv/bin/python", "venv/bin/python3", "venv/bin/python"] {
            let url = cwd.appendingPathComponent(rel)
            if fm.fileExists(atPath: url.path) {
                add(.direct(url))
            }
        }

        // Common brew/system python paths.
        for abs in ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"] {
            let url = URL(fileURLWithPath: abs)
            if fm.fileExists(atPath: url.path) {
                add(.direct(url))
            }
        }

        // Fall back to PATH resolution.
        add(.env("python3"))
        add(.env("python"))

        func run(candidate: CandidateKind) throws -> (Int32, String, String) {
            let pipe = Pipe()
            let process = Process()
            switch candidate {
            case .direct(let pythonURL):
                process.executableURL = pythonURL
                process.arguments = arguments
            case .env(let cmd):
                process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                process.arguments = [cmd] + arguments
            }
            process.standardOutput = pipe
            process.standardError = pipe
            process.currentDirectoryURL = cwd
            process.environment = (ProcessInfo.processInfo.environment).merging([
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            ]) { _, new in new }

            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            return (process.terminationStatus, output, candidate.label)
        }

        var best: (Int32, String, String)?
        var lastError: Error?
        for cand in candidates {
            do {
                let (status, output, label) = try run(candidate: cand)
                if status == 0 {
                    return .success((status, output, label))
                }
                // Keep the most informative output.
                if best == nil || output.count > (best?.1.count ?? 0) {
                    best = (status, output, label)
                }
            } catch {
                lastError = error
                continue
            }
        }

        if let best {
            return .success(best)
        }
        return .failure(lastError ?? NSError(domain: "LiteratureAtlas", code: 1))
    }
#endif

    private func saveChunkIndex() {
        let url = outputRoot.appendingPathComponent("chunks", isDirectory: true).appendingPathComponent("chunks.json")
        do {
            let data = try JSONEncoder().encode(paperChunks)
            try data.write(to: url, options: .atomic)
        } catch {
            logger.error("Failed to save chunks index: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func loadSavedClustersIfNeeded() async {
        let version = currentCorpusVersion()
        let decoder = JSONDecoder()
        let snapshots = clusterSnapshotFiles().compactMap { url -> ClusterSnapshot? in
            guard let data = try? Data(contentsOf: url) else { return nil }
            if let snap = try? decoder.decode(ClusterSnapshot.self, from: data) {
                return snap
            }
            // Legacy format: plain array
            if let legacyClusters = try? decoder.decode([Cluster].self, from: data) {
                return ClusterSnapshot(version: "legacy", k: legacyClusters.count, clusters: legacyClusters)
            }
            return nil
        }

        for snapshot in snapshots {
            let key = ClusterCacheKey(version: snapshot.version, k: snapshot.k)
            var normalized: [Cluster] = []
            for var c in snapshot.clusters {
                c.centroid = normalizeEmbedding(c.centroid)
                normalized.append(c)
            }
            clusterCache[key] = normalized
        }

        // Pick a default set matching current corpus version, preferring k closest to 10.
        let matching = snapshots.filter { $0.version == version }
        if let pick = matching.sorted(by: { abs($0.k - 10) < abs($1.k - 10) }).first,
           let cached = clusterCache[ClusterCacheKey(version: pick.version, k: pick.k)] {
            clusters = cached
           reattachClustersToPapers()
           ingestionLog += "\nLoaded cached clusters (k=\(pick.k)) for current corpus."
       }
   }

    private func clusterSnapshotFiles() -> [URL] {
        var urls: [URL] = []
        let fm = FileManager.default
        for root in dataRoots {
            let folder = root.appendingPathComponent("clusters", isDirectory: true)
            if let files = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
                for file in files where file.lastPathComponent.hasPrefix("clusters_") && file.pathExtension == "json" {
                    urls.append(file)
                }
                // Legacy single snapshot
                let legacy = folder.appendingPathComponent("clusters.json")
                if fm.fileExists(atPath: legacy.path) {
                    urls.append(legacy)
                }
            }
        }
        return urls
    }

    private func reattachClustersToPapers() {
        var paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        for cluster in clusters {
            for pid in cluster.memberPaperIDs {
                if var p = paperByID[pid] {
                    p.clusterIndex = cluster.id
                    paperByID[pid] = p
                }
            }
        }
        papers = Array(paperByID.values)
    }

    private func savePaperIndex() {
        struct Minimal: Codable {
            let id: UUID
            let title: String
            let summary: String
        }

        let index = papers.map { Minimal(id: $0.id, title: $0.title, summary: $0.summary) }
        let url = outputRoot.appendingPathComponent("papers", isDirectory: true).appendingPathComponent("index.json")
        do {
            let data = try JSONEncoder().encode(index)
            try data.write(to: url, options: .atomic)
        } catch {
            logger.error("Failed to save paper index: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Clustering

    func performClustering(k: Int) {
        Task { await runClustering(k: k) }
    }

    private func runClustering(k: Int) async {
        guard !papers.isEmpty else { return }

        let corpusVersion = currentCorpusVersion()
        let clampedK = min(max(1, k), papers.count)
        let cacheKey = ClusterCacheKey(version: corpusVersion, k: clampedK)

        if let cached = clusterCache[cacheKey] {
            clusters = cached
            reattachClustersToPapers()
            ingestionLog += "\nUsing cached clusters for k=\(clampedK)."
            return
        }

        isClustering = true
        clusteringProgress = 0
        clusters.removeAll()
        selectedClusterIDs.removeAll()

        let validPapers = papers.filter { !$0.embedding.isEmpty }
        if validPapers.count < 2 {
            ingestionLog += "\nNot enough embedded papers to cluster."
            isClustering = false
            return
        }

        let embeddings = validPapers.map { $0.embedding }
        let dim = embeddings.first?.count ?? 0
        guard dim > 0 else {
            isClustering = false
            return
        }

        let kClamped = min(max(1, k), embeddings.count)
        ingestionLog += "\n\n[cluster] Clustering into \(kClamped) clusters..."

        let compute = await Task.detached(priority: .userInitiated) { () -> (assignments: [Int], centroids: [[Float]], positions: [Point2D]) in
            let (assignments, centroids) = KMeans.cluster(vectors: embeddings, k: kClamped, iterations: 25)
            let positions = ForceLayout.compute(for: centroids)
            return (assignments, centroids, positions)
        }.value
        let assignments = compute.assignments
        let centroids = compute.centroids
        let positions = compute.positions

        for i in validPapers.indices {
            if let originalIndex = papers.firstIndex(where: { $0.id == validPapers[i].id }) {
                papers[originalIndex].clusterIndex = assignments[i]
            }
        }

        var newClusters: [Cluster] = []

        for clusterID in 0..<kClamped {
            let memberIndices = assignments.enumerated().filter { $0.element == clusterID }.map { $0.offset }
            guard !memberIndices.isEmpty else { continue }
            let members = memberIndices.map { validPapers[$0] }
            let centroid = centroids[clusterID]
            let summaries = members.map { $0.summary }

            do {
                let info = try await clusterSummarizer.summarizeCluster(index: clusterID, summaries: summaries)
                let cluster = Cluster(
                    id: clusterID,
                    name: info.name,
                    metaSummary: info.metaSummary,
                    centroid: centroid,
                    memberPaperIDs: members.map { $0.id },
                    layoutPosition: nil,
                    resolutionK: kClamped,
                    corpusVersion: corpusVersion,
                    subclusters: nil
                )
                newClusters.append(cluster)
            } catch {
                let cluster = Cluster(
                    id: clusterID,
                    name: "Cluster \(clusterID + 1)",
                    metaSummary: "Contains \(members.count) papers.",
                    centroid: centroid,
                    memberPaperIDs: members.map { $0.id },
                    layoutPosition: nil,
                    resolutionK: kClamped,
                    corpusVersion: corpusVersion,
                    subclusters: nil
                )
                newClusters.append(cluster)
            }
        }

        // Apply precomputed force layout positions.
        for idx in newClusters.indices {
            let cid = newClusters[idx].id
            if cid < positions.count {
                newClusters[idx].layoutPosition = positions[cid]
            }
        }
        clusters = newClusters.sorted { $0.id < $1.id }
        ingestionLog += "\nClustering and meta-summaries complete."
        isClustering = false
        clusteringProgress = 1

        // Persist clusters snapshot
        clusterCache[cacheKey] = clusters
        persistClusterSnapshot(clusters, key: cacheKey)
    }

    /// Builds a three-level "Knowledge Galaxy":
    /// - Level 0: 5–8 mega-topics (global clustering on paper embeddings).
    /// - Level 1: 10–20 subtopics within each mega-topic (local clustering).
    /// - Level 2: individual papers within subtopics (paper nodes are shown at high zoom).
    func buildMultiScaleGalaxy(level0Range: ClosedRange<Int> = 5...8, level1Range: ClosedRange<Int> = 10...20) async {
        isClustering = true
        clusteringProgress = 0
        let validPapers = papers.filter { !$0.embedding.isEmpty }
        guard !validPapers.isEmpty else {
            isClustering = false
            return
        }

        let corpusVersion = currentCorpusVersion()
        let inputs: [(id: UUID, embedding: [Float])] = validPapers.map { ($0.id, $0.embedding) }

        let compute = await Task.detached(priority: .userInitiated) { () -> GalaxyComputeResult in
            let count = inputs.count
            let kMega = min(
                max(level0Range.lowerBound, count / max(1, count / 6)),
                min(level0Range.upperBound, count)
            )

            let embeddings = inputs.map { $0.embedding }
            let (megaAssignments, megaCentroids) = KMeans.cluster(vectors: embeddings, k: kMega, iterations: 30)

            var megaInfos: [GalaxyClusterInfo] = []
            var subInfos: [GalaxyClusterInfo] = []
            var assignmentsByID: [UUID: Int] = [:]

            for megaID in 0..<kMega {
                let memberIndices = megaAssignments.enumerated().filter { $0.element == megaID }.map { $0.offset }
                guard !memberIndices.isEmpty else { continue }
                let members = memberIndices.map { inputs[$0] }
                let centroid = megaCentroids[megaID]

                // Default assignment to mega-topic; overwritten by subtopic ids.
                for m in members { assignmentsByID[m.id] = megaID }

                // Subtopics within this mega-topic
                let desiredSubK = min(
                    max(level1Range.lowerBound, members.count / max(1, members.count / 12)),
                    min(level1Range.upperBound, members.count)
                )

                var localSubInfos: [GalaxyClusterInfo] = []
                var localSubIDs: [Int] = []

                if members.count >= 2 && desiredSubK > 1 {
                    let memberEmbeddings = members.map { $0.embedding }
                    let (subAssign, subCentroids) = KMeans.cluster(vectors: memberEmbeddings, k: desiredSubK, iterations: 20)
                    for subID in 0..<desiredSubK {
                        let localIdxs = subAssign.enumerated().filter { $0.element == subID }.map { $0.offset }
                        guard !localIdxs.isEmpty else { continue }
                        let subMembers = localIdxs.map { members[$0] }
                        let subCentroid = subCentroids[subID]

                        let subClusterID = megaID * 1000 + subID
                        localSubIDs.append(subClusterID)
                        for paper in subMembers { assignmentsByID[paper.id] = subClusterID }

                        let subInfo = GalaxyClusterInfo(
                            id: subClusterID,
                            name: "Subtopic \(megaID + 1).\(subID + 1)",
                            metaSummary: "Contains \(subMembers.count) papers.",
                            centroid: subCentroid,
                            memberPaperIDs: subMembers.map { $0.id },
                            layoutPosition: nil,
                            resolutionK: desiredSubK,
                            corpusVersion: corpusVersion,
                            subclusterIDs: nil
                        )
                        localSubInfos.append(subInfo)
                    }
                }

                let subPositions = ForceLayout.compute(for: localSubInfos.map { $0.centroid })
                for i in localSubInfos.indices {
                    localSubInfos[i].layoutPosition = subPositions.indices.contains(i) ? subPositions[i] : nil
                }
                subInfos.append(contentsOf: localSubInfos)

                let megaInfo = GalaxyClusterInfo(
                    id: megaID,
                    name: "Mega-topic \(megaID + 1)",
                    metaSummary: "Contains \(members.count) papers across \(localSubInfos.count) subtopics.",
                    centroid: centroid,
                    memberPaperIDs: members.map { $0.id },
                    layoutPosition: nil,
                    resolutionK: kMega,
                    corpusVersion: corpusVersion,
                    subclusterIDs: localSubIDs
                )
                megaInfos.append(megaInfo)
            }

            let megaPositions = ForceLayout.compute(for: megaInfos.map { $0.centroid })
            for i in megaInfos.indices {
                megaInfos[i].layoutPosition = megaPositions.indices.contains(i) ? megaPositions[i] : nil
            }

            return GalaxyComputeResult(megaClusters: megaInfos, subclusters: subInfos, assignmentsByID: assignmentsByID)
        }.value

        clusteringProgress = 0.55

        var updatedPapers = papers
        for idx in updatedPapers.indices {
            if let assign = compute.assignmentsByID[updatedPapers[idx].id] {
                updatedPapers[idx].clusterIndex = assign
            }
        }
        papers = updatedPapers

        let paperByID = Dictionary(uniqueKeysWithValues: updatedPapers.map { ($0.id, $0) })

        var preservedNames: [Int: (name: String, metaSummary: String)] = [:]
        for mega in megaClusters {
            preservedNames[mega.id] = (mega.name, mega.metaSummary)
            for sub in mega.subclusters ?? [] {
                preservedNames[sub.id] = (sub.name, sub.metaSummary)
            }
        }
        for sub in clusters {
            preservedNames[sub.id] = (sub.name, sub.metaSummary)
        }
        let preservedSources = clusterNameSources
        var newNameSources: [Int: ClusterNameSource] = [:]

        let subBuilt: [Cluster] = compute.subclusters.map { info in
            let heuristic = AppModel.heuristicClusterSummary(
                memberPaperIDs: info.memberPaperIDs,
                papersByID: paperByID,
                fallbackName: info.name,
                paperCount: info.memberPaperIDs.count
            )
            let source = preservedSources[info.id] ?? .heuristic
            let mergedName: String
            let mergedMeta: String
            if (source == .manual || source == .ai),
               let override = preservedNames[info.id] {
                mergedName = override.name
                mergedMeta = override.metaSummary
                newNameSources[info.id] = source
            } else {
                mergedName = heuristic.name
                mergedMeta = heuristic.metaSummary
                newNameSources[info.id] = .heuristic
            }
            return Cluster(
                id: info.id,
                name: mergedName,
                metaSummary: mergedMeta,
                centroid: info.centroid,
                memberPaperIDs: info.memberPaperIDs,
                layoutPosition: info.layoutPosition,
                resolutionK: info.resolutionK,
                corpusVersion: info.corpusVersion,
                subclusters: nil
            )
        }
        let subByID = Dictionary(uniqueKeysWithValues: subBuilt.map { ($0.id, $0) })

        let megaBuilt: [Cluster] = compute.megaClusters.map { info in
            let subs = (info.subclusterIDs ?? []).compactMap { subByID[$0] }
            let heuristic = AppModel.heuristicClusterSummary(
                memberPaperIDs: info.memberPaperIDs,
                papersByID: paperByID,
                fallbackName: info.name,
                paperCount: info.memberPaperIDs.count
            )
            let source = preservedSources[info.id] ?? .heuristic
            let mergedName: String
            let mergedMeta: String
            if (source == .manual || source == .ai),
               let override = preservedNames[info.id] {
                mergedName = override.name
                mergedMeta = override.metaSummary
                newNameSources[info.id] = source
            } else {
                mergedName = heuristic.name
                mergedMeta = heuristic.metaSummary
                newNameSources[info.id] = .heuristic
            }
            return Cluster(
                id: info.id,
                name: mergedName,
                metaSummary: mergedMeta,
                centroid: info.centroid,
                memberPaperIDs: info.memberPaperIDs,
                layoutPosition: info.layoutPosition,
                resolutionK: info.resolutionK,
                corpusVersion: info.corpusVersion,
                subclusters: subs
            )
        }

        megaClusters = megaBuilt
        clusters = subBuilt.sorted { $0.id < $1.id }
        clusterNameSources = newNameSources
        let validIDs = Set(newNameSources.keys)
        pinnedClusterIDs = pinnedClusterIDs.intersection(validIDs)

        clusteringProgress = 0.85
        persistGalaxySnapshot(version: corpusVersion, megaClusters: megaClusters)

        isClustering = false
        clusteringProgress = 1
    }

    // MARK: - Galaxy glossary helpers

    func toggleGalaxyPin(clusterID: Int) {
        if pinnedClusterIDs.contains(clusterID) {
            pinnedClusterIDs.remove(clusterID)
        } else {
            pinnedClusterIDs.insert(clusterID)
        }
        persistGalaxySnapshot(version: currentCorpusVersion(), megaClusters: megaClusters)
    }

    func isGalaxyPinned(clusterID: Int) -> Bool {
        pinnedClusterIDs.contains(clusterID)
    }

    func clusterInsights(for cluster: Cluster, papersByID: [UUID: Paper], keywordLimit: Int = 6, titleLimit: Int = 3) -> ClusterInsights {
        var keywordCounts: [String: Int] = [:]
        var memberPapers: [Paper] = []
        memberPapers.reserveCapacity(cluster.memberPaperIDs.count)

        for pid in cluster.memberPaperIDs {
            guard let paper = papersByID[pid] else { continue }
            memberPapers.append(paper)
            for keyword in paper.keywords ?? [] {
                let cleaned = keyword
                    .lowercased()
                    .replacingOccurrences(of: "_", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard cleaned.count >= 3 else { continue }
                keywordCounts[cleaned, default: 0] += 1
            }
        }

        let topKeywords = keywordCounts
            .sorted { (l, r) in l.value == r.value ? l.key < r.key : l.value > r.value }
            .prefix(keywordLimit)
            .map { $0.key }

        let sortedTitles = memberPapers
            .sorted {
                let ly = $0.year ?? -10_000
                let ry = $1.year ?? -10_000
                if ly != ry { return ly > ry }
                return $0.title < $1.title
            }
            .prefix(titleLimit)
            .map { $0.title }

        return ClusterInsights(topKeywords: topKeywords, topTitles: sortedTitles)
    }

    func nameSubtopicsWithAI(forMegaID megaID: Int, force: Bool = false) async {
        guard let idx = megaClusters.firstIndex(where: { $0.id == megaID }),
              let subs = megaClusters[idx].subclusters,
              !subs.isEmpty else { return }

        let paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        for sub in subs {
            if !force, clusterNameSources[sub.id] == .manual { continue }
            await nameGalaxyClusterWithAI(clusterID: sub.id, paperByID: paperByID, force: force)
        }

        persistGalaxySnapshot(version: currentCorpusVersion(), megaClusters: megaClusters)
    }

    func nameMegaTopicsWithAI(force: Bool = false) async {
        guard !megaClusters.isEmpty else { return }

        let paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        for mega in megaClusters {
            if !force, clusterNameSources[mega.id] == .manual { continue }
            await nameGalaxyClusterWithAI(clusterID: mega.id, paperByID: paperByID, force: force)
        }

        persistGalaxySnapshot(version: currentCorpusVersion(), megaClusters: megaClusters)
    }

    private func setGalaxyClusterFields(clusterID: Int, name: String, metaSummary: String?) {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { return }

        if let idx = clusters.firstIndex(where: { $0.id == clusterID }) {
            clusters[idx].name = trimmedName
            if let metaSummary { clusters[idx].metaSummary = metaSummary }
        }

        for megaIdx in megaClusters.indices {
            if megaClusters[megaIdx].id == clusterID {
                megaClusters[megaIdx].name = trimmedName
                if let metaSummary { megaClusters[megaIdx].metaSummary = metaSummary }
            }
            if megaClusters[megaIdx].subclusters != nil,
               let subIdx = megaClusters[megaIdx].subclusters?.firstIndex(where: { $0.id == clusterID }) {
                megaClusters[megaIdx].subclusters?[subIdx].name = trimmedName
                if let metaSummary { megaClusters[megaIdx].subclusters?[subIdx].metaSummary = metaSummary }
            }
        }
    }

    func renameGalaxyCluster(clusterID: Int, name: String, metaSummary: String?) {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { return }
        setGalaxyClusterFields(clusterID: clusterID, name: trimmedName, metaSummary: metaSummary)

        clusterNameSources[clusterID] = .manual
        persistGalaxySnapshot(version: currentCorpusVersion(), megaClusters: megaClusters)
    }

    func autoNameGalaxyCluster(clusterID: Int, force: Bool = false) async {
        let paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        await nameGalaxyClusterWithAI(clusterID: clusterID, paperByID: paperByID, force: force)
        persistGalaxySnapshot(version: currentCorpusVersion(), megaClusters: megaClusters)
    }

    private func nameGalaxyClusterWithAI(clusterID: Int, paperByID: [UUID: Paper], force: Bool) async {
        if !force, clusterNameSources[clusterID] == .ai {
            return
        }
        guard let cluster = galaxyCluster(for: clusterID) else { return }

        let memberPapers = cluster.memberPaperIDs.compactMap { paperByID[$0] }
        guard !memberPapers.isEmpty else { return }

        let entries: [String] = memberPapers
            .sorted {
                let ly = $0.year ?? -10_000
                let ry = $1.year ?? -10_000
                if ly != ry { return ly > ry }
                return $0.title < $1.title
            }
            .prefix(18)
            .map { paper in
                var parts: [String] = []
                parts.append("Title: \(paper.title)")
                if let year = paper.year { parts.append("Year: \(year)") }
                if let keywords = paper.keywords, !keywords.isEmpty {
                    let list = keywords.prefix(10).joined(separator: ", ")
                    parts.append("Keywords: \(list)")
                }
                var summary = paper.summary.trimmingCharacters(in: .whitespacesAndNewlines)
                if summary.count > 700 {
                    summary = String(summary.prefix(700)) + "…"
                }
                parts.append("Summary: \(summary)")

                if let takeaways = paper.takeaways, !takeaways.isEmpty {
                    let bullets = takeaways.prefix(3).map { "• \($0)" }.joined(separator: "\n")
                    parts.append("Takeaways:\n\(bullets)")
                }

                return parts.joined(separator: "\n")
            }

        do {
            let info = try await clusterSummarizer.summarizeCluster(index: clusterID, summaries: entries)
            setGalaxyClusterFields(clusterID: clusterID, name: info.name, metaSummary: info.metaSummary)
            clusterNameSources[clusterID] = .ai
        } catch {
            logger.error("Failed to name cluster \(clusterID): \(error.localizedDescription, privacy: .public)")
        }
    }

    private func galaxyCluster(for id: Int) -> Cluster? {
        if let mega = megaClusters.first(where: { $0.id == id }) {
            return mega
        }
        if let sub = clusters.first(where: { $0.id == id }) {
            return sub
        }
        for mega in megaClusters {
            if let sub = mega.subclusters?.first(where: { $0.id == id }) {
                return sub
            }
        }
        return nil
    }

    private static func heuristicClusterSummary(
        memberPaperIDs: [UUID],
        papersByID: [UUID: Paper],
        fallbackName: String,
        paperCount: Int
    ) -> ClusterSummary {
        var counts: [String: Int] = [:]
        var exampleTitles: [String] = []

        for pid in memberPaperIDs {
            guard let paper = papersByID[pid] else { continue }
            exampleTitles.append(paper.title)
            let keywords = paper.keywords ?? []
            for keyword in keywords {
                let cleaned = keyword
                    .lowercased()
                    .replacingOccurrences(of: "_", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard cleaned.count >= 3 else { continue }
                counts[cleaned, default: 0] += 1
            }
        }

        let top = counts
            .sorted { (l, r) in l.value == r.value ? l.key < r.key : l.value > r.value }
            .prefix(4)
            .map { $0.key }

        let name: String
        if top.isEmpty {
            name = fallbackName
        } else {
            let pieces = top.prefix(3).map { $0.capitalized }
            name = pieces.joined(separator: " · ")
        }

        let themes = top.prefix(3).joined(separator: ", ")
        let sample = exampleTitles.prefix(2).joined(separator: " · ")
        let meta: String
        if themes.isEmpty {
            meta = "Contains \(paperCount) papers."
        } else if sample.isEmpty {
            meta = "Key themes: \(themes). Contains \(paperCount) papers."
        } else {
            meta = "Key themes: \(themes). Sample: \(sample)."
        }

        return ClusterSummary(name: name, metaSummary: meta)
    }

    private func persistGalaxySnapshot(version: String, megaClusters: [Cluster]) {
        let folder = outputRoot.appendingPathComponent("clusters", isDirectory: true)
        let url = folder.appendingPathComponent("galaxy_\(version).json")
        let encoder = JSONEncoder()
        do {
            let data = try encoder.encode(
                GalaxySnapshot(
                    version: version,
                    megaClusters: megaClusters,
                    pinnedClusterIDs: Array(pinnedClusterIDs).sorted(),
                    clusterNameSources: clusterNameSources
                )
            )
            try data.write(to: url, options: .atomic)
        } catch {
            logger.error("Failed to persist galaxy snapshot: \(error.localizedDescription, privacy: .public)")
        }
    }

    func exportGalaxyArtifacts() -> (jsonURL: URL, reportURL: URL)? {
        let version = currentCorpusVersion()
        guard !megaClusters.isEmpty else { return nil }
        let folder = outputRoot.appendingPathComponent("clusters", isDirectory: true)

        let snapshot = GalaxySnapshot(
            version: version,
            megaClusters: megaClusters,
            pinnedClusterIDs: Array(pinnedClusterIDs).sorted(),
            clusterNameSources: clusterNameSources
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonURL = folder.appendingPathComponent("galaxy_\(version)_with_names.json")
        let reportURL = folder.appendingPathComponent("galaxy_\(version)_report.md")

        do {
            let data = try encoder.encode(snapshot)
            try data.write(to: jsonURL, options: .atomic)
        } catch {
            logger.error("Failed to export galaxy JSON: \(error.localizedDescription, privacy: .public)")
            return nil
        }

        do {
            let papersByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
            var lines: [String] = []
            lines.append("# LiteratureAtlas Galaxy Report")
            lines.append("")
            lines.append("- Corpus version: `\(version)`")
            lines.append("- Exported: \(ISO8601DateFormatter().string(from: Date()))")
            lines.append("- Mega-topics: \(megaClusters.count)")
            lines.append("- Papers: \(papers.count)")
            lines.append("")

            let megaSorted = megaClusters.sorted { l, r in
                let lp = pinnedClusterIDs.contains(l.id)
                let rp = pinnedClusterIDs.contains(r.id)
                if lp != rp { return lp && !rp }
                return l.id < r.id
            }

            for mega in megaSorted {
                let megaPinned = pinnedClusterIDs.contains(mega.id) ? "yes" : "no"
                let megaSource = clusterNameSources[mega.id]?.label ?? "Unknown"
                let insights = clusterInsights(for: mega, papersByID: papersByID)

                lines.append("## \(mega.name)")
                lines.append("")
                lines.append("- ID: \(mega.id)")
                lines.append("- Papers: \(mega.memberPaperIDs.count)")
                lines.append("- Pinned: \(megaPinned)")
                lines.append("- Name source: \(megaSource)")
                if !insights.topKeywords.isEmpty {
                    lines.append("- Top keywords: \(insights.topKeywords.map { "`\($0)`" }.joined(separator: ", "))")
                }
                if !insights.topTitles.isEmpty {
                    lines.append("- Top papers:")
                    for title in insights.topTitles {
                        lines.append("  - \(title)")
                    }
                }
                lines.append("")

                if let subs = mega.subclusters, !subs.isEmpty {
                    let subSorted = subs.sorted { l, r in
                        let lp = pinnedClusterIDs.contains(l.id)
                        let rp = pinnedClusterIDs.contains(r.id)
                        if lp != rp { return lp && !rp }
                        return l.id < r.id
                    }
                    for sub in subSorted {
                        let subPinned = pinnedClusterIDs.contains(sub.id) ? "yes" : "no"
                        let subSource = clusterNameSources[sub.id]?.label ?? "Unknown"
                        let subInsights = clusterInsights(for: sub, papersByID: papersByID, keywordLimit: 5, titleLimit: 3)
                        lines.append("### \(sub.name)")
                        lines.append("")
                        lines.append("- ID: \(sub.id)")
                        lines.append("- Papers: \(sub.memberPaperIDs.count)")
                        lines.append("- Pinned: \(subPinned)")
                        lines.append("- Name source: \(subSource)")
                        if !subInsights.topKeywords.isEmpty {
                            lines.append("- Top keywords: \(subInsights.topKeywords.map { "`\($0)`" }.joined(separator: ", "))")
                        }
                        if !subInsights.topTitles.isEmpty {
                            lines.append("- Top papers:")
                            for title in subInsights.topTitles {
                                lines.append("  - \(title)")
                            }
                        }
                        lines.append("")
                    }
                }
            }

            let report = lines.joined(separator: "\n")
            guard let data = report.data(using: .utf8) else {
                logger.error("Failed to encode galaxy report as UTF-8.")
                return (jsonURL: jsonURL, reportURL: reportURL)
            }
            try data.write(to: reportURL, options: .atomic)
        } catch {
            logger.error("Failed to export galaxy report: \(error.localizedDescription, privacy: .public)")
            return (jsonURL: jsonURL, reportURL: reportURL)
        }

        ingestionLog += "\nExported galaxy report: \(reportURL.lastPathComponent)"
        return (jsonURL: jsonURL, reportURL: reportURL)
    }

    // MARK: - Corpus synthesis (briefing)

    private func corpusBriefingURL(root: URL, version: String) -> URL {
        root.appendingPathComponent("reports", isDirectory: true)
            .appendingPathComponent("corpus_briefing_\(version).md")
    }

    private func loadSavedCorpusBriefingIfPresent() async {
        let version = currentCorpusVersion()
        let fm = FileManager.default
        for root in dataRoots {
            let url = corpusBriefingURL(root: root, version: version)
            guard fm.fileExists(atPath: url.path) else { continue }
            guard let data = try? Data(contentsOf: url),
                  let text = String(data: data, encoding: .utf8),
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            corpusBriefing = text
            corpusBriefingError = nil
            return
        }
    }

    func generateCorpusBriefing(force: Bool = false) {
        Task { await runCorpusBriefing(force: force) }
    }

    private func runCorpusBriefing(force: Bool) async {
        guard !isGeneratingBriefing else { return }
        isGeneratingBriefing = true
        corpusBriefingError = nil
        defer { isGeneratingBriefing = false }

        guard !papers.isEmpty else {
            corpusBriefingError = "Ingest papers first."
            return
        }
        guard !megaClusters.isEmpty else {
            corpusBriefingError = "Run clustering first to generate a topic hierarchy."
            return
        }

        let version = currentCorpusVersion()
        let url = corpusBriefingURL(root: outputRoot, version: version)

        if !force,
           FileManager.default.fileExists(atPath: url.path),
           let data = try? Data(contentsOf: url),
           let cached = String(data: data, encoding: .utf8),
           !cached.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            corpusBriefing = cached
            return
        }

        let context = buildCorpusBriefingContext(version: version)
        do {
            let output = try await corpusBriefingActor.brief(context: context)
            corpusBriefing = output
            guard let data = output.data(using: .utf8) else {
                corpusBriefingError = "Failed to encode briefing as UTF-8."
                return
            }
            try data.write(to: url, options: .atomic)
            ingestionLog += "\n[briefing] Saved corpus briefing: \(url.lastPathComponent)"
        } catch {
            corpusBriefingError = "Failed to generate briefing: \(error.localizedDescription)"
        }
    }

    private func buildCorpusBriefingContext(version: String) -> String {
        let papersByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        var clusterNameByID: [Int: String] = [:]
        for mega in megaClusters {
            clusterNameByID[mega.id] = mega.name
            for sub in mega.subclusters ?? [] {
                clusterNameByID[sub.id] = sub.name
            }
        }

        let years = papers.compactMap { $0.year }
        let yearRange: String = {
            guard let minY = years.min(), let maxY = years.max() else { return "unknown" }
            return "\(minY)–\(maxY)"
        }()

        let done = papers.filter { $0.readingStatus == .done }.count
        let inProgress = papers.filter { $0.readingStatus == .inProgress }.count
        let unread = papers.count - done - inProgress

        var globalKeywordCounts: [String: Int] = [:]
        for paper in papers {
            for keyword in paper.keywords ?? [] {
                let cleaned = keyword
                    .lowercased()
                    .replacingOccurrences(of: "_", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard cleaned.count >= 3 else { continue }
                globalKeywordCounts[cleaned, default: 0] += 1
            }
        }
        let globalTopKeywords = globalKeywordCounts
            .sorted { (l, r) in l.value == r.value ? l.key < r.key : l.value > r.value }
            .prefix(12)
            .map { $0.key }

        var lines: [String] = []
        lines.append("Corpus version: \(version)")
        lines.append("Papers: \(papers.count)")
        lines.append("Years: \(yearRange)")
        lines.append("Reading status: done \(done), in progress \(inProgress), unread \(unread)")
        if !globalTopKeywords.isEmpty {
            lines.append("Global keywords: \(globalTopKeywords.joined(separator: ", "))")
        }

        if let summary = analyticsSummary {
            lines.append("Analytics generated: \(ISO8601DateFormatter().string(from: summary.generatedAt))")
            if let conf = summary.answerConfidence {
                lines.append(String(format: "Analytics answer confidence: %.3f", conf))
            }
            if !summary.factorLabels.isEmpty {
                lines.append("Factor labels: \(summary.factorLabels.prefix(6).joined(separator: ", "))")
            }
            let recTitles = summary.recommendations
                .prefix(6)
                .compactMap { papersByID[$0]?.title }
            if !recTitles.isEmpty {
                lines.append("Recommended reading (analytics): \(recTitles.joined(separator: " · "))")
            }
            let volatile = summary.driftVolatility
                .sorted { $0.volatility > $1.volatility }
                .prefix(5)
                .map { entry in
                    let name = clusterNameByID[entry.clusterID] ?? "Cluster \(entry.clusterID)"
                    return "\(name) (vol \(String(format: "%.2f", entry.volatility)))"
                }
            if !volatile.isEmpty {
                lines.append("Most volatile topics: \(volatile.joined(separator: ", "))")
            }
            let outliers = summary.novelty
                .sorted { $0.novelty > $1.novelty }
                .prefix(5)
                .compactMap { n -> String? in
                    guard let title = papersByID[n.paperID]?.title else { return nil }
                    let c = n.clusterID.flatMap { clusterNameByID[$0] } ?? "n/a"
                    return "\(title) (nov \(String(format: "%.2f", n.novelty)), \(c))"
                }
            if !outliers.isEmpty {
                lines.append("Top novelty outliers: \(outliers.joined(separator: " · "))")
            }
        }
        lines.append("")

        let megaSorted = megaClusters.sorted { l, r in
            let lp = pinnedClusterIDs.contains(l.id)
            let rp = pinnedClusterIDs.contains(r.id)
            if lp != rp { return lp && !rp }
            if l.memberPaperIDs.count != r.memberPaperIDs.count { return l.memberPaperIDs.count > r.memberPaperIDs.count }
            return l.id < r.id
        }

        let megaLimit = megaSorted.prefix(12)
        for mega in megaLimit {
            let pinned = pinnedClusterIDs.contains(mega.id) ? "yes" : "no"
            let source = clusterNameSources[mega.id]?.label ?? "Unknown"
            let insights = clusterInsights(for: mega, papersByID: papersByID, keywordLimit: 6, titleLimit: 3)

            lines.append("Mega-topic: \(mega.name)")
            lines.append("- id: \(mega.id)")
            lines.append("- papers: \(mega.memberPaperIDs.count)")
            lines.append("- pinned: \(pinned)")
            lines.append("- name source: \(source)")
            if !insights.topKeywords.isEmpty {
                lines.append("- keywords: \(insights.topKeywords.joined(separator: ", "))")
            }
            if !insights.topTitles.isEmpty {
                lines.append("- sample titles: \(insights.topTitles.joined(separator: " · "))")
            }
            lines.append("- meta: \(mega.metaSummary)")

            if let subs = mega.subclusters, !subs.isEmpty {
                let subSorted = subs.sorted { l, r in
                    let lp = pinnedClusterIDs.contains(l.id)
                    let rp = pinnedClusterIDs.contains(r.id)
                    if lp != rp { return lp && !rp }
                    if l.memberPaperIDs.count != r.memberPaperIDs.count { return l.memberPaperIDs.count > r.memberPaperIDs.count }
                    return l.id < r.id
                }
                let subLimit = subSorted.prefix(8)
                for sub in subLimit {
                    let sPinned = pinnedClusterIDs.contains(sub.id) ? "yes" : "no"
                    let sSource = clusterNameSources[sub.id]?.label ?? "Unknown"
                    let sInsights = clusterInsights(for: sub, papersByID: papersByID, keywordLimit: 5, titleLimit: 2)
                    lines.append("  Subtopic: \(sub.name)")
                    lines.append("  - id: \(sub.id)")
                    lines.append("  - papers: \(sub.memberPaperIDs.count)")
                    lines.append("  - pinned: \(sPinned)")
                    lines.append("  - name source: \(sSource)")
                    if !sInsights.topKeywords.isEmpty {
                        lines.append("  - keywords: \(sInsights.topKeywords.joined(separator: ", "))")
                    }
                    if !sInsights.topTitles.isEmpty {
                        lines.append("  - sample titles: \(sInsights.topTitles.joined(separator: " · "))")
                    }
                    lines.append("  - meta: \(sub.metaSummary)")
                }
                if subs.count > 8 {
                    lines.append("  (additional subtopics omitted: \(subs.count - 8))")
                }
            }

            lines.append("")
        }

        if megaClusters.count > megaLimit.count {
            lines.append("(additional mega-topics omitted: \(megaClusters.count - megaLimit.count))")
        }

        return lines.joined(separator: "\n")
    }

    private func topicDossierURL(root: URL, version: String, clusterID: Int) -> URL {
        root.appendingPathComponent("reports", isDirectory: true)
            .appendingPathComponent("topic_\(clusterID)_dossier_\(version).md")
    }

    func loadOrGenerateTopicDossier(clusterID: Int, force: Bool = false) async -> String? {
        guard !papers.isEmpty else { return nil }
        guard let cluster = galaxyCluster(for: clusterID) else { return nil }

        let version = currentCorpusVersion()
        let url = topicDossierURL(root: outputRoot, version: version, clusterID: clusterID)
        let fm = FileManager.default
        if !force,
           fm.fileExists(atPath: url.path),
           let data = try? Data(contentsOf: url),
           let cached = String(data: data, encoding: .utf8),
           !cached.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return cached
        }

        let papersByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        let memberPapers = cluster.memberPaperIDs.compactMap { papersByID[$0] }

        let insights = clusterInsights(for: cluster, papersByID: papersByID, keywordLimit: 10, titleLimit: 6)

        // Pick a compact, representative set of papers.
        var selected: [Paper] = []
        let recent = memberPapers.sorted {
            let ly = $0.year ?? -10_000
            let ry = $1.year ?? -10_000
            if ly != ry { return ly > ry }
            return $0.title < $1.title
        }.prefix(6)
        selected.append(contentsOf: recent)

        let oldest = memberPapers.sorted {
            let ly = $0.year ?? 10_000
            let ry = $1.year ?? 10_000
            if ly != ry { return ly < ry }
            return $0.title < $1.title
        }.prefix(2)
        selected.append(contentsOf: oldest)

        let important = memberPapers.filter { paper in
            if paper.isImportant == true { return true }
            let tags = paper.userTags ?? []
            return tags.contains(where: { ["important", "starred", "fav", "favorite"].contains($0.lowercased()) })
        }.prefix(4)
        selected.append(contentsOf: important)

        var unique: [Paper] = []
        var seen = Set<UUID>()
        for paper in selected where !seen.contains(paper.id) {
            seen.insert(paper.id)
            unique.append(paper)
        }
        unique = Array(unique.prefix(12))

        var lines: [String] = []
        lines.append("Topic: \(cluster.name)")
        lines.append("Cluster ID: \(cluster.id)")
        lines.append("Papers: \(cluster.memberPaperIDs.count)")
        lines.append("Meta: \(cluster.metaSummary)")
        if !insights.topKeywords.isEmpty {
            lines.append("Keywords: \(insights.topKeywords.joined(separator: ", "))")
        }

        if let subs = cluster.subclusters, !subs.isEmpty {
            lines.append("")
            lines.append("Subtopics:")
            let subSorted = subs.sorted { $0.memberPaperIDs.count > $1.memberPaperIDs.count }.prefix(12)
            for sub in subSorted {
                let sInsights = clusterInsights(for: sub, papersByID: papersByID, keywordLimit: 6, titleLimit: 2)
                let kw = sInsights.topKeywords.prefix(4).joined(separator: ", ")
                let suffix = kw.isEmpty ? "" : " — \(kw)"
                lines.append("- \(sub.name) (\(sub.memberPaperIDs.count) papers)\(suffix)")
            }
            if subs.count > 12 {
                lines.append("(additional subtopics omitted: \(subs.count - 12))")
            }
        }

        if let summary = analyticsSummary, !summary.paperMetrics.isEmpty {
            let metricsByID = Dictionary(uniqueKeysWithValues: summary.paperMetrics.map { ($0.paperID, $0) })
            var zNovelty: [Double] = []
            var zConsensus: [Double] = []
            var influence: [Double] = []
            for pid in cluster.memberPaperIDs {
                guard let m = metricsByID[pid] else { continue }
                zNovelty.append(m.zNovelty)
                zConsensus.append(m.zConsensus)
                influence.append(m.influenceAbs)
            }
            func avg(_ xs: [Double]) -> Double? {
                guard !xs.isEmpty else { return nil }
                return xs.reduce(0, +) / Double(xs.count)
            }
            if let aN = avg(zNovelty), let aC = avg(zConsensus), let aI = avg(influence) {
                lines.append("")
                lines.append(String(format: "Analytics (avg over member papers): zNovelty %.2f, zConsensus %.2f, influenceAbs %.2f", aN, aC, aI))
            }
        }

        if !unique.isEmpty {
            lines.append("")
            lines.append("Representative paper records:")
            for (idx, paper) in unique.enumerated() {
                lines.append("")
                lines.append("Paper \(idx + 1)")
                lines.append("Title: \(paper.title)")
                if let year = paper.year { lines.append("Year: \(year)") }
                if let keywords = paper.keywords, !keywords.isEmpty {
                    lines.append("Keywords: \(keywords.prefix(10).joined(separator: ", "))")
                }
                var summary = paper.summary.trimmingCharacters(in: .whitespacesAndNewlines)
                if summary.count > 900 {
                    summary = String(summary.prefix(900)) + "…"
                }
                lines.append("Summary: \(summary)")
                if let takeaways = paper.takeaways, !takeaways.isEmpty {
                    let bullets = takeaways.prefix(3).map { "• \($0)" }.joined(separator: "\n")
                    lines.append("Takeaways:\n\(bullets)")
                }
            }
        }

        let context = lines.joined(separator: "\n")

        do {
            let output = try await topicDossierActor.dossier(topicName: cluster.name, context: context)
            guard let data = output.data(using: .utf8) else { return output }
            try data.write(to: url, options: .atomic)
            ingestionLog += "\n[dossier] Saved topic dossier: \(url.lastPathComponent)"
            return output
        } catch {
            return "Failed to generate dossier: \(error.localizedDescription)"
        }
    }

    private func loadSavedGalaxyIfPresent() async {
        let version = currentCorpusVersion()
        let decoder = JSONDecoder()
        let fm = FileManager.default

        for root in dataRoots {
            let folder = root.appendingPathComponent("clusters", isDirectory: true)
            let url = folder.appendingPathComponent("galaxy_\(version).json")
            guard fm.fileExists(atPath: url.path) else { continue }
            guard let data = try? Data(contentsOf: url) else { continue }

            if let snapshot = try? decoder.decode(GalaxySnapshot.self, from: data) {
                applyGalaxySnapshot(snapshot)
                ingestionLog += "\nLoaded cached galaxy for current corpus."
                return
            }

            // Legacy format: mega clusters directly.
            if let legacyMega = try? decoder.decode([Cluster].self, from: data) {
                applyGalaxySnapshot(GalaxySnapshot(version: version, megaClusters: legacyMega))
                ingestionLog += "\nLoaded cached galaxy (legacy) for current corpus."
                return
            }
        }
    }

    private func applyGalaxySnapshot(_ snapshot: GalaxySnapshot) {
        megaClusters = snapshot.megaClusters
        clusters = snapshot.megaClusters.flatMap { $0.subclusters ?? [] }.sorted { $0.id < $1.id }
        pinnedClusterIDs = Set(snapshot.pinnedClusterIDs ?? [])
        clusterNameSources = snapshot.clusterNameSources ?? [:]
        var validIDs = Set<Int>()
        for mega in snapshot.megaClusters {
            validIDs.insert(mega.id)
            for sub in mega.subclusters ?? [] {
                validIDs.insert(sub.id)
            }
        }
        pinnedClusterIDs = pinnedClusterIDs.intersection(validIDs)
        clusterNameSources = clusterNameSources.filter { validIDs.contains($0.key) }

        var paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        for sub in clusters {
            for pid in sub.memberPaperIDs {
                if var p = paperByID[pid] {
                    p.clusterIndex = sub.id
                    paperByID[pid] = p
                }
            }
        }
        papers = Array(paperByID.values)
    }

    private func applyForceLayout(to clusters: [Cluster]) -> [Cluster] {
        guard clusters.count > 1 else { return clusters }
        let points = ForceLayout.compute(for: clusters.map { $0.centroid })
        var updated: [Cluster] = []
        for (idx, var cluster) in clusters.enumerated() {
            cluster.layoutPosition = Point2D(x: points[idx].x, y: points[idx].y)
            updated.append(cluster)
        }
        return updated
    }

    func lensAdjustedClusters(_ clusters: [Cluster], lens: MapLens) -> [Cluster] {
        guard !clusters.isEmpty else { return clusters }
        switch lens {
        case .standard:
            if clusters.allSatisfy({ $0.layoutPosition != nil }) {
                return clusters
            }
            return applyForceLayout(to: clusters)
        case .time:
            let years = clusters.compactMap { averageYear(for: $0) }
            let minYear = years.min() ?? 0
            let maxYear = years.max() ?? 0
            let range = max(1, maxYear - minYear)
            return clusters.enumerated().map { idx, cluster in
                var c = cluster
                let avgYear = averageYear(for: cluster) ?? Double(minYear)
                let radius = 0.25 + 0.55 * (1 - ((avgYear - Double(minYear)) / Double(range)))
                let angle = 2 * Double.pi * Double(idx) / Double(max(1, clusters.count))
                let x = 0.5 + radius * cos(angle) * 0.9
                let y = 0.5 + radius * sin(angle) * 0.9
                c.layoutPosition = Point2D(x: x, y: y)
                return c
            }
        case .method:
            let vectors = clusters.map { methodVector(for: $0) }
            let points = ForceLayout.compute(for: vectors)
            return clusters.enumerated().map { pair in
                var c = pair.element
                let idx = pair.offset
                c.layoutPosition = Point2D(x: points[idx].x, y: points[idx].y)
                return c
            }
        case .dataRegime:
            let vectors = clusters.map { keywordVector(for: $0) }
            let points = ForceLayout.compute(for: vectors)
            return clusters.enumerated().map { pair in
                var c = pair.element
                let idx = pair.offset
                c.layoutPosition = Point2D(x: points[idx].x, y: points[idx].y)
                return c
            }
        case .interest:
            guard let profile = readingProfileVector, !profile.isEmpty else {
                return applyForceLayout(to: clusters)
            }
            guard clusters.allSatisfy({ $0.centroid.count == profile.count }) else {
                return applyForceLayout(to: clusters)
            }
            return clusters.enumerated().map { idx, cluster in
                var c = cluster
                let sim = cosineSimilarity(cluster.centroid, profile)
                let bounded = sim.isNaN ? 0 : max(-1, min(1, sim))
                let norm = (bounded + 1) / 2 // map [-1,1] to [0,1]
                let radius = 0.25 + 0.6 * (1 - Double(norm))
                let angle = 2 * Double.pi * Double(idx) / Double(max(1, clusters.count))
                let x = 0.5 + radius * cos(angle)
                let y = 0.5 + radius * sin(angle)
                c.layoutPosition = Point2D(x: x, y: y)
                return c
            }
        }
    }

    private func averageYear(for cluster: Cluster) -> Double? {
        let years = cluster.memberPaperIDs.compactMap { id in
            papers.first(where: { $0.id == id })?.year
        }
        guard !years.isEmpty else { return nil }
        let total = years.reduce(0, +)
        return Double(total) / Double(years.count)
    }

    private func methodVector(for cluster: Cluster) -> [Float] {
        let text = cluster.memberPaperIDs.compactMap { id in
            papers.first(where: { $0.id == id })?.methodSummary
        }.joined(separator: "\n")
        let vec = normalizeEmbedding(fallbackEmbedding(for: text.isEmpty ? cluster.metaSummary : text))
        return vec.isEmpty ? cluster.centroid : vec
    }

    private func keywordVector(for cluster: Cluster) -> [Float] {
        let text = cluster.memberPaperIDs.compactMap { id in
            papers.first(where: { $0.id == id })?.keywords?.joined(separator: " ")
        }.joined(separator: " ")
        let vec = normalizeEmbedding(fallbackEmbedding(for: text.isEmpty ? cluster.name : text))
        return vec.isEmpty ? cluster.centroid : vec
    }

    private func persistClusterSnapshot(_ clusters: [Cluster], key: ClusterCacheKey) {
        let folder = outputRoot.appendingPathComponent("clusters", isDirectory: true)
        let url = folder.appendingPathComponent("clusters_\(key.version)_k\(key.k).json")
        let encoder = JSONEncoder()
        do {
            let data = try encoder.encode(ClusterSnapshot(version: key.version, k: key.k, clusters: clusters))
            try data.write(to: url, options: .atomic)
            ingestionLog += "\nSaved cluster snapshot: k=\(key.k)."
        } catch {
            logger.error("Failed to persist cluster snapshot k=\(key.k): \(error.localizedDescription, privacy: .public)")
        }
    }

    private func currentCorpusVersion() -> String {
        let signature = papers.sorted { $0.filePath < $1.filePath }
            .map { "\($0.filePath)|v\($0.version)" }
            .joined(separator: "||")
        var hash: UInt64 = 5381
        for byte in signature.utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt64(byte)
        }
        return String(hash)
    }

    func loadSubclusters(for clusterID: Int, desiredK: Int = 4) async -> [Cluster] {
        if let cached = subclusterCache[clusterID] {
            return cached
        }
        guard let parent = clusters.first(where: { $0.id == clusterID }) else { return [] }
        let members = papers.filter { parent.memberPaperIDs.contains($0.id) && !$0.embedding.isEmpty }
        guard members.count >= 2 else { return [] }

        let embeddings = members.map { $0.embedding }
        let kClamped = min(max(2, desiredK), embeddings.count)
        let (assignments, centroids) = KMeans.cluster(vectors: embeddings, k: kClamped, iterations: 20)

        var newClusters: [Cluster] = []
        for cid in 0..<kClamped {
            let idxs = assignments.enumerated().filter { $0.element == cid }.map { $0.offset }
            guard !idxs.isEmpty else { continue }
            let subMembers = idxs.map { members[$0] }
            let summaries = subMembers.map { $0.summary }
            let centroid = centroids[cid]

            do {
                let info = try await clusterSummarizer.summarizeCluster(index: cid, summaries: summaries)
                let c = Cluster(
                    id: clusterID * 1000 + cid,
                    name: info.name,
                    metaSummary: info.metaSummary,
                    centroid: centroid,
                    memberPaperIDs: subMembers.map { $0.id },
                    layoutPosition: nil,
                    resolutionK: kClamped,
                    corpusVersion: parent.corpusVersion,
                    subclusters: nil
                )
                newClusters.append(c)
            } catch {
                let c = Cluster(
                    id: clusterID * 1000 + cid,
                    name: "Subcluster \(cid + 1)",
                    metaSummary: "Contains \(subMembers.count) papers.",
                    centroid: centroid,
                    memberPaperIDs: subMembers.map { $0.id },
                    layoutPosition: nil,
                    resolutionK: kClamped,
                    corpusVersion: parent.corpusVersion,
                    subclusters: nil
                )
                newClusters.append(c)
            }
        }

        let positioned = applyForceLayout(to: newClusters)
        subclusterCache[clusterID] = positioned
        return positioned
    }

    func bridgingPapers(between firstID: Int, and secondID: Int, maxResults: Int = 10) -> [BridgingResult] {
        guard let firstCluster = clusters.first(where: { $0.id == firstID }),
              let secondCluster = clusters.first(where: { $0.id == secondID }) else { return [] }

        var results: [BridgingResult] = []

        for paper in papers {
            guard paper.embedding.count == firstCluster.centroid.count,
                  paper.embedding.count == secondCluster.centroid.count else { continue }
            let s1 = cosineSimilarity(paper.embedding, firstCluster.centroid)
            let s2 = cosineSimilarity(paper.embedding, secondCluster.centroid)
            let combined = min(s1, s2)
            if combined.isNaN { continue }
            results.append(BridgingResult(paper: paper, combinedScore: combined, scoreToFirst: s1, scoreToSecond: s2))
        }

        return Array(results.sorted { $0.combinedScore > $1.combinedScore }.prefix(maxResults))
    }

    // MARK: - Claim graph & pipelines

    /// Build claim-level edges across all papers currently loaded.
    func claimGraphEdges() -> [ClaimEdge] {
        let allClaims = papers.flatMap { $0.claims ?? [] }
        return ClaimRelationInferencer.inferEdges(for: allClaims)
    }

    private struct ClaimEdgeSnapshot: Codable {
        struct Edge: Codable {
            let srcPaperID: UUID
            let dstPaperID: UUID
            let sourceClaimID: UUID
            let targetClaimID: UUID
            let kind: ClaimRelationKind
            let rationale: String?
        }

        let generatedAt: Date
        let edges: [Edge]
    }

    private func persistClaimGraphEdgesSnapshot() {
        let edges = claimGraphEdges()
        guard !edges.isEmpty else { return }

        var paperForClaim: [UUID: UUID] = [:]
        for paper in papers {
            for claim in paper.claims ?? [] {
                paperForClaim[claim.id] = paper.id
            }
        }

        let exportEdges: [ClaimEdgeSnapshot.Edge] = edges.compactMap { edge in
            guard let srcPaper = paperForClaim[edge.sourceClaimID],
                  let dstPaper = paperForClaim[edge.targetClaimID] else { return nil }
            return ClaimEdgeSnapshot.Edge(
                srcPaperID: srcPaper,
                dstPaperID: dstPaper,
                sourceClaimID: edge.sourceClaimID,
                targetClaimID: edge.targetClaimID,
                kind: edge.kind,
                rationale: edge.rationale
            )
        }
        guard !exportEdges.isEmpty else { return }

        let snapshot = ClaimEdgeSnapshot(generatedAt: Date(), edges: exportEdges)
        let url = outputRoot
            .appendingPathComponent("analytics", isDirectory: true)
            .appendingPathComponent("claim_edges.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try encoder.encode(snapshot)
            try data.write(to: url, options: .atomic)
        } catch {
            ingestionLog += "\n[analytics] Failed to persist claim_edges.json: \(error.localizedDescription)"
        }
    }

    private func suggestedStressTests(for assumption: String) -> [String] {
        let a = assumption.lowercased()
        if a.contains("stationary") || a.contains("iid") {
            return [
                "Test under distribution shift / non-stationary regimes.",
                "Add temporal correlation / regime-switching ablations.",
                "Report performance stratified by time period / regime."
            ]
        }
        if a.contains("gaussian") || a.contains("normal") {
            return [
                "Stress-test heavy tails / outliers (robustness).",
                "Swap Gaussian noise for Laplace / t-distribution and compare.",
                "Report sensitivity to rare-event perturbations."
            ]
        }
        if a.contains("risk-neutral") || a.contains("risk neutral") {
            return [
                "Run sensitivity to risk aversion / utility choice.",
                "Compare policies under downside-risk constraints.",
                "Report trade-offs between mean return and tail risk."
            ]
        }
        if a.contains("liquidity") {
            return [
                "Simulate finite liquidity / slippage and re-evaluate.",
                "Ablate execution latency and market impact assumptions.",
                "Report robustness across liquidity regimes."
            ]
        }
        return [
            "Vary the assumption and measure sensitivity.",
            "Find counterexamples / failure cases where it breaks.",
            "Add an ablation removing dependence on the assumption."
        ]
    }

    /// Stress-test a given assumption across the corpus.
    func assumptionStressReport(for assumption: String) -> AssumptionStressReport {
        let report = AssumptionStressTester.stressTest(assumption: assumption, papers: papers)
        let affectedPaperIDs = Set(report.affectedClaims.map(\.paperID))
        var clusterCounts: [String: Int] = [:]
        for pid in affectedPaperIDs {
            if let cid = papers.first(where: { $0.id == pid })?.clusterIndex {
                clusterCounts[String(cid), default: 0] += 1
            }
        }
        appendUserEvent(type: "assumption_stress_test", paperID: nil, extra: [
            "assumption": assumption,
            "affected_claims": report.affectedClaims.count,
            "affected_papers": affectedPaperIDs.count,
            "clusters": clusterCounts,
            "suggested_tests": suggestedStressTests(for: assumption)
        ])
        return report
    }

    /// Generate pseudocode blueprint for a paper's stored method pipeline.
    func methodBlueprint(for paperID: UUID) -> String? {
        guard let paper = papers.first(where: { $0.id == paperID }),
              let pipeline = paper.methodPipeline else { return nil }
        return BlueprintGenerator.generate(for: pipeline, title: paper.title)
    }

    // MARK: - User data & planner

    func updatePaperUserData(id: UUID, notes: String, tags: [String], status: ReadingStatus?) {
        Task { await embedNotesAndPersist(id: id, notes: notes, tags: tags, status: status) }
    }

    private func embedNotesAndPersist(id: UUID, notes: String, tags: [String], status: ReadingStatus?) async {
        guard let idx = papers.firstIndex(where: { $0.id == id }) else { return }
        papers[idx].userNotes = notes.isEmpty ? nil : notes
        papers[idx].userTags = tags.isEmpty ? nil : tags
        papers[idx].readingStatus = status
        let normalizedTags = tags.map { $0.lowercased() }
        papers[idx].isImportant = normalizedTags.contains(where: { $0 == "important" || $0 == "starred" || $0 == "fav" || $0 == "favorite" })
        if status == .done && papers[idx].firstReadAt == nil {
            papers[idx].firstReadAt = Date()
        }
        if papers[idx].ingestedAt == nil {
            papers[idx].ingestedAt = Date()
        }

        if let content = papers[idx].userNotes, !content.isEmpty {
            var emb = normalizeEmbedding(await embedder.encode(for: content) ?? [])
            if emb.isEmpty { emb = normalizeEmbedding(fallbackEmbedding(for: content)) }
            papers[idx].noteEmbedding = emb
        } else {
            papers[idx].noteEmbedding = nil
        }

        _ = try? savePaperJSON(papers[idx])
        _ = try? PaperMarkdownExporter.write(paper: papers[idx], outputRoot: outputRoot)
        savePaperIndex()
        await MainActor.run {
            recomputeReadingProfile(extra: nil)
        }
        appendUserEvent(type: "note_saved", paperID: id, extra: ["tags": tags])
    }

    func generateFlashcards(for paperID: UUID) {
        Task { await runFlashcards(for: paperID) }
    }

    private func runFlashcards(for paperID: UUID) async {
        guard let idx = papers.firstIndex(where: { $0.id == paperID }) else { return }
        let paper = papers[idx]
        let fallbackInstructions = "You write compact study flashcards."
        let instructions = PromptStore.loadText("ui.flashcards.instructions.md", fallback: fallbackInstructions)

        let fallbackTemplate = """
        Create 4 short flashcards (Q/A) to study the paper "{{title}}".
        Base them on the summary and takeaways below. Keep answers <=30 words.

        Summary:
        {{summary}}

        Takeaways:
        {{takeaways}}
        """
        let template = PromptStore.loadText("ui.flashcards.prompt.md", fallback: fallbackTemplate)
        let prompt = PromptStore.render(template: template, variables: [
            "title": paper.title,
            "summary": paper.summary,
            "takeaways": paper.takeaways?.joined(separator: "\n") ?? ""
        ])
        do {
            let session = LanguageModelSession(instructions: instructions)
            let response = try await session.respond(to: prompt)
            let lines = response.content.split(whereSeparator: { $0.isNewline }).map { String($0) }
            var cards: [Flashcard] = []
            var currentQ: String?
            for line in lines {
                if line.lowercased().hasPrefix("q") {
                    currentQ = line.replacingOccurrences(of: "Q:", with: "").trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                } else if line.lowercased().hasPrefix("a"), let q = currentQ {
                    let ans = line.replacingOccurrences(of: "A:", with: "").trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                    cards.append(Flashcard(id: UUID(), question: q, answer: ans))
                    currentQ = nil
                }
            }
            if cards.isEmpty {
                // Fallback: pair consecutive lines
                for chunk in stride(from: 0, to: lines.count, by: 2) {
                    if chunk + 1 < lines.count {
                        cards.append(Flashcard(id: UUID(), question: lines[chunk], answer: lines[chunk + 1]))
                    }
                }
            }
            papers[idx].flashcards = cards.isEmpty ? nil : cards
            if var fcs = papers[idx].flashcards {
                for i in fcs.indices {
                    fcs[i].lastReviewedAt = nil
                    fcs[i].reviewCount = 0
                }
                papers[idx].flashcards = fcs
            }
            _ = try? savePaperJSON(papers[idx])
            _ = try? PaperMarkdownExporter.write(paper: papers[idx], outputRoot: outputRoot)
        } catch {
            ingestionLog += "\nFlashcards failed for \(paper.title): \(error.localizedDescription)"
        }
    }

    func generateStudyQuestions(for paperID: UUID) {
        Task { await runStudyQuestions(for: paperID) }
    }

    private func runStudyQuestions(for paperID: UUID) async {
        guard let idx = papers.firstIndex(where: { $0.id == paperID }) else { return }
        let paper = papers[idx]
        let fallbackInstructions = "You create comprehension questions for research papers."
        let instructions = PromptStore.loadText("ui.study_questions.instructions.md", fallback: fallbackInstructions)

        let fallbackTemplate = """
        Draft 5 questions to check understanding of the paper "{{title}}".
        Use the summary and takeaways. Mix conceptual and application questions.
        """
        let template = PromptStore.loadText("ui.study_questions.prompt.md", fallback: fallbackTemplate)
        let prompt = PromptStore.render(template: template, variables: [
            "title": paper.title
        ])
        do {
            let session = LanguageModelSession(instructions: instructions)
            let response = try await session.respond(to: prompt)
            let questions = response.content
                .split(whereSeparator: { $0.isNewline })
                .map { $0.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            papers[idx].userQuestions = questions
            _ = try? savePaperJSON(papers[idx])
            _ = try? PaperMarkdownExporter.write(paper: papers[idx], outputRoot: outputRoot)
        } catch {
            ingestionLog += "\nQuestion generation failed for \(paper.title): \(error.localizedDescription)"
        }
        await MainActor.run {
            recomputeReadingProfile(extra: nil)
        }
    }

    func dailyQuizCards(limit: Int = 10) -> [(Flashcard, Paper)] {
        let now = Date()
        var pairs: [(Flashcard, Paper, Date)] = []
        for paper in papers {
            guard let flashcards = paper.flashcards, !flashcards.isEmpty else { continue }
            if paper.readingStatus == .done || (paper.isImportant ?? false) || paper.firstReadAt != nil {
                let recency = paper.firstReadAt ?? paper.ingestedAt ?? now
                for card in flashcards {
                    pairs.append((card, paper, recency))
                }
            }
        }
        let sorted = pairs.sorted { $0.2 > $1.2 }
        return Array(sorted.prefix(limit).map { ($0.0, $0.1) })
    }

    func markFlashcardReviewed(paperID: UUID, cardID: UUID) {
        guard let pIdx = papers.firstIndex(where: { $0.id == paperID }) else { return }
        guard var flashcards = papers[pIdx].flashcards,
              let cIdx = flashcards.firstIndex(where: { $0.id == cardID }) else { return }
        flashcards[cIdx].lastReviewedAt = Date()
        flashcards[cIdx].reviewCount = (flashcards[cIdx].reviewCount ?? 0) + 1
        papers[pIdx].flashcards = flashcards
        _ = try? savePaperJSON(papers[pIdx])
        _ = try? PaperMarkdownExporter.write(paper: papers[pIdx], outputRoot: outputRoot)
        appendUserEvent(type: "flashcard_reviewed", paperID: paperID, extra: ["card_id": cardID.uuidString])
    }

    private func recomputeReadingProfile(extra: [Float]?) {
        var vectors: [[Float]] = []

        let important = papers.filter { ($0.isImportant ?? false) || $0.readingStatus == .done }
        for paper in important where !paper.embedding.isEmpty {
            vectors.append(normalizeEmbedding(paper.embedding))
        }

        for paper in papers {
            if let noteVec = paper.noteEmbedding, !noteVec.isEmpty {
                vectors.append(normalizeEmbedding(noteVec))
            }
        }

        if let extra, !extra.isEmpty {
            vectors.append(normalizeEmbedding(extra))
        }

        for emb in questionEmbeddings {
            vectors.append(normalizeEmbedding(emb))
        }

        guard let first = vectors.first else {
            readingProfileVector = nil
            return
        }

        let dim = first.count
        var sum = [Float](repeating: 0, count: dim)
        for vec in vectors where vec.count == dim {
            vDSP.add(sum, vec, result: &sum)
        }
        let inv = 1 / Float(max(1, vectors.count))
        vDSP.multiply(inv, sum, result: &sum)
        readingProfileVector = sum
    }

    func recommendedNextPapers(limit: Int = 5) -> [Paper] {
        let completed = papers.filter { $0.readingStatus == .done }
        let unread = papers.filter { $0.readingStatus != .done }
        guard !unread.isEmpty else { return [] }

        let targetEmbedding: [Float]
        if let profile = readingProfileVector, !profile.isEmpty {
            targetEmbedding = profile
        } else if !completed.isEmpty, let avg = meanEmbedding(for: completed) {
            targetEmbedding = avg
        } else if let avg = meanEmbedding(for: papers) {
            targetEmbedding = avg
        } else {
            return []
        }

        var scored: [ScoredPaper] = []
        for paper in unread where !paper.embedding.isEmpty && paper.embedding.count == targetEmbedding.count {
            let score = cosineSimilarity(paper.embedding, targetEmbedding)
            if score.isNaN { continue }
            scored.append(ScoredPaper(paper: paper, score: score))
        }
        return Array(scored.sorted { $0.score > $1.score }.prefix(limit).map { $0.paper })
    }

    private func meanEmbedding(for papers: [Paper]) -> [Float]? {
        // Use only non-empty embeddings and ignore mismatched dimensions so a single bad entry
        // (e.g. missing embedding) doesn't nullify the aggregate.
        guard let firstValid = papers.first(where: { !$0.embedding.isEmpty })?.embedding else { return nil }

        let dim = firstValid.count
        var sum = [Float](repeating: 0, count: dim)
        var usedCount = 0

        for paper in papers {
            let emb = paper.embedding
            guard emb.count == dim, !emb.isEmpty else { continue }
            vDSP.add(sum, emb, result: &sum)
            usedCount += 1
        }

        guard usedCount > 0 else { return nil }
        let inv = 1 / Float(usedCount)
        vDSP.multiply(inv, sum, result: &sum)
        return sum
    }

    func blindSpots(limit: Int = 5) -> [Paper] {
        guard let profile = readingProfileVector, !profile.isEmpty else { return [] }
        guard let global = meanEmbedding(for: papers) else { return [] }

        let candidates = papers.filter { $0.readingStatus != .done }
        var scored: [(Paper, Float)] = []
        for paper in candidates where paper.embedding.count == profile.count {
            let centrality = cosineSimilarity(paper.embedding, global)
            let personal = cosineSimilarity(paper.embedding, profile)
            if centrality.isNaN || personal.isNaN { continue }
            let blind = max(0, centrality - personal)
            scored.append((paper, blind))
        }

        return Array(scored.sorted { $0.1 > $1.1 }.prefix(limit).map { $0.0 })
    }

    func adaptiveCurriculum(targetClusterID: Int? = nil) -> [CurriculumStep] {
        guard !papers.isEmpty else { return [] }
        let unread = papers.filter { $0.readingStatus != .done }
        guard !unread.isEmpty else { return [] }

        let profile = readingProfileVector ?? meanEmbedding(for: papers) ?? []
        let targetVec: [Float] = {
            if let targetID = targetClusterID,
               let cluster = clusters.first(where: { $0.id == targetID }) {
                return cluster.centroid
            }
            return clusters.first?.centroid ?? profile
        }()

        struct Scored {
            let paper: Paper
            let toProfile: Float
            let toTarget: Float
        }

        let scored: [Scored] = unread.compactMap { paper in
            guard !paper.embedding.isEmpty,
                  paper.embedding.count == profile.count else { return nil }
            let toProfile = cosineSimilarity(paper.embedding, profile)
            let toTarget = targetVec.isEmpty || targetVec.count != paper.embedding.count ? toProfile : cosineSimilarity(paper.embedding, targetVec)
            if toProfile.isNaN || toTarget.isNaN { return nil }
            return Scored(paper: paper, toProfile: toProfile, toTarget: toTarget)
        }

        var foundation = scored
            .sorted { $0.toProfile > $1.toProfile }
            .prefix(3)
            .map { CurriculumStep(id: UUID(), paper: $0.paper, stage: .foundation, score: $0.toProfile) }

        let remainingAfterFoundation = scored.filter { f in
            !foundation.contains(where: { $0.paper.id == f.paper.id })
        }

        var advanced = remainingAfterFoundation
            .sorted { ($0.toTarget - $0.toProfile) > ($1.toTarget - $1.toProfile) }
            .prefix(2)
            .map { CurriculumStep(id: UUID(), paper: $0.paper, stage: .advanced, score: $0.toTarget - $0.toProfile) }

        let remainingAfterAdvanced = remainingAfterFoundation.filter { a in
            !advanced.contains(where: { $0.paper.id == a.paper.id })
        }

        var bridges = remainingAfterAdvanced
            .sorted { min($0.toProfile, $0.toTarget) > min($1.toProfile, $1.toTarget) }
            .prefix(5)
            .map { CurriculumStep(id: UUID(), paper: $0.paper, stage: .bridge, score: min($0.toProfile, $0.toTarget)) }

        // Ensure each stage has at least one if possible.
        if foundation.isEmpty, let fallback = scored.max(by: { $0.toProfile < $1.toProfile }) {
            foundation.append(CurriculumStep(id: UUID(), paper: fallback.paper, stage: .foundation, score: fallback.toProfile))
        }
        if advanced.isEmpty, let fallback = scored.max(by: { ($0.toTarget - $0.toProfile) < ($1.toTarget - $1.toProfile) }) {
            advanced.append(CurriculumStep(id: UUID(), paper: fallback.paper, stage: .advanced, score: fallback.toTarget - fallback.toProfile))
            bridges.removeAll { $0.paper.id == fallback.paper.id }
            foundation.removeAll { $0.paper.id == fallback.paper.id }
        }
        if bridges.isEmpty {
            let candidates = scored.filter { sc in
                !foundation.contains(where: { $0.paper.id == sc.paper.id }) &&
                !advanced.contains(where: { $0.paper.id == sc.paper.id })
            }
            if let fallback = candidates.max(by: { min($0.toProfile, $0.toTarget) < min($1.toProfile, $1.toTarget) }) {
                bridges.append(CurriculumStep(id: UUID(), paper: fallback.paper, stage: .bridge, score: min(fallback.toProfile, fallback.toTarget)))
            } else if let fallbackAny = scored.max(by: { min($0.toProfile, $0.toTarget) < min($1.toProfile, $1.toTarget) }) {
                bridges.append(CurriculumStep(id: UUID(), paper: fallbackAny.paper, stage: .bridge, score: min(fallbackAny.toProfile, fallbackAny.toTarget)))
            }
        }

        return Array(foundation.prefix(3) + bridges.prefix(5) + advanced.prefix(2))
    }

    func knowledgeSnapshot(for topic: String, maxKnown: Int = 5, maxMissing: Int = 5) -> KnowledgeSnapshot? {
        let queryVec = normalizeEmbedding(fallbackEmbedding(for: topic))
        guard !queryVec.isEmpty else { return nil }

        let corpus = explorationPapers
        let knownPapers = corpus.filter { $0.readingStatus == .done || ($0.isImportant ?? false) }
        let unread = corpus.filter { $0.readingStatus != .done }

        let known = knownPapers
            .compactMap { p -> (Paper, Float)? in
                let emb = normalizeEmbedding(p.embedding)
                guard emb.count == queryVec.count else { return nil }
                let score = cosineSimilarity(emb, queryVec)
                return score.isNaN ? nil : (p, score)
            }
            .sorted { $0.1 > $1.1 }
            .prefix(maxKnown)
            .map { $0.0 }

        let missing = unread
            .compactMap { p -> (Paper, Float)? in
                let emb = normalizeEmbedding(p.embedding)
                guard emb.count == queryVec.count else { return nil }
                let score = cosineSimilarity(emb, queryVec)
                return score.isNaN ? nil : (p, score)
            }
            .sorted { $0.1 > $1.1 }
            .prefix(maxMissing)
            .map { $0.0 }

        let summaryLines = known.prefix(3).map { "• \( $0.title )" }
        let summaryText = summaryLines.isEmpty ? "No read papers near this topic yet." : summaryLines.joined(separator: "\n")
        return KnowledgeSnapshot(topic: topic, known: known, missing: missing, summary: summaryText)
    }

    // MARK: - Question answering

    func answerQuestion(_ question: String, topK: Int = 8) {
        Task { await runQuestion(question, topK: topK) }
    }

    private func runQuestion(_ question: String, topK: Int) async {
        guard !papers.isEmpty else { return }
        isAnswering = true
        questionAnswer = ""
        questionTopPapers = []
        questionEvidence = []

        let corpus = explorationPapers
        guard !corpus.isEmpty else {
            if let range = effectiveYearRange {
                questionAnswer = "No papers in the selected year range (\(range.lowerBound)–\(range.upperBound))."
            } else {
                questionAnswer = "No papers available."
            }
            isAnswering = false
            return
        }

        let queryEmbedding = normalizeEmbedding(await embedder.encode(for: question) ?? [])
        let queryVec = !queryEmbedding.isEmpty ? queryEmbedding : normalizeEmbedding(fallbackEmbedding(for: question))
        guard !queryVec.isEmpty else {
            questionAnswer = "Failed to embed question."
            isAnswering = false
            return
        }
        questionHistory.append(question)
        questionEmbeddings.append(queryVec)
        recomputeReadingProfile(extra: nil)

        let paperTitleByID = Dictionary(uniqueKeysWithValues: corpus.map { ($0.id, $0.title) })
        let paperByID = Dictionary(uniqueKeysWithValues: corpus.map { ($0.id, $0) })

        // Two-stage retrieval for scale:
        // 1) Get a short candidate list of papers (ANN if available).
        // 2) Score chunks only inside those papers.
        let candidatePaperK = min(max(topK * 10, 60), corpus.count)

        let candidatePapers: [ScoredPaper] = {
            let allowedIDs: Set<UUID>? = yearFilterEnabled ? explorationPaperIDs : nil
            let annK = min(max(candidatePaperK * 3, candidatePaperK), papers.count)
            if let annTop = annSearch(queryVec: queryVec, k: annK) {
                if let allowedIDs {
                    let filtered = annTop.filter { allowedIDs.contains($0.paper.id) }
                    if !filtered.isEmpty {
                        return Array(filtered.prefix(candidatePaperK))
                    }
                } else {
                    return Array(annTop.prefix(candidatePaperK))
                }
            }
            var scored: [ScoredPaper] = []
            scored.reserveCapacity(min(candidatePaperK, corpus.count))
            for paper in corpus where paper.embedding.count == queryVec.count {
                let score = cosineSimilarity(queryVec, paper.embedding)
                if score.isNaN { continue }
                scored.append(ScoredPaper(paper: paper, score: score))
            }
            scored.sort { $0.score > $1.score }
            return Array(scored.prefix(candidatePaperK))
        }()

        let candidatePaperIDs = candidatePapers.map { $0.paper.id }
        let evidenceLimit = max(12, topK * 3)
        let evidenceChunks = topChunks(
            for: queryVec,
            limit: evidenceLimit,
            candidatePaperIDs: candidatePaperIDs,
            paperTitleByID: paperTitleByID
        )

        if evidenceChunks.isEmpty {
            let top = Array(candidatePapers.prefix(topK))
            questionTopPapers = top
            do {
                let answer = try await questionAnswerer.answer(question: question, topPapers: top)
                questionAnswer = answer
            } catch {
                questionAnswer = "Error from language model: \(error.localizedDescription)"
            }
        } else {
            questionEvidence = evidenceChunks
            let grouped = Dictionary(grouping: evidenceChunks, by: { $0.chunk.paperID })
            let scoredPapers = grouped.compactMap { (key, value) -> ScoredPaper? in
                guard let paper = paperByID[key] else { return nil }
                let best = value.map { $0.score }.max() ?? 0
                return ScoredPaper(paper: paper, score: best)
            }.sorted { $0.score > $1.score }
            var top = Array(scoredPapers.prefix(topK))
            if top.count < topK {
                let used = Set(top.map { $0.paper.id })
                let fill = candidatePapers.filter { !used.contains($0.paper.id) }.prefix(topK - top.count)
                top.append(contentsOf: fill)
            }
            questionTopPapers = top

            do {
                let answer = try await questionAnswerer.answer(question: question, evidence: Array(evidenceChunks.prefix(12)))
                questionAnswer = answer
            } catch {
                questionAnswer = "Error from language model: \(error.localizedDescription)"
            }
        }

        let fileName = question.replacingOccurrences(of: "/", with: "-").prefix(40)
        let qaURL = outputRoot.appendingPathComponent("qa", isDirectory: true).appendingPathComponent("QA_\(fileName).txt")
        do {
            if let data = questionAnswer.data(using: .utf8) {
                try data.write(to: qaURL, options: .atomic)
            }
        } catch {
            logger.error("Failed to save QA answer: \(error.localizedDescription, privacy: .public)")
        }

        // Persist lightweight retrieval diagnostics for offline QA-gap analytics.
        do {
            let topScores = questionTopPapers.prefix(6).map { Double($0.score) }
            let margin: Double = {
                guard topScores.count >= 2 else { return topScores.first ?? 0 }
                return topScores[0] - topScores[1]
            }()
            var clusterCounts: [String: Int] = [:]
            for item in questionTopPapers.prefix(10) {
                if let cid = item.paper.clusterIndex {
                    clusterCounts[String(cid), default: 0] += 1
                }
            }
            let breadth: Double = {
                let total = clusterCounts.values.reduce(0, +)
                guard total > 0, clusterCounts.count > 1 else { return 0 }
                let probs = clusterCounts.values.map { Double($0) / Double(total) }
                let entropy = -probs.reduce(0.0, { acc, p in
                    acc + (p <= 0 ? 0 : p * log(p))
                })
                return entropy / log(Double(clusterCounts.count))
            }()
            appendUserEvent(
                type: "qa_retrieval",
                paperID: nil,
                extra: [
                    "q": question,
                    "top_scores": topScores,
                    "margin": margin,
                    "support_breadth": breadth,
                    "top_clusters": clusterCounts,
                    "evidence_count": questionEvidence.count,
                    "top_paper_count": questionTopPapers.count
                ]
            )
        }
        recordAnswerReady(question)
        appendUserEvent(type: "qa_evidence_viewed", paperID: nil, extra: ["count": questionEvidence.count])

        isAnswering = false
    }

    private func invalidateChunkLookupCache() {
        chunksByPaperID.removeAll(keepingCapacity: true)
        chunksByPaperIDBuiltCount = -1
    }

    private func ensureChunksByPaperID() {
        guard chunksByPaperIDBuiltCount != paperChunks.count || chunksByPaperID.isEmpty else { return }
        var map: [UUID: [PaperChunk]] = [:]
        map.reserveCapacity(min(papers.count, 8_000))
        for chunk in paperChunks {
            map[chunk.paperID, default: []].append(chunk)
        }
        for key in map.keys {
            map[key]?.sort { $0.order < $1.order }
        }
        chunksByPaperID = map
        chunksByPaperIDBuiltCount = paperChunks.count
    }

    private func topChunks(
        for queryVec: [Float],
        limit: Int,
        candidatePaperIDs: [UUID]?,
        paperTitleByID: [UUID: String]
    ) -> [ChunkEvidence] {
        guard limit > 0 else { return [] }
        guard !paperChunks.isEmpty else { return [] }
        ensureChunksByPaperID()

        let dim = queryVec.count
        var results: [ChunkEvidence] = []

        let paperIDs: [UUID]
        if let candidatePaperIDs, !candidatePaperIDs.isEmpty {
            paperIDs = candidatePaperIDs
        } else {
            paperIDs = Array(chunksByPaperID.keys)
        }

        for pid in paperIDs {
            for chunk in chunksByPaperID[pid] ?? [] where chunk.embedding.count == dim {
                let score = cosineSimilarity(chunk.embedding, queryVec)
                if score.isNaN { continue }
                let title = paperTitleByID[pid] ?? "Unknown paper"
                results.append(ChunkEvidence(chunk: chunk, score: score, paperTitle: title))
            }
        }

        return Array(results.sorted { $0.score > $1.score }.prefix(limit))
    }

    private func ensurePaperANNCache(queryDim: Int) -> PaperANNCache? {
        guard AtlasFFI.isAvailable() else {
            paperANNCache = nil
            return nil
        }
        guard queryDim > 0 else { return nil }

        let version = currentCorpusVersion()
        if let cache = paperANNCache,
           cache.version == version,
           cache.dim == queryDim,
           cache.index != nil,
           cache.count > 0 {
            return cache
        }

        let eligible = papers.filter { $0.embedding.count == queryDim }
        guard !eligible.isEmpty else {
            paperANNCache = nil
            return nil
        }
        guard let index = AtlasFFI.buildIndex(vectors: eligible.map { $0.embedding }) else {
            paperANNCache = nil
            return nil
        }
        let ids = eligible.map { $0.id }
        let cache = PaperANNCache(version: version, dim: queryDim, ids: ids, index: index)
        paperANNCache = cache
        return cache
    }

    private func annSearch(queryVec: [Float], k: Int) -> [ScoredPaper]? {
        guard k > 0 else { return [] }
        let dim = queryVec.count
        guard let cache = ensurePaperANNCache(queryDim: dim) else { return nil }

        let results = AtlasFFI.query(index: cache.index, query: queryVec, k: k)
        var scored: [ScoredPaper] = []
        let paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        for res in results {
            let idx = Int(res.index)
            guard idx < cache.ids.count else { continue }
            let pid = cache.ids[idx]
            guard let paper = paperByID[pid] else { continue }
            scored.append(ScoredPaper(paper: paper, score: 1 - res.distance))
        }
        return scored
    }

    // MARK: - Temporal analytics & field evolution

    func topicEvolutionStreams() -> [TopicEvolutionStream] {
        clusters.map { TemporalAnalytics.topicEvolution(for: $0, papers: papers) }
    }

    func methodTakeoverSignals(methodTags: [String]) -> [MethodTakeover] {
        TemporalAnalytics.methodTakeovers(papers: papers, methodTags: methodTags)
    }

    func readingLagAnalytics() -> ReadingLagStats {
        TemporalAnalytics.readingLagStats(papers: papers, clusters: clusters)
    }

    func noveltyHighlights(neighbors: Int = 3) -> [PaperNoveltyScore] {
        TemporalAnalytics.noveltyScores(papers: papers, clusters: clusters, neighbors: neighbors)
    }

    func generateWhatIfPaper(for clusterIDs: [Int]) -> HypotheticalPaper? {
        let selected = clusters.filter { clusterIDs.contains($0.id) }
        guard !selected.isEmpty else { return nil }
        return TemporalAnalytics.generateHypotheticalPaper(from: selected, papers: papers)
    }

    func simulateAuthorPanel(for clusterID: Int, maxSpeakers: Int = 3) -> String {
        guard let cluster = clusters.first(where: { $0.id == clusterID }) else { return "Cluster not found." }
        return TemporalAnalytics.simulatePanel(for: cluster, papers: papers, maxSpeakers: maxSpeakers)
    }

    func simulateClusterDebate(firstID: Int, secondID: Int) -> String {
        guard let first = clusters.first(where: { $0.id == firstID }),
              let second = clusters.first(where: { $0.id == secondID }) else {
            return "Missing clusters."
        }
        return TemporalAnalytics.simulateDebate(between: first, and: second, papers: papers)
    }

    func generateDebateBetweenClusters(firstID: Int, secondID: Int, rounds: Int = 4, useAI: Bool = false) async -> String {
        guard firstID != secondID else { return "Pick two different topics." }
        guard let left = clusters.first(where: { $0.id == firstID }),
              let right = clusters.first(where: { $0.id == secondID }) else {
            return "Missing clusters."
        }

        let paperByID = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        let leftPapers = left.memberPaperIDs.compactMap { paperByID[$0] }
        let rightPapers = right.memberPaperIDs.compactMap { paperByID[$0] }

        let leftRep = representativePaper(for: leftPapers)
        let rightRep = representativePaper(for: rightPapers)

        let leftContext = DebateContext(
            title: left.name,
            summary: leftRep?.summary ?? left.metaSummary,
            exemplars: Array(leftPapers.prefix(2)).map { ($0.title, firstSentence($0.summary)) }
        )
        let rightContext = DebateContext(
            title: right.name,
            summary: rightRep?.summary ?? right.metaSummary,
            exemplars: Array(rightPapers.prefix(2)).map { ($0.title, firstSentence($0.summary)) }
        )

        let roundsClamped = clampDebateRounds(rounds)
        if useAI {
            if SystemLanguageModel.default.availability == .available {
                do {
                    return try await aiDebate(left: leftContext, right: rightContext, rounds: roundsClamped)
                } catch {
                    return heuristicDebate(left: leftContext, right: rightContext, rounds: roundsClamped, note: "AI failed (\(error.localizedDescription)). Falling back to heuristic.")
                }
            } else {
                return heuristicDebate(left: leftContext, right: rightContext, rounds: roundsClamped, note: "On-device model unavailable. Falling back to heuristic.")
            }
        }

        return heuristicDebate(left: leftContext, right: rightContext, rounds: roundsClamped, note: "Heuristic simulation (offline).")
    }

    func generateDebateBetweenPapers(firstID: UUID, secondID: UUID, rounds: Int = 4, useAI: Bool = false) async -> String {
        guard firstID != secondID else { return "Pick two different papers." }
        guard let leftPaper = papers.first(where: { $0.id == firstID }),
              let rightPaper = papers.first(where: { $0.id == secondID }) else {
            return "Missing papers."
        }

        let leftContext = DebateContext(
            title: leftPaper.title,
            summary: leftPaper.summary,
            exemplars: [(leftPaper.title, firstSentence(leftPaper.summary))]
        )
        let rightContext = DebateContext(
            title: rightPaper.title,
            summary: rightPaper.summary,
            exemplars: [(rightPaper.title, firstSentence(rightPaper.summary))]
        )

        let roundsClamped = clampDebateRounds(rounds)
        if useAI {
            if SystemLanguageModel.default.availability == .available {
                do {
                    return try await aiDebate(left: leftContext, right: rightContext, rounds: roundsClamped)
                } catch {
                    return heuristicDebate(left: leftContext, right: rightContext, rounds: roundsClamped, note: "AI failed (\(error.localizedDescription)). Falling back to heuristic.")
                }
            } else {
                return heuristicDebate(left: leftContext, right: rightContext, rounds: roundsClamped, note: "On-device model unavailable. Falling back to heuristic.")
            }
        }

        return heuristicDebate(left: leftContext, right: rightContext, rounds: roundsClamped, note: "Heuristic simulation (offline).")
    }

    private struct DebateContext {
        let title: String
        let summary: String
        let exemplars: [(title: String, snippet: String)]
    }

    private func representativePaper(for papers: [Paper]) -> Paper? {
        guard !papers.isEmpty else { return nil }
        if let summary = analyticsSummary, !summary.centrality.isEmpty {
            let centralityByID = Dictionary(uniqueKeysWithValues: summary.centrality.map { ($0.paperID, $0.weightedDegree) })
            if let best = papers.max(by: { (centralityByID[$0.id] ?? 0) < (centralityByID[$1.id] ?? 0) }) {
                return best
            }
        }
        return papers.max(by: { ($0.year ?? 0, $0.title) < ($1.year ?? 0, $1.title) })
    }

    private func clampDebateRounds(_ rounds: Int) -> Int {
        min(max(rounds, 2), 8)
    }

    private func heuristicDebate(left: DebateContext, right: DebateContext, rounds: Int, note: String) -> String {
        let steps = rounds * 2
        let leftOpening = firstSentence(left.summary)
        let rightOpening = firstSentence(right.summary)

        var lines: [String] = []
        lines.append("Debate — \(left.title) vs \(right.title)")
        lines.append("Rounds: \(rounds) (steps: \(steps))")
        lines.append(note)
        lines.append("")

        for r in 1...rounds {
            lines.append("Round \(r)")
            switch r % 4 {
            case 1:
                lines.append("A (\(left.title)) opening: \(leftOpening)")
                lines.append("B (\(right.title)) opening: \(rightOpening)")
            case 2:
                lines.append("A (\(left.title)) rebuttal: What assumptions does B rely on, and where might it fail?")
                lines.append("B (\(right.title)) rebuttal: What assumptions does A rely on, and where might it fail?")
            case 3:
                lines.append("A (\(left.title)) cross-exam: Which metric/regime is your strongest evidence, and what's the ablation?")
                lines.append("B (\(right.title)) cross-exam: Which metric/regime is your strongest evidence, and what's the ablation?")
            default:
                lines.append("A (\(left.title)) closing: Best use-cases, risks, and what would change your mind.")
                lines.append("B (\(right.title)) closing: Best use-cases, risks, and what would change your mind.")
            }
            lines.append("")
        }

        if !left.exemplars.isEmpty || !right.exemplars.isEmpty {
            lines.append("Context snippets")
            for (t, s) in left.exemplars {
                lines.append("• A exemplar — \(t): \(s)")
            }
            for (t, s) in right.exemplars {
                lines.append("• B exemplar — \(t): \(s)")
            }
            lines.append("")
        }

        lines.append("Next experiments")
        lines.append("• Define evaluation metrics and regimes where each side should win.")
        lines.append("• Run ablations to isolate key mechanism differences.")
        lines.append("• Stress-test assumptions; report failure modes and sensitivity.")
        return lines.joined(separator: "\n")
    }

    private func aiDebate(left: DebateContext, right: DebateContext, rounds: Int) async throws -> String {
        let leftSnippets = left.exemplars.map { "- \($0.title): \($0.snippet)" }.joined(separator: "\n")
        let rightSnippets = right.exemplars.map { "- \($0.title): \($0.snippet)" }.joined(separator: "\n")

        let fallbackInstructions = "You simulate debate transcripts between research ideas; you stay grounded in provided summaries."
        let instructions = PromptStore.loadText("ui.debate.instructions.md", fallback: fallbackInstructions)

        let fallbackTemplate = """
        Simulate a structured debate between two research ideas using ONLY the provided context.

        Side A: {{left_title}}
        Summary A: {{left_summary}}
        Exemplars A:
        {{left_snippets}}

        Side B: {{right_title}}
        Summary B: {{right_summary}}
        Exemplars B:
        {{right_snippets}}

        Output format:
        - Start with: Rounds: {{rounds}} (steps: {{steps}})
        - Then produce {{rounds}} rounds. Each round has:
          - A: 2-3 sentences
          - B: 2-3 sentences
        - End with bullet lists for: Agreements, Disagreements, Next experiments.

        Keep it grounded: do not invent citations or facts not implied by the summaries.
        """
        let template = PromptStore.loadText("ui.debate.prompt.md", fallback: fallbackTemplate)
        let prompt = PromptStore.render(template: template, variables: [
            "left_title": left.title,
            "left_summary": left.summary,
            "left_snippets": leftSnippets.isEmpty ? "(none)" : leftSnippets,
            "right_title": right.title,
            "right_summary": right.summary,
            "right_snippets": rightSnippets.isEmpty ? "(none)" : rightSnippets,
            "rounds": String(rounds),
            "steps": String(rounds * 2)
        ])

        let session = LanguageModelSession(instructions: instructions)
        let response = try await session.respond(to: prompt)
        return response.content
    }

    private func firstSentence(_ text: String, limit: Int = 220) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let stop = [". ", "?\n", "!\n", ".\n", "?", "!"]
        let cut = stop.compactMap { token in trimmed.range(of: token) }.min(by: { $0.lowerBound < $1.lowerBound })
        let sentence = cut.map { String(trimmed[..<$0.lowerBound]) } ?? trimmed
        let collapsed = sentence.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        if collapsed.count <= limit { return collapsed }
        let idx = collapsed.index(collapsed.startIndex, offsetBy: limit)
        return String(collapsed[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines) + "…"
    }

    func recordRecommendationFeedback(paperID: UUID, helpful: Bool) {
        recommendationFeedback[paperID] = helpful
        appendUserEvent(type: helpful ? "rec_helpful" : "rec_not_helpful", paperID: paperID)
    }

    func recordQuestionAsked(_ question: String) {
        appendUserEvent(type: "qa_question", paperID: nil, extra: ["q": question])
    }

    func recordAnswerReady(_ question: String) {
        appendUserEvent(type: "qa_answer_ready", paperID: nil, extra: ["q": question])
    }

    func recordPaperOpened(_ paperID: UUID) {
        appendUserEvent(type: "opened", paperID: paperID)
    }

    private func appendUserEvent(type: String, paperID: UUID?) {
        appendUserEvent(type: type, paperID: paperID, extra: nil)
    }

    private func appendUserEvent(type: String, paperID: UUID?, extra: [String: Any]?) {
        let url = outputRoot.appendingPathComponent("analytics", isDirectory: true).appendingPathComponent("user_events.jsonl")
        var payloadDict: [String: Any] = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "event_type": type
        ]
        if let pid = paperID { payloadDict["paper_id"] = pid.uuidString }
        if let extra {
            for (k, v) in extra { payloadDict[k] = v }
        }
        guard let data = try? JSONSerialization.data(withJSONObject: payloadDict) else { return }

        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            handle.seekToEndOfFile()
            handle.write(data)
            handle.write("\n".data(using: .utf8)!)
        } catch {
            ingestionLog += "\n[event] Failed to persist user event: \(error.localizedDescription)"
        }
    }

    /// Shortest influence path across claim edges between two clusters.
    func influencePath(between firstID: Int, and secondID: Int) -> [String] {
        let edges = claimGraphEdges()
        var paperForClaim: [UUID: UUID] = [:]
        for paper in papers {
            for claim in paper.claims ?? [] {
                paperForClaim[claim.id] = paper.id
            }
        }

        let firstPapers = papers.filter { $0.clusterIndex == firstID }
        let secondPapers = papers.filter { $0.clusterIndex == secondID }
        guard let startPaper = firstPapers.first, let targetPaper = secondPapers.first else { return [] }

        var adj: [UUID: Set<UUID>] = [:]
        for edge in edges {
            guard let srcPaper = paperForClaim[edge.sourceClaimID], let dstPaper = paperForClaim[edge.targetClaimID] else { continue }
            adj[srcPaper, default: []].insert(dstPaper)
        }

        var queue: [UUID] = [startPaper.id]
        var parent: [UUID: UUID] = [:]
        var visited: Set<UUID> = [startPaper.id]
        while let current = queue.first {
            queue.removeFirst()
            if current == targetPaper.id { break }
            for neigh in adj[current] ?? [] where !visited.contains(neigh) {
                visited.insert(neigh)
                parent[neigh] = current
                queue.append(neigh)
            }
        }
        guard visited.contains(targetPaper.id) else { return [] }
        var path: [UUID] = []
        var node = targetPaper.id
        while true {
            path.append(node)
            if node == startPaper.id { break }
            guard let p = parent[node] else { break }
            node = p
        }
        path.reverse()
        let titleMap = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0.title) })
        return path.compactMap { titleMap[$0] }
    }

    /// Claim-level shortest path from any paper in first cluster to any in second (returns claim statements).
    func claimPathBetweenClusters(_ firstID: Int, _ secondID: Int) -> [String] {
        let edges = claimGraphEdges()
        var clusterForClaim: [UUID: Int] = [:]
        var statementForClaim: [UUID: String] = [:]
        for paper in papers {
            for claim in paper.claims ?? [] {
                if let cid = paper.clusterIndex {
                    clusterForClaim[claim.id] = cid
                }
                statementForClaim[claim.id] = claim.statement
            }
        }
        let sources = edges.compactMap { edge in
            clusterForClaim[edge.sourceClaimID] == firstID ? edge.sourceClaimID : nil
        }
        let targets = Set(edges.compactMap { edge in
            clusterForClaim[edge.targetClaimID] == secondID ? edge.targetClaimID : nil
        })
        guard !sources.isEmpty, !targets.isEmpty else { return [] }

        var adj: [UUID: [UUID]] = [:]
        for edge in edges {
            adj[edge.sourceClaimID, default: []].append(edge.targetClaimID)
        }

        var queue: [UUID] = sources
        var parent: [UUID: UUID] = [:]
        var seen = Set(queue)
        var targetHit: UUID?
        while let current = queue.first {
            queue.removeFirst()
            if targets.contains(current) {
                targetHit = current
                break
            }
            for neigh in adj[current] ?? [] where !seen.contains(neigh) {
                seen.insert(neigh)
                parent[neigh] = current
                queue.append(neigh)
            }
        }
        guard let end = targetHit else { return [] }
        var chain: [UUID] = []
        var node = end
        while true {
            chain.append(node)
            if let p = parent[node] {
                node = p
            } else { break }
        }
        chain.reverse()
        return chain.compactMap { statementForClaim[$0] }
    }

    func misconceptionReport(for paperID: UUID, answer: String) -> MisconceptionReport? {
        guard let paper = papers.first(where: { $0.id == paperID }) else { return nil }
        return TemporalAnalytics.detectMisconception(answer: answer, paper: paper)
    }
}

private struct ClusterCacheKey: Hashable {
    let version: String
    let k: Int
}

private struct ClusterSnapshot: Codable {
    let version: String
    let k: Int
    let clusters: [Cluster]
}

private struct GalaxySnapshot: Codable {
    let version: String
    let megaClusters: [Cluster]
    let pinnedClusterIDs: [Int]?
    let clusterNameSources: [Int: ClusterNameSource]?

    init(
        version: String,
        megaClusters: [Cluster],
        pinnedClusterIDs: [Int]? = nil,
        clusterNameSources: [Int: ClusterNameSource]? = nil
    ) {
        self.version = version
        self.megaClusters = megaClusters
        self.pinnedClusterIDs = pinnedClusterIDs
        self.clusterNameSources = clusterNameSources
    }
}

@available(macOS 26, iOS 26, *)
private final class PaperANNCache {
    let version: String
    let dim: Int
    let ids: [UUID]
    let index: UnsafeMutableRawPointer?

    var count: Int { ids.count }

    init(version: String, dim: Int, ids: [UUID], index: UnsafeMutableRawPointer?) {
        self.version = version
        self.dim = dim
        self.ids = ids
        self.index = index
    }

    deinit {
        AtlasFFI.free(index: index)
    }
}

private struct GalaxyClusterInfo: Sendable {
    let id: Int
    let name: String
    let metaSummary: String
    let centroid: [Float]
    let memberPaperIDs: [UUID]
    var layoutPosition: Point2D?
    let resolutionK: Int
    let corpusVersion: String
    let subclusterIDs: [Int]?
}

private struct GalaxyComputeResult: Sendable {
    let megaClusters: [GalaxyClusterInfo]
    let subclusters: [GalaxyClusterInfo]
    let assignmentsByID: [UUID: Int]
}

private enum ForceLayout {
    static func compute(for vectors: [[Float]]) -> [Point2D] {
        let count = vectors.count
        guard count > 0 else { return [] }

        let dim = vectors[0].count
        guard dim > 0 else {
            return Array(repeating: Point2D(x: 0.5, y: 0.5), count: count)
        }
        let normalizedVectors: [[Float]] = vectors.map { v in
            if v.count == dim { return v }
            if v.count > dim { return Array(v.prefix(dim)) }
            var vv = v
            vv.append(contentsOf: repeatElement(0, count: dim - v.count))
            return vv
        }

        var positions: [CGPoint] = (0..<count).map { idx in
            let angle = Double(idx) / Double(count) * 2 * Double.pi
            return CGPoint(x: cos(angle), y: sin(angle))
        }

        let iterations = 200
        let repulsion: Double = 0.02
        let attraction: Double = 0.01
        let step: Double = 0.5

        for _ in 0..<iterations {
            var forces = Array(repeating: CGPoint.zero, count: count)
            for i in 0..<count {
                for j in (i + 1)..<count {
                    let dx = positions[i].x - positions[j].x
                    let dy = positions[i].y - positions[j].y
                    var dist = sqrt(dx*dx + dy*dy)
                    dist = max(dist, 0.001)
                    let rep = repulsion / (dist * dist)
                    let fx = rep * dx / dist
                    let fy = rep * dy / dist
                    forces[i].x += fx
                    forces[i].y += fy
                    forces[j].x -= fx
                    forces[j].y -= fy

                    // attraction based on cosine similarity
                    let sim = max(0.0, Double(cosineSimilarity(normalizedVectors[i], normalizedVectors[j])))
                    let att = attraction * sim
                    forces[i].x -= att * dx
                    forces[i].y -= att * dy
                    forces[j].x += att * dx
                    forces[j].y += att * dy
                }
            }

            for i in 0..<count {
                positions[i].x += CGFloat(step * forces[i].x)
                positions[i].y += CGFloat(step * forces[i].y)
                positions[i].x = min(max(positions[i].x, -2), 2)
                positions[i].y = min(max(positions[i].y, -2), 2)
            }
        }

        // normalize to 0...1 for safer layout in view
        let minX = positions.map { $0.x }.min() ?? 0
        let maxX = positions.map { $0.x }.max() ?? 1
        let minY = positions.map { $0.y }.min() ?? 0
        let maxY = positions.map { $0.y }.max() ?? 1
        let dx = maxX - minX
        let dy = maxY - minY

        return positions.map { point in
            let x = dx == 0 ? 0.5 : Double((point.x - minX) / dx)
            let y = dy == 0 ? 0.5 : Double((point.y - minY) / dy)
            return Point2D(x: x, y: y)
        }
    }
}
