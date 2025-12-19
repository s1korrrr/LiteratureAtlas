import SwiftUI
import Charts
import FoundationModels
import Combine

@available(macOS 26, iOS 26, *)
private struct NoveltyConsensusPoint: Identifiable {
    let id: UUID
    let novelty: Double
    let consensus: Double
    let title: String
    let clusterID: Int?
    let novStd: Double
    let consStd: Double
}

@available(macOS 26, iOS 26, *)
struct AnalyticsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var analyticsContentWidth: CGFloat = 0
    @State private var topicQuery: String = ""
    @State private var snapshot: KnowledgeSnapshot?
    @State private var debateMode: DebateMode = .topicVsTopic
    @State private var debateLeftClusterID: Int?
    @State private var debateRightClusterID: Int?
    @State private var debateLeftPaperQuery: String = ""
    @State private var debateRightPaperQuery: String = ""
    @State private var debateLeftPaperID: UUID?
    @State private var debateRightPaperID: UUID?
    @State private var debateRounds: Int = 4
    @State private var debateUseAI: Bool = false
    @State private var debateTranscript: String = ""
    @State private var isGeneratingDebate: Bool = false
    @State private var panelSpeakerCount: Int = 3
    @State private var hypotheticalTitle: String = ""
    @State private var hypotheticalAbstract: String = ""
    @State private var methodTagsText: String = "q-learning, ppo, sac, dqn"
    @State private var showDailyQuiz: Bool = false
    @State private var dailyQuizIndex: Int = 0
    @State private var dailyQuizReveal: Bool = false
    @State private var focusMyExposure: Bool = false
    @State private var selectedCounterfactual: String?
    @State private var customCutoffs: String = "2010 2015 2020"
    @State private var selectedPaperDetail: Paper?
    @State private var selectedNoveltyPointID: UUID?
    @State private var selectedTradingPointID: UUID?
    @State private var noveltyMode: NoveltyMode = .geometric
    @State private var selectedClusterFilter: Int?
    @State private var showFrontier: Bool = true
    @State private var hoveredTimelineYear: Int?
    @State private var hoveredTimelineCount: Int?
    @State private var yearBrushAnchor: Int?

    @State private var cachedTimeline: [(year: Int, count: Int)] = []
    @State private var cachedTopicStreams: [TopicEvolutionStream] = []
    @State private var cachedMethodSignals: [MethodTakeover] = []
    @State private var cachedReadingStats: ReadingLagStats = ReadingLagStats(averageLagYears: 0, realTimeClusterIDs: [], lateClusterIDs: [], overlay: [])
    @State private var cachedNoveltyConsensus: [NoveltyConsensusPoint] = []
    @State private var cachedNoveltyFrontier: Set<UUID> = []
    @State private var cachedDriftTop: [(cluster: Int, year: Int, drift: Double)] = []
    @State private var cachedFactorExposures: [AnalyticsSummary.FactorExposure] = []
    @State private var cachedInfluenceTop: [(paper: Paper, score: Double)] = []
    @State private var cachedInfluenceTimeline: [(paper: Paper, score: Double, year: Int)] = []
    @State private var cachedIdeaRiver: [Paper] = []

    @State private var timelineTask: Task<Void, Never>?
    @State private var temporalTask: Task<Void, Never>?
    @State private var backendTask: Task<Void, Never>?

    private enum NoveltyMode: String, CaseIterable, Identifiable {
        case geometric, combinatorial, directional
        var id: String { rawValue }
        var label: String {
            switch self {
            case .geometric: return "Geometric"
            case .combinatorial: return "Combinatorial"
            case .directional: return "Directional"
            }
        }
    }

    private enum DebateMode: String, CaseIterable, Identifiable {
        case panel
        case topicVsTopic
        case paperVsPaper

        var id: String { rawValue }

        var label: String {
            switch self {
            case .panel: return "Panel"
            case .topicVsTopic: return "Topics"
            case .paperVsPaper: return "Papers"
            }
        }
    }

    private var timeline: [(year: Int, count: Int)] {
        cachedTimeline
    }

    private var corpusYearDomain: ClosedRange<Int>? {
        model.corpusYearDomain
    }

    private var effectiveYearRange: ClosedRange<Int>? {
        model.effectiveYearRange
    }

    private var yearRangeLabel: String {
        guard let domain = corpusYearDomain else { return "No years" }
        if let range = effectiveYearRange {
            if range == domain { return "All years" }
            return "\(range.lowerBound)–\(range.upperBound)"
        }
        return "\(domain.lowerBound)–\(domain.upperBound)"
    }

    private func chartHeight(_ width: CGFloat, ratio: CGFloat, min: CGFloat, max: CGFloat) -> CGFloat {
        guard width > 0 else { return min }
        return Swift.min(Swift.max(width * ratio, min), max)
    }

    private var papersForAnalytics: [Paper] {
        model.explorationPapers
    }

    private var chartYearDomain: ClosedRange<Int>? {
        model.effectiveYearRange ?? model.corpusYearDomain
    }

    private var clusterAverages: [(name: String, year: Double)] {
        let yearByID = Dictionary(uniqueKeysWithValues: papersForAnalytics.compactMap { paper in
            paper.year.map { (paper.id, $0) }
        })

        return model.clusters.compactMap { cluster in
            let years = cluster.memberPaperIDs.compactMap { yearByID[$0] }
            guard !years.isEmpty else { return nil }
            let total = years.reduce(0, +)
            return (cluster.name, Double(total) / Double(years.count))
        }
    }

    private var topicStreams: [TopicEvolutionStream] {
        cachedTopicStreams
    }

    private var methodSignals: [MethodTakeover] {
        cachedMethodSignals
    }

    private var readingStats: ReadingLagStats {
        cachedReadingStats
    }

    private var methodTags: [String] {
        methodTagsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
    }

    private func scheduleTimelineRecompute() {
        timelineTask?.cancel()
        let papersSnapshot = model.papers

        timelineTask = Task(priority: .userInitiated) {
            typealias TimelineSeries = [(year: Int, count: Int)]
            let computeTask: Task<TimelineSeries, Never> = Task.detached(priority: .userInitiated) { [papersSnapshot] in
                let currentYear = Calendar.current.component(.year, from: Date())
                let years = papersSnapshot.compactMap(\.year).filter { $0 >= 1900 && $0 <= currentYear + 1 }
                var counts: [Int: Int] = [:]
                counts.reserveCapacity(min(years.count, 256))
                for y in years {
                    if Task.isCancelled { return [] }
                    counts[y, default: 0] += 1
                }

                guard let minYear = years.min(), let maxYear = years.max(), minYear <= maxYear else { return [] }
                var series: [(year: Int, count: Int)] = []
                series.reserveCapacity(max(0, maxYear - minYear + 1))
                for year in minYear...maxYear {
                    if Task.isCancelled { return [] }
                    series.append((year: year, count: counts[year] ?? 0))
                }
                return series
            }

            let timeline = await withTaskCancellationHandler(operation: {
                await computeTask.value
            }, onCancel: {
                computeTask.cancel()
            })

            guard !Task.isCancelled else { return }
            cachedTimeline = timeline
        }
    }

    private func scheduleTemporalRecompute(delay: TimeInterval = 0.15) {
        temporalTask?.cancel()
        let papersSnapshot = papersForAnalytics
        let clustersSnapshot = model.clusters
        let tagsSnapshot = methodTags
        let emptyStats = ReadingLagStats(averageLagYears: 0, realTimeClusterIDs: [], lateClusterIDs: [], overlay: [])

        temporalTask = Task(priority: .userInitiated) {
            typealias TemporalResults = (streams: [TopicEvolutionStream], signals: [MethodTakeover], stats: ReadingLagStats)
            if delay > 0 {
                let ns = UInt64(max(0, delay) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: ns)
            }
            guard !Task.isCancelled else { return }

            let computeTask: Task<TemporalResults, Never> = Task.detached(priority: .userInitiated) { [papersSnapshot, clustersSnapshot, tagsSnapshot, emptyStats] in
                if Task.isCancelled { return (streams: [], signals: [], stats: emptyStats) }
                let streams = clustersSnapshot.map { TemporalAnalytics.topicEvolution(for: $0, papers: papersSnapshot) }
                if Task.isCancelled { return (streams: [], signals: [], stats: emptyStats) }
                let signals = TemporalAnalytics.methodTakeovers(papers: papersSnapshot, methodTags: tagsSnapshot)
                if Task.isCancelled { return (streams: [], signals: [], stats: emptyStats) }
                let stats = TemporalAnalytics.readingLagStats(papers: papersSnapshot, clusters: clustersSnapshot)
                return (streams: streams, signals: signals, stats: stats)
            }

            let results: TemporalResults = await withTaskCancellationHandler(operation: {
                await computeTask.value
            }, onCancel: {
                computeTask.cancel()
            })

            guard !Task.isCancelled else { return }
            cachedTopicStreams = results.streams
            cachedMethodSignals = results.signals
            cachedReadingStats = results.stats
        }
    }

    private func scheduleBackendRecompute(delay: TimeInterval = 0.15) {
        backendTask?.cancel()
        let summarySnapshot = analyticsSummary
        let papersSnapshot = model.papers
        let modeSnapshot = noveltyMode
        let clusterFilterSnapshot = selectedClusterFilter
        let yearRangeSnapshot = effectiveYearRange
        let focusSnapshot = focusMyExposure

        backendTask = Task(priority: .userInitiated) {
            if delay > 0 {
                let ns = UInt64(max(0, delay) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: ns)
            }
            guard !Task.isCancelled else { return }

            guard let summarySnapshot else {
                cachedNoveltyConsensus = []
                cachedNoveltyFrontier = []
                cachedDriftTop = []
                cachedFactorExposures = []
                cachedInfluenceTop = []
                cachedInfluenceTimeline = []
                cachedIdeaRiver = []
                return
            }

            typealias BackendResults = (
                novelty: [NoveltyConsensusPoint],
                frontier: Set<UUID>,
                drift: [(cluster: Int, year: Int, drift: Double)],
                factors: [AnalyticsSummary.FactorExposure],
                influenceTop: [(paper: Paper, score: Double)],
                influenceTimeline: [(paper: Paper, score: Double, year: Int)],
                river: [Paper]
            )

            let computeTask: Task<BackendResults, Never> = Task.detached(priority: .userInitiated) { [summarySnapshot, papersSnapshot, modeSnapshot, clusterFilterSnapshot, yearRangeSnapshot, focusSnapshot] in
                if Task.isCancelled {
                    return (novelty: [], frontier: Set<UUID>(), drift: [], factors: [], influenceTop: [], influenceTimeline: [], river: [])
                }
                let novelty = Self.computeNoveltyConsensus(
                    summary: summarySnapshot,
                    papers: papersSnapshot,
                    noveltyMode: modeSnapshot,
                    selectedClusterFilter: clusterFilterSnapshot,
                    yearRange: yearRangeSnapshot
                )
                let frontier = Self.computeNoveltyFrontier(points: novelty)
                let drift = Self.computeDriftTop(summary: summarySnapshot, yearRange: yearRangeSnapshot)
                let factors = Self.computeFactorExposures(summary: summarySnapshot, focusMyExposure: focusSnapshot, yearRange: yearRangeSnapshot)
                let influenceTop = Self.computeInfluenceTop(summary: summarySnapshot, papers: papersSnapshot, yearRange: yearRangeSnapshot)
                let influenceTimeline = Self.computeInfluenceTimeline(summary: summarySnapshot, papers: papersSnapshot, yearRange: yearRangeSnapshot)
                let river = Self.computeIdeaRiver(summary: summarySnapshot, papers: papersSnapshot, yearRange: yearRangeSnapshot)
                return (novelty: novelty, frontier: frontier, drift: drift, factors: factors, influenceTop: influenceTop, influenceTimeline: influenceTimeline, river: river)
            }

            let results: BackendResults = await withTaskCancellationHandler(operation: {
                await computeTask.value
            }, onCancel: {
                computeTask.cancel()
            })

            guard !Task.isCancelled else { return }
            cachedNoveltyConsensus = results.novelty
            cachedNoveltyFrontier = results.frontier
            cachedDriftTop = results.drift
            cachedFactorExposures = results.factors
            cachedInfluenceTop = results.influenceTop
            cachedInfluenceTimeline = results.influenceTimeline
            cachedIdeaRiver = results.river
        }
    }

    private nonisolated static func computeNoveltyConsensus(
        summary: AnalyticsSummary,
        papers: [Paper],
        noveltyMode: NoveltyMode,
        selectedClusterFilter: Int?,
        yearRange: ClosedRange<Int>?
    ) -> [NoveltyConsensusPoint] {
        let titleMap = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0.title) })
        let clusterMap = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0.clusterIndex) })
        let yearMap = Dictionary(uniqueKeysWithValues: papers.compactMap { paper in
            paper.year.map { (paper.id, $0) }
        })

        if !summary.paperMetrics.isEmpty {
            return summary.paperMetrics.compactMap { m in
                guard let title = titleMap[m.paperID] else { return nil }
                if let yearRange {
                    guard let y = yearMap[m.paperID], yearRange.contains(y) else { return nil }
                }
                if let clusterFilter = selectedClusterFilter,
                   clusterMap[m.paperID] != clusterFilter { return nil }

                let noveltyVal: Double = {
                    switch noveltyMode {
                    case .geometric: return m.zNovelty
                    case .combinatorial: return m.novCombinatorial
                    case .directional: return m.novDirectional
                    }
                }()
                return NoveltyConsensusPoint(
                    id: m.paperID,
                    novelty: noveltyVal,
                    consensus: m.zConsensus,
                    title: title,
                    clusterID: clusterMap[m.paperID] ?? nil,
                    novStd: m.noveltyUncertainty,
                    consStd: m.consensusUncertainty
                )
            }
        }

        // Fallback to legacy novelty/centrality
        let centralityMap = Dictionary(uniqueKeysWithValues: summary.centrality.map { ($0.paperID, $0.weightedDegree) })
        return summary.novelty.compactMap { n in
            guard let c = centralityMap[n.paperID] else { return nil }
            if let yearRange {
                guard let y = yearMap[n.paperID], yearRange.contains(y) else { return nil }
            }
            if let clusterFilter = selectedClusterFilter,
               n.clusterID != clusterFilter { return nil }
            return NoveltyConsensusPoint(
                id: n.paperID,
                novelty: n.novelty,
                consensus: c,
                title: titleMap[n.paperID] ?? "Paper",
                clusterID: n.clusterID,
                novStd: 0,
                consStd: 0
            )
        }
    }

    private nonisolated static func computeNoveltyFrontier(points: [NoveltyConsensusPoint]) -> Set<UUID> {
        var frontier: Set<UUID> = []
        let sorted = points.sorted { $0.novelty > $1.novelty }
        var bestConsensus = -Double.infinity
        for pt in sorted {
            if pt.consensus > bestConsensus {
                frontier.insert(pt.id)
                bestConsensus = pt.consensus
            }
        }
        return frontier
    }

    private nonisolated static func computeDriftTop(
        summary: AnalyticsSummary,
        yearRange: ClosedRange<Int>?
    ) -> [(cluster: Int, year: Int, drift: Double)] {
        let rows: [AnalyticsSummary.DriftEntry] = {
            guard let yearRange else { return summary.drift }
            return summary.drift.filter { yearRange.contains($0.year) }
        }()
        return rows
            .sorted { $0.drift > $1.drift }
            .prefix(5)
            .map { (cluster: $0.clusterID, year: $0.year, drift: $0.drift) }
    }

    private nonisolated static func computeFactorExposures(
        summary: AnalyticsSummary,
        focusMyExposure: Bool,
        yearRange: ClosedRange<Int>?
    ) -> [AnalyticsSummary.FactorExposure] {
        let base: [AnalyticsSummary.FactorExposure] = {
            if focusMyExposure, !summary.userFactorExposures.isEmpty {
                return summary.userFactorExposures
            }
            return summary.factorExposures
        }()
        guard let yearRange else { return base }
        return base.filter { yearRange.contains($0.year) }
    }

    private nonisolated static func computeInfluenceTop(
        summary: AnalyticsSummary,
        papers: [Paper],
        yearRange: ClosedRange<Int>?
    ) -> [(paper: Paper, score: Double)] {
        let map = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        var items: [(Paper, Double)] = []
        items.reserveCapacity(min(summary.influence.count, 64))
        for entry in summary.influence {
            guard let paper = map[entry.paperID] else { continue }
            if let yearRange {
                guard let y = paper.year, yearRange.contains(y) else { continue }
            }
            items.append((paper, entry.influence))
        }
        return items
            .sorted { $0.1 > $1.1 }
            .prefix(5)
            .map { (paper: $0.0, score: $0.1) }
    }

    private nonisolated static func computeInfluenceTimeline(
        summary: AnalyticsSummary,
        papers: [Paper],
        yearRange: ClosedRange<Int>?
    ) -> [(paper: Paper, score: Double, year: Int)] {
        let map = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        var rows: [(paper: Paper, score: Double, year: Int)] = []
        rows.reserveCapacity(min(summary.influence.count, 64))
        for entry in summary.influence {
            guard let paper = map[entry.paperID], let year = paper.year else { continue }
            if let yearRange, !yearRange.contains(year) { continue }
            rows.append((paper: paper, score: entry.influence, year: year))
        }
        let top = rows
            .sorted { ($0.year, -$0.score) < ($1.year, -$1.score) }
            .prefix(10)
        return Array(top)
    }

    private nonisolated static func computeIdeaRiver(
        summary: AnalyticsSummary,
        papers: [Paper],
        yearRange: ClosedRange<Int>?
    ) -> [Paper] {
        guard !summary.ideaFlowEdges.isEmpty else { return [] }
        let paperMap = Dictionary(uniqueKeysWithValues: papers.map { ($0.id, $0) })
        let influenceMap = Dictionary(uniqueKeysWithValues: summary.influence.map { ($0.paperID, $0.influence) })
        let filteredInfluence: [UUID: Double] = {
            guard let yearRange else { return influenceMap }
            return influenceMap.filter { id, _ in
                guard let y = paperMap[id]?.year else { return false }
                return yearRange.contains(y)
            }
        }()
        guard let startID = filteredInfluence.max(by: { $0.value < $1.value })?.key,
              let startPaper = paperMap[startID] else { return [] }

        var path: [Paper] = [startPaper]
        var current = startID
        var visited: Set<UUID> = [startID]
        let outEdges = Dictionary(grouping: summary.ideaFlowEdges) { $0.src }

        for _ in 0..<5 {
            guard let options = outEdges[current], !options.isEmpty else { break }
            let next = options
                .filter { edge in
                    guard let dst = edge.dst, !visited.contains(dst) else { return false }
                    guard let yearRange else { return true }
                    guard let y = paperMap[dst]?.year else { return false }
                    return yearRange.contains(y)
                }
                .max(by: { ($0.weight ?? 0) < ($1.weight ?? 0) })
            guard let dst = next?.dst, let paper = paperMap[dst] else { break }
            path.append(paper)
            visited.insert(dst)
            current = dst
        }
        return path
    }

    // MARK: Derived analytics (Python backend)

    private var analyticsSummary: AnalyticsSummary? { model.analyticsSummary }

    private var recommendationPapers: [Paper] {
        guard let ids = analyticsSummary?.recommendations else { return [] }
        let map = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0) })
        return ids.compactMap { map[$0] }
    }

    private var noveltyConsensus: [NoveltyConsensusPoint] {
        cachedNoveltyConsensus
    }

    private var driftTop: [(cluster: Int, year: Int, drift: Double)] {
        cachedDriftTop
    }

    private var selectedNoveltyPoint: NoveltyConsensusPoint? {
        guard let id = selectedNoveltyPointID else { return nil }
        return noveltyConsensus.first(where: { $0.id == id })
    }

    private var selectedNoveltyPaperCandidate: Paper? {
        guard let id = selectedNoveltyPointID else { return nil }
        return model.papers.first(where: { $0.id == id })
    }

    private var tradingLensPoints: [TradingLensPoint] {
        guard let trading = analyticsSummary?.trading,
              trading.available == true,
              let points = trading.scorePoints,
              !points.isEmpty else { return [] }

        let titleMap = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0.title) })
        let yearMap = Dictionary(uniqueKeysWithValues: model.papers.compactMap { paper in
            paper.year.map { (paper.id, $0) }
        })
        let yearRange = effectiveYearRange

        return points.compactMap { pt in
            if let yearRange {
                guard let y = pt.year ?? yearMap[pt.paperID], yearRange.contains(y) else { return nil }
            }
            guard let title = titleMap[pt.paperID] else { return nil }
            return TradingLensPoint(
                id: pt.paperID,
                novelty: pt.novelty,
                usability: pt.usability,
                strategyImpact: pt.strategyImpact ?? 0,
                confidence: pt.confidence ?? 0.0,
                priority: pt.priority ?? 0.0,
                title: title,
                primaryTag: pt.primaryTag,
                primaryAssetClass: pt.primaryAssetClass,
                primaryHorizon: pt.primaryHorizon
            )
        }
    }

    private var selectedTradingPoint: TradingLensPoint? {
        guard let id = selectedTradingPointID else { return nil }
        return tradingLensPoints.first(where: { $0.id == id })
    }

    private var selectedTradingPaperCandidate: Paper? {
        guard let id = selectedTradingPointID else { return nil }
        return model.papers.first(where: { $0.id == id })
    }

    private var driftVolatilityStats: [AnalyticsSummary.DriftVolatility] {
        analyticsSummary?.driftVolatility ?? []
    }

    private var noveltyFrontier: Set<UUID> {
        cachedNoveltyFrontier
    }

    private var factorExposures: [AnalyticsSummary.FactorExposure] { cachedFactorExposures }

    private var influenceTop: [(paper: Paper, score: Double)] { cachedInfluenceTop }

    private var influenceTimeline: [(paper: Paper, score: Double, year: Int)] { cachedInfluenceTimeline }

    private var ideaRiver: [Paper] { cachedIdeaRiver }

    private var debateLeftPaper: Paper? {
        debateLeftPaperID.flatMap { id in model.papers.first(where: { $0.id == id }) }
    }

    private var debateRightPaper: Paper? {
        debateRightPaperID.flatMap { id in model.papers.first(where: { $0.id == id }) }
    }

    private var debateLeftMatches: [Paper] {
        searchPapers(query: debateLeftPaperQuery)
    }

    private var debateRightMatches: [Paper] {
        searchPapers(query: debateRightPaperQuery)
    }

    private func searchPapers(query: String, limit: Int = 12) -> [Paper] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            let suggested = influenceTop.map(\.paper)
            if !suggested.isEmpty { return Array(suggested.prefix(limit)) }
            return Array(model.papers.prefix(limit))
        }

        let needle = trimmed.lowercased()
        return Array(
            model.papers
                .filter { $0.title.lowercased().contains(needle) || $0.originalFilename.lowercased().contains(needle) }
                .prefix(limit)
        )
    }

    private func closestPapersForUncertainty() -> [Paper] {
        // pick high centrality but not recommended yet
        guard let summary = analyticsSummary else { return [] }
        let recSet = Set(summary.recommendations)
        let sorted = summary.centrality.sorted { $0.weightedDegree > $1.weightedDegree }
        let map = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0) })
        return sorted
            .compactMap { map[$0.paperID] }
            .filter { !recSet.contains($0.id) }
            .prefix(5)
            .map { $0 }
    }

    private func filteredFactorExposures() -> [AnalyticsSummary.FactorExposure] {
        guard let summary = analyticsSummary else { return [] }
        let base: [AnalyticsSummary.FactorExposure] = {
            if focusMyExposure, !summary.userFactorExposures.isEmpty {
                return summary.userFactorExposures
            }
            return summary.factorExposures
        }()
        guard let yearRange = effectiveYearRange else { return base }
        return base.filter { yearRange.contains($0.year) }
    }

    private var counterfactualStats: (paperCount: Int, avgCentrality: Double) {
        guard let summary = analyticsSummary else {
            return (model.papers.count, summaryCentralityAverage(analyticsSummary))
        }
        guard let sel = selectedCounterfactual,
              let scenario = summary.counterfactuals.first(where: { $0.name == sel }) else {
            let vals = summary.counterfactuals.first
            return (vals?.paperCount ?? model.papers.count, vals?.avgCentrality ?? summaryCentralityAverage(summary))
        }
        return (scenario.paperCount, scenario.avgCentrality)
    }

    private func summaryCentralityAverage(_ summary: AnalyticsSummary?) -> Double {
        guard let summary else { return 0 }
        let vals = summary.centrality.map { $0.weightedDegree }
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    private func factorLabel(for idx: Int) -> String {
        if let labels = analyticsSummary?.factorLabels, idx < labels.count {
            return labels[idx]
        }
        return "F\(idx)"
    }

    @ViewBuilder private func noveltyCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Novelty vs consensus").font(.headline)
                let points = noveltyConsensus
                HStack {
                    Picker("Mode", selection: $noveltyMode) {
                        ForEach(NoveltyMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    Picker("Cluster", selection: Binding(
                        get: { selectedClusterFilter },
                        set: { selectedClusterFilter = $0 }
                    )) {
                        Text("All clusters").tag(Optional<Int>(nil))
                        ForEach(model.clusters, id: \.id) { c in
                            Text(c.name.prefix(16)).tag(Optional<Int>(c.id))
                        }
                    }
                    .pickerStyle(.menu)
                }
                Toggle("Highlight Pareto frontier", isOn: $showFrontier)
                    .font(.caption)
                    .toggleStyle(.switch)
                NoveltyConsensusChartView(
                    points: points,
                    frontier: noveltyFrontier,
                    showFrontier: showFrontier,
                    selectedPointID: $selectedNoveltyPointID
                )
                .frame(height: chartHeight(analyticsContentWidth, ratio: 0.52, min: 540, max: 920))
                Text("Quadrants: high/high = anchors; high novelty/low consensus = hidden gems. Pareto frontier shows best trade-offs.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if let pt = selectedNoveltyPoint {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(pt.title)
                                .font(.caption.bold())
                            Text(String(format: "Novelty %.2f | Consensus %.2f", pt.novelty, pt.consensus))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let paper = selectedNoveltyPaperCandidate {
                            Button {
                                selectedPaperDetail = paper
                            } label: {
                                Label("Open details", systemImage: "doc.text.magnifyingglass")
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        Button {
                            selectedNoveltyPointID = nil
                        } label: {
                            Label("Clear", systemImage: "xmark")
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
        }
    }

    private func tradingLensCard(_ summary: AnalyticsSummary) -> some View {
        Group {
            if let trading = summary.trading {
                GlassCard {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .firstTextBaseline) {
                            Text("Trading lens").font(.headline)
                            Spacer()
                            if let count = trading.paperCountWithLens {
                                let pct = trading.coveragePct ?? (Double(count) / Double(max(1, summary.paperCount)))
                                Text(String(format: "Coverage %d / %d (%.0f%%)", count, summary.paperCount, pct * 100))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }

                        if trading.available != true {
                            Text(trading.reason ?? "Generate paper trading lens scorecards to populate this section.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            if !tradingLensPoints.isEmpty {
                                TradingLensScatterChartView(points: tradingLensPoints, selectedPointID: $selectedTradingPointID)
                                    .frame(height: chartHeight(analyticsContentWidth, ratio: 0.46, min: 420, max: 760))
                                Text("x=usability · y=novelty · color=strategy impact · opacity=confidence")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("No scored papers in the current filter.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }

                            if let pt = selectedTradingPoint {
                                HStack(alignment: .top, spacing: 10) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(pt.title)
                                            .font(.caption.bold())
                                        Text(String(format: "Novelty %.1f · Usability %.1f · Impact %.1f · Priority %.1f", pt.novelty, pt.usability, pt.strategyImpact, pt.priority))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        if let tag = pt.primaryTag, tag != "Unknown" {
                                            Text("Tag: \(tag)")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    Spacer()
                                    if let paper = selectedTradingPaperCandidate {
                                        Button {
                                            selectedPaperDetail = paper
                                        } label: {
                                            Label("Open details", systemImage: "doc.text.magnifyingglass")
                                        }
                                        .buttonStyle(.borderedProminent)
                                    }
                                    Button {
                                        selectedTradingPointID = nil
                                    } label: {
                                        Label("Clear", systemImage: "xmark")
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }

                            if let tagCounts = trading.tagCounts, !tagCounts.isEmpty {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Top trading tags").font(.subheadline.weight(.semibold))
                                    TradingTagBarChartView(
                                        counts: tagCounts.prefix(14).map { TradingTagCount(tag: $0.tag, count: $0.count) }
                                    )
                                        .frame(height: chartHeight(analyticsContentWidth, ratio: 0.3, min: 300, max: 480))
                                }
                            }

                            if let trends = trading.tagTrends, !trends.isEmpty {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Tag frequency over time").font(.subheadline.weight(.semibold))
                                    TradingTagTrendChartView(
                                        trends: trends.map { TradingTagTrendPoint(tag: $0.tag, year: $0.year, count: $0.count) },
                                        domain: chartYearDomain ?? 1900...Calendar.current.component(.year, from: Date())
                                    )
                                    .frame(height: chartHeight(analyticsContentWidth, ratio: 0.28, min: 260, max: 440))
                                }
                            }

                            if let top = trading.topPriority, !top.isEmpty {
                                let paperMap = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0) })
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Top priority (impact×usability×confidence)").font(.subheadline.weight(.semibold))
                                    ForEach(Array(top.prefix(8)), id: \.paperID) { entry in
                                        let label = paperMap[entry.paperID]?.title ?? "Paper"
                                        HStack(alignment: .firstTextBaseline) {
                                            Button {
                                                if let paper = paperMap[entry.paperID] {
                                                    selectedPaperDetail = paper
                                                }
                                            } label: {
                                                Text(label)
                                                    .font(.caption)
                                                    .lineLimit(2)
                                            }
                                            .buttonStyle(.plain)
                                            Spacer()
                                            Text(String(format: "%.1f", entry.priority ?? 0.0))
                                                .font(.caption2.monospacedDigit())
                                                .foregroundStyle(.secondary)
                                        }
                                        if let verdict = entry.oneLineVerdict, !verdict.isEmpty {
                                            Text(verdict)
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder private func driftCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Topic drift (semantic motion)").font(.headline)
                ForEach(driftTop, id: \.cluster) { entry in
                    Text("Cluster \(entry.cluster) drifted \(String(format: "%.3f", entry.drift)) in \(entry.year)")
                        .font(.caption)
                }
                if !driftVolatilityStats.isEmpty {
                    Text("Volatility (std of yearly drift): " + driftVolatilityStats.map { "C\($0.clusterID)=\(String(format: "%.3f", $0.volatility))" }.joined(separator: " · "))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func factorCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Factor model over time").font(.headline)
                Toggle("Focus on my attention (reads)", isOn: $focusMyExposure)
                    .font(.caption)
                    .toggleStyle(.switch)
                FactorExposureChart(
                    exposures: factorExposures,
                    labels: summary.factorLabels,
                    domain: chartYearDomain ?? 1900...Calendar.current.component(.year, from: Date()),
                    height: chartHeight(analyticsContentWidth, ratio: 0.62, min: 620, max: 960)
                )
                Text("Factors from embeddings + tags (hybrid PCA/NMF). Stacked area shows topic/method mix by year.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if !summary.factorLabels.isEmpty {
                    Text("Factor labels: " + summary.factorLabels.enumerated().map { "F\($0.offset): \($0.element)" }.joined(separator: " · "))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func factorChart() -> some View {
        let rows = factorExposures
        Chart {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                AreaMark(
                    x: .value("Year", row.year),
                    y: .value("Exposure", row.score)
                )
                .foregroundStyle(by: .value("Factor", factorLabel(for: row.factor)))
                .interpolationMethod(.catmullRom)
            }
        }
        .chartXScale(domain: chartYearDomain ?? 1900...Calendar.current.component(.year, from: Date()))
        .frame(height: 520)
        .chartLegend(position: .bottom)
        .animation(.easeInOut(duration: 0.4), value: rows.count)
        .animation(.easeInOut(duration: 0.4), value: focusMyExposure)
    }

    @ViewBuilder private func influenceCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Influential ideas").font(.headline)
                if influenceTop.isEmpty {
                    Text("No influence results yet.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 10) {
                        ForEach(Array(influenceTop.enumerated()), id: \.element.paper.id) { idx, item in
                            PaperActionRow(
                                paper: item.paper,
                                subtitle: item.paper.tradingLens?.oneLineVerdict?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                                    ? item.paper.tradingLens?.oneLineVerdict
                                    : item.paper.summary,
                                leadingBadge: "#\(idx + 1)",
                                trailingPill: String(format: "%.3f", item.score),
                                trailingPillTint: .teal,
                                onOpen: { selectedPaperDetail = item.paper }
                            )
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder private func influenceTimelineCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Idea flow timeline").font(.headline)
                InfluenceTimelineChart(
                    items: influenceTimeline,
                    domain: chartYearDomain ?? 1900...Calendar.current.component(.year, from: Date()),
                    height: chartHeight(analyticsContentWidth, ratio: 0.4, min: 440, max: 760),
                    onOpenPaper: { paper in
                        selectedPaperDetail = paper
                    }
                )
                Text("Directed edges from older → newer similar claims; PageRank highlights semantic influence (not citations).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func ideaRiverCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Idea river (greedy storyline)").font(.headline)
                if ideaRiver.isEmpty {
                    Text("No river computed yet.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 10) {
                        ForEach(Array(ideaRiver.enumerated()), id: \.element.id) { idx, paper in
                            PaperActionRow(
                                paper: paper,
                                subtitle: paper.tradingLens?.oneLineVerdict?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                                    ? paper.tradingLens?.oneLineVerdict
                                    : paper.summary,
                                leadingBadge: "\(idx + 1)",
                                trailingPill: paper.year.map(String.init),
                                trailingPillTint: .mint,
                                onOpen: { selectedPaperDetail = paper }
                            )
                            if idx < ideaRiver.count - 1 {
                                Image(systemName: "arrow.down.right")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Text("Start = most influential, then follow strongest downstream edge (older → newer claims).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func counterfactualCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Counterfactual corpus").font(.headline)
                let scenarios = summary.counterfactuals
                if !scenarios.isEmpty {
                    Picker("Scenario", selection: Binding(
                        get: { selectedCounterfactual ?? scenarios.first?.name },
                        set: { selectedCounterfactual = $0 }
                    )) {
                        ForEach(scenarios, id: \.name) { sc in
                            Text(sc.name).tag(Optional(sc.name))
                        }
                    }
                    .pickerStyle(.menu)
                    HStack {
                        TextField("Custom cutoffs (e.g., 2010 2015 2020)", text: Binding(
                            get: { customCutoffs },
                            set: { customCutoffs = $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                        Button("Recompute") {
                            let nums = customCutoffs.split(separator: " ").compactMap { Int($0) }
                            model.rebuildAnalyticsWithCutoffs(nums.isEmpty ? [2010, 2015, 2020] : nums)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.analyticsRebuildInFlight)
                    }
                    let stats = counterfactualStats
                    Text("Papers kept: \(stats.paperCount)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(String(format: "Avg centrality: %.3f", stats.avgCentrality))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("Scenarios computed in Python (rebuild analytics to refresh).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Run analytics rebuild to compute counterfactual scenarios.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func recommendationCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Personal recommendations & confidence").font(.headline)
                if !recommendationPapers.isEmpty {
                    let migByID: [UUID: Double] = {
                        guard let rows = summary.workflow?.recommendationsMIG?.selected else { return [:] }
                        return Dictionary(uniqueKeysWithValues: rows.map { ($0.paperID, $0.marginalGain ?? 0) })
                    }()
                    Text(summary.workflow?.recommendationsMIG?.available == true ? "Marginal information gain picks:" : "Recommendations:")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(recommendationPapers.prefix(5)) { paper in
                        HStack {
                            Text("• \(paper.title)")
                                .font(.caption2)
                            Spacer()
                            if let gain = migByID[paper.id], gain > 0 {
                                Text(String(format: "+%.2f", gain))
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            Button {
                                model.recordRecommendationFeedback(paperID: paper.id, helpful: true)
                            } label: { Image(systemName: "hand.thumbsup") }
                                .buttonStyle(.borderless)
                            Button {
                                model.recordRecommendationFeedback(paperID: paper.id, helpful: false)
                            } label: { Image(systemName: "hand.thumbsdown") }
                                .buttonStyle(.borderless)
                        }
                    }
                }
                if let conf = summary.answerConfidence {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(String(format: "QA confidence %.0f%%", conf * 100))
                            .font(.caption)
                        ProgressView(value: conf)
                            .tint(conf > 0.7 ? .green : .orange)
                        if conf < 0.6 {
                            Text("Low confidence — consider adding nearby papers:")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            ForEach(closestPapersForUncertainty().prefix(3), id: \.id) { paper in
                                Text("• \(paper.title)")
                                    .font(.caption2)
                            }
                        }
                    }
                }
            }
        }
    }

    private func clusterLabel(_ id: Int) -> String {
        model.clusters.first(where: { $0.id == id })?.name ?? "Cluster \(id)"
    }

    private func paperLabel(_ id: UUID) -> String {
        model.papers.first(where: { $0.id == id })?.title ?? id.uuidString
    }

    @ViewBuilder private func qualityAndStabilityCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Quality & stability").font(.headline)

                if let map = summary.quality?.map, map.available == true {
                    let tw = map.trustworthiness ?? 0
                    let cont = map.continuity ?? 0
                    let overlap = map.avgNeighborOverlap ?? 0
                    Text(String(format: "Map fidelity (clusters): trust %.2f · cont %.2f · overlap %.2f", tw, cont, overlap))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Map fidelity: Not available (run clustering/map export).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let paperMap = summary.quality?.paperMap, paperMap.available == true {
                    let tw = paperMap.trustworthiness ?? 0
                    let cont = paperMap.continuity ?? 0
                    let overlap = paperMap.avgNeighborOverlap ?? 0
                    let p95 = paperMap.distortionQuantiles?.p95 ?? 0
                    Text(String(format: "Map fidelity (papers): trust %.2f · cont %.2f · overlap %.2f · p95 dist %.2f", tw, cont, overlap, p95))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let ing = summary.quality?.ingestion, ing.available == true {
                    let issues = ing.issues?.count ?? 0
                    let missingYear = ing.missingYear ?? 0
                    let lowText = ing.lowTextYield ?? 0
                    Text("Ingestion: \(issues) issues · missing year \(missingYear) · low text yield \(lowText)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if summary.stability?.available == true {
                    let mostAmbiguous = summary.paperMetrics
                        .sorted { ($0.clusterAmbiguity ?? 0) > ($1.clusterAmbiguity ?? 0) }
                        .prefix(5)
                    if let first = mostAmbiguous.first, (first.clusterAmbiguity ?? 0) > 0.1 {
                        Text("Boundary papers (high ambiguity):")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ForEach(Array(mostAmbiguous), id: \.paperID) { row in
                            let conf = row.clusterConfidence ?? 0
                            let amb = row.clusterAmbiguity ?? 0
                            Text(String(format: "• %@ (conf %.2f, amb %.2f)", paperLabel(row.paperID), conf, amb))
                                .font(.caption2)
                        }
                    }
                }

                let mostDistorted = summary.paperMetrics
                    .sorted { ($0.paperLayoutDistortion ?? 0) > ($1.paperLayoutDistortion ?? 0) }
                    .prefix(5)
                if let first = mostDistorted.first, (first.paperLayoutDistortion ?? 0) > 0.25 {
                    Text("Layout distortion (papers):")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(Array(mostDistorted), id: \.paperID) { row in
                        Text(String(format: "• %@ (dist %.2f)", paperLabel(row.paperID), row.paperLayoutDistortion ?? 0))
                            .font(.caption2)
                    }
                }
            }
        }
    }

    @ViewBuilder private func claimsAndEvidenceCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Claims & evidence").font(.headline)

                if let hot = summary.claims?.controversyByCluster, !hot.isEmpty {
                    Text("Debate hotspots (contradictions):")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(Array(hot.prefix(6)), id: \.clusterID) { row in
                        Text(String(format: "• %@ (rate %.2f)", clusterLabel(row.clusterID), row.contradictionRate ?? 0))
                            .font(.caption2)
                    }
                } else {
                    Text("No claim graph controversy yet — ingest papers with extracted claims and rebuild analytics.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let gaps = summary.claims?.stressTestGaps, gaps.available == true,
                   let perCluster = gaps.perCluster, let top = perCluster.first {
                    let tests = top.topSuggestedTests?.prefix(3) ?? []
                    if !tests.isEmpty {
                        Text("Experiment gaps (from stress tests):")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Top cluster: \(clusterLabel(top.clusterID))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        ForEach(Array(tests), id: \.text) { t in
                            Text("• \(t.text)")
                                .font(.caption2)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder private func lifecycleAndBridgesCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Lifecycle & bridges").font(.headline)

                if let perCluster = summary.lifecycle?.perCluster, !perCluster.isEmpty {
                    let emerging = perCluster.filter { ($0.phase ?? "") == "emerging" || ($0.phase ?? "") == "accelerating" }
                        .prefix(6)
                    if !emerging.isEmpty {
                        Text("Emerging / accelerating topics:")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ForEach(Array(emerging), id: \.clusterID) { row in
                            Text("• \(clusterLabel(row.clusterID)) (\(row.phase ?? "unknown"))")
                                .font(.caption2)
                        }
                    }
                }

                if let topicMetrics = summary.bridges?.topicMetrics, !topicMetrics.isEmpty {
                    Text("Bridge topics:")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(Array(topicMetrics.prefix(5)), id: \.clusterID) { row in
                        Text(String(format: "• %@ (bridge %.3f, constraint %.2f)", clusterLabel(row.clusterID), row.bridgingCentrality ?? 0, row.constraint ?? 0))
                            .font(.caption2)
                    }
                }

                let bridgePapers = summary.paperMetrics
                    .sorted { ($0.recombination ?? 0) > ($1.recombination ?? 0) }
                    .prefix(5)
                if let first = bridgePapers.first, (first.recombination ?? 0) > 0.1 {
                    Text("Bridge papers (high recombination):")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(Array(bridgePapers), id: \.paperID) { row in
                        Text(String(format: "• %@ (%.2f)", paperLabel(row.paperID), row.recombination ?? 0))
                            .font(.caption2)
                    }
                }
            }
        }
    }

    @ViewBuilder private func workflowAndHygieneCard(_ summary: AnalyticsSummary) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Workflow & hygiene").font(.headline)

                if let blindspots = summary.workflow?.coverage?.blindspots, !blindspots.isEmpty {
                    Text("Blind spots (high interest, low coverage):")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(Array(blindspots.prefix(6)), id: \.clusterID) { b in
                        Text(String(format: "• %@ (score %.2f)", clusterLabel(b.clusterID), b.blindspotScore ?? 0))
                            .font(.caption2)
                    }
                }

                if let questions = summary.workflow?.qaGaps?.questions {
                    let unresolved = questions.filter { $0.unanswered == true }.prefix(3)
                    if !unresolved.isEmpty {
                        Text("Unanswered questions:")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ForEach(Array(unresolved), id: \.question) { q in
                            Text("• \(q.question)")
                                .font(.caption2)
                        }
                    }
                }

                let dupGroups = summary.hygiene?.duplicates?.groups?.count ?? 0
                let ingestionIssues = summary.quality?.ingestion?.issues?.count ?? 0
                if dupGroups > 0 || ingestionIssues > 0 {
                    Text("Cleanup: \(dupGroups) duplicate groups · \(ingestionIssues) ingestion issues")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func summarySection(_ summary: AnalyticsSummary) -> some View {
        if !noveltyConsensus.isEmpty { noveltyCard() }
        if summary.trading != nil { tradingLensCard(summary) }
        if !driftTop.isEmpty { driftCard() }
        if !factorExposures.isEmpty { factorCard(summary) }
        if !influenceTop.isEmpty { influenceCard() }
        if !influenceTimeline.isEmpty { influenceTimelineCard() }
        if !ideaRiver.isEmpty { ideaRiverCard() }
        qualityAndStabilityCard(summary)
        lifecycleAndBridgesCard(summary)
        claimsAndEvidenceCard(summary)
        workflowAndHygieneCard(summary)
        counterfactualCard(summary)
        if !recommendationPapers.isEmpty || summary.answerConfidence != nil {
            recommendationCard(summary)
        }
    }

    @ViewBuilder private func timelineSection() -> some View {
        if let domain = corpusYearDomain, !timeline.isEmpty {
            TimelineCard(
                domain: domain,
                data: timeline,
                yearRangeLabel: yearRangeLabel,
                width: analyticsContentWidth,
                yearFilterEnabled: $model.yearFilterEnabled,
                yearFilterStart: $model.yearFilterStart,
                yearFilterEnd: $model.yearFilterEnd,
                hoveredYear: $hoveredTimelineYear,
                hoveredCount: $hoveredTimelineCount,
                brushAnchor: $yearBrushAnchor
            )
            .task(id: "\(domain.lowerBound)-\(domain.upperBound)") {
                if model.yearFilterStart == 0 { model.yearFilterStart = domain.lowerBound }
                if model.yearFilterEnd == 0 { model.yearFilterEnd = domain.upperBound }
                model.yearFilterStart = min(max(model.yearFilterStart, domain.lowerBound), domain.upperBound)
                model.yearFilterEnd = min(max(model.yearFilterEnd, domain.lowerBound), domain.upperBound)
            }
        } else {
            GlassCard {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Timeline")
                        .font(.title3.weight(.semibold))
                    Text("No dated papers yet — add publication years to enable the timeline.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func corpusBriefingCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Corpus briefing")
                        .font(.headline)
                    Spacer()
                    Button {
                        model.generateCorpusBriefing()
                    } label: {
                        if model.isGeneratingBriefing {
                            ProgressView().controlSize(.small)
                            Text("Synthesizing…")
                        } else {
                            Label(model.corpusBriefing.isEmpty ? "Generate" : "Refresh", systemImage: "wand.and.stars")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isGeneratingBriefing || model.megaClusters.isEmpty)
                    .contextMenu {
                        Button("Force regenerate") {
                            model.generateCorpusBriefing(force: true)
                        }
                    }
                }

                if model.megaClusters.isEmpty {
                    Text("Run clustering to generate a topic hierarchy first.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if let error = model.corpusBriefingError {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }

                if model.corpusBriefing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text("Create an executive summary of your entire corpus from the current topic hierarchy.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    ScrollView {
                        Text(model.corpusBriefing)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 260)
                }
            }
        }
    }

    @ViewBuilder private func backendAnalyticsCard() -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Backend analytics (DuckDB / Python)")
                    .font(.headline)
                HStack {
                    Button("Reload analytics.json") {
                        model.reloadAnalyticsSummary()
                    }
                    .buttonStyle(.borderedProminent)
                    Text("Reads Output/analytics/analytics.json")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
#if os(macOS)
                    Spacer()
                    Button {
                        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                        let folder = cwd.appendingPathComponent("Output", isDirectory: true)
                            .appendingPathComponent("analytics", isDirectory: true)
                        PlatformOpen.revealInFinder(url: folder)
                    } label: {
                        Label("Open folder", systemImage: "folder")
                    }
                    .buttonStyle(.bordered)
#endif
                }
#if os(macOS)
                HStack {
                    Button {
                        model.rebuildAnalyticsViaPython()
                    } label: {
                        if model.analyticsRebuildInFlight {
                            ProgressView().controlSize(.small)
                            Text("Rebuilding…")
                        } else {
                            Label("Rebuild via Python", systemImage: "hammer.fill")
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.analyticsRebuildInFlight)
                    if let msg = model.analyticsRebuildMessage {
                        Text(msg)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Button {
                        model.installAnalyticsPythonDependencies()
                    } label: {
                        if model.analyticsDepsInstallInFlight {
                            ProgressView().controlSize(.small)
                            Text("Installing…")
                        } else {
                            Label("Install Python deps", systemImage: "square.and.arrow.down")
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.analyticsDepsInstallInFlight || model.analyticsRebuildInFlight)
                    if let msg = model.analyticsDepsInstallMessage {
                        Text(msg)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                if let output = model.analyticsDepsInstallOutput,
                   !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    InteractiveLogPanel(
                        title: "Last install log",
                        text: Binding(
                            get: { model.analyticsDepsInstallOutput ?? "" },
                            set: { newValue in
                                let trimmed = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
                                model.analyticsDepsInstallOutput = trimmed.isEmpty ? nil : newValue
                            }
                        ),
                        minHeight: 160
                    )
                }
                if let output = model.analyticsRebuildOutput,
                   !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    InteractiveLogPanel(
                        title: "Last rebuild log",
                        text: Binding(
                            get: { model.analyticsRebuildOutput ?? "" },
                            set: { newValue in
                                let trimmed = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
                                model.analyticsRebuildOutput = trimmed.isEmpty ? nil : newValue
                            }
                        ),
                        minHeight: 180
                    )
                }
#endif
                if let summary = model.analyticsSummary {
                    Text("Generated at \(formatDate(summary.generatedAt)) · \(summary.paperCount) papers · dim \(summary.vectorDim)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if let topNovel = summary.novelty.first,
                       let paper = model.papers.first(where: { $0.id == topNovel.paperID }) {
                        Text("Top outlier: \(paper.title)")
                            .font(.subheadline)
                        Text(String(format: "Novelty %.3f (cluster %@)", topNovel.novelty, topNovel.clusterID.map(String.init) ?? "n/a"))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    if !summary.topicTrends.isEmpty {
                        let cluster = summary.topicTrends.first?.clusterID ?? 0
                        let years = summary.topicTrends.filter { $0.clusterID == cluster }.map { $0.year }.sorted()
                        if let first = years.first, let last = years.last {
                            Text("Sample trend: cluster \(cluster) covers \(first)–\(last)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Analytics")
                        .font(.title.bold())
                    Text("Position (novelty/consensus) + Flow (influence/drift) across your corpus.")
                        .foregroundStyle(.secondary)

                    corpusBriefingCard()

                    backendAnalyticsCard()

                    timelineSection()

                    // Position block
                    Text("Position · novelty vs consensus")
                        .font(.title3.weight(.semibold))
                    Divider().opacity(0.35)

                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("What do I already know about…")
                                .font(.headline)
                            HStack {
                                TextField("Enter a topic", text: $topicQuery)
                                    .textFieldStyle(.roundedBorder)
                                Button("Analyze") {
                                    snapshot = model.knowledgeSnapshot(for: topicQuery)
                                }
                                .buttonStyle(.borderedProminent)
                                .disabled(topicQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            }
                            if let snap = snapshot {
                                Text("You know (\(snap.known.count))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                ForEach(snap.known) { paper in
                                    Text("• \(paper.title)")
                                        .font(.caption2)
                                }
                                if !snap.missing.isEmpty {
                                    Divider().padding(.vertical, 4)
                                    Text("Missing (\(snap.missing.count))")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    ForEach(snap.missing) { paper in
                                        Text("• \(paper.title)")
                                            .font(.caption2)
                                    }
                                }
                                Text(snap.summary)
                                    .font(.caption2)
                                    .padding(.top, 4)
                            } else {
                                Text("We will summarize your read papers near that topic and surface gaps.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    if !topicStreams.isEmpty {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Topic evolution streams").font(.headline)
                                ForEach(topicStreams.prefix(3), id: \.clusterID) { stream in
                                    if !stream.countsByYear.isEmpty {
                                        TopicStreamChart(
                                            stream: stream,
                                            height: chartHeight(analyticsContentWidth, ratio: 0.34, min: 320, max: 560)
                                        )
                                        Text(stream.narrative)
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    } else {
                                        Text(stream.narrative)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }

                    if !methodSignals.isEmpty {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Method takeovers").font(.headline)
                                HStack {
                                    TextField("Methods (comma separated)", text: $methodTagsText)
                                        .textFieldStyle(.roundedBorder)
                                    Button("Refresh") {
                                        scheduleTemporalRecompute(delay: 0)
                                    }
                                    .buttonStyle(.bordered)
                                }
                                if let first = methodSignals.first {
                                    MethodCrossoverChart(signal: first, papers: papersForAnalytics)
                                        .frame(height: chartHeight(analyticsContentWidth, ratio: 0.34, min: 360, max: 680))
                                }
                                ForEach(methodSignals, id: \.a) { sig in
                                    let detail: String = {
                                        if let year = sig.crossingYear {
                                            return "crossed in \(year)"
                                        }
                                        return "no crossover yet"
                                    }()
                                    Text("\(sig.a) vs \(sig.b): \(detail)")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    if let summary = analyticsSummary {
                        summarySection(summary)
                    }

                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Reading lag & pace")
                                .font(.headline)
                            Text(String(format: "Average publication lag: %.1f years", readingStats.averageLagYears))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if !readingStats.overlay.isEmpty {
                                ReadingLagChart(points: readingStats.overlay)
                                    .frame(height: chartHeight(analyticsContentWidth, ratio: 0.4, min: 420, max: 860))
                            }
                            if !readingStats.realTimeClusterIDs.isEmpty {
                                let list = readingStats.realTimeClusterIDs.map(String.init).joined(separator: ", ")
                                Text("Real-time clusters: \(list)")
                                    .font(.caption2)
                            }
                            if !readingStats.lateClusterIDs.isEmpty {
                                let list = readingStats.lateClusterIDs.map(String.init).joined(separator: ", ")
                                Text("Lagged clusters: \(list)")
                                    .font(.caption2)
                            }
                        }
                    }

                    if let summary = analyticsSummary {
                        let titleMap = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0.title) })
                        if let outlier = summary.paperMetrics.max(by: { $0.zNovelty < $1.zNovelty }),
                           let dense = summary.paperMetrics.max(by: { $0.zConsensus < $1.zConsensus }) {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Novelty & saturation")
                                        .font(.headline)

                                    Text("Weird outlier: \(titleMap[outlier.paperID] ?? "Paper")")
                                        .font(.subheadline)
                                    Text(String(format: "z-novelty %.2f | global novelty %.2f", outlier.zNovelty, outlier.novGlobal))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)

                                    Text("Over-explored pocket: \(titleMap[dense.paperID] ?? "Paper")")
                                        .font(.subheadline)
                                    Text(String(format: "z-consensus %.2f | consensus total %.2f", dense.zConsensus, dense.consensusTotal))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        } else if let outlier = summary.novelty.max(by: { $0.novelty < $1.novelty }),
                                  let dense = summary.centrality.max(by: { $0.weightedDegree < $1.weightedDegree }) {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Novelty & saturation")
                                        .font(.headline)

                                    Text("Weird outlier: \(titleMap[outlier.paperID] ?? "Paper")")
                                        .font(.subheadline)
                                    Text(String(format: "Novelty %.2f", outlier.novelty))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)

                                    Text("Over-explored pocket: \(titleMap[dense.paperID] ?? "Paper")")
                                        .font(.subheadline)
                                    Text(String(format: "Weighted degree %.2f | Avg similarity %.2f", dense.weightedDegree, dense.averageSimilarity))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    if !model.clusters.isEmpty {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 10) {
                                let availability = SystemLanguageModel.default.availability
                                let aiAvailable: Bool = {
                                    switch availability {
                                    case .available: return true
                                    default: return false
                                    }
                                }()

                                HStack {
                                    Text("Debate lab").font(.headline)
                                    Spacer()
                                    Text(aiAvailable ? "AI available" : "AI unavailable")
                                        .font(.caption2.bold())
                                        .foregroundStyle(aiAvailable ? .green : .secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.white.opacity(0.06), in: Capsule())
                                }

                                Text("Heuristic uses a structured offline template from summaries. AI uses the on-device model to synthesize a multi-round debate (still simulated; not citations).")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)

                                Picker("Mode", selection: $debateMode) {
                                    ForEach(DebateMode.allCases) { mode in
                                        Text(mode.label).tag(mode)
                                    }
                                }
                                .pickerStyle(.segmented)

                                if debateMode == .panel {
                                    Picker("Topic", selection: Binding(
                                        get: { debateLeftClusterID ?? model.clusters.first?.id },
                                        set: { debateLeftClusterID = $0 }
                                    )) {
                                        ForEach(model.clusters, id: \.id) { cluster in
                                            Text(cluster.name).tag(Optional(cluster.id))
                                        }
                                    }
                                    .pickerStyle(.menu)

                                    Stepper("Speakers: \(panelSpeakerCount)", value: $panelSpeakerCount, in: 2...5)
                                        .font(.caption)

                                    Button {
                                        if let cid = debateLeftClusterID ?? model.clusters.first?.id {
                                            debateTranscript = model.simulateAuthorPanel(for: cid, maxSpeakers: panelSpeakerCount)
                                        }
                                    } label: {
                                        Label("Simulate panel (heuristic)", systemImage: "person.3.sequence")
                                    }
                                    .buttonStyle(.borderedProminent)
                                } else if debateMode == .topicVsTopic {
                                    HStack {
                                        Picker("A", selection: Binding(
                                            get: { debateLeftClusterID ?? model.clusters.first?.id },
                                            set: { debateLeftClusterID = $0 }
                                        )) {
                                            ForEach(model.clusters, id: \.id) { cluster in
                                                Text(cluster.name).tag(Optional(cluster.id))
                                            }
                                        }
                                        .pickerStyle(.menu)

                                        Picker("B", selection: Binding(
                                            get: { debateRightClusterID ?? model.clusters.dropFirst().first?.id },
                                            set: { debateRightClusterID = $0 }
                                        )) {
                                            ForEach(model.clusters, id: \.id) { cluster in
                                                Text(cluster.name).tag(Optional(cluster.id))
                                            }
                                        }
                                        .pickerStyle(.menu)
                                    }

                                    Stepper("Rounds: \(debateRounds) (steps: \(debateRounds * 2))", value: $debateRounds, in: 2...8)
                                        .font(.caption)

                                    Toggle("AI-assisted (on-device)", isOn: $debateUseAI)
                                        .font(.caption)
                                        .disabled(!aiAvailable)

                                    Button {
                                        guard !isGeneratingDebate else { return }
                                        guard let leftID = debateLeftClusterID ?? model.clusters.first?.id,
                                              let rightID = debateRightClusterID ?? model.clusters.dropFirst().first?.id else { return }
                                        isGeneratingDebate = true
                                        debateTranscript = "Generating…"
                                        Task {
                                            let text = await model.generateDebateBetweenClusters(firstID: leftID, secondID: rightID, rounds: debateRounds, useAI: debateUseAI)
                                            await MainActor.run {
                                                debateTranscript = text
                                                isGeneratingDebate = false
                                            }
                                        }
                                    } label: {
                                        Label(debateUseAI ? "Generate debate (AI)" : "Generate debate (heuristic)", systemImage: "sparkles")
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(isGeneratingDebate)
                                } else {
                                    VStack(alignment: .leading, spacing: 8) {
                                        HStack {
                                            TextField("Search paper A…", text: $debateLeftPaperQuery)
                                                .textFieldStyle(.roundedBorder)
                                            Picker("A", selection: Binding(
                                                get: { debateLeftPaperID ?? debateLeftMatches.first?.id },
                                                set: { debateLeftPaperID = $0 }
                                            )) {
                                                ForEach(debateLeftMatches, id: \.id) { paper in
                                                    Text(paper.title.prefix(46)).tag(Optional(paper.id))
                                                }
                                            }
                                            .pickerStyle(.menu)
                                        }

                                        HStack {
                                            TextField("Search paper B…", text: $debateRightPaperQuery)
                                                .textFieldStyle(.roundedBorder)
                                            Picker("B", selection: Binding(
                                                get: { debateRightPaperID ?? debateRightMatches.first?.id },
                                                set: { debateRightPaperID = $0 }
                                            )) {
                                                ForEach(debateRightMatches, id: \.id) { paper in
                                                    Text(paper.title.prefix(46)).tag(Optional(paper.id))
                                                }
                                            }
                                            .pickerStyle(.menu)
                                        }
                                    }

                                    Stepper("Rounds: \(debateRounds) (steps: \(debateRounds * 2))", value: $debateRounds, in: 2...8)
                                        .font(.caption)

                                    Toggle("AI-assisted (on-device)", isOn: $debateUseAI)
                                        .font(.caption)
                                        .disabled(!aiAvailable)

                                    Button {
                                        guard !isGeneratingDebate else { return }
                                        guard let leftID = debateLeftPaperID ?? debateLeftMatches.first?.id,
                                              let rightID = debateRightPaperID ?? debateRightMatches.first?.id else { return }
                                        isGeneratingDebate = true
                                        debateTranscript = "Generating…"
                                        Task {
                                            let text = await model.generateDebateBetweenPapers(firstID: leftID, secondID: rightID, rounds: debateRounds, useAI: debateUseAI)
                                            await MainActor.run {
                                                debateTranscript = text
                                                isGeneratingDebate = false
                                            }
                                        }
                                    } label: {
                                        Label(debateUseAI ? "Generate debate (AI)" : "Generate debate (heuristic)", systemImage: "sparkles")
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(isGeneratingDebate)
                                }

                                if isGeneratingDebate {
                                    HStack(spacing: 8) {
                                        ProgressView()
                                        Text("Working…")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }

                                if !debateTranscript.isEmpty {
                                    Divider().opacity(0.35)
                                    ScrollView {
                                        Text(debateTranscript)
                                            .font(.caption2)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .textSelection(.enabled)
                                    }
                                    .frame(maxHeight: 320)
                                }
                            }
                        }
                    }

                    if model.clusters.count >= 2 {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("What-if lab").font(.headline)
                                Button("Generate cross-cluster idea") {
                                    let ids = model.clusters.prefix(2).map { $0.id }
                                    if let ghost = model.generateWhatIfPaper(for: ids) {
                                        hypotheticalTitle = ghost.title
                                        hypotheticalAbstract = ghost.abstract
                                    }
                                }
                                .buttonStyle(.bordered)

                                if !hypotheticalTitle.isEmpty {
                                    Text(hypotheticalTitle).font(.subheadline.bold())
                                    Text(hypotheticalAbstract).font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Daily quiz").font(.headline)
                            let deck = model.dailyQuizCards()
                            Text(deck.isEmpty ? "No cards yet — generate flashcards on papers you've read." : "\(deck.count) cards ready from recent reads.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Button {
                                dailyQuizIndex = 0
                                dailyQuizReveal = false
                                showDailyQuiz = true
                            } label: {
                                Label("Start quiz", systemImage: "graduationcap")
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(deck.isEmpty)
                        }
                    }
                }
                .onWidthChange { width in
                    analyticsContentWidth = width
                }
                .task {
                    scheduleTimelineRecompute()
                    scheduleTemporalRecompute(delay: 0)
                    scheduleBackendRecompute(delay: 0)
                }
                .onReceive(model.$papers) { _ in
                    scheduleTimelineRecompute()
                    scheduleTemporalRecompute()
                    scheduleBackendRecompute()
                }
                .onReceive(model.$clusters) { _ in
                    scheduleTemporalRecompute()
                }
                .onReceive(model.$analyticsSummary) { _ in
                    scheduleBackendRecompute()
                }
                .onChange(of: methodTagsText) { _, _ in
                    scheduleTemporalRecompute()
                }
                .onChange(of: model.yearFilterEnabled) { _, _ in
                    scheduleTemporalRecompute()
                    scheduleBackendRecompute()
                }
                .onChange(of: model.yearFilterStart) { _, _ in
                    scheduleTemporalRecompute()
                    scheduleBackendRecompute()
                }
                .onChange(of: model.yearFilterEnd) { _, _ in
                    scheduleTemporalRecompute()
                    scheduleBackendRecompute()
                }
                .onChange(of: noveltyMode) { _, _ in
                    scheduleBackendRecompute()
                }
                .onChange(of: selectedClusterFilter) { _, _ in
                    scheduleBackendRecompute()
                }
                .onChange(of: focusMyExposure) { _, _ in
                    scheduleBackendRecompute()
                }
                .padding()
            }
            .navigationTitle("Analytics")
            .sheet(isPresented: $showDailyQuiz) {
                DailyQuizSheet(
                    deck: model.dailyQuizCards(),
                    index: $dailyQuizIndex,
                    reveal: $dailyQuizReveal,
                    onReveal: { card, paper in
                        model.markFlashcardReviewed(paperID: paper.id, cardID: card.id)
                    }
                )
            }
            #if os(iOS)
            .sheet(item: $selectedPaperDetail) { paper in
                NavigationStack {
                    PaperDetailView(paper: paper)
                        .environmentObject(model)
                        .toolbar {
                            ToolbarItem(placement: .primaryAction) {
                                Button("Close") { selectedPaperDetail = nil }
                            }
                        }
                }
                #if os(iOS)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
                #endif
            }
            #endif
        }
        #if os(macOS)
        .overlay {
            if let paper = selectedPaperDetail {
                DismissibleOverlay(onDismiss: { selectedPaperDetail = nil }) {
                    NavigationStack {
                        PaperDetailView(paper: paper, onClose: { selectedPaperDetail = nil })
                            .environmentObject(model)
                            .toolbar {
                                ToolbarItem(placement: .primaryAction) {
                                    Button {
                                        selectedPaperDetail = nil
                                    } label: {
                                        Label("Close", systemImage: "xmark")
                                    }
                                }
                            }
                    }
                    .frame(minWidth: 640, idealWidth: 920, maxWidth: 980, minHeight: 620, idealHeight: 820, maxHeight: 920)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                            .stroke(Color.white.opacity(0.12), lineWidth: 1)
                    )
                    .padding(24)
                }
                .zIndex(1000)
            }
        }
        #endif
    }
}

@available(macOS 26, iOS 26, *)
private struct TimelineCard: View {
    let domain: ClosedRange<Int>
    let data: [(year: Int, count: Int)]
    let yearRangeLabel: String
    let width: CGFloat

    @Binding var yearFilterEnabled: Bool
    @Binding var yearFilterStart: Int
    @Binding var yearFilterEnd: Int
    @Binding var hoveredYear: Int?
    @Binding var hoveredCount: Int?
    @Binding var brushAnchor: Int?

    private var effectiveRange: ClosedRange<Int> {
        guard yearFilterEnabled else { return domain }
        let start = min(max(yearFilterStart, domain.lowerBound), domain.upperBound)
        let end = min(max(yearFilterEnd, domain.lowerBound), domain.upperBound)
        return min(start, end)...max(start, end)
    }

    private var chartHeight: CGFloat {
        let minHeight: CGFloat = 320
        let maxHeight: CGFloat = 680
        guard width > 0 else { return minHeight }
        return min(max(width * 0.48, minHeight), maxHeight)
    }

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Timeline")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    Text(yearRangeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                TimelineBarChart(
                    domain: domain,
                    data: data,
                    selectedRange: effectiveRange,
                    height: chartHeight,
                    yearFilterEnabled: $yearFilterEnabled,
                    yearFilterStart: $yearFilterStart,
                    yearFilterEnd: $yearFilterEnd,
                    hoveredYear: $hoveredYear,
                    hoveredCount: $hoveredCount,
                    brushAnchor: $brushAnchor
                )

                TimelineYearControls(
                    domain: domain,
                    yearFilterEnabled: $yearFilterEnabled,
                    yearFilterStart: $yearFilterStart,
                    yearFilterEnd: $yearFilterEnd,
                    hoveredYear: $hoveredYear,
                    hoveredCount: $hoveredCount,
                    brushAnchor: $brushAnchor
                )
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct TimelineBarChart: View {
    let domain: ClosedRange<Int>
    let data: [(year: Int, count: Int)]
    let selectedRange: ClosedRange<Int>
    let height: CGFloat

    @Binding var yearFilterEnabled: Bool
    @Binding var yearFilterStart: Int
    @Binding var yearFilterEnd: Int
    @Binding var hoveredYear: Int?
    @Binding var hoveredCount: Int?
    @Binding var brushAnchor: Int?

    private var maxCount: Int { max(1, data.map(\.count).max() ?? 1) }

    var body: some View {
        Chart {
            ForEach(data, id: \.year) { entry in
                let inRange = selectedRange.contains(entry.year)
                BarMark(
                    x: .value("Year", entry.year),
                    y: .value("Papers", entry.count)
                )
                .foregroundStyle(inRange ? .cyan.opacity(0.85) : .cyan.opacity(0.18))
            }

            if let y = hoveredYear,
               let c = hoveredCount,
               domain.contains(y) {
                RuleMark(x: .value("Year", y))
                    .foregroundStyle(.white.opacity(0.28))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    .annotation(position: .top, alignment: .center) {
                        Text("\(y) · \(c)")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.primary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
            }
        }
        .chartXScale(domain: domain)
        .chartYScale(domain: 0...maxCount)
        .chartXAxisLabel("Year")
        .chartYAxisLabel("Papers")
        .frame(height: height)
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    let localX = value.location.x - plotFrame.origin.x
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: localX, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: localX, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    let clampedYear = min(max(year, domain.lowerBound), domain.upperBound)

                                    hoveredYear = clampedYear
                                    hoveredCount = data.first(where: { $0.year == clampedYear })?.count

                                    if brushAnchor == nil {
                                        brushAnchor = clampedYear
                                    } else if let anchor = brushAnchor {
                                        yearFilterStart = min(anchor, clampedYear)
                                        yearFilterEnd = max(anchor, clampedYear)
                                        yearFilterEnabled = true
                                    }
                                }
                                .onEnded { _ in
                                    brushAnchor = nil
                                }
                        )
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: loc.x, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: loc.x, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    let clampedYear = min(max(year, domain.lowerBound), domain.upperBound)
                                    hoveredYear = clampedYear
                                    hoveredCount = data.first(where: { $0.year == clampedYear })?.count
                                },
                                onExit: {
                                    hoveredYear = nil
                                    hoveredCount = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct TimelineYearControls: View {
    let domain: ClosedRange<Int>

    @Binding var yearFilterEnabled: Bool
    @Binding var yearFilterStart: Int
    @Binding var yearFilterEnd: Int
    @Binding var hoveredYear: Int?
    @Binding var hoveredCount: Int?
    @Binding var brushAnchor: Int?

    private var safeDomain: ClosedRange<Int> {
        let lower = min(domain.lowerBound, domain.upperBound)
        let upper = max(domain.lowerBound, domain.upperBound)
        return lower...upper
    }

    private var canSlide: Bool {
        safeDomain.lowerBound < safeDomain.upperBound
    }

    private var startValue: Int {
        let raw = yearFilterStart == 0 ? safeDomain.lowerBound : yearFilterStart
        return min(max(raw, safeDomain.lowerBound), safeDomain.upperBound)
    }

    private var endValue: Int {
        let raw = yearFilterEnd == 0 ? safeDomain.upperBound : yearFilterEnd
        return min(max(raw, safeDomain.lowerBound), safeDomain.upperBound)
    }

    var body: some View {
        HStack(spacing: 10) {
            Toggle("Filter analytics by year", isOn: $yearFilterEnabled)
                .font(.caption)
                .toggleStyle(.switch)
            Spacer()
            Button("All years") {
                yearFilterStart = safeDomain.lowerBound
                yearFilterEnd = safeDomain.upperBound
                yearFilterEnabled = false
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button("Clear") {
                yearFilterEnabled = false
                hoveredYear = nil
                hoveredCount = nil
                brushAnchor = nil
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }

        if canSlide {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("From \(startValue)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Slider(
                        value: Binding(
                            get: { Double(startValue) },
                            set: { newValue in
                                yearFilterStart = min(max(Int(newValue.rounded()), safeDomain.lowerBound), safeDomain.upperBound)
                                yearFilterEnabled = true
                            }
                        ),
                        in: Double(safeDomain.lowerBound)...Double(safeDomain.upperBound),
                        step: 1
                    )
                }
                HStack {
                    Text("To \(endValue)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Slider(
                        value: Binding(
                            get: { Double(endValue) },
                            set: { newValue in
                                yearFilterEnd = min(max(Int(newValue.rounded()), safeDomain.lowerBound), safeDomain.upperBound)
                                yearFilterEnabled = true
                            }
                        ),
                        in: Double(safeDomain.lowerBound)...Double(safeDomain.upperBound),
                        step: 1
                    )
                }
            }
        } else {
            Text("Only year \(safeDomain.lowerBound) in corpus.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }

        Text("Tip: drag across the chart to brush-select a year range.")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}

@available(macOS 26, iOS 26, *)
private struct NoveltyConsensusChartView: View {
    let points: [NoveltyConsensusPoint]
    let frontier: Set<UUID>
    let showFrontier: Bool
    @Binding var selectedPointID: UUID?

    @State private var hoveredPointID: UUID?
    @State private var visibleXDomain: ClosedRange<Double>?
    @State private var visibleYDomain: ClosedRange<Double>?
    @State private var panStartX: ClosedRange<Double>?
    @State private var panStartY: ClosedRange<Double>?
    @State private var zoomStartX: ClosedRange<Double>?
    @State private var zoomStartY: ClosedRange<Double>?

    private var hoveredPoint: NoveltyConsensusPoint? {
        guard let hoveredPointID else { return nil }
        return points.first(where: { $0.id == hoveredPointID })
    }

    private var selectedPoint: NoveltyConsensusPoint? {
        guard let selectedPointID else { return nil }
        return points.first(where: { $0.id == selectedPointID })
    }

    var body: some View {
        let fullXDomain: ClosedRange<Double> = {
            let xs = points.map(\.novelty)
            guard let minX = xs.min(), let maxX = xs.max(), minX < maxX else { return -1.0...1.0 }
            let pad = max(0.25, (maxX - minX) * 0.14)
            return (minX - pad)...(maxX + pad)
        }()
        let fullYDomain: ClosedRange<Double> = {
            let ys = points.map(\.consensus)
            guard let minY = ys.min(), let maxY = ys.max(), minY < maxY else { return -1.0...1.0 }
            let pad = max(0.25, (maxY - minY) * 0.14)
            return (minY - pad)...(maxY + pad)
        }()

        let xDomain = visibleXDomain ?? fullXDomain
        let yDomain = visibleYDomain ?? fullYDomain

        Chart {
            ForEach(points) { pt in
                let isFrontier = showFrontier && frontier.contains(pt.id)
                let isSelected = selectedPointID == pt.id
                let isHovered = hoveredPointID == pt.id
                PointMark(
                    x: .value("Novelty", pt.novelty),
                    y: .value("Consensus", pt.consensus)
                )
                .foregroundStyle(isSelected ? .white : (isFrontier ? .pink : .mint))
                .symbolSize(isSelected ? 180 : (isFrontier ? 90 : 34))
                .opacity(isSelected || isHovered ? 1 : max(0.35, 1 - pt.novStd * 4))
            }

            if let focus = selectedPoint ?? hoveredPoint {
                RuleMark(x: .value("Novelty", focus.novelty))
                    .foregroundStyle(Color.white.opacity(0.14))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                RuleMark(y: .value("Consensus", focus.consensus))
                    .foregroundStyle(Color.white.opacity(0.14))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                PointMark(
                    x: .value("Novelty", focus.novelty),
                    y: .value("Consensus", focus.consensus)
                )
                .foregroundStyle(.white)
                .symbolSize(220)
                .annotation(position: .top, alignment: .leading) {
                    Text("\(focus.title)\nNovelty \(String(format: "%.2f", focus.novelty)) · Consensus \(String(format: "%.2f", focus.consensus))")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .lineLimit(3)
                        .frame(maxWidth: 320, alignment: .leading)
                }
            }
        }
        .chartXScale(domain: xDomain)
        .chartYScale(domain: yDomain)
        .chartXAxisLabel("Novelty (z / combo / directional)")
        .chartYAxisLabel("Consensus (z)")
        .overlay(alignment: .topTrailing) {
            ChartZoomControls(
                onZoomIn: {
                    visibleXDomain = ChartZoomPan.zoom(domain: xDomain, by: 1.35, within: fullXDomain)
                    visibleYDomain = ChartZoomPan.zoom(domain: yDomain, by: 1.35, within: fullYDomain)
                },
                onZoomOut: {
                    visibleXDomain = ChartZoomPan.zoom(domain: xDomain, by: 0.74, within: fullXDomain)
                    visibleYDomain = ChartZoomPan.zoom(domain: yDomain, by: 0.74, within: fullYDomain)
                },
                onReset: {
                    visibleXDomain = fullXDomain
                    visibleYDomain = fullYDomain
                }
            )
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        .gesture(
                            SpatialTapGesture()
                                .onEnded { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    selectedPointID = nearestPoint(at: value.location, plotFrame: plotFrame, proxy: proxy)?.id
                                }
                        )
                        #if os(macOS)
                        .simultaneousGesture(
                            DragGesture(minimumDistance: 6)
                                .onChanged { value in
                                    if panStartX == nil { panStartX = xDomain }
                                    if panStartY == nil { panStartY = yDomain }
                                    guard let startX = panStartX, let startY = panStartY else { return }
                                    let width = max(1, plotFrame.width)
                                    let height = max(1, plotFrame.height)
                                    let spanX = startX.upperBound - startX.lowerBound
                                    let spanY = startY.upperBound - startY.lowerBound
                                    let deltaX = -Double(value.translation.width / width) * spanX
                                    let deltaY = Double(value.translation.height / height) * spanY
                                    visibleXDomain = ChartZoomPan.pan(domain: startX, by: deltaX, within: fullXDomain)
                                    visibleYDomain = ChartZoomPan.pan(domain: startY, by: deltaY, within: fullYDomain)
                                }
                                .onEnded { _ in
                                    panStartX = nil
                                    panStartY = nil
                                }
                        )
                        .simultaneousGesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if zoomStartX == nil { zoomStartX = xDomain }
                                    if zoomStartY == nil { zoomStartY = yDomain }
                                    guard let startX = zoomStartX, let startY = zoomStartY else { return }
                                    visibleXDomain = ChartZoomPan.zoom(domain: startX, by: Double(value), within: fullXDomain)
                                    visibleYDomain = ChartZoomPan.zoom(domain: startY, by: Double(value), within: fullYDomain)
                                }
                                .onEnded { _ in
                                    zoomStartX = nil
                                    zoomStartY = nil
                                }
                        )
                        .onTapGesture(count: 2) {
                            visibleXDomain = fullXDomain
                            visibleYDomain = fullYDomain
                        }
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let absLoc = CGPoint(x: plotFrame.origin.x + loc.x, y: plotFrame.origin.y + loc.y)
                                    hoveredPointID = nearestPoint(at: absLoc, plotFrame: plotFrame, proxy: proxy)?.id
                                },
                                onExit: {
                                    hoveredPointID = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
        .onAppear {
            if visibleXDomain == nil { visibleXDomain = fullXDomain }
            if visibleYDomain == nil { visibleYDomain = fullYDomain }
        }
        .onChange(of: fullXDomain) { _, newValue in
            visibleXDomain = newValue
        }
        .onChange(of: fullYDomain) { _, newValue in
            visibleYDomain = newValue
        }
        .animation(.easeInOut(duration: 0.25), value: selectedPointID)
        .animation(.easeInOut(duration: 0.25), value: hoveredPointID)
    }

    private func nearestPoint(at location: CGPoint, plotFrame: CGRect, proxy: ChartProxy) -> NoveltyConsensusPoint? {
        var best: (NoveltyConsensusPoint, CGFloat)?
        for pt in points {
            guard let x = proxy.position(forX: pt.novelty),
                  let y = proxy.position(forY: pt.consensus) else { continue }
            let point = CGPoint(x: plotFrame.origin.x + x, y: plotFrame.origin.y + y)
            let dist = hypot(point.x - location.x, point.y - location.y)
            if let existing = best {
                if dist < existing.1 { best = (pt, dist) }
            } else {
                best = (pt, dist)
            }
        }
        guard let best else { return nil }
        return best.1 <= 30 ? best.0 : nil
    }
}

@available(macOS 26, iOS 26, *)
private struct TopicStreamChart: View {
    let stream: TopicEvolutionStream
    let height: CGFloat

    @State private var hoveredYear: Int?
    @State private var hoveredCount: Int?
    @State private var visibleDomain: ClosedRange<Int>?
    @State private var panStartDomain: ClosedRange<Int>?
    @State private var zoomStartDomain: ClosedRange<Int>?

    var body: some View {
        let years = stream.countsByYear.keys.sorted()
        let fullDomain: ClosedRange<Int> = {
            guard let first = years.first, let last = years.last, first <= last else { return 1900...Calendar.current.component(.year, from: Date()) }
            return first...last
        }()
        let domain = visibleDomain ?? fullDomain
        let maxCount = max(1, stream.countsByYear.values.max() ?? 1)

        Chart {
            ForEach(years, id: \.self) { year in
                let count = stream.countsByYear[year] ?? 0
                let isBurst = stream.burstYears.contains(year)
                let isDecay = stream.decayYears.contains(year)
                let base: Color = isBurst ? .red : (isDecay ? .orange : .blue)
                let emphasis = hoveredYear == nil || hoveredYear == year

                BarMark(
                    x: .value("Year", year),
                    y: .value("Papers", count)
                )
                .foregroundStyle(base.opacity(emphasis ? 0.8 : 0.22))
            }

            if let y = hoveredYear,
               let c = hoveredCount,
               domain.contains(y) {
                RuleMark(x: .value("Year", y))
                    .foregroundStyle(.white.opacity(0.22))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    .annotation(position: .top, alignment: .center) {
                        Text("\(y) · \(c)")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
            }
        }
        .chartXScale(domain: domain)
        .chartYScale(domain: 0...maxCount)
        .chartXAxisLabel("Year")
        .chartYAxisLabel("Count")
        .frame(height: height)
        .overlay(alignment: .topTrailing) {
            ChartZoomControls(
                onZoomIn: { visibleDomain = ChartZoomPan.zoom(domain: domain, by: 1.35, within: fullDomain) },
                onZoomOut: { visibleDomain = ChartZoomPan.zoom(domain: domain, by: 0.74, within: fullDomain) },
                onReset: { visibleDomain = fullDomain }
            )
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        #if os(iOS)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    let localX = value.location.x - plotFrame.origin.x
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: localX, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: localX, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    let clampedYear = min(max(year, domain.lowerBound), domain.upperBound)
                                    hoveredYear = clampedYear
                                    hoveredCount = stream.countsByYear[clampedYear]
                                }
                        )
                        #endif
                        #if os(macOS)
                        .gesture(
                            DragGesture(minimumDistance: 6)
                                .onChanged { value in
                                    if panStartDomain == nil { panStartDomain = domain }
                                    guard let startDomain = panStartDomain else { return }
                                    let width = max(1, plotFrame.width)
                                    let span = Double(startDomain.upperBound - startDomain.lowerBound)
                                    let delta = -Double(value.translation.width / width) * span
                                    visibleDomain = ChartZoomPan.pan(domain: startDomain, by: delta, within: fullDomain)
                                }
                                .onEnded { _ in
                                    panStartDomain = nil
                                }
                        )
                        .simultaneousGesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if zoomStartDomain == nil { zoomStartDomain = domain }
                                    guard let startDomain = zoomStartDomain else { return }
                                    visibleDomain = ChartZoomPan.zoom(domain: startDomain, by: Double(value), within: fullDomain)
                                }
                                .onEnded { _ in
                                    zoomStartDomain = nil
                                }
                        )
                        .onTapGesture(count: 2) {
                            visibleDomain = fullDomain
                        }
                        #endif
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: loc.x, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: loc.x, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    let clampedYear = min(max(year, domain.lowerBound), domain.upperBound)
                                    hoveredYear = clampedYear
                                    hoveredCount = stream.countsByYear[clampedYear]
                                },
                                onExit: {
                                    hoveredYear = nil
                                    hoveredCount = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
        .onAppear {
            if visibleDomain == nil { visibleDomain = fullDomain }
        }
        .onChange(of: fullDomain) { _, newValue in
            visibleDomain = newValue
        }
        .animation(.easeInOut(duration: 0.25), value: hoveredYear)
    }
}

@available(macOS 26, iOS 26, *)
private struct MethodCrossoverChart: View {
    let signal: MethodTakeover
    let papers: [Paper]

    @State private var hoveredYear: Int?
    @State private var visibleDomain: ClosedRange<Int>?
    @State private var panStartDomain: ClosedRange<Int>?
    @State private var zoomStartDomain: ClosedRange<Int>?

    var body: some View {
        let aCounts = yearlyCounts(for: signal.a)
        let bCounts = yearlyCounts(for: signal.b)
        let years = Array(Set(aCounts.keys).union(bCounts.keys)).sorted()
        let fullDomain: ClosedRange<Int> = {
            guard let first = years.first, let last = years.last, first <= last else { return 1900...Calendar.current.component(.year, from: Date()) }
            return first...last
        }()
        let domain = visibleDomain ?? fullDomain
        Chart {
            ForEach(years, id: \.self) { year in
                LineMark(
                    x: .value("Year", year),
                    y: .value(signal.a, aCounts[year] ?? 0)
                )
                .foregroundStyle(.blue)
                LineMark(
                    x: .value("Year", year),
                    y: .value(signal.b, bCounts[year] ?? 0)
                )
                .foregroundStyle(.orange)
            }

            if let y = hoveredYear, years.contains(y) {
                let a = aCounts[y] ?? 0
                let b = bCounts[y] ?? 0

                RuleMark(x: .value("Year", y))
                    .foregroundStyle(.white.opacity(0.22))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))

                PointMark(x: .value("Year", y), y: .value(signal.a, a))
                    .foregroundStyle(.blue)
                    .symbolSize(90)

                PointMark(x: .value("Year", y), y: .value(signal.b, b))
                    .foregroundStyle(.orange)
                    .symbolSize(90)
                    .annotation(position: .top, alignment: .center) {
                        Text("\(y) · \(signal.a)=\(a) · \(signal.b)=\(b)")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
            }
        }
        .chartXAxisLabel("Year")
        .chartYAxisLabel("Count")
        .chartXScale(domain: domain)
        .overlay(alignment: .topTrailing) {
            ChartZoomControls(
                onZoomIn: { visibleDomain = ChartZoomPan.zoom(domain: domain, by: 1.35, within: fullDomain) },
                onZoomOut: { visibleDomain = ChartZoomPan.zoom(domain: domain, by: 0.74, within: fullDomain) },
                onReset: { visibleDomain = fullDomain }
            )
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        #if os(iOS)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    let localX = value.location.x - plotFrame.origin.x
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: localX, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: localX, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    hoveredYear = nearestYear(to: year, years: years)
                                }
                                .onEnded { _ in }
                        )
                        #endif
                        #if os(macOS)
                        .gesture(
                            DragGesture(minimumDistance: 6)
                                .onChanged { value in
                                    if panStartDomain == nil { panStartDomain = domain }
                                    guard let startDomain = panStartDomain else { return }
                                    let width = max(1, plotFrame.width)
                                    let span = Double(startDomain.upperBound - startDomain.lowerBound)
                                    let delta = -Double(value.translation.width / width) * span
                                    visibleDomain = ChartZoomPan.pan(domain: startDomain, by: delta, within: fullDomain)
                                }
                                .onEnded { _ in
                                    panStartDomain = nil
                                }
                        )
                        .simultaneousGesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if zoomStartDomain == nil { zoomStartDomain = domain }
                                    guard let startDomain = zoomStartDomain else { return }
                                    visibleDomain = ChartZoomPan.zoom(domain: startDomain, by: Double(value), within: fullDomain)
                                }
                                .onEnded { _ in
                                    zoomStartDomain = nil
                                }
                        )
                        .onTapGesture(count: 2) {
                            visibleDomain = fullDomain
                        }
                        #endif
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: loc.x, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: loc.x, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    hoveredYear = nearestYear(to: year, years: years)
                                },
                                onExit: {
                                    hoveredYear = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
        .onAppear {
            if visibleDomain == nil { visibleDomain = fullDomain }
        }
        .onChange(of: fullDomain) { _, newValue in
            visibleDomain = newValue
        }
        .animation(.easeInOut(duration: 0.25), value: hoveredYear)
    }

    private func nearestYear(to year: Int, years: [Int]) -> Int? {
        guard let first = years.first else { return nil }
        var best = first
        var bestDist = abs(first - year)
        for y in years.dropFirst() {
            let d = abs(y - year)
            if d < bestDist {
                best = y
                bestDist = d
            }
        }
        return best
    }

    private func yearlyCounts(for tag: String) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for paper in papers {
            guard let year = paper.year else { continue }
            if paperContains(paper, tag: tag) {
                counts[year, default: 0] += 1
            }
        }
        return counts
    }

    private func paperContains(_ paper: Paper, tag: String) -> Bool {
        let needle = tag.lowercased()
        let haystacks = [
            paper.methodSummary?.lowercased() ?? "",
            paper.summary.lowercased(),
            (paper.keywords ?? []).joined(separator: " ").lowercased()
        ]
        return haystacks.contains { $0.contains(needle) }
    }
}

@available(macOS 26, iOS 26, *)
private struct FilterChip<Label: StringProtocol>: View {
    let label: Label
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.caption)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(isOn ? Color.accentColor.opacity(0.18) : Color.white.opacity(0.06))
                .foregroundStyle(isOn ? Color.accentColor : .primary)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isOn)
    }
}

@available(macOS 26, iOS 26, *)
private struct ReadingLagChart: View {
    let points: [ReadingOverlayPoint]

    @State private var hoveredPubYear: Int?
    @State private var hoveredReadYear: Int?
    @State private var visibleXDomain: ClosedRange<Int>?
    @State private var visibleYDomain: ClosedRange<Int>?
    @State private var panStartX: ClosedRange<Int>?
    @State private var panStartY: ClosedRange<Int>?
    @State private var zoomStartX: ClosedRange<Int>?
    @State private var zoomStartY: ClosedRange<Int>?

    var body: some View {
        let minPub = points.map(\.publicationYear).min() ?? 0
        let minRead = points.map(\.readYear).min() ?? 0
        let maxPub = points.map(\.publicationYear).max() ?? 0
        let maxRead = points.map(\.readYear).max() ?? 0
        let minYear = min(minPub, minRead)
        let maxYear = max(maxPub, maxRead)
        let fullDomain: ClosedRange<Int> = minYear...maxYear
        let xDomain = visibleXDomain ?? fullDomain
        let yDomain = visibleYDomain ?? fullDomain

        Chart {
            if minYear <= maxYear {
                ForEach([minYear, maxYear], id: \.self) { y in
                    LineMark(
                        x: .value("Year", y),
                        y: .value("Year", y)
                    )
                }
                .foregroundStyle(Color.white.opacity(0.14))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
            }

            ForEach(points.indices, id: \.self) { idx in
                let p = points[idx]
                PointMark(
                    x: .value("Published", p.publicationYear),
                    y: .value("Read", p.readYear)
                )
                .foregroundStyle(.mint.opacity(0.75))
                .symbolSize(24)
            }

            if let pub = hoveredPubYear, let read = hoveredReadYear {
                RuleMark(x: .value("Published", pub))
                    .foregroundStyle(.white.opacity(0.18))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                RuleMark(y: .value("Read", read))
                    .foregroundStyle(.white.opacity(0.18))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                PointMark(x: .value("Published", pub), y: .value("Read", read))
                    .foregroundStyle(.white)
                    .symbolSize(90)
                    .annotation(position: .top, alignment: .center) {
                        Text("Pub \(pub) · Read \(read) · Lag \(read - pub)y")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
            }
        }
        .chartXScale(domain: xDomain)
        .chartYScale(domain: yDomain)
        .chartXAxisLabel("Publication year")
        .chartYAxisLabel("Read year")
        .overlay(alignment: .topTrailing) {
            ChartZoomControls(
                onZoomIn: {
                    visibleXDomain = ChartZoomPan.zoom(domain: xDomain, by: 1.35, within: fullDomain)
                    visibleYDomain = ChartZoomPan.zoom(domain: yDomain, by: 1.35, within: fullDomain)
                },
                onZoomOut: {
                    visibleXDomain = ChartZoomPan.zoom(domain: xDomain, by: 0.74, within: fullDomain)
                    visibleYDomain = ChartZoomPan.zoom(domain: yDomain, by: 0.74, within: fullDomain)
                },
                onReset: {
                    visibleXDomain = fullDomain
                    visibleYDomain = fullDomain
                }
            )
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        #if os(iOS)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    let localX = value.location.x - plotFrame.origin.x
                                    let localY = value.location.y - plotFrame.origin.y
                                    let pub: Int? = {
                                        if let yr = proxy.value(atX: localX, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: localX, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    let read: Int? = {
                                        if let yr = proxy.value(atY: localY, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atY: localY, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    if let pub { hoveredPubYear = pub }
                                    if let read { hoveredReadYear = read }
                                }
                        )
                        #endif
                        #if os(macOS)
                        .gesture(
                            DragGesture(minimumDistance: 6)
                                .onChanged { value in
                                    if panStartX == nil { panStartX = xDomain }
                                    if panStartY == nil { panStartY = yDomain }
                                    guard let startX = panStartX, let startY = panStartY else { return }
                                    let width = max(1, plotFrame.width)
                                    let height = max(1, plotFrame.height)
                                    let spanX = Double(startX.upperBound - startX.lowerBound)
                                    let spanY = Double(startY.upperBound - startY.lowerBound)
                                    let deltaX = -Double(value.translation.width / width) * spanX
                                    let deltaY = Double(value.translation.height / height) * spanY
                                    visibleXDomain = ChartZoomPan.pan(domain: startX, by: deltaX, within: fullDomain)
                                    visibleYDomain = ChartZoomPan.pan(domain: startY, by: deltaY, within: fullDomain)
                                }
                                .onEnded { _ in
                                    panStartX = nil
                                    panStartY = nil
                                }
                        )
                        .simultaneousGesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if zoomStartX == nil { zoomStartX = xDomain }
                                    if zoomStartY == nil { zoomStartY = yDomain }
                                    guard let startX = zoomStartX, let startY = zoomStartY else { return }
                                    visibleXDomain = ChartZoomPan.zoom(domain: startX, by: Double(value), within: fullDomain)
                                    visibleYDomain = ChartZoomPan.zoom(domain: startY, by: Double(value), within: fullDomain)
                                }
                                .onEnded { _ in
                                    zoomStartX = nil
                                    zoomStartY = nil
                                }
                        )
                        .onTapGesture(count: 2) {
                            visibleXDomain = fullDomain
                            visibleYDomain = fullDomain
                        }
                        #endif
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let pub: Int? = {
                                        if let yr = proxy.value(atX: loc.x, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: loc.x, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    let read: Int? = {
                                        if let yr = proxy.value(atY: loc.y, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atY: loc.y, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    if let pub { hoveredPubYear = pub }
                                    if let read { hoveredReadYear = read }
                                },
                                onExit: {
                                    hoveredPubYear = nil
                                    hoveredReadYear = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
        .onAppear {
            if visibleXDomain == nil { visibleXDomain = fullDomain }
            if visibleYDomain == nil { visibleYDomain = fullDomain }
        }
        .onChange(of: fullDomain) { _, newValue in
            visibleXDomain = newValue
            visibleYDomain = newValue
        }
        .animation(.easeInOut(duration: 0.25), value: hoveredPubYear)
    }
}

@available(macOS 26, iOS 26, *)
private struct FactorExposureChart: View {
    let exposures: [AnalyticsSummary.FactorExposure]
    let labels: [String]
    let domain: ClosedRange<Int>
    let height: CGFloat

    @State private var hoveredYear: Int?
    @State private var hoveredSummary: String?
    @State private var visibleDomain: ClosedRange<Int>?
    @State private var panStartDomain: ClosedRange<Int>?
    @State private var zoomStartDomain: ClosedRange<Int>?

    var body: some View {
        let fullDomain = domain
        let xDomain = visibleDomain ?? fullDomain
        Chart {
            ForEach(Array(exposures.enumerated()), id: \.offset) { _, row in
                AreaMark(
                    x: .value("Year", row.year),
                    y: .value("Exposure", row.score)
                )
                .foregroundStyle(by: .value("Factor", label(for: row.factor)))
                .interpolationMethod(.catmullRom)
            }

            if let y = hoveredYear, xDomain.contains(y) {
                RuleMark(x: .value("Year", y))
                    .foregroundStyle(.white.opacity(0.18))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    .annotation(position: .top, alignment: .leading) {
                        if let hoveredSummary {
                            Text(hoveredSummary)
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }
                    }
            }
        }
        .chartXScale(domain: xDomain)
        .frame(height: height)
        .chartLegend(position: .bottom)
        .overlay(alignment: .topTrailing) {
            ChartZoomControls(
                onZoomIn: { visibleDomain = ChartZoomPan.zoom(domain: xDomain, by: 1.35, within: fullDomain) },
                onZoomOut: { visibleDomain = ChartZoomPan.zoom(domain: xDomain, by: 0.74, within: fullDomain) },
                onReset: { visibleDomain = fullDomain }
            )
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        #if os(iOS)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    let localX = value.location.x - plotFrame.origin.x
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: localX, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: localX, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    let clampedYear = min(max(year, xDomain.lowerBound), xDomain.upperBound)
                                    hoveredYear = clampedYear
                                    hoveredSummary = summaryText(for: clampedYear)
                                }
                        )
                        #endif
                        #if os(macOS)
                        .gesture(
                            DragGesture(minimumDistance: 6)
                                .onChanged { value in
                                    if panStartDomain == nil { panStartDomain = xDomain }
                                    guard let startDomain = panStartDomain else { return }
                                    let width = max(1, plotFrame.width)
                                    let span = Double(startDomain.upperBound - startDomain.lowerBound)
                                    let delta = -Double(value.translation.width / width) * span
                                    visibleDomain = ChartZoomPan.pan(domain: startDomain, by: delta, within: fullDomain)
                                }
                                .onEnded { _ in
                                    panStartDomain = nil
                                }
                        )
                        .simultaneousGesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if zoomStartDomain == nil { zoomStartDomain = xDomain }
                                    guard let startDomain = zoomStartDomain else { return }
                                    visibleDomain = ChartZoomPan.zoom(domain: startDomain, by: Double(value), within: fullDomain)
                                }
                                .onEnded { _ in
                                    zoomStartDomain = nil
                                }
                        )
                        .onTapGesture(count: 2) {
                            visibleDomain = fullDomain
                        }
                        #endif
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let year: Int? = {
                                        if let yr = proxy.value(atX: loc.x, as: Int.self) { return yr }
                                        if let yrD = proxy.value(atX: loc.x, as: Double.self) { return Int(yrD.rounded()) }
                                        return nil
                                    }()
                                    guard let year else { return }
                                    let clampedYear = min(max(year, xDomain.lowerBound), xDomain.upperBound)
                                    hoveredYear = clampedYear
                                    hoveredSummary = summaryText(for: clampedYear)
                                },
                                onExit: {
                                    hoveredYear = nil
                                    hoveredSummary = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
        .onAppear {
            if visibleDomain == nil { visibleDomain = fullDomain }
        }
        .onChange(of: fullDomain) { _, newValue in
            visibleDomain = newValue
        }
        .animation(.easeInOut(duration: 0.25), value: hoveredYear)
    }

    private func label(for idx: Int) -> String {
        if idx < labels.count { return labels[idx] }
        return "F\(idx)"
    }

    private func summaryText(for year: Int) -> String? {
        let rows = exposures.filter { $0.year == year }
        guard !rows.isEmpty else { return "\(year)" }
        let top = rows
            .sorted { abs($0.score) > abs($1.score) }
            .prefix(3)
            .map { "\(label(for: $0.factor)): \(String(format: "%.2f", $0.score))" }
            .joined(separator: " · ")
        return "\(year) · \(top)"
    }
}

@available(macOS 26, iOS 26, *)
private struct InfluenceTimelineChart: View {
    let items: [(paper: Paper, score: Double, year: Int)]
    let domain: ClosedRange<Int>
    let height: CGFloat
    let onOpenPaper: (Paper) -> Void

    @State private var hoveredPaperID: UUID?
    @State private var visibleXDomain: ClosedRange<Int>?
    @State private var visibleYDomain: ClosedRange<Double>?
    @State private var panStartX: ClosedRange<Int>?
    @State private var panStartY: ClosedRange<Double>?
    @State private var zoomStartX: ClosedRange<Int>?
    @State private var zoomStartY: ClosedRange<Double>?

    private var hoveredItem: (paper: Paper, score: Double, year: Int)? {
        guard let hoveredPaperID else { return nil }
        return items.first(where: { $0.paper.id == hoveredPaperID })
    }

    var body: some View {
        let fullXDomain = domain
        let fullYDomain: ClosedRange<Double> = {
            let vals = items.map(\.score)
            guard let minV = vals.min(), let maxV = vals.max(), minV < maxV else { return 0...1 }
            let pad = (maxV - minV) * 0.12
            return (minV - pad)...(maxV + pad)
        }()
        let xDomain = visibleXDomain ?? fullXDomain
        let yDomain = visibleYDomain ?? fullYDomain

        Chart {
            ForEach(items, id: \.paper.id) { item in
                let isHovered = item.paper.id == hoveredPaperID
                PointMark(
                    x: .value("Year", item.year),
                    y: .value("Influence", item.score)
                )
                .foregroundStyle(isHovered ? .white : .orange.opacity(0.85))
                .symbolSize(isHovered ? 180 : 70)
            }

            if let hoveredItem {
                RuleMark(x: .value("Year", hoveredItem.year))
                    .foregroundStyle(.white.opacity(0.22))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("\(hoveredItem.paper.title) · \(String(format: "%.3f", hoveredItem.score))")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                            .lineLimit(2)
                            .frame(maxWidth: 320, alignment: .leading)
                    }
            }
        }
        .chartXScale(domain: xDomain)
        .chartYScale(domain: yDomain)
        .chartXAxisLabel("Year")
        .chartYAxisLabel("Influence")
        .frame(height: height)
        .overlay(alignment: .topTrailing) {
            ChartZoomControls(
                onZoomIn: {
                    visibleXDomain = ChartZoomPan.zoom(domain: xDomain, by: 1.35, within: fullXDomain)
                    visibleYDomain = ChartZoomPan.zoom(domain: yDomain, by: 1.35, within: fullYDomain)
                },
                onZoomOut: {
                    visibleXDomain = ChartZoomPan.zoom(domain: xDomain, by: 0.74, within: fullXDomain)
                    visibleYDomain = ChartZoomPan.zoom(domain: yDomain, by: 0.74, within: fullYDomain)
                },
                onReset: {
                    visibleXDomain = fullXDomain
                    visibleYDomain = fullYDomain
                }
            )
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        .gesture(
                            SpatialTapGesture()
                                .onEnded { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    if let nearest = nearestItem(at: value.location, plotFrame: plotFrame, proxy: proxy) {
                                        hoveredPaperID = nearest.paper.id
                                        onOpenPaper(nearest.paper)
                                    }
                                }
                        )
                        #if os(macOS)
                        .simultaneousGesture(
                            DragGesture(minimumDistance: 6)
                                .onChanged { value in
                                    if panStartX == nil { panStartX = xDomain }
                                    if panStartY == nil { panStartY = yDomain }
                                    guard let startX = panStartX, let startY = panStartY else { return }
                                    let width = max(1, plotFrame.width)
                                    let height = max(1, plotFrame.height)
                                    let spanX = Double(startX.upperBound - startX.lowerBound)
                                    let spanY = startY.upperBound - startY.lowerBound
                                    let deltaX = -Double(value.translation.width / width) * spanX
                                    let deltaY = Double(value.translation.height / height) * spanY
                                    visibleXDomain = ChartZoomPan.pan(domain: startX, by: deltaX, within: fullXDomain)
                                    visibleYDomain = ChartZoomPan.pan(domain: startY, by: deltaY, within: fullYDomain)
                                }
                                .onEnded { _ in
                                    panStartX = nil
                                    panStartY = nil
                                }
                        )
                        .simultaneousGesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if zoomStartX == nil { zoomStartX = xDomain }
                                    if zoomStartY == nil { zoomStartY = yDomain }
                                    guard let startX = zoomStartX, let startY = zoomStartY else { return }
                                    visibleXDomain = ChartZoomPan.zoom(domain: startX, by: Double(value), within: fullXDomain)
                                    visibleYDomain = ChartZoomPan.zoom(domain: startY, by: Double(value), within: fullYDomain)
                                }
                                .onEnded { _ in
                                    zoomStartX = nil
                                    zoomStartY = nil
                                }
                        )
                        .onTapGesture(count: 2) {
                            visibleXDomain = fullXDomain
                            visibleYDomain = fullYDomain
                        }
                        #endif
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let absLoc = CGPoint(x: plotFrame.origin.x + loc.x, y: plotFrame.origin.y + loc.y)
                                    hoveredPaperID = nearestItem(at: absLoc, plotFrame: plotFrame, proxy: proxy)?.paper.id
                                },
                                onExit: {
                                    hoveredPaperID = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
        .onAppear {
            if visibleXDomain == nil { visibleXDomain = fullXDomain }
            if visibleYDomain == nil { visibleYDomain = fullYDomain }
        }
        .onChange(of: fullXDomain) { _, newValue in
            visibleXDomain = newValue
        }
        .onChange(of: fullYDomain) { _, newValue in
            visibleYDomain = newValue
        }
        .animation(.easeInOut(duration: 0.25), value: hoveredPaperID)
    }

    private func nearestItem(at location: CGPoint, plotFrame: CGRect, proxy: ChartProxy) -> (paper: Paper, score: Double, year: Int)? {
        var best: ((paper: Paper, score: Double, year: Int), CGFloat)?
        for item in items {
            guard let x = proxy.position(forX: item.year),
                  let y = proxy.position(forY: item.score) else { continue }
            let point = CGPoint(x: plotFrame.origin.x + x, y: plotFrame.origin.y + y)
            let dist = hypot(point.x - location.x, point.y - location.y)
            if let existing = best {
                if dist < existing.1 { best = (item, dist) }
            } else {
                best = (item, dist)
            }
        }
        guard let best else { return nil }
        return best.1 <= 34 ? best.0 : nil
    }
}

@available(macOS 26, iOS 26, *)
private struct DailyQuizSheet: View {
    let deck: [(Flashcard, Paper)]
    @Binding var index: Int
    @Binding var reveal: Bool
    let onReveal: (Flashcard, Paper) -> Void

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Daily quiz").font(.headline)
                if deck.isEmpty {
                    Text("No cards available.")
                } else {
                    let item = deck[min(index, deck.count - 1)]
                    Text(item.1.title)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Q: \(item.0.question)")
                        .font(.title3.bold())
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if reveal {
                        Text("A: \(item.0.answer)")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    } else {
                        Button("Show answer") {
                            reveal = true
                            onReveal(item.0, item.1)
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    Spacer()
                    HStack {
                        Text("Card \(index + 1) of \(deck.count)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Previous") {
                            index = max(0, index - 1)
                            reveal = false
                        }.disabled(index == 0)
                        Button("Next") {
                            index = min(deck.count - 1, index + 1)
                            reveal = false
                        }
                        .disabled(index >= deck.count - 1)
                    }
                }
            }
            .padding()
            .frame(minWidth: 420, minHeight: 360)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }

    @Environment(\.dismiss) private var dismiss
}
