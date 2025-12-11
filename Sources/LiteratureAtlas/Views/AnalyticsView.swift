import SwiftUI
import Charts

@available(macOS 26, iOS 26, *)
struct AnalyticsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var topicQuery: String = ""
    @State private var snapshot: KnowledgeSnapshot?
    @State private var panelClusterID: Int?
    @State private var panelTranscript: String = ""
    @State private var hypotheticalTitle: String = ""
    @State private var hypotheticalAbstract: String = ""
    @State private var methodTagsText: String = "q-learning, ppo, sac, dqn"
    @State private var showDailyQuiz: Bool = false
    @State private var dailyQuizIndex: Int = 0
    @State private var dailyQuizReveal: Bool = false
    @State private var focusMyExposure: Bool = false
    @State private var selectedCounterfactual: String?
    @State private var customCutoffs: String = "2010 2015 2020"

    private var timeline: [(year: Int, count: Int)] {
        let counts = Dictionary(grouping: model.papers.compactMap { $0.year }) { $0 }
            .mapValues { $0.count }
        return counts.keys.sorted().map { ($0, counts[$0] ?? 0) }
    }

    private var clusterAverages: [(name: String, year: Double)] {
        model.clusters.compactMap { cluster in
            let years = cluster.memberPaperIDs.compactMap { id in
                model.papers.first(where: { $0.id == id })?.year
            }
            guard !years.isEmpty else { return nil }
            let avg = years.reduce(0, +)
            return (cluster.name, Double(avg) / Double(years.count))
        }
    }

    private var topicStreams: [TopicEvolutionStream] {
        model.topicEvolutionStreams()
    }

    private var methodSignals: [MethodTakeover] {
        model.methodTakeoverSignals(methodTags: methodTags)
    }

    private var readingStats: ReadingLagStats {
        model.readingLagAnalytics()
    }

    private var noveltyScores: [PaperNoveltyScore] {
        model.noveltyHighlights()
    }

    private var methodTags: [String] {
        methodTagsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
    }

    // MARK: Derived analytics (Python backend)

    private var analyticsSummary: AnalyticsSummary? { model.analyticsSummary }

    private var recommendationPapers: [Paper] {
        guard let ids = analyticsSummary?.recommendations else { return [] }
        let map = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0) })
        return ids.compactMap { map[$0] }
    }

    private struct NoveltyConsensusPoint: Identifiable {
        let id: UUID
        let novelty: Double
        let centrality: Double
        let title: String
    }

    private var noveltyConsensus: [NoveltyConsensusPoint] {
        guard let summary = analyticsSummary else { return [] }
        let centralityMap = Dictionary(uniqueKeysWithValues: summary.centrality.map { ($0.paperID, $0.weightedDegree) })
        let titleMap = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0.title) })
        return summary.novelty.compactMap { n in
            guard let c = centralityMap[n.paperID] else { return nil }
            return NoveltyConsensusPoint(id: n.paperID, novelty: n.novelty, centrality: c, title: titleMap[n.paperID] ?? "Paper")
        }
    }

    private var driftTop: [(cluster: Int, year: Int, drift: Double)] {
        guard let summary = analyticsSummary else { return [] }
        return summary.drift.sorted { $0.drift > $1.drift }.prefix(5).map { ($0.clusterID, $0.year, $0.drift) }
    }

    private var factorExposureSample: [AnalyticsSummary.FactorExposure] {
        let base = filteredFactorExposures()
        return Array(base.prefix(80))
    }

    private var influenceTop: [(paper: Paper, score: Double)] {
        guard let summary = analyticsSummary else { return [] }
        let map = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0) })
        return summary.influence
            .compactMap { entry in map[entry.paperID].map { ($0, entry.influence) } }
            .sorted { $0.1 > $1.1 }
            .prefix(5)
            .map { $0 }
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
        guard focusMyExposure, let _ = analyticsSummary else {
            return analyticsSummary?.factorExposures ?? []
        }
        // Use only exposures for papers the user has marked done/important
        let keep = Set(model.papers.filter { $0.readingStatus == .done || ($0.isImportant ?? false) }.compactMap { $0.year })
        return (analyticsSummary?.factorExposures ?? []).filter { keep.contains($0.year) }
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

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Step 4 - Timeline & Drift")
                        .font(.title2.bold())
                    Text("See when topics heat up and how clusters evolve over time.")
                        .foregroundStyle(.secondary)

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
                            } else if let error = model.analyticsLoadError {
                                Text(error)
                                    .font(.caption2)
                                    .foregroundStyle(.red)
                            } else {
                                Text("Run the Python script to populate analytics.json, then reload.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    if timeline.isEmpty {
                        Text("Add papers with year metadata to populate the timeline.")
                            .foregroundStyle(.secondary)
                    } else {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Timeline (papers per year)").font(.headline)
                                ForEach(timeline, id: \.year) { entry in
                                    HStack {
                                        Text("\(entry.year)")
                                            .frame(width: 60, alignment: .leading)
                                        GeometryReader { geo in
                                            let width = max(6, geo.size.width * CGFloat(entry.count) / CGFloat((timeline.map { $0.count }.max() ?? 1)))
                                            RoundedRectangle(cornerRadius: 4)
                                                .fill(Color.blue.opacity(0.4))
                                                .frame(width: width, height: 10, alignment: .leading)
                                                .alignmentGuide(.leading) { _ in 0 }
                                        }
                                        .frame(height: 12)
                                        Text("\(entry.count)")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }

                    if !clusterAverages.isEmpty {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Cluster average year").font(.headline)
                                ForEach(clusterAverages, id: \.name) { entry in
                                    HStack {
                                        Text(entry.name).font(.subheadline)
                                        Spacer()
                                        Text(String(format: "%.1f", entry.year))
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }

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
                                        Chart {
                                            ForEach(stream.countsByYear.keys.sorted(), id: \.self) { year in
                                                let count = stream.countsByYear[year] ?? 0
                                                BarMark(
                                                    x: .value("Year", year),
                                                    y: .value("Papers", count)
                                                )
                                                .foregroundStyle(stream.burstYears.contains(year) ? .red.opacity(0.7) : .blue.opacity(0.7))
                                            }
                                        }
                                        .chartXAxisLabel("Year")
                                        .chartYAxisLabel("Count")
                                        .frame(height: 140)
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
                                        // recompute via state change
                                        methodTagsText = methodTagsText
                                    }
                                    .buttonStyle(.bordered)
                                }
                                if let first = methodSignals.first {
                                    MethodCrossoverChart(signal: first)
                                        .frame(height: 160)
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
                        if !noveltyConsensus.isEmpty {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Novelty vs consensus").font(.headline)
                                    Chart {
                                        ForEach(noveltyConsensus) { pt in
                                            PointMark(
                                                x: .value("Novelty", pt.novelty),
                                                y: .value("Centrality", pt.centrality)
                                            )
                                            .foregroundStyle(.mint)
                                            .annotation(position: .topLeading) {
                                                Text(pt.title.prefix(18))
                                                    .font(.caption2)
                                            }
                                        }
                                    }
                                    .chartXAxisLabel("Novelty (farther from centroid)")
                                    .chartYAxisLabel("Centrality (weighted degree)")
                                    .frame(height: 220)
                                    Text("Quadrants: high/high = anchors; high novelty/low centrality = hidden gems.")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }

                        if !driftTop.isEmpty {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Topic drift (semantic motion)").font(.headline)
                                    ForEach(driftTop, id: \.cluster) { entry in
                                        Text("Cluster \(entry.cluster) drifted \(String(format: "%.3f", entry.drift)) in \(entry.year)")
                                            .font(.caption)
                                    }
                                }
                            }
                        }

                        if !factorExposureSample.isEmpty {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Factor exposures over time").font(.headline)
                                    Toggle("Focus on my read/important papers", isOn: $focusMyExposure)
                                        .font(.caption)
                                        .toggleStyle(.switch)
                                    Chart {
                                        ForEach(factorExposureSample, id: \.year) { row in
                                            LineMark(
                                                x: .value("Year", row.year),
                                                y: .value("Score", row.score),
                                                series: .value("Factor", "F\(row.factor)")
                                            )
                                        }
                                    }
                                    .frame(height: 200)
                                    Text("Factors come from embedding PCA; exposes your corpus/topic mix by year.")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }

                        if !influenceTop.isEmpty {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Influential ideas").font(.headline)
                                    ForEach(influenceTop, id: \.paper.id) { item in
                                        Text("\(item.paper.title) — score \(String(format: "%.3f", item.score))")
                                            .font(.caption)
                                    }
                                }
                            }
                        }

                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Counterfactual corpus").font(.headline)
                                if let scenarios = analyticsSummary?.counterfactuals, !scenarios.isEmpty {
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

                        if !recommendationPapers.isEmpty || summary.answerConfidence != nil {
                            GlassCard {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Personal recommendations & confidence").font(.headline)
                                    if !recommendationPapers.isEmpty {
                                        Text("Bandit picks:")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        ForEach(recommendationPapers.prefix(5)) { paper in
                                            HStack {
                                                Text("• \(paper.title)")
                                                    .font(.caption2)
                                                Spacer()
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
                                    .frame(height: 170)
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

                    if !noveltyScores.isEmpty {
                        let sorted = noveltyScores.sorted { $0.novelty > $1.novelty }
                        let saturated = noveltyScores.sorted { $0.saturation > $1.saturation }
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Novelty & saturation").font(.headline)
                                if let topOutlier = sorted.first,
                                   let paper = model.papers.first(where: { $0.id == topOutlier.paperID }) {
                                    Text("Weird outlier: \(paper.title)")
                                        .font(.subheadline)
                                    Text(String(format: "Novelty %.2f | Saturation %.2f", topOutlier.novelty, topOutlier.saturation))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                if let dense = saturated.first,
                                   let paper = model.papers.first(where: { $0.id == dense.paperID }) {
                                    Text("Over-explored pocket: \(paper.title)")
                                        .font(.subheadline)
                                    Text(String(format: "Novelty %.2f | Saturation %.2f", dense.novelty, dense.saturation))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    if !model.clusters.isEmpty {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Panel of authors").font(.headline)
                                Picker("Cluster", selection: Binding(
                                    get: { panelClusterID ?? model.clusters.first?.id },
                                    set: { panelClusterID = $0 }
                                )) {
                                    ForEach(model.clusters, id: \.id) { cluster in
                                        Text(cluster.name).tag(Optional(cluster.id))
                                    }
                                }
                                .pickerStyle(.menu)

                                Button("Simulate debate") {
                                    if let cid = panelClusterID ?? model.clusters.first?.id {
                                        panelTranscript = model.simulateAuthorPanel(for: cid, maxSpeakers: 3)
                                    }
                                }
                                .buttonStyle(.borderedProminent)

                                if !panelTranscript.isEmpty {
                                    Text(panelTranscript)
                                        .font(.caption2)
                                        .frame(maxWidth: .infinity, alignment: .leading)
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
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct MethodCrossoverChart: View {
    @EnvironmentObject private var model: AppModel
    let signal: MethodTakeover

    var body: some View {
        let aCounts = yearlyCounts(for: signal.a)
        let bCounts = yearlyCounts(for: signal.b)
        let years = Array(Set(aCounts.keys).union(bCounts.keys)).sorted()
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
        }
        .chartXAxisLabel("Year")
        .chartYAxisLabel("Count")
    }

    private func yearlyCounts(for tag: String) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for paper in model.papers {
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
private struct ReadingLagChart: View {
    let points: [ReadingOverlayPoint]

    var body: some View {
        let minYear = points.map { $0.publicationYear }.min() ?? 0
        let maxYear = points.map { $0.readYear }.max() ?? 0
        Chart {
            ForEach(points.indices, id: \.self) { idx in
                let p = points[idx]
                PointMark(
                    x: .value("Published", p.publicationYear),
                    y: .value("Read", p.readYear)
                )
                .foregroundStyle(.mint)
                RuleMark(y: .value("Read year", p.publicationYear))
                    .lineStyle(StrokeStyle(lineWidth: 0.4, dash: [2]))
                    .foregroundStyle(Color.white.opacity(0.1))
            }
        }
        .chartXScale(domain: minYear...maxYear)
        .chartYScale(domain: minYear...maxYear)
        .chartXAxisLabel("Publication year")
        .chartYAxisLabel("Read year")
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
