import SwiftUI

@available(macOS 26, iOS 26, *)
struct TradingLensView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var nav: AppNavigation

    @State private var selectedPointID: UUID?
    @State private var selectedPaperDetail: Paper?
    @State private var searchQuery: String = ""
    @State private var sort: TradingSort = .priority
    @State private var statusFilter: StatusFilter = .all
    @State private var selectedTradingTags: Set<String> = []
    @State private var selectedAssetClasses: Set<String> = []
    @State private var selectedHorizons: Set<String> = []
    @State private var showHypotheses: Bool = true

    private enum TradingSort: String, CaseIterable, Identifiable {
        case priority
        case impact
        case usability
        case novelty
        case confidence
        case recency
        case title

        var id: String { rawValue }

        var label: String {
            switch self {
            case .priority: return "Priority"
            case .impact: return "Impact"
            case .usability: return "Usability"
            case .novelty: return "Novelty"
            case .confidence: return "Confidence"
            case .recency: return "Recency"
            case .title: return "Title"
            }
        }
    }

    private enum StatusFilter: String, CaseIterable, Identifiable {
        case all
        case unread
        case inProgress
        case done

        var id: String { rawValue }

        var label: String {
            switch self {
            case .all: return "All"
            case .unread: return "Unread"
            case .inProgress: return "In progress"
            case .done: return "Done"
            }
        }

        func matches(_ paper: Paper) -> Bool {
            guard self != .all else { return true }
            let status = paper.readingStatus ?? .unread
            switch self {
            case .all: return true
            case .unread: return status == .unread
            case .inProgress: return status == .inProgress
            case .done: return status == .done
            }
        }
    }

    fileprivate struct HypothesisCard: Identifiable {
        let id: String
        let paperID: UUID
        let paperTitle: String
        let hypothesis: String
        let features: [String]
        let target: String?
        let horizon: String?
        let primaryTag: String?
        let primaryAssetClass: String?
    }

    fileprivate struct FailureItem: Identifiable {
        let id: UUID
        let title: String
        let error: String
    }

    private var papers: [Paper] {
        model.explorationPapers
    }

    private var failureItems: [FailureItem] {
        model.tradingLensFailures
            .map { paperID, error in
                let title = papers.first(where: { $0.id == paperID })?.title ?? paperID.uuidString
                return FailureItem(id: paperID, title: title, error: error)
            }
            .sorted { lhs, rhs in
                lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
            }
    }

    private var papersWithLens: [Paper] {
        papers.filter { $0.tradingLens != nil }
    }

    private var papersMissingLensCount: Int {
        papers.count - papersWithLens.count
    }

    private var papersWithScores: [Paper] {
        papersWithLens.filter { ($0.tradingScores ?? $0.tradingLens?.scores) != nil }
    }

    private func normalizedValues(_ raw: [String]?) -> Set<String> {
        let cleaned = (raw ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if cleaned.isEmpty { return ["Unknown"] }
        return Set(cleaned)
    }

    private func matches(selected: Set<String>, values: Set<String>) -> Bool {
        guard !selected.isEmpty else { return true }
        return !values.isDisjoint(with: selected)
    }

    private func primaryValue(from values: [String]?) -> String? {
        let cleaned = (values ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if cleaned.isEmpty { return nil }
        if cleaned.count == 1 { return cleaned.first }
        if let preferred = cleaned.first(where: { $0.lowercased() != "unknown" }) { return preferred }
        return cleaned.first
    }

    private func tradingTags(for paper: Paper) -> Set<String> {
        normalizedValues(paper.tradingLens?.tradingTags)
    }

    private func assetClasses(for paper: Paper) -> Set<String> {
        normalizedValues(paper.tradingLens?.assetClasses)
    }

    private func horizons(for paper: Paper) -> Set<String> {
        normalizedValues(paper.tradingLens?.horizons)
    }

    private var availableTradingTags: [String] {
        uniqueOptions(from: papersWithLens) { Array(tradingTags(for: $0)) }
    }

    private var availableAssetClasses: [String] {
        uniqueOptions(from: papersWithLens) { Array(assetClasses(for: $0)) }
    }

    private var availableHorizons: [String] {
        uniqueOptions(from: papersWithLens) { Array(horizons(for: $0)) }
    }

    private func uniqueOptions(from papers: [Paper], extract: (Paper) -> [String]) -> [String] {
        var set: Set<String> = []
        for paper in papers {
            for item in extract(paper) {
                let cleaned = item.trimmingCharacters(in: .whitespacesAndNewlines)
                if cleaned.isEmpty { continue }
                set.insert(cleaned)
            }
        }
        return set.sorted { lhs, rhs in
            if lhs.lowercased() == "unknown" { return false }
            if rhs.lowercased() == "unknown" { return true }
            return lhs.localizedCaseInsensitiveCompare(rhs) == .orderedAscending
        }
    }

    private var filteredPapers: [Paper] {
        let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var base = papersWithScores

        if statusFilter != .all {
            base = base.filter { statusFilter.matches($0) }
        }

        if !query.isEmpty {
            base = base.filter { paper in
                if paper.title.lowercased().contains(query) { return true }
                if paper.summary.lowercased().contains(query) { return true }
                if let verdict = paper.tradingLens?.oneLineVerdict, verdict.lowercased().contains(query) { return true }
                if let keywords = paper.keywords, keywords.joined(separator: " ").lowercased().contains(query) { return true }
                return false
            }
        }

        base = base.filter { paper in
            matches(selected: selectedTradingTags, values: tradingTags(for: paper))
                && matches(selected: selectedAssetClasses, values: assetClasses(for: paper))
                && matches(selected: selectedHorizons, values: horizons(for: paper))
        }

        return base.sorted(by: comparator)
    }

    private var comparator: (Paper, Paper) -> Bool {
        { lhs, rhs in
            let ls = lhs.tradingScores ?? lhs.tradingLens?.scores
            let rs = rhs.tradingScores ?? rhs.tradingLens?.scores

            let lImpact = ls?.strategyImpact ?? 0
            let rImpact = rs?.strategyImpact ?? 0
            let lUsability = ls?.usability ?? 0
            let rUsability = rs?.usability ?? 0
            let lNovelty = ls?.novelty ?? 0
            let rNovelty = rs?.novelty ?? 0
            let lConfidence = ls?.confidence ?? 0
            let rConfidence = rs?.confidence ?? 0

            let lPriority = lImpact * lUsability * lConfidence
            let rPriority = rImpact * rUsability * rConfidence

            switch sort {
            case .priority:
                if lPriority != rPriority { return lPriority > rPriority }
                if lNovelty != rNovelty { return lNovelty > rNovelty }
            case .impact:
                if lImpact != rImpact { return lImpact > rImpact }
            case .usability:
                if lUsability != rUsability { return lUsability > rUsability }
            case .novelty:
                if lNovelty != rNovelty { return lNovelty > rNovelty }
            case .confidence:
                if lConfidence != rConfidence { return lConfidence > rConfidence }
            case .recency:
                break
            case .title:
                break
            }

            let ly = lhs.year ?? -10_000
            let ry = rhs.year ?? -10_000
            if sort == .recency, ly != ry { return ly > ry }
            if ly != ry { return ly > ry }
            return lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
        }
    }

    private var lensPoints: [TradingLensPoint] {
        filteredPapers.compactMap { paper in
            guard let lens = paper.tradingLens else { return nil }
            guard let scores = paper.tradingScores ?? lens.scores else { return nil }
            let novelty = scores.novelty ?? 0
            let usability = scores.usability ?? 0
            let impact = scores.strategyImpact ?? 0
            let confidence = scores.confidence ?? 0
            return TradingLensPoint(
                id: paper.id,
                novelty: novelty,
                usability: usability,
                strategyImpact: impact,
                confidence: confidence,
                priority: impact * usability * confidence,
                title: paper.title,
                primaryTag: primaryValue(from: lens.tradingTags),
                primaryAssetClass: primaryValue(from: lens.assetClasses),
                primaryHorizon: primaryValue(from: lens.horizons)
            )
        }
    }

    private var selectedTradingPoint: TradingLensPoint? {
        guard let selectedPointID else { return nil }
        return lensPoints.first(where: { $0.id == selectedPointID })
    }

    private var tagCounts: [TradingTagCount] {
        var counts: [String: Int] = [:]
        for paper in filteredPapers {
            let tags = tradingTags(for: paper)
            for tag in tags {
                counts[tag, default: 0] += 1
            }
        }
        return counts
            .map { TradingTagCount(tag: $0.key, count: $0.value) }
            .sorted { lhs, rhs in
                if lhs.count != rhs.count { return lhs.count > rhs.count }
                return lhs.tag < rhs.tag
            }
    }

    private var tagTrends: [TradingTagTrendPoint] {
        var counts: [String: Int] = [:]
        for entry in tagCounts.prefix(6) {
            counts[entry.tag] = entry.count
        }
        let includeTags = Set(counts.keys)
        guard !includeTags.isEmpty else { return [] }

        var trend: [String: Int] = [:]
        for paper in filteredPapers {
            guard let year = paper.year else { continue }
            for tag in tradingTags(for: paper) where includeTags.contains(tag) {
                let key = "\(tag)#\(year)"
                trend[key, default: 0] += 1
            }
        }
        return trend.compactMap { key, count in
            guard let sep = key.lastIndex(of: "#") else { return nil }
            let tag = String(key[..<sep])
            let yearText = String(key[key.index(after: sep)...])
            guard let year = Int(yearText) else { return nil }
            return TradingTagTrendPoint(tag: tag, year: year, count: count)
        }
    }

    private var yearDomain: ClosedRange<Int> {
        let currentYear = Calendar.current.component(.year, from: Date())
        let years = papers.compactMap(\.year).filter { $0 >= 1900 && $0 <= currentYear + 1 }
        guard let minY = years.min(), let maxY = years.max(), minY <= maxY else { return 1900...currentYear }
        return minY...maxY
    }

    private var hypotheses: [HypothesisCard] {
        var out: [HypothesisCard] = []
        for paper in filteredPapers {
            guard let lens = paper.tradingLens else { continue }
            let primaryTag = primaryValue(from: lens.tradingTags)
            let primaryAsset = primaryValue(from: lens.assetClasses)
            for (idx, h) in (lens.alphaHypotheses ?? []).enumerated() {
                let text = (h.hypothesis ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if text.isEmpty { continue }
                out.append(
                    HypothesisCard(
                        id: "\(paper.id.uuidString)#\(idx)",
                        paperID: paper.id,
                        paperTitle: paper.title,
                        hypothesis: text,
                        features: (h.features ?? []).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty },
                        target: h.target,
                        horizon: h.horizon,
                        primaryTag: primaryTag,
                        primaryAssetClass: primaryAsset
                    )
                )
            }
        }
        return out
    }

    private var hasActiveFilters: Bool {
        statusFilter != .all
            || !searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !selectedTradingTags.isEmpty
            || !selectedAssetClasses.isEmpty
            || !selectedHorizons.isEmpty
    }

    private func clearFilters() {
        withAnimation(.easeInOut(duration: 0.12)) {
            searchQuery = ""
            statusFilter = .all
            selectedTradingTags.removeAll()
            selectedAssetClasses.removeAll()
            selectedHorizons.removeAll()
        }
    }

    private func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[mid]
        }
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    headerCard
                    scatterCard
                    tagsCard
                    papersCard
                    if showHypotheses { hypothesesCard }
                    Spacer(minLength: 24)
                }
                .padding()
            }
            .navigationTitle("Trading lens")
        }
        .sheet(item: $selectedPaperDetail) { paper in
            PaperDetailView(paper: paper)
                .environmentObject(model)
        }
    }

    private var headerCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Trading lens").font(.headline)
                        Text("Filter, rank, and harvest actionable hypotheses from paper scorecards.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Toggle("Show hypotheses", isOn: $showHypotheses)
                        .toggleStyle(.switch)
                        .font(.caption)
                }

                let withLens = papersWithLens.count
                let total = papers.count
                let coverage = total > 0 ? Double(withLens) / Double(total) : 0
                HStack(spacing: 12) {
                    StatPill(label: "Lens", value: "\(withLens)/\(total) (\(Int(coverage * 100))%)")
                    StatPill(label: "Missing", value: "\(papersMissingLensCount)")
                    StatPill(label: "Scored", value: "\(papersWithScores.count)")
                    if !model.tradingLensFailures.isEmpty {
                        Menu {
                            Button("Retry all failures") {
                                for paperID in model.tradingLensFailures.keys {
                                    model.generateTradingLens(for: paperID)
                                }
                            }
                            Divider()
                            ForEach(failureItems.prefix(12), id: \.id) { item in
                                Button("Retry: \(item.title)") {
                                    model.generateTradingLens(for: item.id)
                                }
                            }
                            if model.tradingLensFailures.count > 12 {
                                Text("…and \(model.tradingLensFailures.count - 12) more")
                            }
                        } label: {
                            StatPill(label: "Failures", value: "\(model.tradingLensFailures.count)")
                        }
                    }
                }

                if !lensPoints.isEmpty {
                    let priorities = lensPoints.map(\.priority)
                    let avgN = average(lensPoints.map(\.novelty)) ?? 0
                    let avgU = average(lensPoints.map(\.usability)) ?? 0
                    let avgI = average(lensPoints.map(\.strategyImpact)) ?? 0
                    let avgC = average(lensPoints.map(\.confidence)) ?? 0
                    HStack(spacing: 12) {
                        StatPill(label: "Median P", value: String(format: "%.1f", median(priorities) ?? 0))
                        StatPill(label: "Top P", value: String(format: "%.1f", priorities.max() ?? 0))
                        StatPill(label: "Avg N/U/I/C", value: String(format: "%.1f/%.1f/%.1f/%.2f", avgN, avgU, avgI, avgC))
                    }
                }

                if model.tradingLensBackfillInFlight || papersMissingLensCount > 0 {
                    Divider().padding(.vertical, 4)
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Generate missing scorecards")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            if model.tradingLensBackfillInFlight {
                                Button("Stop") { model.cancelTradingLensBackfill() }
                                    .buttonStyle(.bordered)
                            } else {
                                Menu {
                                    Button("Backfill missing (all)") { model.backfillTradingLensForMissingPapers() }
                                    Button("Backfill missing (25 newest)") { model.backfillTradingLensForMissingPapers(limit: 25) }
                                } label: {
                                    Label("Backfill", systemImage: "sparkles")
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }

                        if model.tradingLensBackfillInFlight {
                            ProgressView(value: model.tradingLensBackfillProgress)
                                .tint(.mint)
                            Text("\(model.tradingLensBackfillCompletedCount)/\(max(1, model.tradingLensBackfillTotalCount)) · \(model.tradingLensBackfillCurrentPaper)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else if papersMissingLensCount > 0 {
                            Text("\(papersMissingLensCount) papers missing a trading lens scorecard in the current year filter.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    private var scatterCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Novelty vs usability").font(.headline)
                    Spacer()
                    if let pt = selectedTradingPoint {
                        Button {
                            if let paper = papers.first(where: { $0.id == pt.id }) {
                                selectedPaperDetail = paper
                            }
                        } label: {
                            Label("Open paper", systemImage: "doc.text.magnifyingglass")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    if selectedPointID != nil {
                        Button {
                            selectedPointID = nil
                        } label: {
                            Label("Clear", systemImage: "xmark")
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if lensPoints.isEmpty {
                    Text("No scored papers match your filters yet.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    TradingLensScatterChartView(points: lensPoints, selectedPointID: $selectedPointID)
                        .frame(minHeight: 420)
                    Text("x=usability · y=novelty · color=strategy impact · opacity=confidence")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    if let pt = selectedTradingPoint,
                       let paper = papers.first(where: { $0.id == pt.id }) {
                        Divider().padding(.vertical, 4)
                        TradingSelectedPaperCard(
                            paper: paper,
                            onOpenDetails: { selectedPaperDetail = paper },
                            selectedTradingTags: $selectedTradingTags,
                            selectedAssetClasses: $selectedAssetClasses,
                            selectedHorizons: $selectedHorizons
                        )
                    }
                }
            }
        }
    }

    private var tagsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Tags").font(.headline)

                if tagCounts.isEmpty {
                    Text("No tags available in the current filter.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    TradingTagBarChartView(counts: Array(tagCounts.prefix(14)))
                        .frame(minHeight: 280)

                    Divider().padding(.vertical, 4)
                    Text("Quick tag filters").font(.subheadline.weight(.semibold))
                    TradingTagChipGrid(
                        counts: Array(tagCounts.prefix(18)),
                        selection: $selectedTradingTags
                    )
                }

                if !tagTrends.isEmpty {
                    Divider().padding(.vertical, 4)
                    Text("Tag frequency over time (top tags)").font(.subheadline.weight(.semibold))
                    TradingTagTrendChartView(trends: tagTrends, domain: yearDomain)
                        .frame(minHeight: 280)
                }
            }
        }
    }

    private var papersCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Papers").font(.headline)
                    Spacer()
                    Picker("Status", selection: $statusFilter) {
                        ForEach(StatusFilter.allCases) { option in
                            Text(option.label).tag(option)
                        }
                    }
                    .pickerStyle(.menu)

                    Picker("Sort", selection: $sort) {
                        ForEach(TradingSort.allCases) { option in
                            Text(option.label).tag(option)
                        }
                    }
                    .pickerStyle(.menu)

                    if hasActiveFilters {
                        Button {
                            clearFilters()
                        } label: {
                            Label("Clear", systemImage: "xmark.circle")
                        }
                        .buttonStyle(.bordered)
                    }
                }

                SearchField(text: $searchQuery, placeholder: "Search title, summary, verdict…")

                HStack(spacing: 10) {
                    MultiSelectMenu(title: "Tags", emptyLabel: "Any tag", options: availableTradingTags, selection: $selectedTradingTags)
                    MultiSelectMenu(title: "Assets", emptyLabel: "Any asset", options: availableAssetClasses, selection: $selectedAssetClasses)
                    MultiSelectMenu(title: "Horizon", emptyLabel: "Any horizon", options: availableHorizons, selection: $selectedHorizons)
                    Spacer()
                }
                .font(.caption)

                Divider().overlay(Color.white.opacity(0.12))

                if filteredPapers.isEmpty {
                    Text("No scored papers match your filters.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(filteredPapers) { paper in
                            TradingPaperRow(paper: paper, onOpen: {
                                selectedPaperDetail = paper
                            }, onCreateProject: {
                                if let project = model.createStrategyProject(from: paper.id) {
                                    nav.selectedTab = .projects
                                    nav.requestedStrategyProjectID = project.id
                                }
                            })
                            .contextMenu {
                                Button("Open details") { selectedPaperDetail = paper }
                                Button("Create project") {
                                    if let project = model.createStrategyProject(from: paper.id) {
                                        nav.selectedTab = .projects
                                        nav.requestedStrategyProjectID = project.id
                                    }
                                }
                                if paper.tradingLens == nil {
                                    Button("Generate trading lens") { model.generateTradingLens(for: paper.id) }
                                } else {
                                    Button("Re-generate trading lens") { model.generateTradingLens(for: paper.id) }
                                }
                                Button("Generate strategy blueprint") { model.generateStrategyBlueprint(for: paper.id) }
                                Button("Audit backtest") { model.auditBacktest(for: paper.id) }
                            }
                        }
                    }
                }
            }
        }
    }

    private var hypothesesCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Alpha hypotheses").font(.headline)
                    Spacer()
                    Text("\(hypotheses.count)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text("Harvested from paper trading lens scorecards (not deduplicated yet).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                if hypotheses.isEmpty {
                    Text("No hypotheses available in the current filter.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(hypotheses.prefix(40)) { h in
                            HypothesisRow(card: h, onOpenPaper: {
                                if let paper = papers.first(where: { $0.id == h.paperID }) {
                                    selectedPaperDetail = paper
                                }
                            }, onCreateProject: {
                                if let project = model.createStrategyProject(from: h.paperID) {
                                    nav.selectedTab = .projects
                                    nav.requestedStrategyProjectID = project.id
                                }
                            })
                        }
                        if hypotheses.count > 40 {
                            Text("Showing 40 of \(hypotheses.count). Narrow filters to see more.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct StatPill: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.bold())
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption2.monospacedDigit())
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.thinMaterial, in: Capsule())
        .overlay(Capsule().stroke(Color.white.opacity(0.14), lineWidth: 1))
    }
}

@available(macOS 26, iOS 26, *)
private struct TradingPaperRow: View {
    @EnvironmentObject private var model: AppModel

    let paper: Paper
    var onOpen: (() -> Void)?
    var onCreateProject: (() -> Void)? = nil

    @State private var isHovering = false

    private var scores: TradingLensScores? {
        paper.tradingScores ?? paper.tradingLens?.scores
    }

    private var priority: Double {
        let impact = scores?.strategyImpact ?? 0
        let usability = scores?.usability ?? 0
        let confidence = scores?.confidence ?? 0
        return impact * usability * confidence
    }

    private func firstNonUnknown(_ values: [String]?) -> String? {
        for raw in (values ?? []) {
            let cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if cleaned.isEmpty { continue }
            if cleaned.lowercased() == "unknown" { continue }
            return cleaned
        }
        return nil
    }

    private var isMissingTradingLens: Bool {
        if paper.tradingLens == nil { return true }
        return (paper.tradingScores ?? paper.tradingLens?.scores) == nil
    }

    private var background: Color {
        if isHovering { return Color.white.opacity(0.08) }
        return Color.white.opacity(0.04)
    }

    var body: some View {
        let primaryTag = firstNonUnknown(paper.tradingLens?.tradingTags)
        let primaryAsset = firstNonUnknown(paper.tradingLens?.assetClasses)
        let primaryHorizon = firstNonUnknown(paper.tradingLens?.horizons)
        let shape = RoundedRectangle(cornerRadius: 14, style: .continuous)

        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(paper.title)
                    .font(.subheadline.bold())
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                Spacer()
                if let year = paper.year {
                    Text("\(year)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            if let verdict = paper.tradingLens?.oneLineVerdict, !verdict.isEmpty {
                Text(verdict)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            } else {
                Text(paper.summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            if primaryTag != nil || primaryAsset != nil || primaryHorizon != nil {
                HStack(spacing: 6) {
                    if let primaryTag {
                        miniPill(primaryTag, tint: .teal)
                    }
                    if let primaryAsset {
                        miniPill(primaryAsset, tint: .mint)
                    }
                    if let primaryHorizon {
                        miniPill(primaryHorizon, tint: .orange)
                    }
                    Spacer()
                }
            }

            HStack(alignment: .center, spacing: 12) {
                TradingScoreStrip(scores: scores, priority: priority)
                Spacer()
                if let status = paper.readingStatus {
                    Text(status.label)
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.blue.opacity(0.12), in: Capsule())
                        .foregroundStyle(.blue)
                }
#if os(macOS)
                if isHovering {
                    quickActions
                        .transition(.opacity.combined(with: .move(edge: .trailing)))
                }
#endif
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(background, in: shape)
        .overlay(shape.stroke(Color.white.opacity(isHovering ? 0.22 : 0.10), lineWidth: 1))
        .contentShape(shape)
        .onTapGesture { onOpen?() }
#if os(macOS)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.12)) {
                isHovering = hovering
            }
        }
#endif
        .accessibilityAddTraits(.isButton)
    }

    @ViewBuilder
    private func miniPill(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(tint.opacity(0.14), in: Capsule())
            .foregroundStyle(tint)
            .lineLimit(1)
    }

#if os(macOS)
    private var quickActions: some View {
        HStack(spacing: 8) {
            if isMissingTradingLens {
                Button {
                    model.generateTradingLens(for: paper.id)
                } label: {
                    Image(systemName: "sparkles")
                }
                .buttonStyle(.borderless)
                .help("Generate trading lens")
            }

            if let onCreateProject {
                Button {
                    onCreateProject()
                } label: {
                    Image(systemName: "point.3.connected.trianglepath")
                }
                .buttonStyle(.borderless)
                .help("Create project")
            }

            Button {
                PlatformOpen.open(url: paper.fileURL)
            } label: {
                Image(systemName: "doc.richtext")
            }
            .buttonStyle(.borderless)
            .help("Open PDF")

            if let noteURL = model.obsidianNoteURL(for: paper.id) {
                Button {
                    PlatformOpen.open(url: noteURL)
                } label: {
                    Image(systemName: "note.text")
                }
                .buttonStyle(.borderless)
                .help("Open Obsidian note")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
#endif
}

@available(macOS 26, iOS 26, *)
private struct SearchField: View {
    @Binding var text: String
    let placeholder: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
    }
}

@available(macOS 26, iOS 26, *)
private struct MultiSelectMenu: View {
    let title: String
    let emptyLabel: String
    let options: [String]
    @Binding var selection: Set<String>

    private var labelText: String {
        selection.isEmpty ? emptyLabel : "\(title) (\(selection.count))"
    }

    var body: some View {
        Menu {
            if selection.isEmpty == false {
                Button("Clear") { selection.removeAll() }
                Divider()
            }
            if options.isEmpty {
                Text("No data")
            } else {
                ForEach(options, id: \.self) { option in
                    Button {
                        if selection.contains(option) {
                            selection.remove(option)
                        } else {
                            selection.insert(option)
                        }
                    } label: {
                        HStack {
                            Text(option)
                            Spacer()
                            if selection.contains(option) {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
        } label: {
            Text(labelText)
        }
        .buttonStyle(.bordered)
        .tint(Color.white.opacity(0.12))
        .disabled(options.isEmpty)
    }
}

@available(macOS 26, iOS 26, *)
private struct TradingTagChipGrid: View {
    let counts: [TradingTagCount]
    @Binding var selection: Set<String>

    private let columns: [GridItem] = [
        GridItem(.adaptive(minimum: 160), spacing: 8, alignment: .leading)
    ]

    var body: some View {
        LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
            ForEach(counts) { entry in
                TradingFilterChip(
                    label: entry.tag,
                    value: "\(entry.count)",
                    tint: .teal,
                    isSelected: selection.contains(entry.tag),
                    onTap: {
                        if selection.contains(entry.tag) {
                            selection.remove(entry.tag)
                        } else {
                            selection.insert(entry.tag)
                        }
                    }
                )
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct TradingFilterChip: View {
    let label: String
    var value: String? = nil
    let tint: Color
    let isSelected: Bool
    var onTap: (() -> Void)?

    var body: some View {
        Button {
            onTap?()
        } label: {
            HStack(spacing: 8) {
                Text(label)
                    .font(.caption2.weight(.semibold))
                    .lineLimit(1)
                Spacer(minLength: 6)
                if let value {
                    Text(value)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(tint)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                (isSelected ? tint.opacity(0.20) : Color.white.opacity(0.06)),
                in: Capsule()
            )
            .overlay(
                Capsule().stroke(
                    (isSelected ? tint.opacity(0.5) : Color.white.opacity(0.12)),
                    lineWidth: 1
                )
            )
        }
        .buttonStyle(.plain)
        .help(isSelected ? "Remove filter" : "Filter")
    }
}

@available(macOS 26, iOS 26, *)
private struct TradingSelectedPaperCard: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var nav: AppNavigation

    let paper: Paper
    var onOpenDetails: (() -> Void)? = nil
    @Binding var selectedTradingTags: Set<String>
    @Binding var selectedAssetClasses: Set<String>
    @Binding var selectedHorizons: Set<String>

    private let columns: [GridItem] = [
        GridItem(.adaptive(minimum: 160), spacing: 8, alignment: .leading)
    ]

    private var scores: TradingLensScores? {
        paper.tradingScores ?? paper.tradingLens?.scores
    }

    private var priority: Double {
        let impact = scores?.strategyImpact ?? 0
        let usability = scores?.usability ?? 0
        let confidence = scores?.confidence ?? 0
        return impact * usability * confidence
    }

    private var titleLine: String {
        if let year = paper.year { return "\(paper.title) (\(year))" }
        return paper.title
    }

    private func cleaned(_ values: [String]?) -> [String] {
        (values ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && $0.lowercased() != "unknown" }
    }

    private func toggle(_ value: String, in selection: Binding<Set<String>>) {
        if selection.wrappedValue.contains(value) {
            selection.wrappedValue.remove(value)
        } else {
            selection.wrappedValue.insert(value)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("Selected")
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "P %.1f", priority))
                    .font(.caption2.monospacedDigit())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.16), in: Capsule())
                    .foregroundStyle(.orange)
            }

            Button {
                onOpenDetails?()
            } label: {
                Text(titleLine)
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)

            if let verdict = paper.tradingLens?.oneLineVerdict,
               !verdict.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(verdict)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }

            TradingScoreStrip(scores: scores, priority: priority)

            HStack(spacing: 10) {
                Button {
                    onOpenDetails?()
                } label: {
                    Label("Details", systemImage: "doc.text.magnifyingglass")
                }
                .buttonStyle(.borderedProminent)

                Button {
                    PlatformOpen.open(url: paper.fileURL)
                } label: {
                    Label("PDF", systemImage: "doc.richtext")
                }
                .buttonStyle(.bordered)

                if let noteURL = model.obsidianNoteURL(for: paper.id) {
                    Button {
                        PlatformOpen.open(url: noteURL)
                    } label: {
                        Label("Note", systemImage: "note.text")
                    }
                    .buttonStyle(.bordered)
                }

                Button {
                    if let project = model.createStrategyProject(from: paper.id) {
                        nav.selectedTab = .projects
                        nav.requestedStrategyProjectID = project.id
                    }
                } label: {
                    Label("Project", systemImage: "point.3.connected.trianglepath")
                }
                .buttonStyle(.bordered)

                Spacer()
            }
            .font(.caption)

            let tags = cleaned(paper.tradingLens?.tradingTags).prefix(10)
            let assets = cleaned(paper.tradingLens?.assetClasses).prefix(8)
            let horizons = cleaned(paper.tradingLens?.horizons).prefix(8)

            if !tags.isEmpty || !assets.isEmpty || !horizons.isEmpty {
                Divider().padding(.vertical, 2)
            }

            if !tags.isEmpty {
                Text("Tags").font(.caption.bold()).foregroundStyle(.secondary)
                LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                    ForEach(Array(tags), id: \.self) { tag in
                        TradingFilterChip(
                            label: tag,
                            tint: .teal,
                            isSelected: selectedTradingTags.contains(tag),
                            onTap: { toggle(tag, in: $selectedTradingTags) }
                        )
                    }
                }
            }

            if !assets.isEmpty {
                Text("Assets").font(.caption.bold()).foregroundStyle(.secondary)
                LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                    ForEach(Array(assets), id: \.self) { asset in
                        TradingFilterChip(
                            label: asset,
                            tint: .mint,
                            isSelected: selectedAssetClasses.contains(asset),
                            onTap: { toggle(asset, in: $selectedAssetClasses) }
                        )
                    }
                }
            }

            if !horizons.isEmpty {
                Text("Horizons").font(.caption.bold()).foregroundStyle(.secondary)
                LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                    ForEach(Array(horizons), id: \.self) { horizon in
                        TradingFilterChip(
                            label: horizon,
                            tint: .orange,
                            isSelected: selectedHorizons.contains(horizon),
                            onTap: { toggle(horizon, in: $selectedHorizons) }
                        )
                    }
                }
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
    }
}

@available(macOS 26, iOS 26, *)
private struct HypothesisRow: View {
    let card: TradingLensView.HypothesisCard
    var onOpenPaper: (() -> Void)?
    var onCreateProject: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(card.hypothesis)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(3)
                Spacer()
                if let tag = card.primaryTag, !tag.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, tag.lowercased() != "unknown" {
                    Text(tag)
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.teal.opacity(0.14), in: Capsule())
                        .foregroundStyle(.teal)
                }
            }

            if !card.features.isEmpty {
                Text("features: \(card.features.prefix(6).joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            if let target = card.target, !target.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("target: \(target)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if let horizon = card.horizon, !horizon.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("horizon: \(horizon)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                Button { onOpenPaper?() } label: {
                    Text(card.paperTitle)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .buttonStyle(.plain)

                Spacer()

                Button {
                    let parts: [String] = [
                        card.hypothesis,
                        card.features.isEmpty ? nil : "features: \(card.features.joined(separator: ", "))",
                        card.target?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false ? "target: \(card.target ?? "")" : nil,
                        card.horizon?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false ? "horizon: \(card.horizon ?? "")" : nil
                    ].compactMap { $0 }
                    PlatformClipboard.copy(parts.joined(separator: "\n"))
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .help("Copy hypothesis")

                if onCreateProject != nil {
                    Button { onCreateProject?() } label: {
                        Image(systemName: "point.3.connected.trianglepath")
                    }
                    .buttonStyle(.borderless)
                    .help("Create project")
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        )
    }
}
