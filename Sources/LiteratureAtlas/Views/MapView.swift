import SwiftUI
import FoundationModels

private enum MapPalette {
    static let backdrop = LinearGradient(
        colors: [
            Color(red: 0.08, green: 0.08, blue: 0.16),
            Color(red: 0.12, green: 0.09, blue: 0.22),
            Color(red: 0.10, green: 0.14, blue: 0.30)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let canvas = LinearGradient(
        colors: [
            Color(red: 0.14, green: 0.16, blue: 0.42),
            Color(red: 0.22, green: 0.16, blue: 0.55),
            Color(red: 0.10, green: 0.26, blue: 0.62)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let glow = RadialGradient(
        colors: [Color.white.opacity(0.22), .clear],
        center: .center,
        startRadius: 10,
        endRadius: 220
    )

    static let panel = Color.white.opacity(0.08)
    static let panelStroke = Color.white.opacity(0.14)

    static func nodeGradient(for index: Int) -> LinearGradient {
        let palette: [[Color]] = [
            [Color(red: 0.98, green: 0.78, blue: 0.18), Color(red: 0.93, green: 0.58, blue: 0.08)],
            [Color(red: 0.50, green: 0.83, blue: 0.98), Color(red: 0.20, green: 0.67, blue: 0.95)],
            [Color(red: 0.74, green: 0.65, blue: 1.00), Color(red: 0.54, green: 0.38, blue: 0.98)],
            [Color(red: 0.71, green: 0.93, blue: 0.76), Color(red: 0.34, green: 0.78, blue: 0.52)],
            [Color(red: 1.00, green: 0.64, blue: 0.79), Color(red: 0.95, green: 0.36, blue: 0.59)]
        ]
        let colors = palette[index % palette.count]
        return LinearGradient(colors: colors, startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    static func categoricalTint(for key: String) -> Color {
        let cleaned = key.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty || cleaned.lowercased() == "unknown" {
            return Color.white.opacity(0.55)
        }
        let palette: [Color] = [
            Color(red: 0.98, green: 0.78, blue: 0.18), // gold
            Color(red: 0.50, green: 0.83, blue: 0.98), // cyan
            Color(red: 0.74, green: 0.65, blue: 1.00), // lavender
            Color(red: 0.71, green: 0.93, blue: 0.76), // mint
            Color(red: 1.00, green: 0.64, blue: 0.79), // pink
            Color(red: 0.96, green: 0.50, blue: 0.28), // orange
            Color(red: 0.45, green: 0.93, blue: 0.88), // teal
            Color(red: 0.92, green: 0.92, blue: 0.46), // lemon
            Color(red: 0.64, green: 0.80, blue: 0.31), // lime
            Color(red: 0.62, green: 0.54, blue: 0.98)  // violet
        ]
        let idx = abs(stableHash(cleaned)) % palette.count
        return palette[idx].opacity(0.85)
    }

    private static func stableHash(_ s: String) -> Int {
        var h: UInt64 = 5381
        for scalar in s.unicodeScalars {
            h = ((h << 5) &+ h) &+ UInt64(scalar.value)
        }
        return Int(truncatingIfNeeded: h)
    }
}

private enum ClusterResolution: String, CaseIterable {
    case low, medium, high

    var label: String {
        switch self {
        case .low: return "Low (10-12)"
        case .medium: return "Medium (12-16)"
        case .high: return "High (16-20)"
        }
    }

    var subtopicRange: ClosedRange<Int> {
        switch self {
        case .low: return 10...12
        case .medium: return 12...16
        case .high: return 16...20
        }
    }
}

private enum ZoomLevel: Int, CaseIterable {
    case mega = 0
    case topics = 1
    case papers = 2

    var label: String {
        switch self {
        case .mega: return "Mega-topics"
        case .topics: return "Subtopics"
        case .papers: return "Papers"
        }
    }
}

private enum PaperStatusFilter: String, CaseIterable, Identifiable {
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

private enum PaperSort: String, CaseIterable, Identifiable {
    case recommended
    case recency
    case title
    case noveltyZ
    case consensusZ
    case influencePos
    case clusterConfidence
    case recombination
    case rigor
    case openness
    case tradingPriority
    case tradingImpact
    case tradingUsability
    case tradingNovelty

    var id: String { rawValue }

    var label: String {
        switch self {
        case .recommended: return "Recommended"
        case .recency: return "Recency"
        case .title: return "Title"
        case .noveltyZ: return "Novelty (z)"
        case .consensusZ: return "Consensus (z)"
        case .influencePos: return "Influence (+)"
        case .clusterConfidence: return "Cluster confidence"
        case .recombination: return "Recombination"
        case .rigor: return "Rigor proxy"
        case .openness: return "Openness"
        case .tradingPriority: return "Trading priority"
        case .tradingImpact: return "Trading impact"
        case .tradingUsability: return "Trading usability"
        case .tradingNovelty: return "Trading novelty"
        }
    }
}

private enum PaperColorBy: String, CaseIterable, Identifiable {
    case novelty
    case tradingTag
    case assetClass
    case horizon

    var id: String { rawValue }

    var label: String {
        switch self {
        case .novelty: return "Novelty"
        case .tradingTag: return "Trading tag"
        case .assetClass: return "Asset class"
        case .horizon: return "Horizon"
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct MapBackdropView: View {
    @State private var phase: Double = 0

    var body: some View {
        MapPalette.backdrop
            .overlay(
                AngularGradient(
                    colors: [
                        Color.purple.opacity(0.22),
                        Color.cyan.opacity(0.18),
                        Color.pink.opacity(0.20),
                        Color.purple.opacity(0.22)
                    ],
                    center: .center
                )
                .blur(radius: 160)
                .opacity(0.35)
                .rotationEffect(.degrees(phase * 360))
                .blendMode(.screen)
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)
            .onAppear {
                guard phase == 0 else { return }
                withAnimation(.linear(duration: 28).repeatForever(autoreverses: false)) {
                    phase = 1
                }
            }
    }
}

@available(macOS 26, iOS 26, *)
struct MapView: View {
    @EnvironmentObject private var model: AppModel
    @State private var resolution: ClusterResolution = .medium
    @State private var lens: MapLens = .standard
    @State private var zoomLevel: ZoomLevel = .mega
    @State private var selectedMegaID: Int?
    @State private var selectedSubtopicID: Int?
    @State private var selectedClusterIDs: Set<Int> = []
    @State private var selectedPaper: Paper?
    @State private var paperHighlights: [UUID: PaperNoveltyScore] = [:]
    @State private var paperSearchQuery: String = ""
    @State private var paperSort: PaperSort = .recommended
    @State private var paperStatusFilter: PaperStatusFilter = .all
    @State private var selectedTradingTags: Set<String> = []
    @State private var selectedAssetClasses: Set<String> = []
    @State private var selectedHorizons: Set<String> = []
    @State private var paperColorBy: PaperColorBy = .novelty
    @State private var isNamingSubtopics: Bool = false
    @State private var isNamingMegaTopics: Bool = false
    @State private var debateText: String = ""
    @State private var showDebate: Bool = false
    @State private var showGlossary: Bool = false
    @State private var showExportAlert: Bool = false
    @State private var exportMessage: String = ""
    @State private var hypothetical: HypotheticalPaper?
    @State private var showHypothetical: Bool = false

    private var selectedMegaCluster: Cluster? {
        if let id = selectedMegaID {
            return model.megaClusters.first(where: { $0.id == id })
        }
        return nil
    }

    private var selectedSubtopicCluster: Cluster? {
        if let id = selectedSubtopicID {
            if let mega = selectedMegaCluster, let sub = mega.subclusters?.first(where: { $0.id == id }) {
                return sub
            }
            return model.clusters.first(where: { $0.id == id })
        }
        return nil
    }

    private var activeClusters: [Cluster] {
        switch zoomLevel {
        case .mega:
            return model.lensAdjustedClusters(model.megaClusters, lens: lens)
        case .topics:
            let base = (selectedMegaCluster ?? model.megaClusters.first)?.subclusters ?? []
            return model.lensAdjustedClusters(base, lens: lens)
        case .papers:
            return []
        }
    }

    private var paperSubtopics: [Cluster] {
        let fallback = model.megaClusters.first?.subclusters ?? model.clusters
        let base = selectedMegaCluster?.subclusters ?? fallback
        return model.lensAdjustedClusters(base, lens: lens)
    }

    private var activePapers: [Paper] {
        guard zoomLevel == .papers else { return [] }
        guard let sub = selectedSubtopicCluster else { return [] }
        return model.explorationPapers(in: sub).sorted {
            let ly = $0.year ?? -10_000
            let ry = $1.year ?? -10_000
            if ly != ry { return ly > ry }
            return $0.title < $1.title
        }
    }

    private var activePaperDriftVector: (dx: Double, dy: Double)? {
        guard let sub = selectedSubtopicCluster else { return nil }
        guard let drift = model.analyticsSummary?.drift else { return nil }
        return drift.last(where: { $0.clusterID == sub.id }).map { ($0.dx ?? 0, $0.dy ?? 0) }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                MapBackdropView()

                VStack(alignment: .leading, spacing: 14) {
                    header

                    if model.papers.isEmpty {
                        Text("Ingest some PDFs first on the Ingest tab.")
                            .foregroundStyle(.secondary)
                        Spacer()
                    } else {
                        controlDeck

                        if zoomLevel == .mega, !model.megaClusters.isEmpty {
                            HStack {
                                Spacer()
                                Button {
                                    guard !isNamingMegaTopics else { return }
                                    isNamingMegaTopics = true
                                    Task {
                                        await model.nameMegaTopicsWithAI()
                                        await MainActor.run { isNamingMegaTopics = false }
                                    }
                                } label: {
                                    if isNamingMegaTopics {
                                        Label("Naming…", systemImage: "sparkles")
                                    } else {
                                        Label("Name mega-topics", systemImage: "sparkles")
                                    }
                                }
                                .buttonStyle(.bordered)
                                .disabled(isNamingMegaTopics)
                                .contextMenu {
                                    Button("Force AI re-name mega-topics") {
                                        guard !isNamingMegaTopics else { return }
                                        isNamingMegaTopics = true
                                        Task {
                                            await model.nameMegaTopicsWithAI(force: true)
                                            await MainActor.run { isNamingMegaTopics = false }
                                        }
                                    }
                                }
                            }
                        }

                        if zoomLevel == .topics, let mega = selectedMegaCluster {
                            HStack {
                                breadcrumb(for: mega)
                                Spacer()
                                Button {
                                    guard !isNamingSubtopics else { return }
                                    isNamingSubtopics = true
                                    Task {
                                        await model.nameSubtopicsWithAI(forMegaID: mega.id)
                                        await MainActor.run { isNamingSubtopics = false }
                                    }
                                } label: {
                                    if isNamingSubtopics {
                                        Label("Naming…", systemImage: "sparkles")
                                    } else {
                                        Label("Name subtopics", systemImage: "sparkles")
                                    }
                                }
                                .buttonStyle(.bordered)
                                .disabled(isNamingSubtopics)
                                .contextMenu {
                                    Button("Force AI re-name subtopics") {
                                        guard !isNamingSubtopics else { return }
                                        isNamingSubtopics = true
                                        Task {
                                            await model.nameSubtopicsWithAI(forMegaID: mega.id, force: true)
                                            await MainActor.run { isNamingSubtopics = false }
                                        }
                                    }
                                }
                            }
                        }
                        if zoomLevel == .papers, let sub = selectedSubtopicCluster {
                            paperBreadcrumb(for: sub)
                        } else if zoomLevel == .papers {
                            Text("Zoomed to papers: pick a subtopic node first.")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.7))
                        }

                        if zoomLevel != .papers && activeClusters.isEmpty {
                            Text("Run clustering to see the map.")
                                .foregroundStyle(.white.opacity(0.7))
                            Spacer()
                        } else if zoomLevel == .papers {
	                            PaperMapAndSidebar(
	                                cluster: selectedSubtopicCluster,
	                                subtopics: paperSubtopics,
	                                papers: activePapers,
	                                highlights: paperHighlights,
	                                driftVector: activePaperDriftVector,
	                                searchQuery: $paperSearchQuery,
	                                statusFilter: $paperStatusFilter,
	                                sort: $paperSort,
	                                selectedTradingTags: $selectedTradingTags,
	                                selectedAssetClasses: $selectedAssetClasses,
	                                selectedHorizons: $selectedHorizons,
	                                colorBy: $paperColorBy,
	                                onSelectPaper: { paper in
	                                    withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
	                                        selectedPaper = paper
	                                    }
                                },
                                onSelectSubtopic: { cluster in
                                    selectSubtopic(cluster)
                                }
                            )
                            .frame(minHeight: 360)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                        } else {
                            ClusterMapAndSidebar(
                                clusters: activeClusters,
                                selectedClusterIDs: $selectedClusterIDs,
                                isZoomed: zoomLevel == .topics,
                                showBridging: zoomLevel == .topics,
                                onZoomOut: zoomLevel == .mega ? nil : {
                                    withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                        zoomLevel = .mega
                                        selectedMegaID = nil
                                        selectedSubtopicID = nil
                                        selectedClusterIDs.removeAll()
                                    }
                                },
                                onZoom: zoomLevel == .mega ? { cluster in
                                    selectMega(cluster)
                                } : { cluster in
                                    selectSubtopic(cluster)
                                },
                                onSelect: zoomLevel == .mega ? { cluster in
                                    selectMega(cluster)
                                } : { cluster in
                                    if zoomLevel == .topics { selectSubtopic(cluster) }
                                },
                                lensLabel: lens.label
                            )
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                        }

                        if selectedClusterIDs.count == 2 {
                            let pair = Array(selectedClusterIDs)
                            GlassPanel {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Compare & brainstorm").font(.headline).foregroundStyle(.white)
                                    HStack {
                                        Button {
                                            showDebate = true
                                            debateText = "Generating…"
                                            Task {
                                                let text = await model.generateDebateBetweenClusters(firstID: pair[0], secondID: pair[1], rounds: 4, useAI: false)
                                                await MainActor.run { debateText = text }
                                            }
                                        } label: {
                                            Label("Simulate debate", systemImage: "person.3.sequence")
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .contextMenu {
                                            Button("Generate debate (AI)") {
                                                showDebate = true
                                                debateText = "Generating…"
                                                Task {
                                                    let text = await model.generateDebateBetweenClusters(firstID: pair[0], secondID: pair[1], rounds: 4, useAI: true)
                                                    await MainActor.run { debateText = text }
                                                }
                                            }
                                        }
                                        Button {
                                            if let ghost = model.generateWhatIfPaper(for: pair) {
                                                hypothetical = ghost
                                                showHypothetical = true
                                            }
                                        } label: {
                                            Label("Generate what-if paper", systemImage: "lightbulb")
                                        }
                                        .buttonStyle(.bordered)
                                    }
                                }
                            }
                        }
                    }
                }
                .padding()
                .animation(.spring(response: 0.35, dampingFraction: 0.85), value: zoomLevel)
                .task(id: PaperHighlightTaskKey(
                    zoomLevel: zoomLevel,
                    subtopicID: selectedSubtopicID,
                    papersCount: model.papers.count,
                    yearFilterEnabled: model.yearFilterEnabled,
                    yearFilterStart: model.yearFilterStart,
                    yearFilterEnd: model.yearFilterEnd
                )) {
                    guard zoomLevel == .papers, let sub = selectedSubtopicCluster else {
                        await MainActor.run { paperHighlights = [:] }
                        return
                    }
                    let papers = activePapers
                    guard papers.count >= 2 else {
                        await MainActor.run { paperHighlights = [:] }
                        return
                    }

                    let clusterSnapshot = sub
                    let papersSnapshot = papers
                    let highlights = await Task.detached(priority: .userInitiated) {
                        let scores = TemporalAnalytics.noveltyScores(papers: papersSnapshot, clusters: [clusterSnapshot], neighbors: 3)
                        return Dictionary(uniqueKeysWithValues: scores.map { ($0.paperID, $0) })
                    }.value

                    await MainActor.run { paperHighlights = highlights }
                }
            }
            .navigationTitle("Map")
            .alert("Export", isPresented: $showExportAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(exportMessage)
            }
            .sheet(isPresented: $showDebate) {
                NavigationStack {
                    ScrollView {
                        Text(debateText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                    }
                    .navigationTitle("Cluster debate")
                    .toolbar {
                        ToolbarItem(placement: .primaryAction) {
                            Button("Close") { showDebate = false }
                        }
                    }
                }
                .frame(minWidth: 420, minHeight: 360)
            }
            .sheet(isPresented: $showHypothetical) {
                if let ghost = hypothetical {
                    NavigationStack {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(ghost.title).font(.title3.bold())
                            Text(ghost.abstract)
                                .font(.body)
                                .foregroundStyle(.secondary)
                            let anchors = ghost.anchorClusterIDs.map(String.init).joined(separator: ", ")
                            Text("Anchors: \(anchors)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding()
                        .navigationTitle("What-if paper")
                        .toolbar {
                            ToolbarItem(placement: .primaryAction) {
                                Button("Close") { showHypothetical = false }
                            }
                        }
                    }
                    .frame(minWidth: 420, minHeight: 360)
                }
            }
            .sheet(isPresented: $showGlossary) {
                TopicGlossaryView(
                    onSelectMega: { cluster in
                        selectMega(cluster)
                    },
                    onSelectSubtopic: { cluster in
                        selectSubtopic(cluster)
                    }
                )
            }
            #if os(iOS)
            .sheet(item: $selectedPaper) { paper in
                NavigationStack {
                    PaperDetailView(paper: paper)
                        .navigationTitle("Paper")
                        .toolbar {
                            ToolbarItem(placement: .primaryAction) {
                                Button("Close") { selectedPaper = nil }
                            }
                        }
                }
                #if os(iOS)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
                #endif
                .frame(minWidth: 520, minHeight: 620)
            }
            #endif
        }
        #if os(macOS)
        .overlay {
            if let paper = selectedPaper {
                DismissibleOverlay(onDismiss: { selectedPaper = nil }) {
                    NavigationStack {
                        PaperDetailView(paper: paper, onClose: { selectedPaper = nil })
                            .environmentObject(model)
                            .toolbar {
                                ToolbarItem(placement: .primaryAction) {
                                    Button {
                                        selectedPaper = nil
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

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Step 2 — Knowledge Galaxy")
                    .font(.title.bold())
                    .foregroundStyle(.white)
                HStack(spacing: 8) {
                    let papersValue: String = {
                        guard model.yearFilterEnabled else { return "\(model.papers.count)" }
                        return "\(model.explorationPapers.count)/\(model.papers.count)"
                    }()
                    StatPill(label: "Papers", value: papersValue)
                    if model.yearFilterEnabled, let range = model.effectiveYearRange {
                        StatPill(label: "Years", value: "\(range.lowerBound)–\(range.upperBound)")
                    }
                    StatPill(label: "Clusters", value: "\(max(model.megaClusters.count, model.clusters.count))")
                    StatPill(label: "Lens", value: lens.label)
                }
            }
            Spacer()
            HStack(spacing: 10) {
                Button {
                    showGlossary = true
                } label: {
                    Label("Glossary", systemImage: "text.book.closed")
                }
                .buttonStyle(.bordered)

                if model.yearFilterEnabled {
                    Button {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            model.resetYearFilter()
                        }
                    } label: {
                        Label("Clear years", systemImage: "calendar.badge.minus")
                    }
                    .buttonStyle(.bordered)
                }

                Button {
                    if let result = model.exportGalaxyArtifacts() {
                        exportMessage = "Wrote:\n\(result.jsonURL.lastPathComponent)\n\(result.reportURL.lastPathComponent)"
                    } else {
                        exportMessage = "Nothing to export yet. Run clustering first."
                    }
                    showExportAlert = true
                } label: {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)

                if model.isClustering {
                    VStack(alignment: .trailing, spacing: 4) {
                        ProgressView(value: model.clusteringProgress)
                            .frame(width: 140)
                        Text(String(format: "%.0f%%", model.clusteringProgress * 100))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var resolutionControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Picker("Resolution", selection: $resolution) {
                    ForEach(ClusterResolution.allCases, id: \.self) { res in
                        Text(res.label).tag(res)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(model.papers.count < 3)

                Spacer()

                Button {
                    triggerClustering()
                } label: {
                    Label("Run clustering", systemImage: "sparkles")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isIngesting || model.isClustering || model.papers.count < 3)
            }

            if model.papers.count < 3 {
                Text("Need at least 3 papers to cluster.")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            } else {
                Text("Resolution tunes the number of subtopics inside each mega-topic.")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
    }

    private var lensControls: some View {
        HStack {
            Picker("Lens", selection: $lens) {
                ForEach(MapLens.allCases) { lens in
                    Text(lens.label).tag(lens)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 420)

            Spacer()
        }
    }

    private var zoomControls: some View {
        HStack(spacing: 12) {
            Text("Zoom")
                .font(.subheadline.bold())
            Picker("Zoom", selection: $zoomLevel) {
                ForEach(ZoomLevel.allCases, id: \.self) { level in
                    Text(level.label).tag(level)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 460)
            Spacer()
        }
    }

    private var controlDeck: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    resolutionControls
                }

                HStack(spacing: 16) {
                    lensControls
                    lensLegend
                }

                zoomControls
            }
        }
    }

    private var lensLegend: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(Color.cyan.opacity(0.5))
                .frame(width: 12, height: 12)
            Text(lens.label)
                .font(.caption.bold())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.white.opacity(0.04), in: Capsule())
        .overlay(Capsule().stroke(Color.white.opacity(0.1), lineWidth: 1))
    }

    private func breadcrumb(for cluster: Cluster) -> some View {
        HStack(spacing: 6) {
            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    zoomLevel = .mega
                    selectedMegaID = nil
                    selectedSubtopicID = nil
                    selectedClusterIDs.removeAll()
                }
            } label: {
                Label("All topics", systemImage: "circle.grid.3x3")
            }
            .buttonStyle(.bordered)

            Image(systemName: "chevron.right")
                .foregroundStyle(.white.opacity(0.7))

            Text(cluster.name)
                .font(.headline)
                .foregroundStyle(.white)
        }
    }

    private func paperBreadcrumb(for cluster: Cluster) -> some View {
        HStack(spacing: 6) {
            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    zoomLevel = .topics
                }
            } label: {
                Label("Subtopics", systemImage: "circle.grid.3x3")
            }
            .buttonStyle(.bordered)

            Image(systemName: "chevron.right")
                .foregroundStyle(.white.opacity(0.7))

            Text(cluster.name)
                .font(.headline)
                .foregroundStyle(.white)
        }
    }

    private func triggerClustering() {
        selectedClusterIDs.removeAll()
        selectedMegaID = nil
        selectedSubtopicID = nil
        zoomLevel = .mega
        Task {
            await model.buildMultiScaleGalaxy(level0Range: 5...8, level1Range: resolution.subtopicRange)
        }
    }

    private func selectMega(_ cluster: Cluster) {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
            selectedMegaID = cluster.id
            selectedSubtopicID = nil
            selectedClusterIDs = [cluster.id]
            zoomLevel = .topics
        }
    }

    private func megaID(containingSubtopic subtopicID: Int) -> Int? {
        for mega in model.megaClusters {
            if mega.subclusters?.contains(where: { $0.id == subtopicID }) == true {
                return mega.id
            }
        }
        return nil
    }

    private func selectSubtopic(_ cluster: Cluster) {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
            selectedMegaID = megaID(containingSubtopic: cluster.id) ?? selectedMegaID
            selectedSubtopicID = cluster.id
            selectedClusterIDs = [cluster.id]
            zoomLevel = .papers
            paperSearchQuery = ""
        }
    }
}

private struct PaperHighlightTaskKey: Hashable {
    let zoomLevel: ZoomLevel
    let subtopicID: Int?
    let papersCount: Int
    let yearFilterEnabled: Bool
    let yearFilterStart: Int
    let yearFilterEnd: Int
}

@available(macOS 26, iOS 26, *)
struct ClusterMapAndSidebar: View {
    @EnvironmentObject private var model: AppModel
    let clusters: [Cluster]
    @Binding var selectedClusterIDs: Set<Int>
    let isZoomed: Bool
    let showBridging: Bool
    let onZoomOut: (() -> Void)?
    let onZoom: ((Cluster) -> Void)?
    let onSelect: ((Cluster) -> Void)?
    let lensLabel: String
    private var driftMagnitudes: [Int: Double] {
        guard let drift = model.analyticsSummary?.drift else { return [:] }
        var latest: [Int: (year: Int, val: Double)] = [:]
        for entry in drift {
            if let existing = latest[entry.clusterID] {
                if entry.year > existing.year { latest[entry.clusterID] = (entry.year, entry.drift) }
            } else {
                latest[entry.clusterID] = (entry.year, entry.drift)
            }
        }
        return Dictionary(uniqueKeysWithValues: latest.map { ($0.key, $0.value.val) })
    }

    private var driftVectors: [Int: (dx: Double, dy: Double)] {
        guard let drift = model.analyticsSummary?.drift else { return [:] }
        var latest: [Int: (year: Int, dx: Double, dy: Double)] = [:]
        for entry in drift {
            if let existing = latest[entry.clusterID] {
                if entry.year > existing.year { latest[entry.clusterID] = (entry.year, entry.dx ?? 0, entry.dy ?? 0) }
            } else {
                latest[entry.clusterID] = (entry.year, entry.dx ?? 0, entry.dy ?? 0)
            }
        }
        return Dictionary(uniqueKeysWithValues: latest.map { ($0.key, (dx: $0.value.dx, dy: $0.value.dy)) })
    }

    private var ideaEdges: [(Int, Int, Double)] {
        guard let edges = model.analyticsSummary?.ideaFlowEdges else { return [] }
        let clusterMap = Dictionary(uniqueKeysWithValues: model.papers.compactMap { paper in
            paper.clusterIndex.map { ($0, paper.id) }.map { ($0.1, $0.0) }
        })
        var agg: [PairKey: Double] = [:]
        for e in edges {
            guard let s = e.src, let d = e.dst else { continue }
            guard let cs = clusterMap[s], let cd = clusterMap[d] else { continue }
            if !clusters.contains(where: { $0.id == cs }) || !clusters.contains(where: { $0.id == cd }) { continue }
            let key = PairKey(a: cs, b: cd)
            agg[key, default: 0] += e.weight ?? 1.0
        }
        return agg.map { ($0.key.a, $0.key.b, $0.value) }
    }

    private struct PairKey: Hashable {
        let a: Int
        let b: Int
    }

    var body: some View {
        HStack(spacing: 16) {
            GlassPanel {
                ClusterGraphView(
                    clusters: clusters,
                    selectedClusterIDs: $selectedClusterIDs,
                    lensLabel: lensLabel,
                    onSelect: onSelect,
                    driftMagnitudes: driftMagnitudes,
                    driftVectors: driftVectors,
                    ideaEdges: ideaEdges
                )
                .frame(minHeight: 360)
            }

            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text(isZoomed ? "Subtopic details" : "Cluster details")
                            .font(.headline)
                            .foregroundStyle(.white)
                        Spacer()
                        if let onZoomOut {
                            Button("Back") { onZoomOut() }
                                .buttonStyle(.bordered)
                        }
                    }

                    if selectedClusterIDs.isEmpty {
                        Text("Tap a cluster node in the map to see details.")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.7))
                    } else {
                        ForEach(clusters.filter { selectedClusterIDs.contains($0.id) }) { cluster in
                            ClusterDetailCard(cluster: cluster, onZoom: onZoom != nil && !isZoomed ? {
                                onZoom?(cluster)
                            } : nil)
                        }
                    }

                    if showBridging {
                        Divider().padding(.vertical, 4)
                        BridgingSection(selectedClusterIDs: $selectedClusterIDs)
                    }
                }
            }
            .frame(width: 380)
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct PaperMapAndSidebar: View {
    @EnvironmentObject private var model: AppModel
    let cluster: Cluster?
    let subtopics: [Cluster]
    let papers: [Paper]
    let highlights: [UUID: PaperNoveltyScore]
    let driftVector: (dx: Double, dy: Double)?
	    @Binding var searchQuery: String
	    @Binding var statusFilter: PaperStatusFilter
	    @Binding var sort: PaperSort
	    @Binding var selectedTradingTags: Set<String>
	    @Binding var selectedAssetClasses: Set<String>
	    @Binding var selectedHorizons: Set<String>
	    @Binding var colorBy: PaperColorBy
	    var onSelectPaper: ((Paper) -> Void)?
	    var onSelectSubtopic: ((Cluster) -> Void)?

    private var metricsByID: [UUID: AnalyticsSummary.PaperMetric] {
        Dictionary(uniqueKeysWithValues: (model.analyticsSummary?.paperMetrics ?? []).map { ($0.paperID, $0) })
    }

	    private var recommendationRankByID: [UUID: Int] {
	        guard let recs = model.analyticsSummary?.recommendations, !recs.isEmpty else { return [:] }
	        return Dictionary(uniqueKeysWithValues: recs.enumerated().map { ($0.element, $0.offset) })
	    }

	    private var hasAnyTradingFilter: Bool {
	        !selectedTradingTags.isEmpty || !selectedAssetClasses.isEmpty || !selectedHorizons.isEmpty
	    }

	    private func tradingTags(for paper: Paper) -> Set<String> {
	        let tags = paper.tradingLens?.tradingTags ?? []
	        return Set(tags.isEmpty ? ["Unknown"] : tags)
	    }

	    private func assetClasses(for paper: Paper) -> Set<String> {
	        let assets = paper.tradingLens?.assetClasses ?? []
	        return Set(assets.isEmpty ? ["Unknown"] : assets)
	    }

	    private func horizons(for paper: Paper) -> Set<String> {
	        let horizons = paper.tradingLens?.horizons ?? []
	        return Set(horizons.isEmpty ? ["Unknown"] : horizons)
	    }

	    private func matches(selected: Set<String>, values: Set<String>) -> Bool {
	        guard !selected.isEmpty else { return true }
	        return !values.isDisjoint(with: selected)
	    }

	    private func primaryValue(from values: Set<String>) -> String {
	        if values.count == 1, let only = values.first { return only }
	        if let preferred = values.first(where: { $0.lowercased() != "unknown" }) { return preferred }
	        return values.first ?? "Unknown"
	    }

	    private func tint(for paper: Paper) -> Color? {
	        switch colorBy {
	        case .novelty:
	            return nil
	        case .tradingTag:
	            return MapPalette.categoricalTint(for: primaryValue(from: tradingTags(for: paper)))
	        case .assetClass:
	            return MapPalette.categoricalTint(for: primaryValue(from: assetClasses(for: paper)))
	        case .horizon:
	            return MapPalette.categoricalTint(for: primaryValue(from: horizons(for: paper)))
	        }
	    }

	    private var tintByID: [UUID: Color] {
	        guard colorBy != .novelty else { return [:] }
	        var map: [UUID: Color] = [:]
	        for paper in filteredPapers {
	            if let tint = tint(for: paper) {
	                map[paper.id] = tint
	            }
	        }
	        return map
	    }

	    private var availableTradingTags: [String] {
	        uniqueOptions(from: papers) { Array(tradingTags(for: $0)) }
	    }

	    private var availableAssetClasses: [String] {
	        uniqueOptions(from: papers) { Array(assetClasses(for: $0)) }
	    }

	    private var availableHorizons: [String] {
	        uniqueOptions(from: papers) { Array(horizons(for: $0)) }
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
        let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = query.lowercased()

        var base = papers
        if statusFilter != .all {
            base = base.filter { statusFilter.matches($0) }
        }

	        if !lowered.isEmpty {
	            base = base.filter { paper in
	                if paper.title.lowercased().contains(lowered) { return true }
	                if paper.summary.lowercased().contains(lowered) { return true }
	                if let keywords = paper.keywords, keywords.joined(separator: " ").lowercased().contains(lowered) { return true }
	                return false
	            }
	        }

	        if hasAnyTradingFilter {
	            base = base.filter { paper in
	                matches(selected: selectedTradingTags, values: tradingTags(for: paper))
	                    && matches(selected: selectedAssetClasses, values: assetClasses(for: paper))
	                    && matches(selected: selectedHorizons, values: horizons(for: paper))
	            }
	        }

	        return base.sorted(by: comparator)
	    }

	    private var comparator: (Paper, Paper) -> Bool {
	        { lhs, rhs in
	            let lm = metricsByID[lhs.id]
	            let rm = metricsByID[rhs.id]
	            let ls = lhs.tradingScores ?? lhs.tradingLens?.scores
	            let rs = rhs.tradingScores ?? rhs.tradingLens?.scores
	            switch sort {
            case .recommended:
                let lRank = recommendationRankByID[lhs.id] ?? Int.max
                let rRank = recommendationRankByID[rhs.id] ?? Int.max
                if lRank != rRank { return lRank < rRank }
                fallthrough
            case .recency:
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
            case .title:
                return lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
            case .noveltyZ:
                let lv = lm?.zNovelty ?? Double(highlights[lhs.id]?.novelty ?? 0)
                let rv = rm?.zNovelty ?? Double(highlights[rhs.id]?.novelty ?? 0)
                if lv != rv { return lv > rv }
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
            case .consensusZ:
                let lv = lm?.zConsensus ?? Double(highlights[lhs.id]?.saturation ?? 0)
                let rv = rm?.zConsensus ?? Double(highlights[rhs.id]?.saturation ?? 0)
                if lv != rv { return lv > rv }
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
            case .influencePos:
                let lv = lm?.influencePos ?? 0
                let rv = rm?.influencePos ?? 0
                if lv != rv { return lv > rv }
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
            case .clusterConfidence:
                let lv = lm?.clusterConfidence ?? 0
                let rv = rm?.clusterConfidence ?? 0
                if lv != rv { return lv > rv }
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
            case .recombination:
                let lv = lm?.recombination ?? 0
                let rv = rm?.recombination ?? 0
                if lv != rv { return lv > rv }
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
            case .rigor:
                let lv = lm?.rigorProxy ?? 0
                let rv = rm?.rigorProxy ?? 0
                if lv != rv { return lv > rv }
                let ly = lhs.year ?? -10_000
                let ry = rhs.year ?? -10_000
                if ly != ry { return ly > ry }
                return lhs.title < rhs.title
	            case .openness:
	                let lv = lm?.opennessScore ?? 0
	                let rv = rm?.opennessScore ?? 0
	                if lv != rv { return lv > rv }
	                let ly = lhs.year ?? -10_000
	                let ry = rhs.year ?? -10_000
	                if ly != ry { return ly > ry }
	                return lhs.title < rhs.title
	            case .tradingPriority:
	                let lv = (ls?.strategyImpact ?? 0) * (ls?.usability ?? 0) * (ls?.confidence ?? 0)
	                let rv = (rs?.strategyImpact ?? 0) * (rs?.usability ?? 0) * (rs?.confidence ?? 0)
	                if lv != rv { return lv > rv }
	                let ly = lhs.year ?? -10_000
	                let ry = rhs.year ?? -10_000
	                if ly != ry { return ly > ry }
	                return lhs.title < rhs.title
	            case .tradingImpact:
	                let lv = ls?.strategyImpact ?? 0
	                let rv = rs?.strategyImpact ?? 0
	                if lv != rv { return lv > rv }
	                let ly = lhs.year ?? -10_000
	                let ry = rhs.year ?? -10_000
	                if ly != ry { return ly > ry }
	                return lhs.title < rhs.title
	            case .tradingUsability:
	                let lv = ls?.usability ?? 0
	                let rv = rs?.usability ?? 0
	                if lv != rv { return lv > rv }
	                let ly = lhs.year ?? -10_000
	                let ry = rhs.year ?? -10_000
	                if ly != ry { return ly > ry }
	                return lhs.title < rhs.title
	            case .tradingNovelty:
	                let lv = ls?.novelty ?? 0
	                let rv = rs?.novelty ?? 0
	                if lv != rv { return lv > rv }
	                let ly = lhs.year ?? -10_000
	                let ry = rhs.year ?? -10_000
	                if ly != ry { return ly > ry }
	                return lhs.title < rhs.title
	            }
	        }
	    }

    private var emptyMessage: String {
        if cluster == nil {
            return "Select a subtopic to see its papers."
        }
        if papers.isEmpty {
            return "No papers in this subtopic (after filters)."
        }
        if filteredPapers.isEmpty {
            return "No papers match your filters."
        }
        return "Select a paper to see details."
    }

    var body: some View {
        HStack(spacing: 16) {
	            GlassPanel {
	                PaperScatterView(
	                    papers: filteredPapers,
	                    highlights: highlights,
	                    tintByPaperID: tintByID,
	                    driftVector: driftVector,
	                    emptyMessage: emptyMessage,
	                    onSelectPaper: onSelectPaper
	                )
                .frame(minHeight: 360)
            }

            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Papers")
                            .font(.headline)
                            .foregroundStyle(.white)
                        Spacer()
                        if let cluster {
                            Circle()
                                .fill(MapPalette.nodeGradient(for: cluster.id))
                                .frame(width: 14, height: 14)
                                .shadow(color: .white.opacity(0.45), radius: 6)
                        }
                    }

                    if let cluster {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(cluster.name)
                                .font(.subheadline.bold())
                                .foregroundStyle(.white)
                            if !cluster.metaSummary.isEmpty {
                                Text(cluster.metaSummary)
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.78))
                                    .lineLimit(3)
                            }
                            if let lens = cluster.tradingLens, !lens.isEmpty {
                                Text(lens)
                                    .font(.caption2)
                                    .foregroundStyle(.white.opacity(0.72))
                                    .lineLimit(4)
                            }
                        }
                    } else {
                        Text("Pick a subtopic to explore its papers.")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.72))
                    }

                    HStack(spacing: 10) {
                        Menu {
                            if subtopics.isEmpty {
                                Text("No subtopics yet.")
                            } else {
                                ForEach(subtopics) { sub in
                                    Button(sub.name) { onSelectSubtopic?(sub) }
                                }
                            }
                        } label: {
                            Label(cluster?.name ?? "Choose subtopic", systemImage: "circle.grid.3x3")
                        }
                        .buttonStyle(.bordered)
                        .tint(Color.white.opacity(0.12))

                        Spacer()

	                        if !searchQuery.isEmpty || statusFilter != .all || sort != .recommended || hasAnyTradingFilter || colorBy != .novelty {
	                            Button("Reset") {
	                                searchQuery = ""
	                                statusFilter = .all
	                                sort = .recommended
	                                selectedTradingTags = []
	                                selectedAssetClasses = []
	                                selectedHorizons = []
	                                colorBy = .novelty
	                            }
	                            .buttonStyle(.bordered)
	                            .tint(Color.white.opacity(0.12))
	                        }
                    }

                    SearchField(text: $searchQuery, placeholder: "Search title, keywords, summary")

                    Picker("Status", selection: $statusFilter) {
                        ForEach(PaperStatusFilter.allCases) { filter in
                            Text(filter.label).tag(filter)
                        }
                    }
                    .pickerStyle(.segmented)

	                    HStack {
	                        Text("\(filteredPapers.count) of \(papers.count) papers")
	                            .font(.caption)
	                            .foregroundStyle(.white.opacity(0.75))
	                        Spacer()
	                        Picker("Color", selection: $colorBy) {
	                            ForEach(PaperColorBy.allCases) { option in
	                                Text(option.label).tag(option)
	                            }
	                        }
	                        .pickerStyle(.menu)
	                        Picker("Sort", selection: $sort) {
	                            ForEach(PaperSort.allCases) { option in
	                                Text(option.label).tag(option)
	                            }
	                        }
	                        .pickerStyle(.menu)
	                    }

	                    HStack(spacing: 10) {
	                        MultiSelectMenu(
	                            title: "Tags",
	                            emptyLabel: "Any tag",
	                            options: availableTradingTags,
	                            selection: $selectedTradingTags
	                        )
	                        MultiSelectMenu(
	                            title: "Assets",
	                            emptyLabel: "Any asset",
	                            options: availableAssetClasses,
	                            selection: $selectedAssetClasses
	                        )
	                        MultiSelectMenu(
	                            title: "Horizon",
	                            emptyLabel: "Any horizon",
	                            options: availableHorizons,
	                            selection: $selectedHorizons
	                        )
	                        Spacer()
	                    }
	                    .font(.caption)

	                    Divider().overlay(Color.white.opacity(0.12))

                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            if filteredPapers.isEmpty {
                                Text(emptyMessage)
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.7))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.top, 6)
                            } else {
                                ForEach(filteredPapers) { paper in
                                    Button {
                                        onSelectPaper?(paper)
	                                    } label: {
	                                        PaperRowCard(
	                                            paper: paper,
	                                            accentTint: tintByID[paper.id],
	                                            metric: metricsByID[paper.id],
	                                            highlight: highlights[paper.id],
	                                            recommendationRank: recommendationRankByID[paper.id]
	                                        )
	                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .frame(width: 390)
        }
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
private struct SearchField: View {
    @Binding var text: String
    let placeholder: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.white.opacity(0.65))
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .foregroundStyle(.white)
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.65))
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
private struct PaperRowCard: View {
    let paper: Paper
    let accentTint: Color?
    let metric: AnalyticsSummary.PaperMetric?
    let highlight: PaperNoveltyScore?
    let recommendationRank: Int?

    private func zText(_ value: Double) -> String {
        String(format: "%+.2f", value)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                if let accentTint {
                    Circle()
                        .fill(accentTint)
                        .frame(width: 10, height: 10)
                        .overlay(Circle().stroke(Color.white.opacity(0.35), lineWidth: 1))
                        .padding(.top, 4)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text(paper.title)
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                        .lineLimit(2)

                    Text(paper.summary)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.78))
                        .lineLimit(2)
                }

                Spacer(minLength: 0)

                if recommendationRank != nil {
                    Image(systemName: "star.fill")
                        .font(.caption.bold())
                        .foregroundStyle(Color.yellow.opacity(0.9))
                        .padding(.top, 2)
                }
            }

            HStack(spacing: 6) {
                if let year = paper.year {
                    MiniPill(label: "\(year)", tint: Color.white.opacity(0.10))
                }

                if let status = paper.readingStatus {
                    MiniPill(
                        label: status.label,
                        tint: status == .done ? Color.green.opacity(0.22) : Color.white.opacity(0.10)
                    )
                }

                if let m = metric {
                    MiniPill(label: "N z \(zText(m.zNovelty))", tint: Color.pink.opacity(0.20))
                    MiniPill(label: "C z \(zText(m.zConsensus))", tint: Color.mint.opacity(0.18))

                    if let conf = m.clusterConfidence, conf < 0.65 {
                        MiniPill(label: "Boundary", tint: Color.yellow.opacity(0.20))
                    }

                    if m.hasCodeLink == true {
                        Image(systemName: "chevron.left.forwardslash.chevron.right")
                            .font(.caption2.bold())
                            .foregroundStyle(.white.opacity(0.72))
                    }
                    if m.hasDataLink == true {
                        Image(systemName: "externaldrive")
                            .font(.caption2.bold())
                            .foregroundStyle(.white.opacity(0.72))
                    }
                    if m.duplicateGroup != nil {
                        Image(systemName: "doc.on.doc")
                            .font(.caption2.bold())
                            .foregroundStyle(.white.opacity(0.62))
                    }
                    if let flags = m.ingestionFlags, !flags.isEmpty {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption2.bold())
                            .foregroundStyle(Color.orange.opacity(0.85))
                    }
                } else if let h = highlight {
                    if h.novelty > 0.6 {
                        MiniPill(label: "Outlier", tint: Color.purple.opacity(0.22))
                    }
                    if h.saturation > 0.6 {
                        MiniPill(label: "Dense", tint: Color.orange.opacity(0.22))
                    }
                }

                Spacer(minLength: 0)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
    }
}

@available(macOS 26, iOS 26, *)
private struct MiniPill: View {
    let label: String
    let tint: Color

    var body: some View {
        Text(label)
            .font(.caption2.bold())
            .foregroundStyle(.white.opacity(0.85))
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(tint, in: Capsule())
    }
}

@available(macOS 26, iOS 26, *)
struct ClusterGraphView: View {
    let clusters: [Cluster]
    @Binding var selectedClusterIDs: Set<Int>
    let lensLabel: String
    let onSelect: ((Cluster) -> Void)?
    var driftMagnitudes: [Int: Double] = [:]
    var driftVectors: [Int: (dx: Double, dy: Double)] = [:]
    var ideaEdges: [(Int, Int, Double)] = []
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var layoutSignature: String {
        clusters.map { cluster in
            guard let layout = cluster.layoutPosition else { return "\(cluster.id):nil" }
            return "\(cluster.id):\(Int(layout.x * 1000)):\(Int(layout.y * 1000))"
        }.joined(separator: "|")
    }

    var body: some View {
        TimelineView(.periodic(from: .now, by: reduceMotion ? 3600 : 1.0 / 18.0)) { timeline in
            GeometryReader { geo in
                let size = geo.size
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                let radius = min(size.width, size.height) / 2 - 80
                let time = timeline.date.timeIntervalSinceReferenceDate
                let wobble = reduceMotion ? 0 : min(10, radius * 0.03)
                let spin = reduceMotion ? 0 : time * 0.015
                let positions = Dictionary(uniqueKeysWithValues: clusters.enumerated().map { idx, cluster in
                    (cluster.id, position(for: cluster, fallbackIndex: idx, total: clusters.count, center: center, radius: radius, time: time, wobble: wobble))
                })

                ZStack {
                    RoundedRectangle(cornerRadius: 24)
                        .fill(MapPalette.canvas)
                        .overlay(MapPalette.glow)
                        .overlay(
                            AngularGradient(
                                colors: [.clear, Color.white.opacity(0.12), .clear],
                                center: .center
                            )
                            .blur(radius: 80)
                            .rotationEffect(.radians(spin))
                            .blendMode(.screen)
                        )
                        .overlay(
                            ZStack {
                                ForEach(0..<6, id: \.self) { idx in
                                    Circle()
                                        .stroke(Color.white.opacity(0.05), lineWidth: 1)
                                        .frame(width: radius * 2 * CGFloat(0.3 + 0.1 * Double(idx)), height: radius * 2 * CGFloat(0.3 + 0.1 * Double(idx)))
                                        .offset(y: 8)
                                }
                            }
                        )
                        .shadow(color: .black.opacity(0.28), radius: 20, x: 0, y: 12)

                    if clusters.isEmpty {
                        Text("No clusters yet.")
                            .foregroundStyle(.white.opacity(0.7))
                    } else {
                        VStack {
                            HStack {
                                Text("Lens: \(lensLabel)")
                                    .font(.caption.bold())
                                    .foregroundStyle(.white.opacity(0.8))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(Color.white.opacity(0.08), in: Capsule())
                                Spacer()
                            }
                            .padding(12)
                            Spacer()
                        }

                        if selectedClusterIDs.count == 2 {
                            let ids = Array(selectedClusterIDs)
                            if let p1 = positions[ids[0]], let p2 = positions[ids[1]] {
                                Path { path in
                                    path.move(to: p1)
                                    path.addLine(to: p2)
                                }
                                .stroke(style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [6, 6]))
                                .foregroundColor(.yellow.opacity(0.9))
                            }
                        }

                        // Idea-flow edges
                        ForEach(Array(ideaEdges.enumerated()), id: \.offset) { _, edge in
                            if let p1 = positions[edge.0], let p2 = positions[edge.1] {
                                let width = max(1, min(4, CGFloat(edge.2) * 0.5))
                                Path { path in
                                    path.move(to: p1)
                                    path.addLine(to: p2)
                                }
                                .stroke(Color.orange.opacity(0.35), style: StrokeStyle(lineWidth: width, lineCap: .round))
                            }
                        }

                        ForEach(clusters, id: \.id) { cluster in
                            if let pos = positions[cluster.id] {
                                ClusterNodeView(cluster: cluster, isSelected: selectedClusterIDs.contains(cluster.id))
                                    .position(pos)
                                    .onTapGesture {
                                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                            if selectedClusterIDs.contains(cluster.id) {
                                                selectedClusterIDs.remove(cluster.id)
                                            } else {
                                                if selectedClusterIDs.count >= 2 {
                                                    selectedClusterIDs.removeAll()
                                                }
                                                selectedClusterIDs.insert(cluster.id)
                                            }
                                        }
                                        onSelect?(cluster)
                                    }

                                if let drift = driftMagnitudes[cluster.id], drift > 0 {
                                    let vec = driftVectors[cluster.id] ?? (dx: Double(cos(hashAngle(for: cluster.id))), dy: Double(sin(hashAngle(for: cluster.id))))
                                    let length = CGFloat(min(drift, 0.6)) * radius * 0.35
                                    let end = CGPoint(
                                        x: pos.x + length * CGFloat(vec.dx),
                                        y: pos.y + length * CGFloat(vec.dy)
                                    )
                                    ArrowShape(start: pos, end: end)
                                        .stroke(Color.cyan.opacity(0.75), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                                }
                            }
                        }
                    }
                }
                .animation(.spring(response: 0.55, dampingFraction: 0.85), value: layoutSignature)
            }
        }
    }

    private func position(for cluster: Cluster, fallbackIndex: Int, total: Int, center: CGPoint, radius: CGFloat, time: TimeInterval, wobble: CGFloat) -> CGPoint {
        let base: CGPoint
        if let layout = cluster.layoutPosition {
            let x = center.x + (CGFloat(layout.x) - 0.5) * radius * 2
            let y = center.y + (CGFloat(layout.y) - 0.5) * radius * 2
            base = CGPoint(x: x, y: y)
        } else {
            let angle = 2 * Double.pi * Double(fallbackIndex) / Double(max(total, 1))
            base = CGPoint(
                x: center.x + radius * CGFloat(cos(angle)),
                y: center.y + radius * CGFloat(sin(angle))
            )
        }

        guard wobble > 0 else { return base }
        let phase = CGFloat(time * 0.9) + hashAngle(for: cluster.id)
        let dx = wobble * CGFloat(cos(phase))
        let dy = wobble * CGFloat(sin(phase * 1.18))
        return CGPoint(x: base.x + dx, y: base.y + dy)
    }

    private func hashAngle(for id: Int) -> CGFloat {
        // Deterministic pseudo-angle per cluster id
        let seed = UInt64(truncatingIfNeeded: id) &* 6364136223846793005 &+ 1
        let frac = Double(seed % 10_000) / 10_000.0
        return CGFloat(frac * 2 * Double.pi)
    }
}

@available(macOS 26, iOS 26, *)
struct ClusterNodeView: View {
    @EnvironmentObject private var model: AppModel
    let cluster: Cluster
    let isSelected: Bool

    private var paperCount: Int {
        model.paperCount(for: cluster)
    }

    private var size: CGFloat {
        let base: CGFloat = 130
        let growth = CGFloat(min(paperCount, 14)) * 4
        return base + growth
    }

    var body: some View {
        VStack(spacing: 4) {
            Text(cluster.name)
                .font(.headline.weight(.semibold))
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .foregroundStyle(.white)
            Text("\(paperCount) papers")
                .font(.caption.bold())
                .foregroundStyle(.white.opacity(0.8))
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .frame(width: size)
        .background(
            ZStack {
                if isSelected {
                    Circle()
                        .fill(MapPalette.nodeGradient(for: cluster.id))
                        .blur(radius: 22)
                        .opacity(0.6)
                        .scaleEffect(1.15)
                }
                RoundedRectangle(cornerRadius: 18)
                    .fill(MapPalette.nodeGradient(for: cluster.id))
                RoundedRectangle(cornerRadius: 18)
                    .strokeBorder(Color.white.opacity(isSelected ? 0.55 : 0.25), lineWidth: isSelected ? 2 : 1)
            }
        )
        .shadow(color: Color.black.opacity(isSelected ? 0.45 : 0.22), radius: isSelected ? 18 : 10, x: 0, y: 8)
        .scaleEffect(isSelected ? 1.06 : 1.0)
        .opacity(model.yearFilterEnabled && paperCount == 0 ? 0.42 : 1)
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: isSelected)
    }
}

@available(macOS 26, iOS 26, *)
struct ClusterDetailCard: View {
    @EnvironmentObject private var model: AppModel
    let cluster: Cluster
    let onZoom: (() -> Void)?

    @State private var isEditing: Bool = false
    @State private var draftName: String = ""
    @State private var draftMeta: String = ""
    @State private var isNaming: Bool = false
    @State private var showDossier: Bool = false
    @State private var dossierText: String = ""
    @State private var isGeneratingDossier: Bool = false

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(cluster.name)
                            .font(.headline)
                            .foregroundStyle(.white)
                        let filteredCount = model.paperCount(for: cluster)
                        let countLabel: String = {
                            guard model.yearFilterEnabled else { return "\(filteredCount) papers" }
                            return "\(filteredCount)/\(cluster.memberPaperIDs.count) papers"
                        }()
                        Text(countLabel)
                            .font(.caption.bold())
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    Spacer()
                    Button {
                        if isEditing {
                            draftName = ""
                            draftMeta = ""
                        } else {
                            draftName = cluster.name
                            draftMeta = cluster.metaSummary
                        }
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            isEditing.toggle()
                        }
                    } label: {
                        Image(systemName: isEditing ? "xmark" : "pencil")
                    }
                    .buttonStyle(.bordered)
                    .tint(Color.white.opacity(0.12))

                    Button {
                        guard !isNaming else { return }
                        isNaming = true
                        Task {
                            await model.autoNameGalaxyCluster(clusterID: cluster.id)
                            await MainActor.run { isNaming = false }
                        }
                    } label: {
                        if isNaming {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "sparkles")
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(Color.white.opacity(0.12))
                    .contextMenu {
                        Button("Force AI re-name") {
                            guard !isNaming else { return }
                            isNaming = true
                            Task {
                                await model.autoNameGalaxyCluster(clusterID: cluster.id, force: true)
                                await MainActor.run { isNaming = false }
                            }
                        }
                    }

                    Circle()
                        .fill(MapPalette.nodeGradient(for: cluster.id))
                        .frame(width: 14, height: 14)
                        .shadow(color: .white.opacity(0.5), radius: 6)
                }

                if isEditing {
                    VStack(alignment: .leading, spacing: 8) {
                        TextField("Topic name", text: $draftName)
                            .textFieldStyle(.roundedBorder)
                        TextEditor(text: $draftMeta)
                            .frame(minHeight: 72, maxHeight: 120)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.16)))

                        HStack {
                            Button("Save") {
                                model.renameGalaxyCluster(clusterID: cluster.id, name: draftName, metaSummary: draftMeta.isEmpty ? nil : draftMeta)
                                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                    isEditing = false
                                }
                            }
                            .buttonStyle(.borderedProminent)

                            Button("Cancel") {
                                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                    isEditing = false
                                }
                            }
                            .buttonStyle(.bordered)

                            Spacer()
                        }
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                } else {
                    Text(cluster.metaSummary)
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.9))
                        .lineLimit(6)
                        .transition(.opacity)
                }

                HStack {
                    Button {
                        showDossier = true
                        if dossierText.isEmpty {
                            loadDossier(force: false)
                        }
                    } label: {
                        Label("Dossier", systemImage: "doc.text")
                            .font(.subheadline.bold())
                    }
                    .buttonStyle(.bordered)
                    .tint(Color.white.opacity(0.12))

                    if isGeneratingDossier {
                        ProgressView()
                            .controlSize(.small)
                    }

                    Spacer()
                }

                if let onZoom {
                    Button {
                        onZoom()
                    } label: {
                        Label("Zoom into subtopics", systemImage: "magnifyingglass")
                            .font(.subheadline.bold())
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.white.opacity(0.16))
                }

                let sampleID: UUID? = {
                    if model.yearFilterEnabled,
                       let pid = cluster.memberPaperIDs.first(where: { model.explorationPaperIDs.contains($0) }) {
                        return pid
                    }
                    return cluster.memberPaperIDs.first
                }()
                if let pid = sampleID,
                   let example = model.papers.first(where: { $0.id == pid }) {
                    Text("Sample: \(example.title)")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.8))
                }
            }
        }
        .onChange(of: cluster.id) { _, _ in
            dossierText = ""
            isGeneratingDossier = false
        }
        .sheet(isPresented: $showDossier) {
            NavigationStack {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text(cluster.name)
                            .font(.title3.bold())
                            .foregroundStyle(.primary)
                        Spacer()
                        Button {
                            loadDossier(force: true)
                        } label: {
                            Label("Regenerate", systemImage: "arrow.clockwise")
                        }
                        .buttonStyle(.bordered)
                        .disabled(isGeneratingDossier)
                    }

                    if isGeneratingDossier {
                        HStack {
                            ProgressView()
                            Text("Synthesizing…")
                                .foregroundStyle(.secondary)
                        }
                    }

                    ScrollView {
                        Text(dossierText.isEmpty ? "No dossier yet — tap Regenerate or close and reopen." : dossierText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }

                    Spacer()
                }
                .padding()
                .navigationTitle("Topic dossier")
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button("Close") { showDossier = false }
                    }
                }
            }
            #if os(iOS)
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
            #endif
            .frame(minWidth: 520, minHeight: 520)
        }
    }

    private func loadDossier(force: Bool) {
        guard !isGeneratingDossier else { return }
        isGeneratingDossier = true
        Task {
            let text = await model.loadOrGenerateTopicDossier(clusterID: cluster.id, force: force) ?? ""
            await MainActor.run {
                dossierText = text
                isGeneratingDossier = false
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
struct PaperScatterView: View {
    let papers: [Paper]
    let highlights: [UUID: PaperNoveltyScore]
    let tintByPaperID: [UUID: Color]
    let driftVector: (dx: Double, dy: Double)?
    let emptyMessage: String
    var onSelectPaper: ((Paper) -> Void)?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var expandedPaperID: UUID?
    @State private var panOffset: CGSize = .zero
    @GestureState private var gesturePan: CGSize = .zero
    @State private var zoomScale: CGFloat = 1.0
    @GestureState private var gestureZoom: CGFloat = 1.0

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - 60
            let usesDotMode = shouldUseDotMode(total: papers.count, size: size)
            let scale = clampScale(zoomScale * gestureZoom)
            let offset = CGSize(width: panOffset.width + gesturePan.width, height: panOffset.height + gesturePan.height)
            let basePositions: [CGPoint] = papers.enumerated().map { idx, _ in
                position(for: idx, total: papers.count, center: center, radius: radius)
            }
            let resolvedPositions: [CGPoint] = usesDotMode
                ? basePositions
                : relaxedPositions(
                    base: basePositions,
                    center: center,
                    radius: radius,
                    expandedPaperID: expandedPaperID
                )
            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(MapPalette.canvas)
                    .overlay(MapPalette.glow)
                    .shadow(color: .black.opacity(0.25), radius: 16, x: 0, y: 10)

                if papers.isEmpty {
                    Text(emptyMessage)
                        .foregroundStyle(.white.opacity(0.7))
                } else {
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) {
                            withAnimation(.spring(response: 0.45, dampingFraction: 0.85)) {
                                zoomScale = 1.0
                                panOffset = .zero
                            }
                        }
                        .onTapGesture {
                            withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                expandedPaperID = nil
                            }
                        }

                    ZStack {
                        if let drift = driftVector {
                            let len = radius * 0.25
                            let end = CGPoint(x: center.x + len * CGFloat(drift.dx), y: center.y + len * CGFloat(drift.dy))
                            ArrowShape(start: center, end: end)
                                .stroke(Color.cyan.opacity(0.7), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                                .opacity(usesDotMode ? 0.45 : 0.7)
                        }

                        if usesDotMode {
	                            ForEach(Array(papers.enumerated()), id: \.1.id) { idx, paper in
	                                let pos = resolvedPositions[idx]
	                                let score = highlights[paper.id]
	                                let isSelected = expandedPaperID == paper.id
	                                PaperDotView(highlight: score, isSelected: isSelected, overrideTint: tintByPaperID[paper.id])
	                                    .position(pos)
	                                    .contentShape(Circle())
	                                    .onTapGesture {
	                                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
	                                            expandedPaperID = isSelected ? nil : paper.id
                                        }
                                    }
                            }

                            if let selectedID = expandedPaperID,
                               let idx = papers.firstIndex(where: { $0.id == selectedID }) {
                                let paper = papers[idx]
                                let pos = resolvedPositions[idx]
	                                PaperNodeView(
	                                    paper: paper,
	                                    highlight: highlights[paper.id],
	                                    isExpanded: true,
	                                    accentTint: tintByPaperID[paper.id],
	                                    onOpen: { onSelectPaper?(paper) }
	                                )
                                .position(pos)
                                .zIndex(10)
                                .transition(.opacity.combined(with: .scale(scale: 0.98)))
                            }
                        } else {
                            ForEach(Array(papers.enumerated()), id: \.1.id) { idx, paper in
                                let pos = resolvedPositions[idx]
                                let score = highlights[paper.id]
                                let isExpanded = expandedPaperID == paper.id
	                                PaperNodeView(
	                                    paper: paper,
	                                    highlight: score,
	                                    isExpanded: isExpanded,
	                                    accentTint: tintByPaperID[paper.id],
	                                    onOpen: { onSelectPaper?(paper) }
	                                )
                                .position(pos)
                                .zIndex(isExpanded ? 10 : 0)
                                .onTapGesture {
                                    withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                        expandedPaperID = isExpanded ? nil : paper.id
                                    }
                                }
                                .animation(.spring(response: 0.45, dampingFraction: 0.85), value: pos)
                            }
                        }
                    }
                    .scaleEffect(scale, anchor: .center)
                    .offset(offset)
                    .animation(reduceMotion ? nil : .spring(response: 0.45, dampingFraction: 0.85), value: scale)
                }
            }
            .simultaneousGesture(panAndZoomGesture())
        }
    }

    private func position(for index: Int, total: Int, center: CGPoint, radius: CGFloat) -> CGPoint {
        let goldenAngle = Double.pi * (3 - sqrt(5))
        let i = Double(index + 1)
        let denom = Double(max(total, 1))
        let radial = radius * CGFloat(sqrt(i / denom))
        let angle = i * goldenAngle
        return CGPoint(
            x: center.x + radial * CGFloat(cos(angle)),
            y: center.y + radial * CGFloat(sin(angle))
        )
    }

    private func relaxedPositions(
        base: [CGPoint],
        center: CGPoint,
        radius: CGFloat,
        expandedPaperID: UUID?
    ) -> [CGPoint] {
        guard base.count > 1 else { return base }
        let n = base.count
        let cardRadius: CGFloat = 96
        let expandedRadius: CGFloat = 210
        let iterations = min(30, max(14, n + 6))

        func nodeRadius(at index: Int) -> CGFloat {
            guard let expandedPaperID, papers[index].id == expandedPaperID else { return cardRadius }
            return expandedRadius
        }

        var positions = base
        for _ in 0..<iterations {
            for i in 0..<n {
                for j in (i + 1)..<n {
                    let ri = nodeRadius(at: i)
                    let rj = nodeRadius(at: j)
                    var dx = positions[j].x - positions[i].x
                    var dy = positions[j].y - positions[i].y
                    let dist = max(0.0001, sqrt(dx * dx + dy * dy))
                    let minDist = ri + rj
                    if dist < minDist {
                        let overlap = (minDist - dist) * 0.5
                        dx /= dist
                        dy /= dist
                        positions[i].x -= dx * overlap
                        positions[i].y -= dy * overlap
                        positions[j].x += dx * overlap
                        positions[j].y += dy * overlap
                    }
                }
            }

            for i in 0..<n {
                let ri = nodeRadius(at: i)
                positions[i].x = positions[i].x * 0.92 + base[i].x * 0.08
                positions[i].y = positions[i].y * 0.92 + base[i].y * 0.08

                var dx = positions[i].x - center.x
                var dy = positions[i].y - center.y
                let dist = max(0.0001, sqrt(dx * dx + dy * dy))
                let maxDist = max(0, radius - ri)
                if dist > maxDist {
                    dx /= dist
                    dy /= dist
                    positions[i].x = center.x + dx * maxDist
                    positions[i].y = center.y + dy * maxDist
                }
            }
        }
        return positions
    }

    private func shouldUseDotMode(total: Int, size: CGSize) -> Bool {
        guard total > 0 else { return false }
        let area = size.width * size.height
        let cardFootprint: CGFloat = 220 * 160
        let capacity = Int(area / cardFootprint)
        let threshold = max(8, min(48, capacity))
        return total > threshold
    }

    private func panAndZoomGesture() -> some Gesture {
        let pan = DragGesture(minimumDistance: 4)
            .updating($gesturePan) { value, state, _ in
                state = value.translation
            }
            .onEnded { value in
                panOffset = CGSize(width: panOffset.width + value.translation.width, height: panOffset.height + value.translation.height)
            }

        let zoom = MagnificationGesture()
            .updating($gestureZoom) { value, state, _ in
                state = value
            }
            .onEnded { value in
                zoomScale = clampScale(zoomScale * value)
            }

        return SimultaneousGesture(pan, zoom)
    }

    private func clampScale(_ value: CGFloat) -> CGFloat {
        min(max(value, 0.65), 2.6)
    }
}

@available(macOS 26, iOS 26, *)
private struct PaperDotView: View {
    let highlight: PaperNoveltyScore?
    let isSelected: Bool
    let overrideTint: Color?

    private var tint: Color {
        if let overrideTint { return overrideTint }
        guard let highlight else { return Color.white.opacity(0.55) }
        if highlight.novelty > 0.6 { return Color.purple.opacity(0.85) }
        if highlight.saturation > 0.6 { return Color.orange.opacity(0.85) }
        return Color.mint.opacity(0.75)
    }

    var body: some View {
        Circle()
            .fill(tint)
            .frame(width: isSelected ? 14 : 9, height: isSelected ? 14 : 9)
            .overlay(
                Circle()
                    .stroke(Color.white.opacity(isSelected ? 0.55 : 0.18), lineWidth: isSelected ? 2 : 1)
            )
            .shadow(color: tint.opacity(isSelected ? 0.45 : 0.22), radius: isSelected ? 10 : 4, x: 0, y: 3)
            .animation(.spring(response: 0.35, dampingFraction: 0.85), value: isSelected)
    }
}

@available(macOS 26, iOS 26, *)
struct PaperNodeView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openURL) private var openURL

    let paper: Paper
    let highlight: PaperNoveltyScore?
    let isExpanded: Bool
    let accentTint: Color?
    var onOpen: (() -> Void)?

    @State private var isHovering: Bool = false
    @State private var isShowingTagPrompt: Bool = false
    @State private var newTag: String = ""
    @State private var isShowingQuestion: Bool = false
    @State private var questionText: String = ""
    @State private var questionAnswer: String = ""
    @State private var isAnswering: Bool = false

    private var cardWidth: CGFloat { isExpanded ? 360 : 160 }
    private var latestPaper: Paper {
        model.papers.first(where: { $0.id == paper.id }) ?? paper
    }

    var body: some View {
	        VStack(alignment: .leading, spacing: 8) {
	            HStack(alignment: .top, spacing: 8) {
	                if let accentTint {
	                    Circle()
	                        .fill(accentTint)
	                        .frame(width: 10, height: 10)
	                        .overlay(Circle().stroke(Color.white.opacity(0.35), lineWidth: 1))
	                        .padding(.top, 4)
	                }
	                Text(paper.title)
	                    .font(.subheadline.bold())
	                    .foregroundStyle(.white)
                    .lineLimit(isExpanded ? 3 : 2)
                Spacer(minLength: 0)
                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    .font(.caption.bold())
                    .foregroundStyle(.white.opacity(0.7))
                    .padding(.top, 2)
            }

            if isExpanded {
                HStack(spacing: 10) {
                    if let year = paper.year {
                        Label("\(year)", systemImage: "calendar")
                    }
                    if let pages = paper.pageCount {
                        Label("\(pages)p", systemImage: "doc.text")
                    }
                    Spacer()
                }
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.75))
            }

            Text(paper.summary)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.85))
                .lineLimit(isExpanded ? 8 : 3)

            if isExpanded {
                quickActions
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            HStack(spacing: 6) {
                if let status = paper.readingStatus {
                    Text(status.label)
                        .font(.caption2.bold())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.green.opacity(0.2), in: Capsule())
                }
                if let h = highlight {
                    if h.novelty > 0.6 {
                        Label("Outlier", systemImage: "sparkle")
                            .font(.caption2.bold())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(Color.purple.opacity(0.2), in: Capsule())
                    }
                    if h.saturation > 0.6 {
                        Label("Dense", systemImage: "circle.hexagonpath")
                            .font(.caption2.bold())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(Color.orange.opacity(0.2), in: Capsule())
                    }
                }
            }

            if isExpanded {
                if let takeaways = paper.takeaways, !takeaways.isEmpty {
                    Divider().overlay(Color.white.opacity(0.12))
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Takeaways")
                            .font(.caption.bold())
                            .foregroundStyle(.white.opacity(0.85))
                        ForEach(takeaways.prefix(4), id: \.self) { item in
                            Text("• \(item)")
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.85))
                                .lineLimit(2)
                        }
                    }
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                if let keywords = paper.keywords, !keywords.isEmpty {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 70), spacing: 6)],
                        alignment: .leading,
                        spacing: 6
                    ) {
                        ForEach(keywords.prefix(8), id: \.self) { kw in
                            Text(kw)
                                .font(.caption2.bold())
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(Color.white.opacity(0.08), in: Capsule())
                                .foregroundStyle(.white.opacity(0.9))
                        }
                    }
                    .transition(.opacity)
                }

                if let onOpen {
                    Button {
                        onOpen()
                    } label: {
                        Label("Open full details", systemImage: "doc.text.magnifyingglass")
                            .font(.caption.bold())
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.white.opacity(0.16))
                    .padding(.top, 2)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
        }
        .padding(10)
        .frame(width: cardWidth, alignment: .leading)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(.thinMaterial)
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.black.opacity(isExpanded ? 0.28 : 0.22))
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.06), Color.white.opacity(0.025)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .blendMode(.overlay)
            }
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.white.opacity(isExpanded ? 0.28 : 0.12), lineWidth: isExpanded ? 2 : 1)
        )
        .shadow(color: .black.opacity(isExpanded ? 0.28 : 0.15), radius: isExpanded ? 12 : 6, x: 0, y: isExpanded ? 8 : 3)
        .scaleEffect(isExpanded ? 1.06 : (isHovering ? 1.02 : 1.0))
        .onHover { hovering in
            isHovering = hovering
        }
        .animation(.spring(response: 0.35, dampingFraction: 0.85), value: isExpanded)
        .alert("Add a tag", isPresented: $isShowingTagPrompt) {
            TextField("Tag", text: $newTag)
            Button("Add") {
                addTag()
            }
            Button("Cancel", role: .cancel) {
                newTag = ""
            }
        } message: {
            Text("Tags help you filter and prioritize papers.")
        }
        .sheet(isPresented: $isShowingQuestion) {
            NavigationStack {
                VStack(alignment: .leading, spacing: 12) {
                    Text(latestPaper.title)
                        .font(.headline)
                    TextField("Ask a question about this paper…", text: $questionText, axis: .vertical)
                        .textFieldStyle(.roundedBorder)

                    HStack {
                        Button {
                            let trimmed = questionText.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { return }
                            Task { await answerQuestion(trimmed) }
                        } label: {
                            Label(isAnswering ? "Asking…" : "Ask", systemImage: "paperplane.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isAnswering || questionText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                        Button("Clear") {
                            questionText = ""
                            questionAnswer = ""
                        }
                        .buttonStyle(.bordered)

                        Spacer()
                    }

                    if isAnswering {
                        HStack {
                            ProgressView()
                            Text("Thinking…")
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !questionAnswer.isEmpty {
                        Divider()
                        ScrollView {
                            Text(questionAnswer)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    Spacer()
                }
                .padding()
                .navigationTitle("Ask about paper")
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button("Close") { isShowingQuestion = false }
                    }
                }
            }
            #if os(iOS)
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
            #endif
            .frame(minWidth: 520, minHeight: 460)
        }
    }

    private var quickActions: some View {
        HStack(spacing: 8) {
            Menu {
                Button("Unread") { setReadingStatus(.unread) }
                Button("In progress") { setReadingStatus(.inProgress) }
                Button("Done") { setReadingStatus(.done) }
                Divider()
                Button("Clear status") { setReadingStatus(nil) }
            } label: {
                Label(latestPaper.readingStatus?.label ?? "Status", systemImage: "checkmark.circle")
                    .font(.caption.bold())
            }
            .buttonStyle(.bordered)
            .tint(Color.white.opacity(0.12))
            .help("Set reading status")

            Button {
                toggleImportant()
            } label: {
                Image(systemName: isStarred ? "star.fill" : "star")
            }
            .buttonStyle(.bordered)
            .tint(Color.white.opacity(0.12))
            .help(isStarred ? "Unstar" : "Star")

            Button {
                newTag = ""
                isShowingTagPrompt = true
            } label: {
                Image(systemName: "tag")
            }
            .buttonStyle(.bordered)
            .tint(Color.white.opacity(0.12))
            .help("Add tag")

            Button {
                questionText = ""
                questionAnswer = ""
                isShowingQuestion = true
            } label: {
                Image(systemName: "questionmark.bubble")
            }
            .buttonStyle(.bordered)
            .tint(Color.white.opacity(0.12))
            .help("Ask about this paper")

            Button {
                openURL(latestPaper.fileURL)
            } label: {
                Image(systemName: "doc.richtext")
            }
            .buttonStyle(.bordered)
            .tint(Color.white.opacity(0.12))
            .help("Open PDF")
        }
    }

    private var isStarred: Bool {
        if latestPaper.isImportant == true { return true }
        let tags = latestPaper.userTags ?? []
        return tags.contains(where: { tag in
            let norm = tag.lowercased()
            return norm == "important" || norm == "starred" || norm == "fav" || norm == "favorite"
        })
    }

    private func setReadingStatus(_ status: ReadingStatus?) {
        let notes = latestPaper.userNotes ?? ""
        let tags = latestPaper.userTags ?? []
        model.updatePaperUserData(id: paper.id, notes: notes, tags: tags, status: status)
    }

    private func toggleImportant() {
        let notes = latestPaper.userNotes ?? ""
        var tags = latestPaper.userTags ?? []
        let importantKeys: Set<String> = ["important", "starred", "fav", "favorite"]

        if isStarred {
            tags = tags.filter { !importantKeys.contains($0.lowercased()) }
        } else {
            if !tags.contains(where: { $0.lowercased() == "important" }) {
                tags.append("important")
            }
        }

        model.updatePaperUserData(id: paper.id, notes: notes, tags: tags, status: latestPaper.readingStatus)
    }

    private func addTag() {
        let tag = newTag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !tag.isEmpty else { return }
        let notes = latestPaper.userNotes ?? ""
        var tags = latestPaper.userTags ?? []
        if !tags.contains(where: { $0.lowercased() == tag.lowercased() }) {
            tags.append(tag)
        }
        model.updatePaperUserData(id: paper.id, notes: notes, tags: tags, status: latestPaper.readingStatus)
        newTag = ""
    }

    private func answerQuestion(_ question: String) async {
        isAnswering = true
        questionAnswer = ""
        defer { isAnswering = false }

        let takeaways = latestPaper.takeaways?.prefix(6).joined(separator: "\n• ") ?? ""
        let keywords = latestPaper.keywords?.prefix(12).joined(separator: ", ") ?? ""
        let fallbackInstructions = "You answer questions about a single research paper using only provided context."
        let instructions = PromptStore.loadText("ui.single_paper_qa.instructions.md", fallback: fallbackInstructions)

        let fallbackTemplate = """
        Answer the question using ONLY the information in the paper context. If the context doesn't contain the answer, say what is missing.

        Paper title:
        {{title}}

        Keywords:
        {{keywords}}

        Summary:
        {{summary}}

        Takeaways:
        {{takeaways}}

        Question:
        {{question}}

        Write a concise answer in 5-10 bullet points.
        """
        let template = PromptStore.loadText("ui.single_paper_qa.prompt.md", fallback: fallbackTemplate)
        let prompt = PromptStore.render(template: template, variables: [
            "title": latestPaper.title,
            "keywords": keywords,
            "summary": LLMText.clip(latestPaper.summary, maxChars: 2400),
            "takeaways": takeaways.isEmpty ? "(none provided)" : "• \(takeaways)",
            "question": LLMText.clip(question, maxChars: 700)
        ])

        do {
            let session = LanguageModelSession(instructions: instructions)
            let response = try await session.respond(to: prompt)
            questionAnswer = response.content
        } catch {
            questionAnswer = "Failed to answer: \(error.localizedDescription)"
        }
    }
}

private struct GlassPanel<Content: View>: View {
    let content: () -> Content

    init(@ViewBuilder content: @escaping () -> Content) {
        self.content = content
    }

    var body: some View {
        content()
            .padding(14)
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(.thinMaterial)
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.black.opacity(0.22))
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(MapPalette.panel)
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(MapPalette.panelStroke, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: .black.opacity(0.25), radius: 14, x: 0, y: 8)
    }
}

private struct StatPill: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 6) {
            Text(label.uppercased())
                .font(.caption2.bold())
                .foregroundStyle(.white.opacity(0.6))
            Text(value)
                .font(.caption.bold())
                .foregroundStyle(.white)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(Color.white.opacity(0.05), in: Capsule())
        .overlay(Capsule().stroke(Color.white.opacity(0.12), lineWidth: 1))
    }
}

@available(macOS 26, iOS 26, *)
struct BridgingSection: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selectedClusterIDs: Set<Int>

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Bridging papers")
                .font(.headline)
                .foregroundStyle(.white)

            let selected = Array(selectedClusterIDs)
            if selected.count != 2 {
                Text("Select exactly two clusters in the map to find bridging papers.")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            } else {
                if let path = influencePath(selected: selected), !path.isEmpty {
                    Text("Influence path")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                    Text(path.joined(separator: " → "))
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.8))
                        .lineLimit(3)
                }
                if let claims = claimPath(selected: selected), !claims.isEmpty {
                    Text("Claim path")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                    ForEach(claims.prefix(4), id: \.self) { stmt in
                        Text("• \(stmt)")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }

                let bridges = model.bridgingPapers(between: selected[0], and: selected[1])
                if bridges.isEmpty {
                    Text("No strong bridging papers found.")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.7))
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(bridges) { result in
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(result.paper.title)
                                        .font(.subheadline.bold())
                                        .foregroundStyle(.white)
                                    Text(
                                        String(
                                            format: "Bridge score: %.3f (c1=%.3f, c2=%.3f)",
                                            result.combinedScore,
                                            result.scoreToFirst,
                                            result.scoreToSecond
                                        )
                                    )
                                    .font(.caption2)
                                    .foregroundStyle(.white.opacity(0.7))
                                    Text(result.paper.summary)
                                        .font(.caption2)
                                        .foregroundStyle(.white.opacity(0.85))
                                        .lineLimit(4)
                                }
                                .padding(8)
                                .background(Color.white.opacity(0.05))
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    }
                    .frame(maxHeight: 200)
                }
            }
        }
    }

    private func influencePath(selected: [Int]) -> [String]? {
        guard selected.count == 2 else { return nil }
        return model.influencePath(between: selected[0], and: selected[1])
    }

    private func claimPath(selected: [Int]) -> [String]? {
        guard selected.count == 2 else { return nil }
        return model.claimPathBetweenClusters(selected[0], selected[1])
    }
}

struct ArrowShape: Shape {
    let start: CGPoint
    let end: CGPoint

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: start)
        path.addLine(to: end)

        let angle = atan2(end.y - start.y, end.x - start.x)
        let arrowLength: CGFloat = 8
        let arrowAngle: CGFloat = .pi / 7

        let p1 = CGPoint(
            x: end.x - arrowLength * cos(angle - arrowAngle),
            y: end.y - arrowLength * sin(angle - arrowAngle)
        )
        let p2 = CGPoint(
            x: end.x - arrowLength * cos(angle + arrowAngle),
            y: end.y - arrowLength * sin(angle + arrowAngle)
        )
        path.addLine(to: p1)
        path.move(to: end)
        path.addLine(to: p2)
        return path
    }
}
