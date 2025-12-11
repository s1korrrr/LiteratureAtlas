import SwiftUI

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

    static let panel = Color.white.opacity(0.06)
    static let panelStroke = Color.white.opacity(0.12)

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

@available(macOS 26, iOS 26, *)
struct MapView: View {
    @EnvironmentObject private var model: AppModel
    @State private var resolution: ClusterResolution = .medium
    @State private var lens: MapLens = .standard
    @State private var zoomLevel: ZoomLevel = .mega
    @State private var selectedMega: Cluster?
    @State private var selectedSubtopic: Cluster?
    @State private var selectedClusterIDs: Set<Int> = []
    @State private var debateText: String = ""
    @State private var showDebate: Bool = false
    @State private var hypothetical: HypotheticalPaper?
    @State private var showHypothetical: Bool = false

    private var activeClusters: [Cluster] {
        switch zoomLevel {
        case .mega:
            return model.lensAdjustedClusters(model.megaClusters, lens: lens)
        case .topics:
            let base = (selectedMega ?? model.megaClusters.first)?.subclusters ?? []
            return model.lensAdjustedClusters(base, lens: lens)
        case .papers:
            return []
        }
    }

    private var activePapers: [Paper] {
        guard zoomLevel == .papers else { return [] }
        guard let sub = selectedSubtopic ?? selectedMega?.subclusters?.first else { return [] }
        return model.papers.filter { $0.clusterIndex == sub.id }
    }

    private var activePaperDriftVector: (dx: Double, dy: Double)? {
        guard let sub = selectedSubtopic ?? selectedMega?.subclusters?.first else { return nil }
        guard let drift = model.analyticsSummary?.drift else { return nil }
        return drift.last(where: { $0.clusterID == sub.id }).map { ($0.dx ?? 0, $0.dy ?? 0) }
    }

    private var paperHighlights: [UUID: PaperNoveltyScore] {
        let ids = Set(activePapers.map { $0.id })
        let scores = model.noveltyHighlights(neighbors: 3)
        return Dictionary(uniqueKeysWithValues: scores.filter { ids.contains($0.paperID) }.map { ($0.paperID, $0) })
    }

    var body: some View {
        NavigationStack {
            ZStack {
                MapPalette.backdrop.ignoresSafeArea()

                VStack(alignment: .leading, spacing: 14) {
                    header

                    if model.papers.isEmpty {
                        Text("Ingest some PDFs first on the Ingest tab.")
                            .foregroundStyle(.secondary)
                        Spacer()
                    } else {
                        controlDeck

                        if zoomLevel == .topics, let mega = selectedMega {
                            breadcrumb(for: mega)
                        }
                    if zoomLevel == .papers, let sub = selectedSubtopic {
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
                        PaperScatterView(papers: activePapers, highlights: paperHighlights, driftVector: activePaperDriftVector)
                            .frame(minHeight: 360)
                    } else {
                        ClusterMapAndSidebar(
                            clusters: activeClusters,
                            selectedClusterIDs: $selectedClusterIDs,
                            isZoomed: zoomLevel == .topics,
                            showBridging: zoomLevel == .topics,
                                onZoomOut: zoomLevel == .mega ? nil : {
                                    zoomLevel = .mega
                                    selectedMega = nil
                                    selectedSubtopic = nil
                                    selectedClusterIDs.removeAll()
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
                        }

                        if selectedClusterIDs.count == 2 {
                            let pair = Array(selectedClusterIDs)
                            GlassPanel {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Compare & brainstorm").font(.headline).foregroundStyle(.white)
                                    HStack {
                                        Button {
                                            debateText = model.simulateClusterDebate(firstID: pair[0], secondID: pair[1])
                                            showDebate = true
                                        } label: {
                                            Label("Simulate debate", systemImage: "person.3.sequence")
                                        }
                                        .buttonStyle(.borderedProminent)
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
            }
            .navigationTitle("Map")
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
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Step 2 — Knowledge Galaxy")
                    .font(.title.bold())
                    .foregroundStyle(.white)
                HStack(spacing: 8) {
                    StatPill(label: "Papers", value: "\(model.papers.count)")
                    StatPill(label: "Clusters", value: "\(max(model.megaClusters.count, model.clusters.count))")
                    StatPill(label: "Lens", value: lens.label)
                }
            }
            Spacer()
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
                zoomLevel = .mega
                selectedMega = nil
                selectedSubtopic = nil
                selectedClusterIDs.removeAll()
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
                zoomLevel = .topics
                selectedSubtopic = nil
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
        selectedMega = nil
        selectedSubtopic = nil
        zoomLevel = .mega
        Task {
            await model.buildMultiScaleGalaxy(level0Range: 5...8, level1Range: resolution.subtopicRange)
        }
    }

    private func selectMega(_ cluster: Cluster) {
        selectedMega = cluster
        selectedSubtopic = nil
        selectedClusterIDs = [cluster.id]
        zoomLevel = .topics
    }

    private func selectSubtopic(_ cluster: Cluster) {
        selectedSubtopic = cluster
        selectedClusterIDs = [cluster.id]
        zoomLevel = .papers
    }
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
struct ClusterGraphView: View {
    let clusters: [Cluster]
    @Binding var selectedClusterIDs: Set<Int>
    let lensLabel: String
    let onSelect: ((Cluster) -> Void)?
    var driftMagnitudes: [Int: Double] = [:]
    var driftVectors: [Int: (dx: Double, dy: Double)] = [:]
    var ideaEdges: [(Int, Int, Double)] = []

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - 80

            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(MapPalette.canvas)
                    .overlay(MapPalette.glow)
                    .overlay(
                        AngularGradient(
                            colors: [.clear, Color.white.opacity(0.1), .clear],
                            center: .center
                        )
                        .blur(radius: 80)
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
                        if let firstIndex = clusters.firstIndex(where: { $0.id == ids[0] }),
                           let secondIndex = clusters.firstIndex(where: { $0.id == ids[1] }) {
                            let p1 = position(for: clusters[firstIndex], fallbackIndex: firstIndex, total: clusters.count, center: center, radius: radius)
                            let p2 = position(for: clusters[secondIndex], fallbackIndex: secondIndex, total: clusters.count, center: center, radius: radius)
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
                        if let sIdx = clusters.firstIndex(where: { $0.id == edge.0 }),
                           let dIdx = clusters.firstIndex(where: { $0.id == edge.1 }) {
                            let p1 = position(for: clusters[sIdx], fallbackIndex: sIdx, total: clusters.count, center: center, radius: radius)
                            let p2 = position(for: clusters[dIdx], fallbackIndex: dIdx, total: clusters.count, center: center, radius: radius)
                            let width = max(1, min(4, CGFloat(edge.2) * 0.5))
                            Path { path in
                                path.move(to: p1)
                                path.addLine(to: p2)
                            }
                            .stroke(Color.orange.opacity(0.35), style: StrokeStyle(lineWidth: width, lineCap: .round))
                        }
                    }

                    ForEach(Array(clusters.enumerated()), id: \.1.id) { index, cluster in
                        let pos = position(for: cluster, fallbackIndex: index, total: clusters.count, center: center, radius: radius)
                        ClusterNodeView(cluster: cluster, isSelected: selectedClusterIDs.contains(cluster.id))
                            .position(pos)
                            .onTapGesture {
                                if selectedClusterIDs.contains(cluster.id) {
                                    selectedClusterIDs.remove(cluster.id)
                                } else {
                                    if selectedClusterIDs.count >= 2 {
                                        selectedClusterIDs.removeAll()
                                    }
                                    selectedClusterIDs.insert(cluster.id)
                                }
                                onSelect?(cluster)
                            }

                        if let drift = driftMagnitudes[cluster.id], drift > 0, let layout = cluster.layoutPosition {
                            let vec = driftVectors[cluster.id] ?? (dx: Double(cos(hashAngle(for: cluster.id))), dy: Double(sin(hashAngle(for: cluster.id))))
                            let length = CGFloat(min(drift, 0.6)) * radius * 0.35
                            let start = CGPoint(
                                x: center.x + (CGFloat(layout.x) - 0.5) * radius * 2,
                                y: center.y + (CGFloat(layout.y) - 0.5) * radius * 2
                            )
                            let end = CGPoint(
                                x: start.x + length * CGFloat(vec.dx),
                                y: start.y + length * CGFloat(vec.dy)
                            )
                            ArrowShape(start: start, end: end)
                                .stroke(Color.cyan.opacity(0.8), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        }
                    }
                }
            }
        }
    }

    private func position(for cluster: Cluster, fallbackIndex: Int, total: Int, center: CGPoint, radius: CGFloat) -> CGPoint {
        if let layout = cluster.layoutPosition {
            let x = center.x + (CGFloat(layout.x) - 0.5) * radius * 2
            let y = center.y + (CGFloat(layout.y) - 0.5) * radius * 2
            return CGPoint(x: x, y: y)
        }

        let angle = 2 * Double.pi * Double(fallbackIndex) / Double(max(total, 1))
        return CGPoint(
            x: center.x + radius * CGFloat(cos(angle)),
            y: center.y + radius * CGFloat(sin(angle))
        )
    }

    private func hashAngle(for id: Int) -> CGFloat {
        // Deterministic pseudo-angle per cluster id
        let seed = UInt64(id &* 6364136223846793005 &+ 1)
        let frac = Double(seed % 10_000) / 10_000.0
        return CGFloat(frac * 2 * Double.pi)
    }
}

@available(macOS 26, iOS 26, *)
struct ClusterNodeView: View {
    let cluster: Cluster
    let isSelected: Bool

    private var size: CGFloat {
        let base: CGFloat = 130
        let growth = CGFloat(min(cluster.memberPaperIDs.count, 14)) * 4
        return base + growth
    }

    var body: some View {
        VStack(spacing: 4) {
            Text(cluster.name)
                .font(.headline.weight(.semibold))
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .foregroundStyle(.white)
            Text("\(cluster.memberPaperIDs.count) papers")
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
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: isSelected)
    }
}

@available(macOS 26, iOS 26, *)
struct ClusterDetailCard: View {
    @EnvironmentObject private var model: AppModel
    let cluster: Cluster
    let onZoom: (() -> Void)?

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(cluster.name)
                            .font(.headline)
                            .foregroundStyle(.white)
                        Text("\(cluster.memberPaperIDs.count) papers")
                            .font(.caption.bold())
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    Spacer()
                    Circle()
                        .fill(MapPalette.nodeGradient(for: cluster.id))
                        .frame(width: 14, height: 14)
                        .shadow(color: .white.opacity(0.5), radius: 6)
                }

                Text(cluster.metaSummary)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.9))
                    .lineLimit(6)

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

                if let example = model.papers.first(where: { $0.clusterIndex == cluster.id }) {
                    Text("Sample: \(example.title)")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.8))
                }
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
struct PaperScatterView: View {
    let papers: [Paper]
    let highlights: [UUID: PaperNoveltyScore]
    let driftVector: (dx: Double, dy: Double)?

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - 60
            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(MapPalette.canvas)
                    .overlay(MapPalette.glow)
                    .shadow(color: .black.opacity(0.25), radius: 16, x: 0, y: 10)
                if let drift = driftVector {
                    let len = radius * 0.25
                    let end = CGPoint(x: center.x + len * CGFloat(drift.dx), y: center.y + len * CGFloat(drift.dy))
                    ArrowShape(start: center, end: end)
                        .stroke(Color.cyan.opacity(0.7), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                }

                if papers.isEmpty {
                    Text("Select a subtopic to see its papers.")
                        .foregroundStyle(.white.opacity(0.7))
                } else {
                    ForEach(Array(papers.enumerated()), id: \.1.id) { idx, paper in
                        let pos = position(for: idx, total: papers.count, center: center, radius: radius)
                        let score = highlights[paper.id]
                        PaperNodeView(paper: paper, highlight: score)
                            .position(pos)
                    }
                }
            }
        }
    }

    private func position(for index: Int, total: Int, center: CGPoint, radius: CGFloat) -> CGPoint {
        let angle = 2 * Double.pi * Double(index) / Double(max(total, 1))
        let radial = radius * CGFloat(0.4 + 0.6 * sqrt(Double(index + 1) / Double(max(total, 1))))
        return CGPoint(
            x: center.x + radial * CGFloat(cos(angle)),
            y: center.y + radial * CGFloat(sin(angle))
        )
    }
}

@available(macOS 26, iOS 26, *)
struct PaperNodeView: View {
    let paper: Paper
    let highlight: PaperNoveltyScore?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(paper.title)
                .font(.subheadline.bold())
                .foregroundStyle(.white)
                .lineLimit(2)
            Text(paper.summary)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.85))
                .lineLimit(3)
            if let status = paper.readingStatus {
                Text(status.label)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(Color.green.opacity(0.2), in: Capsule())
            }
            if let h = highlight {
                HStack(spacing: 6) {
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
        }
        .padding(10)
        .frame(width: 180, alignment: .leading)
        .background(
            LinearGradient(
                colors: [Color.white.opacity(0.06), Color.white.opacity(0.02)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .shadow(color: .black.opacity(0.15), radius: 6, x: 0, y: 3)
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
            .background(MapPalette.panel)
            .background(.ultraThinMaterial.opacity(0.4))
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
