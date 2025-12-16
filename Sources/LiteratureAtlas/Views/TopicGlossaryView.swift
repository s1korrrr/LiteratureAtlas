import SwiftUI

@available(macOS 26, iOS 26, *)
struct TopicGlossaryView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var onSelectMega: ((Cluster) -> Void)?
    var onSelectSubtopic: ((Cluster) -> Void)?

    @State private var expandedMegaIDs: Set<Int> = []
    @State private var renameClusterID: Int?
    @State private var renameText: String = ""
    @State private var isShowingRename: Bool = false
    @State private var namingInFlight: Set<Int> = []

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.08, green: 0.08, blue: 0.16),
                        Color(red: 0.12, green: 0.09, blue: 0.22),
                        Color(red: 0.10, green: 0.14, blue: 0.30)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        if model.megaClusters.isEmpty {
                            Text("Run clustering to generate a galaxy.")
                                .foregroundStyle(.white.opacity(0.75))
                                .padding(.top, 20)
                        } else {
                            let papersByID = Dictionary(uniqueKeysWithValues: model.papers.map { ($0.id, $0) })
                            ForEach(sortedMegaClusters()) { mega in
                                megaCard(mega, papersByID: papersByID)
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Topic glossary")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .alert("Rename topic", isPresented: $isShowingRename) {
            TextField("Topic name", text: $renameText)
            Button("Save") {
                guard let id = renameClusterID else { return }
                model.renameGalaxyCluster(clusterID: id, name: renameText, metaSummary: nil)
                renameClusterID = nil
            }
            Button("Cancel", role: .cancel) {
                renameClusterID = nil
            }
        } message: {
            Text("Pick a short, descriptive name.")
        }
    }

    private func sortedMegaClusters() -> [Cluster] {
        model.megaClusters.sorted { l, r in
            let lp = model.isGalaxyPinned(clusterID: l.id)
            let rp = model.isGalaxyPinned(clusterID: r.id)
            if lp != rp { return lp && !rp }
            return l.id < r.id
        }
    }

    private func sortedSubclusters(_ subs: [Cluster]) -> [Cluster] {
        subs.sorted { l, r in
            let lp = model.isGalaxyPinned(clusterID: l.id)
            let rp = model.isGalaxyPinned(clusterID: r.id)
            if lp != rp { return lp && !rp }
            return l.id < r.id
        }
    }

    private func megaCard(_ mega: Cluster, papersByID: [UUID: Paper]) -> some View {
        let insights = model.clusterInsights(for: mega, papersByID: papersByID)
        return GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text(mega.name)
                                .font(.headline)
                                .foregroundStyle(.white)
                            NameSourcePill(source: model.clusterNameSources[mega.id] ?? .heuristic)
                        }
                        Text("\(mega.memberPaperIDs.count) papers")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.75))
                    }
                    Spacer()

                    if let onSelectMega {
                        Button {
                            dismiss()
                            onSelectMega(mega)
                        } label: {
                            Image(systemName: "arrow.right.circle")
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.white.opacity(0.85))
                    }

                    rowActions(for: mega)
                }

                if !insights.topKeywords.isEmpty {
                    KeywordChips(keywords: insights.topKeywords)
                }

                if !insights.topTitles.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Top papers")
                            .font(.caption.bold())
                            .foregroundStyle(.white.opacity(0.85))
                        ForEach(insights.topTitles, id: \.self) { title in
                            Text("• \(title)")
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.85))
                                .lineLimit(2)
                        }
                    }
                }

                if let subs = mega.subclusters, !subs.isEmpty {
                    let expanded = Binding(
                        get: { expandedMegaIDs.contains(mega.id) },
                        set: { value in
                            if value {
                                expandedMegaIDs.insert(mega.id)
                            } else {
                                expandedMegaIDs.remove(mega.id)
                            }
                        }
                    )
                    DisclosureGroup(isExpanded: expanded) {
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(sortedSubclusters(subs)) { sub in
                                subtopicRow(sub, papersByID: papersByID)
                            }
                        }
                        .padding(.top, 8)
                    } label: {
                        Text("Subtopics (\(subs.count))")
                            .font(.subheadline.bold())
                            .foregroundStyle(.white)
                    }
                    .tint(.white.opacity(0.85))
                }
            }
        }
    }

    private func subtopicRow(_ sub: Cluster, papersByID: [UUID: Paper]) -> some View {
        let insights = model.clusterInsights(for: sub, papersByID: papersByID, keywordLimit: 5, titleLimit: 3)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(sub.name)
                            .font(.subheadline.bold())
                            .foregroundStyle(.white)
                        NameSourcePill(source: model.clusterNameSources[sub.id] ?? .heuristic)
                    }
                    Text("\(sub.memberPaperIDs.count) papers")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.75))
                }
                Spacer()

                if let onSelectSubtopic {
                    Button {
                        dismiss()
                        onSelectSubtopic(sub)
                    } label: {
                        Image(systemName: "arrow.right.circle")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.white.opacity(0.85))
                }

                rowActions(for: sub, compact: true)
            }

            if !insights.topKeywords.isEmpty {
                KeywordChips(keywords: insights.topKeywords)
            }

            if !insights.topTitles.isEmpty {
                ForEach(insights.topTitles, id: \.self) { title in
                    Text("• \(title)")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.85))
                        .lineLimit(1)
                }
            }
        }
        .padding(10)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        )
    }

    private func rowActions(for cluster: Cluster, compact: Bool = false) -> some View {
        HStack(spacing: compact ? 6 : 8) {
            Button {
                model.toggleGalaxyPin(clusterID: cluster.id)
            } label: {
                Image(systemName: model.isGalaxyPinned(clusterID: cluster.id) ? "pin.fill" : "pin")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.white.opacity(0.85))

            Button {
                renameClusterID = cluster.id
                renameText = cluster.name
                isShowingRename = true
            } label: {
                Image(systemName: "pencil")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.white.opacity(0.85))

            Button {
                guard !namingInFlight.contains(cluster.id) else { return }
                namingInFlight.insert(cluster.id)
                Task {
                    await model.autoNameGalaxyCluster(clusterID: cluster.id)
                    _ = await MainActor.run { namingInFlight.remove(cluster.id) }
                }
            } label: {
                if namingInFlight.contains(cluster.id) {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "sparkles")
                }
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.white.opacity(0.85))
            .contextMenu {
                Button("Force AI re-name") {
                    guard !namingInFlight.contains(cluster.id) else { return }
                    namingInFlight.insert(cluster.id)
                    Task {
                        await model.autoNameGalaxyCluster(clusterID: cluster.id, force: true)
                        _ = await MainActor.run { namingInFlight.remove(cluster.id) }
                    }
                }
            }
        }
    }
}

private struct NameSourcePill: View {
    let source: ClusterNameSource

    var body: some View {
        Text(source.label)
            .font(.caption2.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.white.opacity(0.10), in: Capsule())
            .foregroundStyle(.white.opacity(0.85))
    }
}

private struct KeywordChips: View {
    let keywords: [String]

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 6)], alignment: .leading, spacing: 6) {
            ForEach(keywords, id: \.self) { kw in
                Text(kw.capitalized)
                    .font(.caption2.bold())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.08), in: Capsule())
                    .foregroundStyle(.white.opacity(0.9))
            }
        }
    }
}
