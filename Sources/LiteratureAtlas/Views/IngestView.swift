import SwiftUI
import UniformTypeIdentifiers
import FoundationModels

@available(macOS 26, iOS 26, *)
struct IngestView: View {
    @EnvironmentObject private var model: AppModel
    @State private var showFolderPicker = false
    @State private var assumptionQuery: String = ""
    @State private var assumptionNarrative: String = ""
    @State private var topEdges: [ClaimEdge] = []
    @State private var showStyleTips = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    headerCard

                    GlassCard {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack(spacing: 12) {
                                Button {
                                    showFolderPicker = true
                                } label: {
                                    Label("Select Folder of PDFs", systemImage: "folder")
                                    .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(.borderedProminent)
                                .disabled(model.isIngesting)

                                Button("Stop") {
                                    model.cancelIngestion()
                                }
                                .buttonStyle(.bordered)
                                .disabled(!model.isIngesting)
                            }

                            if let folder = model.selectedFolder {
                                Label("Selected folder: \(folder.lastPathComponent)", systemImage: "checkmark.folder")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }

                            ProgressView(value: model.ingestionProgress)
                                .tint(.mint)
                                .animation(.easeInOut, value: model.ingestionProgress)
                            HStack {
                                Text("\(model.ingestionCompletedCount)/\(max(1, model.ingestionTotalCount)) files")
                                Spacer()
                                if !model.ingestionCurrentFile.isEmpty {
                                    Text("Now: \(model.ingestionCurrentFile)")
                                }
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }

                    GlassCard {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Activity")
                        .font(.headline)
                    ingestStatusRow
                    if let latest = latestIngestedPaper {
                        Divider().padding(.vertical, 4)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Last processed").font(.caption).foregroundStyle(.secondary)
                            Text(latest.title).font(.subheadline.bold())
                            HStack(spacing: 10) {
                                        if let year = latest.year {
                                            Label("Year \(year)", systemImage: "calendar")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                        if let pages = latest.pageCount {
                                            Label("\(pages) pages", systemImage: "doc.on.doc")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                }
                            }
                        }
                    }

                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Log")
                                .font(.headline)
                            if showStyleTips {
                                Text("Tip: keep this view open; clustering can run in parallel; hover status pills for detail.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            ScrollView {
                                Text(model.ingestionLog.isEmpty ? "No logs yet." : model.ingestionLog)
                                    .font(.system(.footnote, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .frame(minHeight: 200)
                        }
                    }

                    if !model.papers.isEmpty {
                        readingPlannerCard
                        claimGraphCard
                        assumptionStressCard
                    }

                    Spacer(minLength: 24)
                }
                .padding()
            }
            .navigationTitle("Literature Atlas")
        }
        .fileImporter(
            isPresented: $showFolderPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let folder = urls.first {
                    model.ingestFolder(url: folder)
                    showStyleTips = true
                }
            case .failure(let error):
                model.ingestionLog += "\nFolder picker error: \(error.localizedDescription)"
            }
        }
    }

    private var headerCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Step 1 - Ingest PDFs")
                            .font(.title.bold())
                        Text("Summarize first pages on-device, embed with NLContextualEmbedding, and write JSON files into the repo Output folder (no data leaves the app).")
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    availabilityBadge
                }
            }
        }
    }

    private var availabilityBadge: some View {
        let availability = SystemLanguageModel.default.availability
        switch availability {
        case .available:
            return AnyView(
                Label("On-device model ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .padding(8)
                    .background(.green.opacity(0.15), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            )
        case .unavailable(let reason):
            return AnyView(
                Label("Model unavailable", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .padding(8)
                    .background(.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        Text(String(describing: reason))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.top, 30), alignment: .topLeading
                    )
            )
        }
    }

    private var ingestStatusRow: some View {
        HStack(spacing: 12) {
            statusPill(title: model.isIngesting ? "Ingesting" : "Idle", color: model.isIngesting ? .mint : .gray)
            statusPill(title: model.isClustering ? "Clustering" : "Not clustering", color: model.isClustering ? .blue : .gray.opacity(0.8))
            statusPill(title: "Papers: \(model.papers.count)", color: .purple.opacity(0.8))
        }
    }

    private func statusPill(title: String, color: Color) -> some View {
        Text(title)
            .font(.caption.bold())
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .background(color.opacity(0.15), in: Capsule())
            .overlay(Capsule().stroke(color.opacity(0.3), lineWidth: 1))
            .foregroundStyle(color)
    }

    private var readingPlannerCard: some View {
        let nextPapers = model.recommendedNextPapers()
        let blind = model.blindSpots(limit: 3)
        let curriculum = model.adaptiveCurriculum().prefix(5)
        return GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Reading planner")
                    .font(.headline)
                Text("Adaptive picks based on your done/important papers and question history.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if nextPapers.isEmpty {
                    Text("Mark some papers as done to unlock recommendations.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Next up (\(nextPapers.count))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(nextPapers.prefix(5)) { paper in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(paper.title).font(.subheadline.bold())
                                Text(paper.summary).font(.caption2).lineLimit(2)
                            }
                            Spacer()
                            if let status = paper.readingStatus {
                                Text(status.label)
                                    .font(.caption2)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 4)
                                    .background(Color.blue.opacity(0.1), in: Capsule())
                            }
                        }
                    }
                }

                if !blind.isEmpty {
                    Divider().padding(.vertical, 4)
                    Text("Blind spots (central but far from you)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(blind) { paper in
                        Text("• \(paper.title)")
                            .font(.caption2)
                    }
                }

                if !curriculum.isEmpty {
                    Divider().padding(.vertical, 4)
                    Text("Adaptive curriculum")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(curriculum, id: \.id) { step in
                        HStack {
                            Text(step.stage.label)
                                .font(.caption2.bold())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 3)
                                .background(Color.orange.opacity(0.1), in: Capsule())
                            Text(step.paper.title)
                                .font(.caption2)
                                .lineLimit(1)
                            Spacer()
                        }
                    }
                }
            }
        }
    }

    private var claimGraphCard: some View {
        let edges = topEdges.isEmpty ? model.claimGraphEdges() : topEdges
        return GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Claim graph")
                    .font(.headline)
                Text("Shows how claims support, extend, or contradict each other across papers.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if edges.isEmpty {
                    Text("No claim relations yet. Ingest papers to build the evidence graph.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Top relations")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(edges.prefix(5), id: \.id) { edge in
                        if let source = model.papers.first(where: { $0.claims?.contains(where: { $0.id == edge.sourceClaimID }) == true })?.title,
                           let target = model.papers.first(where: { $0.claims?.contains(where: { $0.id == edge.targetClaimID }) == true })?.title {
                            HStack(alignment: .top, spacing: 6) {
                                Text(edge.kind.rawValue.capitalized)
                                    .font(.caption2.bold())
                                    .padding(6)
                                    .background(Color.blue.opacity(0.1), in: Capsule())
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("From \(source) → \(target)")
                                        .font(.subheadline)
                                    if let rationale = edge.rationale {
                                        Text(rationale)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                    Button {
                        topEdges = model.claimGraphEdges()
                    } label: {
                        Label("Refresh relations", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    private var assumptionStressCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Assumption stress test")
                    .font(.headline)
                Text("Pick an assumption (e.g., “infinite liquidity”, “Poisson arrivals”) and see which claims rely on it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(alignment: .top) {
                    TextField("e.g., infinite liquidity", text: $assumptionQuery, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                    Button {
                        let trimmed = assumptionQuery.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else { return }
                        let report = model.assumptionStressReport(for: trimmed)
                        assumptionNarrative = report.narrative
                    } label: {
                        Label("Run", systemImage: "exclamationmark.shield")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.papers.isEmpty)
                }
                if !assumptionNarrative.isEmpty {
                    Text(assumptionNarrative)
                        .font(.footnote)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    @ViewBuilder
    private var availabilityBanner: some View {
        let availability = SystemLanguageModel.default.availability
        switch availability {
        case .available:
            Text("On-device language model: available")
                .font(.caption)
                .foregroundStyle(.green)
        case .unavailable(let reason):
            Text("On-device language model unavailable: \(String(describing: reason))")
                .font(.caption)
                .foregroundStyle(.red)
        }
    }
    
    private var latestIngestedPaper: Paper? {
        let papers = model.papers
        return papers.max(by: { ($0.ingestedAt ?? Date.distantPast) < ($1.ingestedAt ?? Date.distantPast) })
    }
}
