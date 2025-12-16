import SwiftUI
import UniformTypeIdentifiers
import FoundationModels

@available(macOS 26, iOS 26, *)
struct PaperDetailView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let paper: Paper
    var onClose: (() -> Void)? = nil

    @State private var explainLevel: ExplainLevel = .expert
    @State private var generatedELI: String = ""
    @State private var notesText: String = ""
    @State private var tagsText: String = ""
    @State private var statusSelection: ReadingStatus = .unread
    @State private var blueprintText: String = ""
    @State private var showQuiz: Bool = false
    @State private var quizIndex: Int = 0
    @State private var revealAnswer: Bool = false

    private var metrics: AnalyticsSummary.PaperMetric? {
        model.analyticsSummary?.paperMetrics.first(where: { $0.paperID == paper.id })
    }

    private var dominantFactorLabel: String? {
        guard let summary = model.analyticsSummary else { return nil }
        guard let entry = summary.factorLoadings.first(where: { $0.paperID == paper.id }) else { return nil }
        guard let maxIdx = entry.scores.enumerated().max(by: { $0.element < $1.element })?.offset else { return nil }
        guard maxIdx < summary.factorLabels.count else { return "F\(maxIdx)" }
        return "F\(maxIdx): \(summary.factorLabels[maxIdx])"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header

                GlassCard {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Summary").font(.headline)
                        Text(paper.summary)
                            .font(.body)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                if let m = metrics {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Impact signals").font(.headline)
                            if let dom = dominantFactorLabel {
                                Text("Dominant factor: \(dom)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            HStack {
                                MetricPill(label: "Novelty z", value: m.zNovelty, tint: .pink)
                                MetricPill(label: "Consensus z", value: m.zConsensus, tint: .mint)
                                MetricPill(label: "Comb novelty", value: m.novCombinatorial, tint: .orange)
                            }
                            HStack {
                                MetricPill(label: "Influence +", value: m.influencePos, tint: .green)
                                MetricPill(label: "Influence −", value: m.influenceNeg, tint: .red)
                                MetricPill(label: "Drift contrib", value: m.driftContrib, tint: .cyan)
                            }
                            HStack(spacing: 10) {
                                RoleBadge(label: "Originator", value: m.roleSource)
                                RoleBadge(label: "Bridge", value: m.roleBridge)
                                RoleBadge(label: "Consolidator", value: m.roleSink)
                            }
                            Text(String(format: "Uncertainty: novelty ±%.3f; consensus ±%.3f", m.noveltyUncertainty, m.consensusUncertainty))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if let takeaways = paper.takeaways, !takeaways.isEmpty {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Takeaways").font(.headline)
                            ForEach(takeaways, id: \.self) { item in
                                HStack(alignment: .top, spacing: 6) {
                                    Image(systemName: "checkmark.circle.fill").font(.caption2)
                                    Text(item)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .font(.footnote)
                            }
                        }
                    }
                }

                if let intro = paper.introSummary {
                    sectionCard(title: "Introduction", text: intro)
                }
                if let methods = paper.methodSummary {
                    sectionCard(title: "Methods", text: methods)
                }
                if let results = paper.resultsSummary {
                    sectionCard(title: "Results", text: results)
                }

                if let claims = latestPaper()?.claims, !claims.isEmpty {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Claims").font(.headline)
                            ForEach(claims, id: \.id) { claim in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("• \(claim.statement)")
                                        .font(.subheadline)
                                    if !claim.assumptions.isEmpty {
                                        Text("Assumptions: \(claim.assumptions.joined(separator: ", "))")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    if let eval = claim.evaluation {
                                        let metrics = eval.metrics.joined(separator: ", ")
                                        Text("Eval: \(eval.dataset ?? "Unknown dataset") \(eval.period ?? "") • \(metrics)")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                } else if let assumptions = latestPaper()?.assumptions, !assumptions.isEmpty {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Assumptions").font(.headline)
                            Text(assumptions.joined(separator: ", "))
                                .font(.subheadline)
                        }
                    }
                }

                if let eval = latestPaper()?.evaluationContext {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Evaluation context").font(.headline)
                            if let dataset = eval.dataset { Label(dataset, systemImage: "externaldrive").font(.subheadline) }
                            if let period = eval.period { Label("Period: \(period)", systemImage: "calendar").font(.caption) }
                            if !eval.metrics.isEmpty { Label("Metrics: \(eval.metrics.joined(separator: ", "))", systemImage: "chart.bar.doc.horizontal").font(.caption) }
                        }
                    }
                }

                if let pipeline = latestPaper()?.methodPipeline {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Method pipeline").font(.headline)
                            ForEach(Array(pipeline.steps.enumerated()), id: \.offset) { idx, step in
                                HStack(alignment: .top, spacing: 8) {
                                    Text("\(idx + 1).").font(.caption.bold())
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("[\(step.stage.rawValue)] \(step.label)").font(.subheadline)
                                        if let detail = step.detail {
                                            Text(detail).font(.caption).foregroundStyle(.secondary)
                                        }
                                    }
                                }
                            }
                            Button {
                                blueprintText = model.methodBlueprint(for: paper.id) ?? "Blueprint unavailable."
                            } label: {
                                Label("Generate pseudocode", systemImage: "terminal")
                            }
                            .buttonStyle(.borderedProminent)

                            if !blueprintText.isEmpty {
                                Divider().padding(.vertical, 4)
                                ScrollView {
                                    Text(blueprintText)
                                        .font(.system(.footnote, design: .monospaced))
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .frame(minHeight: 120, maxHeight: 220)
                            }
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Explain like I'm...").font(.headline)
                            Picker("Level", selection: $explainLevel) {
                                ForEach(ExplainLevel.allCases, id: \.self) { level in
                                    Text(level.label).tag(level)
                                }
                            }
                            .pickerStyle(.segmented)
                        }

                        Button("Generate") { Task { await generateELI() } }
                            .buttonStyle(.borderedProminent)

                        if !generatedELI.isEmpty {
                            Text(generatedELI)
                                .font(.body)
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Notes & tags").font(.headline)
                        TextEditor(text: $notesText)
                            .frame(minHeight: 120)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2)))
                            .onChange(of: notesText) { _, newValue in
                                saveUserData(notes: newValue, tags: tagsText)
                            }
                        HStack {
                            TextField("Tags (comma separated)", text: $tagsText)
                                .textFieldStyle(.roundedBorder)
                                .onChange(of: tagsText) { _, newValue in
                                    saveUserData(notes: notesText, tags: newValue)
                                }
                            Picker("Status", selection: $statusSelection) {
                                ForEach(ReadingStatus.allCases, id: \.self) { status in
                                    Text(status.label).tag(status)
                                }
                            }
                            .pickerStyle(.menu)
                            .onChange(of: statusSelection) { _, newValue in
                                saveUserData(notes: notesText, tags: tagsText, status: newValue)
                            }
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Study aids").font(.headline)
                        HStack {
                            Button {
                                model.generateFlashcards(for: paper.id)
                            } label: {
                                Label("Generate flashcards", systemImage: "square.and.pencil")
                            }
                            .buttonStyle(.bordered)

                            Button {
                                model.generateStudyQuestions(for: paper.id)
                            } label: {
                                Label("Check-understanding questions", systemImage: "questionmark.bubble")
                            }
                            .buttonStyle(.bordered)

                            if let flashcards = latestPaper()?.flashcards, !flashcards.isEmpty {
                                Button {
                                    quizIndex = 0
                                    revealAnswer = false
                                    showQuiz = true
                                } label: {
                                    Label("Quiz me", systemImage: "graduationcap")
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }

                        if let flashcards = latestPaper()?.flashcards, !flashcards.isEmpty {
                            Divider().padding(.vertical, 4)
                            Text("Flashcards")
                                .font(.subheadline.bold())
                            ForEach(flashcards) { card in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(card.question).font(.body.bold())
                                    Text(card.answer).font(.footnote)
                                }
                                .padding(8)
                                .background(Color.blue.opacity(0.06))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                        if let questions = latestPaper()?.userQuestions, !questions.isEmpty {
                            Divider().padding(.vertical, 4)
                            Text("Questions to check understanding")
                                .font(.subheadline.bold())
                            ForEach(Array(questions.enumerated()), id: \.0) { _, q in
                                Text("• \(q)")
                                    .font(.footnote)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                Spacer()
                Button {
                    if let onClose {
                        onClose()
                    } else {
                        dismiss()
                    }
                } label: {
                    Label("Close", systemImage: "xmark")
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
        }
        .navigationTitle(paper.title)
        .onAppear {
            notesText = paper.userNotes ?? ""
            tagsText = (paper.userTags ?? []).joined(separator: ", ")
            statusSelection = paper.readingStatus ?? .unread
            blueprintText = ""
        }
        .sheet(isPresented: $showQuiz) {
            if let cards = latestPaper()?.flashcards, !cards.isEmpty {
                FlashcardQuizView(
                    cards: cards,
                    index: $quizIndex,
                    reveal: $revealAnswer,
                    onClose: { showQuiz = false }
                )
            }
        }
        .onAppear {
            model.recordPaperOpened(paper.id)
        }
    }

    private var header: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text(paper.title).font(.title2.bold())
                Text(paper.originalFilename)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                if let clusterID = latestPaper()?.clusterIndex,
                   let cluster = model.clusters.first(where: { $0.id == clusterID }) {
                    Label(cluster.name, systemImage: "circle.grid.3x3")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                if let year = paper.year {
                    Label("Year: \(year)", systemImage: "calendar")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let pages = paper.pageCount {
                    Label("\(pages) pages", systemImage: "doc.on.doc")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let ingested = paper.ingestedAt {
                    Label("Ingested \(relativeDate(ingested))", systemImage: "tray.and.arrow.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if let firstRead = paper.firstReadAt {
                    Label("First read \(relativeDate(firstRead))", systemImage: "book")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Button {
                    openPDF(at: paper.fileURL)
                } label: {
                    Label("Open PDF", systemImage: "doc.richtext")
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private func sectionCard(title: String, text: String) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(.headline)
                Text(text).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func openPDF(at url: URL) {
#if os(macOS)
        NSWorkspace.shared.open(url)
#else
        // On iPadOS a Link can be used.
#endif
    }

    private func saveUserData(notes: String, tags: String, status: ReadingStatus? = nil) {
        let cleanedTags = tags.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        model.updatePaperUserData(id: paper.id, notes: notes, tags: cleanedTags, status: status ?? statusSelection)
    }

    private func latestPaper() -> Paper? {
        model.papers.first(where: { $0.id == paper.id })
    }

    private func generateELI() async {
        generatedELI = "Generating..."
        let levelText = explainLevel.prompt
        let fallbackInstructions = "You simplify research papers for different audiences."
        let instructions = PromptStore.loadText("ui.eli.instructions.md", fallback: fallbackInstructions)

        let fallbackTemplate = """
        Explain this paper to a {{level}} reader using the summary below.
        Keep it concise (5-7 sentences).

        Summary:
        {{summary}}
        """
        let template = PromptStore.loadText("ui.eli.prompt.md", fallback: fallbackTemplate)
        let prompt = PromptStore.render(template: template, variables: [
            "level": levelText,
            "summary": paper.summary
        ])
        do {
            let session = LanguageModelSession(instructions: instructions)
            let response = try await session.respond(to: prompt)
            generatedELI = response.content
        } catch {
            generatedELI = "Failed: \(error.localizedDescription)"
        }
    }

    private func relativeDate(_ date: Date) -> String {
        let fmt = RelativeDateTimeFormatter()
        fmt.unitsStyle = .short
        return fmt.localizedString(for: date, relativeTo: Date())
    }
}

enum ExplainLevel: CaseIterable {
    case undergrad, senior, expert

    var label: String {
        switch self {
        case .undergrad: return "Undergrad"
        case .senior: return "Senior"
        case .expert: return "Expert"
        }
    }

    var prompt: String {
        switch self {
        case .undergrad: return "third-year undergraduate"
        case .senior: return "senior engineer"
        case .expert: return "domain expert"
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct FlashcardQuizView: View {
    let cards: [Flashcard]
    @Binding var index: Int
    @Binding var reveal: Bool
    let onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Flashcard quiz").font(.headline)
                Spacer()
                Button("Close") { onClose() }
            }
            Divider()
            if cards.isEmpty {
                Text("No flashcards yet.")
            } else {
                let card = cards[min(index, cards.count - 1)]
                Text("Q: \(card.question)")
                    .font(.title3.bold())
                    .frame(maxWidth: .infinity, alignment: .leading)
                if reveal {
                    Text("A: \(card.answer)")
                        .font(.body)
                        .foregroundStyle(.secondary)
                } else {
                    Button("Show answer") { reveal = true }
                        .buttonStyle(.borderedProminent)
                }
                Spacer()
                HStack {
                    Text("Card \(index + 1) of \(cards.count)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Previous") {
                        index = max(0, index - 1)
                        reveal = false
                    }.disabled(index == 0)
                    Button("Next") {
                        index = min(cards.count - 1, index + 1)
                        reveal = false
                    }
                    .disabled(index >= cards.count - 1)
                }
            }
        }
        .padding()
        .frame(minWidth: 360, minHeight: 320)
    }
}

// MARK: - Metric helper views

private struct MetricPill: View {
    let label: String
    let value: Double
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "%.2f", value))
                .font(.caption.bold())
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(tint.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}

private struct RoleBadge: View {
    let label: String
    let value: Double
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(Color.white.opacity(0.2)).frame(width: 10, height: 10)
            Text(label)
                .font(.caption2)
            Text(String(format: "%.2f", value))
                .font(.caption2.bold())
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
