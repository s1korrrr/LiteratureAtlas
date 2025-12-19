import SwiftUI
import Charts
import Foundation

@available(macOS 26, iOS 26, *)
struct StrategyProjectDetailView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss

    let strategyID: UUID

    @State private var draft: StrategyProject?
    @State private var showPaperPicker: Bool = false
    @State private var showDeleteConfirm: Bool = false
    @State private var selectedPaperDetail: Paper?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if draft != nil {
                        let project = Binding<StrategyProject>(
                            get: { draft! },
                            set: { draft = $0 }
                        )
                        headerCard(project)
                        dashboardCard(project)
                        linkedPapersCard(project)
                        ideaCard(project)
                        featuresCard(project)
                        modelCard(project)
                        tradeCard(project)
                        decisionsCard(project)
                        outcomesCard(project)
                        feedbackCard(project)
                    } else {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Strategy not found").font(.headline)
                                Text("It may have been deleted or not loaded yet.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .padding(16)
            }
            .navigationTitle("Strategy")
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        model.exportQuantKnowledgeGraphSnapshot()
                    } label: {
                        Label("Export KG", systemImage: "square.and.arrow.up")
                    }
                    Button {
                        save()
                    } label: {
                        Label("Save", systemImage: "tray.and.arrow.down")
                    }
                    .buttonStyle(.borderedProminent)

                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
            .confirmationDialog("Delete this strategy project?", isPresented: $showDeleteConfirm) {
                Button("Delete", role: .destructive) {
                    model.deleteStrategyProject(strategyID)
                    dismiss()
                }
            }
            .sheet(isPresented: $showPaperPicker) {
                PaperPickerSheet(
                    title: "Link paper",
                    papers: model.papers.sorted(by: { ($0.year ?? 0, $0.title) > ($1.year ?? 0, $1.title) }),
                    onPick: { pickedID in
                        guard var d = draft else { return }
                        if !d.paperIDs.contains(pickedID) { d.paperIDs.append(pickedID) }
                        draft = d
                    }
                )
                .presentationDetents([.medium, .large])
            }
            .sheet(item: $selectedPaperDetail) { paper in
                PaperDetailView(paper: paper)
                    .environmentObject(model)
            }
            .task(id: strategyID) {
                draft = model.strategyProjects.first(where: { $0.id == strategyID })
            }
        }
    }

    private func save() {
        guard let draft else { return }
        model.updateStrategyProject(draft)
        self.draft = model.strategyProjects.first(where: { $0.id == strategyID }) ?? draft
    }

    private func headerCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                TextField("Title", text: project.title)
                    .textFieldStyle(.roundedBorder)

                HStack(spacing: 10) {
                    Toggle("Archived", isOn: boolOptional(project.archived))
                        .toggleStyle(.switch)

                    Spacer()

                    Button {
                        PlatformClipboard.copy(project.id.wrappedValue.uuidString.uppercased())
                    } label: {
                        Label("Copy ID", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)

                    if let noteURL = model.obsidianStrategyNoteURL(for: project.id.wrappedValue) {
                        Button {
                            PlatformOpen.open(url: noteURL)
                        } label: {
                            Label("Open note", systemImage: "note.text")
                        }
                        .buttonStyle(.borderedProminent)
#if os(macOS)
                        Button {
                            PlatformOpen.revealInFinder(url: noteURL)
                        } label: {
                            Label("Reveal", systemImage: "folder")
                        }
                        .buttonStyle(.bordered)
#endif
                    }
                }

                TextField("Tags (comma separated)", text: tagsText(project.tags))
                    .textFieldStyle(.roundedBorder)

                let stages = stageStatus(for: project.wrappedValue)
                HStack(spacing: 10) {
                    StrategyStageDot(label: "Idea", done: stages.idea)
                    StrategyStageDot(label: "Features", done: stages.features)
                    StrategyStageDot(label: "Model", done: stages.model)
                    StrategyStageDot(label: "Trade", done: stages.tradePlan)
                    StrategyStageDot(label: "Outcome", done: stages.outcomes)
                    Spacer()
                    let pct = Int(Double(stages.completedCount) / 5.0 * 100.0)
                    Text("\(pct)%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                HStack(spacing: 12) {
                    Text("Created \(relativeDate(project.createdAt.wrappedValue))")
                    Text("Updated \(relativeDate(project.updatedAt.wrappedValue))")
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
    }

    private func dashboardCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Dashboard").font(.headline)
                    Spacer()
                    if project.outcomes.wrappedValue.isEmpty {
                        Button {
                            project.outcomes.wrappedValue.append(QuantOutcome(kind: .backtest, metrics: QuantBacktestMetrics(), notes: nil))
                        } label: {
                            Label("Add outcome", systemImage: "plus")
                        }
                        .buttonStyle(.bordered)
                    }
                }

                let stages = stageStatus(for: project.wrappedValue)
                let missing: [String] = [
                    stages.idea ? nil : "Write idea",
                    stages.features ? nil : "Add features",
                    stages.model ? nil : "Define model",
                    stages.tradePlan ? nil : "Draft trade plan",
                    stages.outcomes ? nil : "Record outcome"
                ].compactMap { $0 }
                if !missing.isEmpty {
                    Text("Next: \(missing.prefix(3).joined(separator: " · "))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let latest = project.outcomes.wrappedValue.max(by: { $0.measuredAt < $1.measuredAt }) {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Latest \(latest.kind.label)")
                            .font(.subheadline.weight(.semibold))
                        Spacer()
                        Text(relativeDate(latest.measuredAt))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    if let m = latest.metrics {
                        StrategyMetricRow(metrics: m)
                    }

                    if let series = latest.pnlSeries?.points, series.count >= 3 {
                        StrategyPnLChart(points: series)
                            .frame(height: 220)
                    }

                    if let artifacts = latest.artifactPaths, !artifacts.isEmpty {
                        Divider().padding(.vertical, 2)
                        Text("Artifacts").font(.caption.bold()).foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(artifacts.prefix(6), id: \.self) { path in
                                Text(path)
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            if artifacts.count > 6 {
                                Text("…and \(artifacts.count - 6) more")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } else {
                    Text("No outcomes recorded yet. Add one to start tracking Sharpe, drawdowns, and PnL curves.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func linkedPapersCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Research").font(.headline)
                    Spacer()
                    Button {
                        showPaperPicker = true
                    } label: {
                        Label("Link paper", systemImage: "link.badge.plus")
                    }
                    .buttonStyle(.bordered)
                }

                if project.paperIDs.wrappedValue.isEmpty {
                    Text("No linked papers yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 10) {
                        ForEach(project.paperIDs.wrappedValue, id: \.self) { pid in
                            if let paper = model.papers.first(where: { $0.id == pid }) {
                                HStack(spacing: 10) {
                                    PaperActionRow(
                                        paper: paper,
                                        subtitle: paper.tradingLens?.oneLineVerdict?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                                            ? paper.tradingLens?.oneLineVerdict
                                            : paper.summary,
                                        trailingPill: paper.year.map(String.init),
                                        trailingPillTint: .mint,
                                        onOpen: { selectedPaperDetail = paper }
                                    )
                                    Button(role: .destructive) {
                                        project.paperIDs.wrappedValue.removeAll(where: { $0 == pid })
                                    } label: {
                                        Image(systemName: "link.badge.minus")
                                    }
                                    .buttonStyle(.borderless)
                                    .help("Unlink paper")
                                }
                            } else {
                                HStack {
                                    Text(pid.uuidString)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Button(role: .destructive) {
                                        project.paperIDs.wrappedValue.removeAll(where: { $0 == pid })
                                    } label: {
                                        Image(systemName: "link.badge.minus")
                                    }
                                    .buttonStyle(.borderless)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func ideaCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Idea").font(.headline)
                TextEditor(text: Binding(
                    get: { project.idea.wrappedValue?.text ?? "" },
                    set: { newValue in
                        var idea = project.idea.wrappedValue ?? StrategyIdea(text: "", hypotheses: nil, assumptions: nil)
                        idea.text = newValue
                        project.idea.wrappedValue = idea
                    }
                ))
                .frame(minHeight: 90)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2)))
            }
        }
    }

    private func featuresCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Features").font(.headline)
                    Spacer()
                    Button {
                        project.features.wrappedValue.append(QuantFeature(name: ""))
                    } label: {
                        Label("Add", systemImage: "plus")
                    }
                    .buttonStyle(.bordered)
                }

                if project.features.wrappedValue.isEmpty {
                    Text("No features yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(project.features.wrappedValue.indices, id: \.self) { idx in
                        VStack(alignment: .leading, spacing: 8) {
                            TextField("Feature name", text: project.features[idx].name)
                                .textFieldStyle(.roundedBorder)

                            TextField("Description (optional)", text: optionalText(project.features[idx].description))
                                .textFieldStyle(.roundedBorder)

                            HStack {
                                Spacer()
                                Button(role: .destructive) {
                                    project.features.wrappedValue.remove(at: idx)
                                } label: {
                                    Label("Remove", systemImage: "trash")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        if idx != project.features.wrappedValue.indices.last { Divider().padding(.vertical, 6) }
                    }
                }
            }
        }
    }

    private func modelCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Model").font(.headline)
                    Spacer()
                    if project.model.wrappedValue == nil {
                        Button {
                            project.model.wrappedValue = QuantModel(name: "Model")
                        } label: {
                            Label("Add model", systemImage: "plus")
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button(role: .destructive) {
                            project.model.wrappedValue = nil
                        } label: {
                            Label("Remove", systemImage: "trash")
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if project.model.wrappedValue != nil {
                    let model = unwrap(project.model, fallback: QuantModel(name: "Model"))
                    TextField("Model name", text: model.name)
                        .textFieldStyle(.roundedBorder)

                    TextField("Description (optional)", text: optionalText(model.description))
                        .textFieldStyle(.roundedBorder)

                    Text("Tip: Link features by selecting IDs later (UI intentionally minimal for now).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    Text("No model yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func tradeCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Trade").font(.headline)
                    Spacer()
                    if project.tradePlan.wrappedValue == nil {
                        Button {
                            project.tradePlan.wrappedValue = QuantTradePlan()
                        } label: {
                            Label("Add trade plan", systemImage: "plus")
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button(role: .destructive) {
                            project.tradePlan.wrappedValue = nil
                        } label: {
                            Label("Remove", systemImage: "trash")
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if project.tradePlan.wrappedValue != nil {
                    let trade = unwrap(project.tradePlan, fallback: QuantTradePlan())
                    TextField("Universe (optional)", text: optionalText(trade.universe))
                        .textFieldStyle(.roundedBorder)

                    TextField("Horizon (optional)", text: optionalText(trade.horizon))
                        .textFieldStyle(.roundedBorder)

                    labeledEditor(
                        "Signal definition",
                        text: optionalText(trade.signalDefinition)
                    )
                    labeledEditor(
                        "Portfolio construction",
                        text: optionalText(trade.portfolioConstruction)
                    )
                    labeledEditor(
                        "Costs & slippage",
                        text: optionalText(trade.costsAndSlippage)
                    )
                    labeledEditor(
                        "Constraints",
                        text: optionalText(trade.constraints)
                    )
                    labeledEditor(
                        "Execution notes",
                        text: optionalText(trade.executionNotes)
                    )
                } else {
                    Text("No trade plan yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func decisionsCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Decisions").font(.headline)
                    Spacer()
                    Button {
                        project.decisions.wrappedValue.append(QuantDecision(kind: .build, rationale: ""))
                    } label: {
                        Label("Add", systemImage: "plus")
                    }
                    .buttonStyle(.bordered)
                }

                if project.decisions.wrappedValue.isEmpty {
                    Text("No decisions yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(project.decisions.wrappedValue.indices, id: \.self) { idx in
                        VStack(alignment: .leading, spacing: 8) {
                            Picker("Kind", selection: project.decisions[idx].kind) {
                                ForEach(QuantDecisionKind.allCases, id: \.self) { kind in
                                    Text(kind.label).tag(kind)
                                }
                            }
                            .pickerStyle(.menu)

                            labeledEditor(
                                "Rationale",
                                text: project.decisions[idx].rationale,
                                minHeight: 70
                            )

                            HStack {
                                Spacer()
                                Button(role: .destructive) {
                                    project.decisions.wrappedValue.remove(at: idx)
                                } label: {
                                    Label("Remove", systemImage: "trash")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        if idx != project.decisions.wrappedValue.indices.last { Divider().padding(.vertical, 6) }
                    }
                }
            }
        }
    }

    private func outcomesCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("PnL / Outcomes").font(.headline)
                    Spacer()
                    Button {
                        project.outcomes.wrappedValue.append(QuantOutcome(kind: .backtest, metrics: QuantBacktestMetrics(), notes: nil))
                    } label: {
                        Label("Add", systemImage: "plus")
                    }
                    .buttonStyle(.bordered)
                }

                if project.outcomes.wrappedValue.isEmpty {
                    Text("No outcomes yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(project.outcomes.wrappedValue.indices, id: \.self) { idx in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Measured \(relativeDate(project.outcomes[idx].measuredAt.wrappedValue))")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }

                            Picker("Kind", selection: project.outcomes[idx].kind) {
                                ForEach(QuantOutcomeKind.allCases, id: \.self) { kind in
                                    Text(kind.label).tag(kind)
                                }
                            }
                            .pickerStyle(.menu)

                            HStack(spacing: 10) {
                                TextField("Sharpe", text: metricBinding(project.outcomes[idx].metrics, keyPath: \.sharpe))
                                    .textFieldStyle(.roundedBorder)
                                TextField("PnL", text: metricBinding(project.outcomes[idx].metrics, keyPath: \.pnl))
                                    .textFieldStyle(.roundedBorder)
                                TextField("Max DD", text: metricBinding(project.outcomes[idx].metrics, keyPath: \.maxDrawdown))
                                    .textFieldStyle(.roundedBorder)
                            }

                            HStack(spacing: 10) {
                                TextField("CAGR", text: metricBinding(project.outcomes[idx].metrics, keyPath: \.cagr))
                                    .textFieldStyle(.roundedBorder)
                                TextField("Turnover", text: metricBinding(project.outcomes[idx].metrics, keyPath: \.turnover))
                                    .textFieldStyle(.roundedBorder)
                                TextField("Hit rate", text: metricBinding(project.outcomes[idx].metrics, keyPath: \.hitRate))
                                    .textFieldStyle(.roundedBorder)
                            }

                            if let m = project.outcomes[idx].metrics.wrappedValue {
                                StrategyMetricRow(metrics: m)
                            }

                            if let series = project.outcomes[idx].pnlSeries.wrappedValue?.points, series.count >= 3 {
                                StrategyPnLChart(points: series)
                                    .frame(height: 160)
                            }

                            if let artifacts = project.outcomes[idx].artifactPaths.wrappedValue, !artifacts.isEmpty {
                                Text("Artifacts").font(.caption.bold()).foregroundStyle(.secondary)
                                VStack(alignment: .leading, spacing: 6) {
                                    ForEach(artifacts.prefix(8), id: \.self) { path in
                                        Text(path)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                    }
                                }
                            }

                            labeledEditor(
                                "Notes",
                                text: optionalText(project.outcomes[idx].notes),
                                minHeight: 70
                            )

                            HStack {
                                Spacer()
                                Button(role: .destructive) {
                                    project.outcomes.wrappedValue.remove(at: idx)
                                } label: {
                                    Label("Remove", systemImage: "trash")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        if idx != project.outcomes.wrappedValue.indices.last { Divider().padding(.vertical, 6) }
                    }
                }
            }
        }
    }

    private func feedbackCard(_ project: Binding<StrategyProject>) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Feedback").font(.headline)
                    Spacer()
                    Button {
                        project.feedback.wrappedValue.append(QuantFeedback(text: ""))
                    } label: {
                        Label("Add", systemImage: "plus")
                    }
                    .buttonStyle(.bordered)
                }

                if project.feedback.wrappedValue.isEmpty {
                    Text("No feedback yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(project.feedback.wrappedValue.indices, id: \.self) { idx in
                        VStack(alignment: .leading, spacing: 8) {
                            labeledEditor(
                                "Note",
                                text: project.feedback[idx].text,
                                minHeight: 70
                            )
                            HStack {
                                Spacer()
                                Button(role: .destructive) {
                                    project.feedback.wrappedValue.remove(at: idx)
                                } label: {
                                    Label("Remove", systemImage: "trash")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        if idx != project.feedback.wrappedValue.indices.last { Divider().padding(.vertical, 6) }
                    }
                }
            }
        }
    }

    private func stageStatus(for project: StrategyProject) -> (idea: Bool, features: Bool, model: Bool, tradePlan: Bool, outcomes: Bool, completedCount: Int) {
        let idea = {
            let text = project.idea?.text.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let hyps = project.idea?.hypotheses ?? []
            return !text.isEmpty || !hyps.isEmpty
        }()
        let features = !project.features.isEmpty
        let model = project.model != nil
        let tradePlan = project.tradePlan != nil
        let outcomes = !project.outcomes.isEmpty
        let completed = [idea, features, model, tradePlan, outcomes].filter { $0 }.count
        return (idea, features, model, tradePlan, outcomes, completed)
    }

    private func relativeDate(_ date: Date) -> String {
        let fmt = RelativeDateTimeFormatter()
        fmt.unitsStyle = .short
        return fmt.localizedString(for: date, relativeTo: Date())
    }

    private func tagsText(_ binding: Binding<[String]?>) -> Binding<String> {
        Binding(
            get: {
                let tags = (binding.wrappedValue ?? [])
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                return tags.joined(separator: ", ")
            },
            set: { newValue in
                let parts = newValue
                    .split(whereSeparator: { $0 == "," || $0.isNewline })
                    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                if parts.isEmpty {
                    binding.wrappedValue = nil
                } else {
                    binding.wrappedValue = Array(NSOrderedSet(array: parts)).compactMap { $0 as? String }
                }
            }
        )
    }

    private func labeledEditor(_ title: String, text: Binding<String>, minHeight: CGFloat = 90) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            TextEditor(text: text)
                .frame(minHeight: minHeight)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2)))
        }
    }

    private func boolOptional(_ binding: Binding<Bool?>) -> Binding<Bool> {
        Binding(
            get: { binding.wrappedValue ?? false },
            set: { binding.wrappedValue = $0 }
        )
    }

    private func optionalText(_ binding: Binding<String?>) -> Binding<String> {
        Binding(
            get: { binding.wrappedValue ?? "" },
            set: { newValue in
                let trimmed = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
                binding.wrappedValue = trimmed.isEmpty ? nil : newValue
            }
        )
    }

    private func unwrap<T>(_ binding: Binding<T?>, fallback: T) -> Binding<T> {
        Binding(
            get: { binding.wrappedValue ?? fallback },
            set: { binding.wrappedValue = $0 }
        )
    }

    private func metricBinding(_ metrics: Binding<QuantBacktestMetrics?>, keyPath: WritableKeyPath<QuantBacktestMetrics, Double?>) -> Binding<String> {
        Binding(
            get: {
                guard let v = metrics.wrappedValue?[keyPath: keyPath] else { return "" }
                return String(v)
            },
            set: { newValue in
                var m = metrics.wrappedValue ?? QuantBacktestMetrics()
                let trimmed = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    m[keyPath: keyPath] = nil
                } else if let v = Double(trimmed) {
                    m[keyPath: keyPath] = v
                }
                metrics.wrappedValue = m
            }
        )
    }
}

@available(macOS 26, iOS 26, *)
private struct StrategyStageDot: View {
    let label: String
    let done: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(done ? Color.green.opacity(0.95) : Color.white.opacity(0.12))
                .frame(width: 8, height: 8)
                .overlay(Circle().stroke(Color.white.opacity(0.14), lineWidth: 1))
            Text(label)
                .font(.caption2)
                .foregroundStyle(done ? Color.primary.opacity(0.9) : Color.secondary)
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct StrategyMetricRow: View {
    let metrics: QuantBacktestMetrics

    var body: some View {
        HStack(spacing: 10) {
            StrategyMetricPill(label: "Sharpe", value: metrics.sharpe, tint: .mint)
            StrategyMetricPill(label: "CAGR", value: metrics.cagr, tint: .teal)
            StrategyMetricPill(label: "MaxDD", value: metrics.maxDrawdown, tint: .red)
            StrategyMetricPill(label: "Turn", value: metrics.turnover, tint: .orange)
            StrategyMetricPill(label: "Hit", value: metrics.hitRate, tint: .indigo)
            StrategyMetricPill(label: "PnL", value: metrics.pnl, tint: .gray)
            Spacer()
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct StrategyMetricPill: View {
    let label: String
    let value: Double?
    let tint: Color

    private var display: String {
        guard let value else { return "—" }
        if abs(value) >= 1000 { return String(format: "%.0f", value) }
        if abs(value) >= 100 { return String(format: "%.1f", value) }
        return String(format: "%.2f", value)
    }

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.bold())
                .foregroundStyle(.secondary)
            Text(display)
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.primary.opacity(0.9))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(tint.opacity(0.12), in: Capsule())
        .overlay(Capsule().stroke(Color.white.opacity(0.12), lineWidth: 1))
    }
}

@available(macOS 26, iOS 26, *)
private struct StrategyPnLChart: View {
    let points: [QuantTimeSeriesPoint]

    private var sorted: [QuantTimeSeriesPoint] {
        points.sorted { $0.t < $1.t }
    }

    private var tint: Color {
        guard let first = sorted.first, let last = sorted.last else { return .mint }
        return last.v >= first.v ? .mint : .red
    }

    var body: some View {
        Chart {
            ForEach(sorted.indices, id: \.self) { i in
                let p = sorted[i]
                LineMark(
                    x: .value("t", p.t),
                    y: .value("PnL", p.v)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(tint.opacity(0.92))
                .lineStyle(StrokeStyle(lineWidth: 2))
            }
            if let last = sorted.last {
                PointMark(x: .value("t", last.t), y: .value("PnL", last.v))
                    .foregroundStyle(tint)
                    .symbolSize(46)
            }
        }
        .chartLegend(.hidden)
        .background(Color.black.opacity(0.14), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        )
        .padding(.top, 2)
    }
}

@available(macOS 26, iOS 26, *)
private struct PaperPickerSheet: View {
    @Environment(\.dismiss) private var dismiss

    let title: String
    let papers: [Paper]
    let onPick: (UUID) -> Void

    @State private var query: String = ""

    private var filtered: [Paper] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !q.isEmpty else { return papers }
        return papers.filter { paper in
            if paper.title.lowercased().contains(q) { return true }
            if paper.summary.lowercased().contains(q) { return true }
            if (paper.keywords ?? []).joined(separator: " ").lowercased().contains(q) { return true }
            return false
        }
    }

    var body: some View {
        NavigationStack {
            List(filtered) { paper in
                Button {
                    onPick(paper.id)
                    dismiss()
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(paper.title).font(.headline)
                        if let year = paper.year {
                            Text("\(year) · \(paper.id.uuidString)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else {
                            Text(paper.id.uuidString)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle(title)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .searchable(text: $query, placement: .toolbar, prompt: "Search papers")
        }
    }
}
