import Foundation
import SwiftUI

@available(macOS 26, iOS 26, *)
struct ReadingPlannerCard: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var nav: AppNavigation

    @Binding var selectedPaper: Paper?

    private struct TradingLensFailure: Identifiable {
        let id: UUID
        let paper: Paper
        let message: String
    }

    private struct TradingPriorityRow: Identifiable {
        let id: UUID
        let paper: Paper
        let subtitle: String
        let pill: String
    }

    var body: some View {
        let nextUp = Array(model.recommendedNextPapers().prefix(5))
        let tradingPriority = model.tradingPriorityPapers(limit: 5)
        let missingTradingLensCount = model.papers.filter { $0.tradingLens == nil }.count
        let blindSpots = model.blindSpots(limit: 3)
        let curriculum = Array(model.adaptiveCurriculum().prefix(5))
        let failures = tradingLensFailures()

        return GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                header

                tradingLensSection(
                    missingTradingLensCount: missingTradingLensCount,
                    failures: failures
                )

                nextUpSection(nextUp)

                tradingPrioritySection(tradingPriority)

                blindSpotsSection(blindSpots)

                curriculumSection(curriculum)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Reading planner")
                .font(.headline)
            Text("Adaptive picks based on your done/important papers and question history.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                Button { nav.selectedTab = .map } label: {
                    Label("Open Map", systemImage: "circle.grid.3x3")
                }
                .buttonStyle(.bordered)

                Button { nav.selectedTab = .trading } label: {
                    Label("Open Trading", systemImage: "dollarsign.circle")
                }
                .buttonStyle(.bordered)

                Button { nav.selectedTab = .qa } label: {
                    Label("Ask Q&A", systemImage: "questionmark.circle")
                }
                .buttonStyle(.bordered)
            }
            .font(.caption)
        }
    }

    @ViewBuilder
    private func tradingLensSection(missingTradingLensCount: Int, failures: [TradingLensFailure]) -> some View {
        if model.tradingLensBackfillInFlight || missingTradingLensCount > 0 || !failures.isEmpty {
            Divider().padding(.vertical, 4)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Trading lens scorecards")
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
                } else if missingTradingLensCount > 0 {
                    Text("\(missingTradingLensCount) papers missing trading lens. Generate to enable trading priority + analytics filters.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if !failures.isEmpty {
                    DisclosureGroup("Failures (\(failures.count))") {
                        VStack(spacing: 8) {
                            ForEach(failures) { entry in
                                PaperActionRow(
                                    paper: entry.paper,
                                    subtitle: entry.message,
                                    trailingPill: "Retry",
                                    trailingPillTint: .red,
                                    onOpen: { selectedPaper = entry.paper }
                                )
                            }
                            Button("Clear failures") { model.tradingLensFailures = [:] }
                                .buttonStyle(.bordered)
                        }
                        .padding(.top, 6)
                    }
                    .font(.caption)
                }
            }
        }
    }

    @ViewBuilder
    private func nextUpSection(_ papers: [Paper]) -> some View {
        if papers.isEmpty {
            Text("Mark some papers as done to unlock recommendations.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Text("Next up")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(papers) { paper in
                PaperActionRow(
                    paper: paper,
                    subtitle: paper.summary,
                    trailingPill: paper.readingStatus?.label,
                    trailingPillTint: .blue,
                    onOpen: { selectedPaper = paper }
                )
            }
        }
    }

    @ViewBuilder
    private func tradingPrioritySection(_ papers: [Paper]) -> some View {
        if !papers.isEmpty {
            let rows: [TradingPriorityRow] = papers.map { paper in
                let verdict = paper.tradingLens?.oneLineVerdict?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let subtitle = verdict.isEmpty ? paper.summary : verdict

                let scores = paper.tradingScores ?? paper.tradingLens?.scores
                let priority = (scores?.strategyImpact ?? 0) * (scores?.usability ?? 0) * (scores?.confidence ?? 0)
                let pill = String(format: "P %.1f", priority)

                return TradingPriorityRow(id: paper.id, paper: paper, subtitle: subtitle, pill: pill)
            }

            Divider().padding(.vertical, 4)
            Text("Trading priority (impact × usability × confidence)")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(rows) { row in
                PaperActionRow(
                    paper: row.paper,
                    subtitle: row.subtitle,
                    trailingPill: row.pill,
                    trailingPillTint: .mint,
                    onOpen: { selectedPaper = row.paper }
                )
            }
        }
    }

    @ViewBuilder
    private func blindSpotsSection(_ papers: [Paper]) -> some View {
        if !papers.isEmpty {
            Divider().padding(.vertical, 4)
            Text("Blind spots (central but far from you)")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(papers) { paper in
                PaperActionRow(
                    paper: paper,
                    subtitle: paper.summary,
                    trailingPill: paper.year.map(String.init),
                    trailingPillTint: .cyan,
                    onOpen: { selectedPaper = paper }
                )
            }
        }
    }

    @ViewBuilder
    private func curriculumSection(_ steps: [CurriculumStep]) -> some View {
        if !steps.isEmpty {
            Divider().padding(.vertical, 4)
            Text("Adaptive curriculum")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(steps, id: \.id) { step in
                PaperActionRow(
                    paper: step.paper,
                    title: step.paper.title,
                    subtitle: nil,
                    leadingBadge: step.stage.label,
                    trailingPill: step.paper.year.map(String.init),
                    trailingPillTint: .orange,
                    onOpen: { selectedPaper = step.paper }
                )
            }
        }
    }

    private func tradingLensFailures() -> [TradingLensFailure] {
        model.tradingLensFailures
            .compactMap { (id, message) in
                guard let paper = model.papers.first(where: { $0.id == id }) else { return nil }
                return TradingLensFailure(id: id, paper: paper, message: message)
            }
            .sorted { $0.paper.title.localizedCaseInsensitiveCompare($1.paper.title) == .orderedAscending }
    }
}
