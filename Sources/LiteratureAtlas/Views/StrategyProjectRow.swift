import SwiftUI
import Charts

@available(macOS 26, iOS 26, *)
struct StrategyProjectRow: View {
    @EnvironmentObject private var model: AppModel

    let project: StrategyProject
    var onOpen: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            stageStrip
            if let metrics = latestMetrics {
                metricsRow(metrics)
            }
            if let series = latestPnLPoints, series.count >= 3 {
                Sparkline(points: series)
                    .frame(height: 54)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .onTapGesture { onOpen?() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(project.title.isEmpty ? "Untitled strategy" : project.title)
                    .font(.headline)
                    .lineLimit(2)
                Spacer()
                Text(relativeDate(project.updatedAt))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let idea = project.idea?.text, !idea.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(idea)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            HStack(spacing: 8) {
                if !project.paperIDs.isEmpty { MetricChip(label: "Papers", value: project.paperIDs.count) }
                MetricChip(label: "Features", value: project.features.count)
                MetricChip(label: "Decisions", value: project.decisions.count)
                MetricChip(label: "Outcomes", value: project.outcomes.count)
                Spacer()

                if project.archived ?? false {
                    Text("Archived")
                        .font(.caption2.bold())
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.gray.opacity(0.22), in: Capsule())
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption2)
        }
    }

    private var stageStrip: some View {
        let stages = stageStatus
        return HStack(spacing: 10) {
            StageDot(label: "Idea", done: stages.idea)
            StageDot(label: "Features", done: stages.features)
            StageDot(label: "Model", done: stages.model)
            StageDot(label: "Trade", done: stages.tradePlan)
            StageDot(label: "Outcome", done: stages.outcomes)
            Spacer()

            let pct = Int(Double(stages.completedCount) / 5.0 * 100.0)
            Text("\(pct)%")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func metricsRow(_ m: QuantBacktestMetrics) -> some View {
        HStack(spacing: 10) {
            MetricPill(label: "Sharpe", value: m.sharpe, tint: .mint)
            MetricPill(label: "CAGR", value: m.cagr, tint: .teal)
            MetricPill(label: "MaxDD", value: m.maxDrawdown, tint: .red)
            MetricPill(label: "Turn", value: m.turnover, tint: .orange)
            MetricPill(label: "Hit", value: m.hitRate, tint: .indigo)
            Spacer()
        }
    }

    private var stageStatus: (idea: Bool, features: Bool, model: Bool, tradePlan: Bool, outcomes: Bool, completedCount: Int) {
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

    private var latestOutcome: QuantOutcome? {
        project.outcomes.max(by: { $0.measuredAt < $1.measuredAt })
    }

    private var latestMetrics: QuantBacktestMetrics? {
        latestOutcome?.metrics
    }

    private var latestPnLPoints: [QuantTimeSeriesPoint]? {
        latestOutcome?.pnlSeries?.points
    }

    private func relativeDate(_ date: Date) -> String {
        let fmt = RelativeDateTimeFormatter()
        fmt.unitsStyle = .short
        return fmt.localizedString(for: date, relativeTo: Date())
    }
}

@available(macOS 26, iOS 26, *)
private struct MetricChip: View {
    let label: String
    let value: Int

    var body: some View {
        Text("\(label): \(value)")
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

@available(macOS 26, iOS 26, *)
private struct StageDot: View {
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
private struct MetricPill: View {
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
private struct Sparkline: View {
    let points: [QuantTimeSeriesPoint]

    var body: some View {
        Chart {
            ForEach(points.indices, id: \.self) { i in
                let p = points[i]
                LineMark(
                    x: .value("t", p.t),
                    y: .value("v", p.v)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(.mint.opacity(0.9))
                .lineStyle(StrokeStyle(lineWidth: 2))
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartLegend(.hidden)
        .background(Color.black.opacity(0.14), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        )
        .padding(.top, 2)
    }
}

