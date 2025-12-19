import SwiftUI

@available(macOS 26, iOS 26, *)
struct TradingScoreStrip: View {
    let scores: TradingLensScores?
    let priority: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                ScoreBar(label: "N", value: scores?.novelty, maxValue: 10, tint: .pink)
                ScoreBar(label: "U", value: scores?.usability, maxValue: 10, tint: .mint)
                ScoreBar(label: "I", value: scores?.strategyImpact, maxValue: 10, tint: .teal)
                ScoreBar(label: "C", value: scores?.confidence, maxValue: 1, tint: .indigo)
                ScoreBar(label: "P", value: priority, maxValue: 100, tint: .orange)
            }
        }
    }
}

@available(macOS 26, iOS 26, *)
private struct ScoreBar: View {
    let label: String
    let value: Double?
    let maxValue: Double
    let tint: Color

    private var ratio: Double {
        guard let value, maxValue > 0 else { return 0 }
        return Swift.min(Swift.max(value / maxValue, 0), 1)
    }

    private var display: String {
        guard let value else { return "—" }
        if maxValue <= 1.01 { return String(format: "%.2f", value) }
        return String(format: "%.1f", value)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Text(label)
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)
                Text(display)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.primary.opacity(0.9))
            }

            GeometryReader { geo in
                let width = geo.size.width
                let filled = Swift.max(0, width * ratio)
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.white.opacity(0.07))
                    Capsule()
                        .fill(tint.opacity(0.95))
                        .frame(width: filled)
                }
            }
            .frame(height: 6)
        }
        .frame(width: 74)
    }
}
