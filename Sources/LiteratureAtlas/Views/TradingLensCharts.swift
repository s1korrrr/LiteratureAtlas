import SwiftUI
import Charts
import Foundation

@available(macOS 26, iOS 26, *)
struct TradingLensPoint: Identifiable, Equatable {
    let id: UUID
    let novelty: Double
    let usability: Double
    let strategyImpact: Double
    let confidence: Double
    let priority: Double
    let title: String
    let primaryTag: String?
    let primaryAssetClass: String?
    let primaryHorizon: String?
}

@available(macOS 26, iOS 26, *)
struct TradingLensScatterChartView: View {
    let points: [TradingLensPoint]
    @Binding var selectedPointID: UUID?

    @State private var hoveredPointID: UUID?

    private var hoveredPoint: TradingLensPoint? {
        guard let hoveredPointID else { return nil }
        return points.first(where: { $0.id == hoveredPointID })
    }

    private var selectedPoint: TradingLensPoint? {
        guard let selectedPointID else { return nil }
        return points.first(where: { $0.id == selectedPointID })
    }

    private func color(for impact: Double) -> Color {
        let x = min(max(impact / 10.0, 0.0), 1.0)
        return Color(hue: 0.58 - 0.48 * x, saturation: 0.86, brightness: 0.95)
    }

    private func opacity(for confidence: Double) -> Double {
        let x = min(max(confidence, 0.0), 1.0)
        return 0.22 + 0.78 * x
    }

    var body: some View {
        let fullXDomain: ClosedRange<Double> = {
            let xs = points.map(\.usability)
            guard let minX = xs.min(), let maxX = xs.max(), minX < maxX else { return 0...10 }
            let pad = max(0.35, (maxX - minX) * 0.12)
            return max(-0.5, minX - pad)...min(10.5, maxX + pad)
        }()
        let fullYDomain: ClosedRange<Double> = {
            let ys = points.map(\.novelty)
            guard let minY = ys.min(), let maxY = ys.max(), minY < maxY else { return 0...10 }
            let pad = max(0.35, (maxY - minY) * 0.12)
            return max(-0.5, minY - pad)...min(10.5, maxY + pad)
        }()

        Chart {
            ForEach(points) { pt in
                let isSelected = selectedPointID == pt.id
                let isHovered = hoveredPointID == pt.id
                let scaledPriority = min(max(pt.priority / 100.0, 0.0), 1.0)
                let baseSize = 34.0 + scaledPriority * 110.0

                PointMark(
                    x: .value("Usability", pt.usability),
                    y: .value("Novelty", pt.novelty)
                )
                .foregroundStyle(isSelected ? Color.white : color(for: pt.strategyImpact))
                .symbolSize(isSelected ? 240 : (isHovered ? baseSize * 1.45 : baseSize))
                .opacity(isSelected || isHovered ? 1.0 : opacity(for: pt.confidence))
            }

            if let focus = selectedPoint ?? hoveredPoint {
                RuleMark(x: .value("Usability", focus.usability))
                    .foregroundStyle(Color.white.opacity(0.14))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                RuleMark(y: .value("Novelty", focus.novelty))
                    .foregroundStyle(Color.white.opacity(0.14))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                PointMark(
                    x: .value("Usability", focus.usability),
                    y: .value("Novelty", focus.novelty)
                )
                .foregroundStyle(.white)
                .symbolSize(260)
                .annotation(position: .top, alignment: .leading) {
                    Text("\(focus.title)\nN \(String(format: "%.1f", focus.novelty)) · U \(String(format: "%.1f", focus.usability)) · I \(String(format: "%.1f", focus.strategyImpact))")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .lineLimit(3)
                        .frame(maxWidth: 320, alignment: .leading)
                }
            }
        }
        .chartXScale(domain: fullXDomain)
        .chartYScale(domain: fullYDomain)
        .chartXAxisLabel("Usability (0–10)")
        .chartYAxisLabel("Novelty (0–10)")
        .chartOverlay { (proxy: ChartProxy) in
            GeometryReader { geo in
                if let plotAnchor = proxy.plotFrame {
                    let plotFrame = geo[plotAnchor]
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        .gesture(
                            SpatialTapGesture()
                                .onEnded { value in
                                    guard plotFrame.contains(value.location) else { return }
                                    selectedPointID = nearestPoint(at: value.location, plotFrame: plotFrame, proxy: proxy)?.id
                                }
                        )
                        #if os(macOS)
                        .overlay(
                            PointerTrackingView(
                                onMove: { loc in
                                    let absLoc = CGPoint(x: plotFrame.origin.x + loc.x, y: plotFrame.origin.y + loc.y)
                                    hoveredPointID = nearestPoint(at: absLoc, plotFrame: plotFrame, proxy: proxy)?.id
                                },
                                onExit: {
                                    hoveredPointID = nil
                                }
                            )
                            .frame(width: plotFrame.width, height: plotFrame.height)
                            .position(x: plotFrame.midX, y: plotFrame.midY)
                        )
                        #endif
                }
            }
        }
    }

    private func nearestPoint(at location: CGPoint, plotFrame: CGRect, proxy: ChartProxy) -> TradingLensPoint? {
        var best: (TradingLensPoint, CGFloat)?
        for pt in points {
            guard let x = proxy.position(forX: pt.usability),
                  let y = proxy.position(forY: pt.novelty) else { continue }
            let point = CGPoint(x: plotFrame.origin.x + x, y: plotFrame.origin.y + y)
            let dist = hypot(point.x - location.x, point.y - location.y)
            if let existing = best {
                if dist < existing.1 { best = (pt, dist) }
            } else {
                best = (pt, dist)
            }
        }
        guard let best else { return nil }
        return best.1 <= 30 ? best.0 : nil
    }
}

@available(macOS 26, iOS 26, *)
struct TradingTagCount: Identifiable, Equatable {
    let tag: String
    let count: Int

    var id: String { tag }
}

@available(macOS 26, iOS 26, *)
struct TradingTagTrendPoint: Identifiable, Equatable {
    let tag: String
    let year: Int
    let count: Int

    var id: String { "\(tag)#\(year)" }
}

@available(macOS 26, iOS 26, *)
struct TradingTagBarChartView: View {
    let counts: [TradingTagCount]

    var body: some View {
        Chart {
            ForEach(counts) { entry in
                BarMark(
                    x: .value("Papers", entry.count),
                    y: .value("Tag", entry.tag)
                )
                .foregroundStyle(.teal.opacity(0.75))
            }
        }
        .chartXAxisLabel("Papers")
        .chartYAxisLabel("Tag")
    }
}

@available(macOS 26, iOS 26, *)
struct TradingTagTrendChartView: View {
    let trends: [TradingTagTrendPoint]
    let domain: ClosedRange<Int>

    var body: some View {
        let ordered = trends.sorted { lhs, rhs in
            if lhs.tag == rhs.tag { return lhs.year < rhs.year }
            return lhs.tag < rhs.tag
        }
        Chart {
            ForEach(ordered) { entry in
                LineMark(
                    x: .value("Year", entry.year),
                    y: .value("Count", entry.count)
                )
                .foregroundStyle(by: .value("Tag", entry.tag))
                .interpolationMethod(.catmullRom)
            }
        }
        .chartXScale(domain: domain)
        .chartXAxisLabel("Year")
        .chartYAxisLabel("Papers")
        .chartLegend(position: .bottom)
    }
}

