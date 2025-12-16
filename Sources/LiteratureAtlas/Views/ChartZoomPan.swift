import Foundation

enum ChartZoomPan {
    static func pan(domain: ClosedRange<Double>, by delta: Double, within bounds: ClosedRange<Double>) -> ClosedRange<Double> {
        let len = domain.upperBound - domain.lowerBound
        guard len > 0 else { return bounds }
        var lower = domain.lowerBound + delta
        var upper = domain.upperBound + delta
        if lower < bounds.lowerBound {
            let shift = bounds.lowerBound - lower
            lower += shift
            upper += shift
        }
        if upper > bounds.upperBound {
            let shift = bounds.upperBound - upper
            lower += shift
            upper += shift
        }
        lower = max(bounds.lowerBound, min(lower, bounds.upperBound))
        upper = max(bounds.lowerBound, min(upper, bounds.upperBound))
        if upper <= lower { upper = min(bounds.upperBound, lower + max(0.0001, len)) }
        return lower...upper
    }

    static func zoom(domain: ClosedRange<Double>, by scale: Double, around anchor: Double? = nil, within bounds: ClosedRange<Double>, minLength: Double = 0.25) -> ClosedRange<Double> {
        let len = domain.upperBound - domain.lowerBound
        guard len > 0 else { return bounds }
        let boundedScale = max(0.25, min(scale, 4))
        let newLen = max(minLength, len / boundedScale)
        let center = anchor ?? (domain.lowerBound + domain.upperBound) / 2
        var lower = center - newLen / 2
        var upper = center + newLen / 2
        if lower < bounds.lowerBound {
            let shift = bounds.lowerBound - lower
            lower += shift
            upper += shift
        }
        if upper > bounds.upperBound {
            let shift = bounds.upperBound - upper
            lower += shift
            upper += shift
        }
        lower = max(bounds.lowerBound, min(lower, bounds.upperBound))
        upper = max(bounds.lowerBound, min(upper, bounds.upperBound))
        if upper <= lower {
            upper = min(bounds.upperBound, lower + minLength)
        }
        return lower...upper
    }

    static func pan(domain: ClosedRange<Int>, by delta: Double, within bounds: ClosedRange<Int>) -> ClosedRange<Int> {
        let len = Double(domain.upperBound - domain.lowerBound)
        guard len > 0 else { return bounds }
        var lower = Double(domain.lowerBound) + delta
        var upper = Double(domain.upperBound) + delta
        if lower < Double(bounds.lowerBound) {
            let shift = Double(bounds.lowerBound) - lower
            lower += shift
            upper += shift
        }
        if upper > Double(bounds.upperBound) {
            let shift = Double(bounds.upperBound) - upper
            lower += shift
            upper += shift
        }
        let intLower = max(bounds.lowerBound, min(Int(lower.rounded()), bounds.upperBound))
        let intUpper = max(bounds.lowerBound, min(Int(upper.rounded()), bounds.upperBound))
        if intUpper <= intLower {
            let fallbackUpper = min(bounds.upperBound, intLower + max(1, Int(len.rounded())))
            return intLower...fallbackUpper
        }
        return intLower...intUpper
    }

    static func zoom(domain: ClosedRange<Int>, by scale: Double, around anchor: Double? = nil, within bounds: ClosedRange<Int>, minLength: Int = 1) -> ClosedRange<Int> {
        let len = Double(domain.upperBound - domain.lowerBound)
        guard len > 0 else { return bounds }
        let boundedScale = max(0.25, min(scale, 4))
        let newLen = max(Double(minLength), len / boundedScale)
        let center = anchor ?? (Double(domain.lowerBound) + Double(domain.upperBound)) / 2
        var lower = center - newLen / 2
        var upper = center + newLen / 2
        if lower < Double(bounds.lowerBound) {
            let shift = Double(bounds.lowerBound) - lower
            lower += shift
            upper += shift
        }
        if upper > Double(bounds.upperBound) {
            let shift = Double(bounds.upperBound) - upper
            lower += shift
            upper += shift
        }
        let intLower = max(bounds.lowerBound, min(Int(lower.rounded()), bounds.upperBound))
        let intUpper = max(bounds.lowerBound, min(Int(upper.rounded()), bounds.upperBound))
        if intUpper <= intLower {
            return intLower...min(bounds.upperBound, intLower + minLength)
        }
        return intLower...intUpper
    }
}

