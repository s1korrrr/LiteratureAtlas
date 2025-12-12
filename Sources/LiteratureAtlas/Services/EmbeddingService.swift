import Foundation
@preconcurrency import NaturalLanguage
import Accelerate

@available(macOS 26, iOS 26, *)
@MainActor
final class SentenceEmbedder {
    private let model: NLContextualEmbedding?
    private var isLoaded: Bool = false

    init(language: NLLanguage = .english) {
        guard let embedding = NLContextualEmbedding(language: language) else {
            model = nil
            return
        }
        self.model = embedding

        if embedding.hasAvailableAssets {
            do {
                try embedding.load()
                isLoaded = true
            } catch {
                isLoaded = false
            }
        } else {
            embedding.requestAssets { [weak self] result, _ in
                guard result == .available else { return }
                do { try embedding.load() } catch { return }
                Task { @MainActor in
                    self?.isLoaded = true
                }
            }
        }
    }

    /// Mean-pools token vectors to a single embedding.
    func encode(for text: String) async -> [Float]? {
        guard !text.isEmpty else { return nil }
        guard let model else { return nil }
        if !isLoaded {
            await ensureLoaded(model: model)
            guard isLoaded else { return nil }
        }
        guard let result = try? model.embeddingResult(for: text, language: nil) else {
            return nil
        }

        var mean = [Float](repeating: 0, count: model.dimension)
        var tokenCount = 0

        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { tokenVector, _ in
            let vector = vDSP.doubleToFloat(tokenVector)
            mean = vDSP.add(mean, vector)
            tokenCount += 1
            return true
        }

        guard tokenCount > 0 else { return nil }
        let scalar = 1 / Float(tokenCount)
        mean = vDSP.multiply(scalar, mean)
        return mean
    }

    var dimension: Int { model?.dimension ?? 0 }

    private func ensureLoaded(model: NLContextualEmbedding, timeoutSeconds: Double = 4) async {
        if isLoaded { return }
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while !isLoaded && Date() < deadline {
            try? await Task.sleep(nanoseconds: 150_000_000)
        }
        if !isLoaded, model.hasAvailableAssets {
            do {
                try model.load()
                isLoaded = true
            } catch {
                isLoaded = false
            }
        }
    }
}

// MARK: - Similarity helpers

func squaredDistance(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return .greatestFiniteMagnitude }
    var sum: Float = 0
    for i in 0..<a.count {
        let d = a[i] - b[i]
        sum += d * d
    }
    return sum
}

func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return .nan }
    let dot = vDSP.dot(a, b)
    let na = sqrt(vDSP.dot(a, a))
    let nb = sqrt(vDSP.dot(b, b))
    guard na > 0, nb > 0 else { return .nan }
    return dot / (na * nb)
}

// MARK: - KMeans (small, deterministic enough for UI use)

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1
        return state
    }
}

struct KMeans {
    static func cluster(vectors: [[Float]], k: Int, iterations: Int = 25) -> (assignments: [Int], centroids: [[Float]]) {
        let n = vectors.count
        guard n > 0 else { return ([], []) }
        let dim = vectors[0].count
        guard dim > 0 else {
            return (Array(repeating: 0, count: n), Array(repeating: [Float](), count: max(1, k)))
        }

        let normalizedVectors: [[Float]] = vectors.map { v in
            if v.count == dim { return v }
            if v.count > dim { return Array(v.prefix(dim)) }
            var vv = v
            vv.append(contentsOf: repeatElement(0, count: dim - v.count))
            return vv
        }

        let kClamped = min(max(1, k), n)

        var rng = SeededGenerator(seed: 0xC0FFEE)
        let shuffled = Array(0..<n).shuffled(using: &rng)
        var centroids: [[Float]] = []
        for i in 0..<kClamped {
            centroids.append(normalizedVectors[shuffled[i]])
        }

        var assignments = Array(repeating: 0, count: n)

        for _ in 0..<iterations {
            // Assignment step
            for i in 0..<n {
                var bestIndex = 0
                var bestDist = squaredDistance(normalizedVectors[i], centroids[0])
                if kClamped > 1 {
                    for c in 1..<kClamped {
                        let dist = squaredDistance(normalizedVectors[i], centroids[c])
                        if dist < bestDist {
                            bestDist = dist
                            bestIndex = c
                        }
                    }
                }
                assignments[i] = bestIndex
            }

            // Update step
            var newCentroids = Array(repeating: [Float](repeating: 0, count: dim), count: kClamped)
            var counts = Array(repeating: 0, count: kClamped)

            for i in 0..<n {
                let cid = assignments[i]
                counts[cid] += 1
                let vector = normalizedVectors[i]
                for j in 0..<dim {
                    newCentroids[cid][j] += vector[j]
                }
            }

            for c in 0..<kClamped {
                if counts[c] > 0 {
                    let inv = 1 / Float(counts[c])
                    vDSP.multiply(inv, newCentroids[c], result: &newCentroids[c])
                } else {
                    let randomIndex = Int(rng.next() % UInt64(n))
                    newCentroids[c] = normalizedVectors[randomIndex]
                }
            }

            centroids = newCentroids
        }

        return (assignments, centroids)
    }
}
