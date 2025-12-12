import Foundation
import Accelerate

/// Minimal in-process vector index. Acts as a placeholder for a future Rust-backed ANN.
struct VectorIndex {
    private let dim: Int
    private let vectors: [[Float]]

    init?(vectors: [[Float]]) {
        guard let first = vectors.first else { return nil }
        dim = first.count
        guard dim > 0 else { return nil }
        self.vectors = vectors
    }

    func query(_ needle: [Float], k: Int) -> [(index: Int, score: Float)] {
        guard needle.count == dim else { return [] }
        guard k > 0 else { return [] }
        var scored: [(Int, Float)] = []
        for (idx, vec) in vectors.enumerated() where vec.count == dim {
            let score = cosineSimilarity(needle, vec)
            if score.isNaN { continue }
            scored.append((idx, score))
        }
        return Array(scored.sorted { $0.1 > $1.1 }.prefix(k))
    }
}
