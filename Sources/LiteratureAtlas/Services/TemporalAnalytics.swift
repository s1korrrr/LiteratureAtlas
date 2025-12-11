import Foundation
import Accelerate

@available(macOS 26, iOS 26, *)
enum TemporalAnalytics {

    // MARK: Topic evolution

    static func topicEvolution(for cluster: Cluster, papers: [Paper]) -> TopicEvolutionStream {
        let relevant = papers.filter { $0.clusterIndex == cluster.id && $0.year != nil }
        guard !relevant.isEmpty else {
            return TopicEvolutionStream(clusterID: cluster.id, countsByYear: [:], burstYears: [], decayYears: [], narrative: "No dated papers for \(cluster.name).")
        }

        let years = relevant.compactMap { $0.year }
        guard let minYear = years.min(), let maxYear = years.max() else {
            return TopicEvolutionStream(clusterID: cluster.id, countsByYear: [:], burstYears: [], decayYears: [], narrative: "No dated papers for \(cluster.name).")
        }

        var counts: [Int: Int] = [:]
        for year in minYear...maxYear { counts[year] = 0 }
        for paper in relevant {
            if let y = paper.year { counts[y, default: 0] += 1 }
        }

        var burstYears: [Int] = []
        var decayYears: [Int] = []
        var prevCount = counts[minYear] ?? 0
        if minYear < maxYear {
            for year in (minYear + 1)...maxYear {
                let current = counts[year] ?? 0
                if (prevCount == 0 && current >= 2) || (prevCount > 0 && current >= prevCount * 2) {
                    burstYears.append(year)
                }
                if prevCount > 0 && current <= prevCount / 2 {
                    decayYears.append(year)
                }
                prevCount = current
            }
        }

        let peak = counts.max(by: { $0.value < $1.value })?.key ?? minYear
        let narrative = "Between \(minYear) and \(maxYear), \(cluster.name) peaked in \(peak) with \(counts[peak] ?? 0) papers; bursts: \(burstYears.sorted()); decay: \(decayYears.sorted())."
        return TopicEvolutionStream(clusterID: cluster.id, countsByYear: counts, burstYears: burstYears, decayYears: decayYears, narrative: narrative)
    }

    // MARK: Method takeover / replacement

    static func methodTakeovers(papers: [Paper], methodTags: [String]) -> [MethodTakeover] {
        let lowerTags = methodTags.map { $0.lowercased() }
        var counts: [String: [Int: Int]] = [:] // method -> year -> count

        for paper in papers {
            guard let year = paper.year else { continue }
            for tag in lowerTags where paper.contains(methodTag: tag) {
                counts[tag, default: [:]][year, default: 0] += 1
            }
        }

        guard let minYear = counts.values.compactMap({ $0.keys.min() }).min(),
              let maxYear = counts.values.compactMap({ $0.keys.max() }).max(),
              minYear <= maxYear else { return [] }

        var results: [MethodTakeover] = []
        for i in 0..<lowerTags.count {
            for j in (i + 1)..<lowerTags.count {
                let a = lowerTags[i]
                let b = lowerTags[j]
                var cumulativeA = 0
                var cumulativeB = 0
                var crossingYear: Int?
                var prevDiff: Int?

                for year in minYear...maxYear {
                    cumulativeA += counts[a]?[year] ?? 0
                    cumulativeB += counts[b]?[year] ?? 0
                    let diff = cumulativeA - cumulativeB
                    if let prev = prevDiff,
                       (prev >= 0 && diff < 0) || (prev <= 0 && diff > 0) {
                        crossingYear = year
                        break
                    }
                    prevDiff = diff
                }

                let leader: String?
                if cumulativeA > cumulativeB {
                    leader = a
                } else if cumulativeB > cumulativeA {
                    leader = b
                } else if let prev = prevDiff, prev > 0 {
                    leader = a
                } else if let prev = prevDiff, prev < 0 {
                    leader = b
                } else {
                    leader = nil
                }
                results.append(MethodTakeover(a: a, b: b, crossingYear: crossingYear, leadingAtEnd: leader))
            }
        }
        return results
    }

    // MARK: Reading over time

    static func readingLagStats(papers: [Paper], clusters: [Cluster]) -> ReadingLagStats {
        var lags: [Double] = []
        var overlay: [ReadingOverlayPoint] = []
        var perCluster: [Int: [Double]] = [:]
        let calendar = Calendar.current

        for paper in papers {
            guard let pubYear = paper.year,
                  let readDate = paper.firstReadAt else { continue }
            let readYear = calendar.component(.year, from: readDate)
            let lag = Double(readYear - pubYear)
            lags.append(lag)
            overlay.append(ReadingOverlayPoint(clusterID: paper.clusterIndex, publicationYear: pubYear, readYear: readYear))
            if let cid = paper.clusterIndex {
                perCluster[cid, default: []].append(lag)
            }
        }

        let averageLag = lags.isEmpty ? 0 : lags.reduce(0, +) / Double(lags.count)
        var realTime: [Int] = []
        var late: [Int] = []
        for (cid, values) in perCluster {
            let med = median(values)
            if med <= 1.0 {
                realTime.append(cid)
            } else if med >= 2.0 {
                late.append(cid)
            }
        }

        return ReadingLagStats(
            averageLagYears: averageLag,
            realTimeClusterIDs: realTime.sorted(),
            lateClusterIDs: late.sorted(),
            overlay: overlay
        )
    }

    // MARK: Novelty & saturation

    static func noveltyScores(papers: [Paper], clusters: [Cluster], neighbors: Int = 3) -> [PaperNoveltyScore] {
        let clusterByID = Dictionary(uniqueKeysWithValues: clusters.map { ($0.id, $0) })
        var grouped = Dictionary(grouping: papers) { $0.clusterIndex ?? -1 }
        grouped.removeValue(forKey: -1)

        var results: [PaperNoveltyScore] = []

        for (cid, group) in grouped {
            guard let centroid = clusterByID[cid]?.centroid, !centroid.isEmpty else { continue }
            for paper in group where paper.embedding.count == centroid.count {
                let sim = cosineSimilarity(paper.embedding, centroid)
                let novelty = max(0, 1 - ((sim + 1) / 2))
                let neighborsVecs = group.filter { $0.id != paper.id && $0.embedding.count == paper.embedding.count }
                let sims = neighborsVecs.map { cosineSimilarity(paper.embedding, $0.embedding) }.filter { !$0.isNaN }
                let top = sims.sorted(by: >).prefix(max(1, neighbors))
                let density = top.isEmpty ? 0 : top.reduce(0, +) / Float(top.count)
                let saturation = max(0, min(1, (density + 1) / 2))
                let nearest = neighborsVecs
                    .map { (id: $0.id, score: cosineSimilarity(paper.embedding, $0.embedding)) }
                    .sorted { $0.score > $1.score }
                    .prefix(max(1, neighbors))
                    .map { $0.id }

                results.append(PaperNoveltyScore(paperID: paper.id, novelty: novelty, saturation: saturation, nearestNeighbors: Array(nearest)))
            }
        }
        return results
    }

    // MARK: Hypothetical paper generator

    static func generateHypotheticalPaper(from clusters: [Cluster], papers: [Paper]) -> HypotheticalPaper {
        let titleParts = clusters.map { $0.name }
        let title = titleParts.isEmpty ? "Hypothetical paper" : "Bridge: " + titleParts.joined(separator: " × ")
        let abstractSeeds = clusters.map { $0.metaSummary }.filter { !$0.isEmpty }
        var abstract = "This hypothetical paper bridges \(titleParts.joined(separator: ", "))."
        if !abstractSeeds.isEmpty {
            abstract += " It leverages: \(abstractSeeds.joined(separator: " "))"
        }

        var vectors = clusters.map { $0.centroid }.filter { !$0.isEmpty }
        if vectors.isEmpty {
            vectors = papers.prefix(3).map { $0.embedding }
        }
        let embedding = normalizeAndAverage(vectors)

        return HypotheticalPaper(id: UUID(), title: title, abstract: abstract, embedding: embedding, anchorClusterIDs: clusters.map { $0.id })
    }

    // MARK: Panel simulation

    static func simulatePanel(for cluster: Cluster, papers: [Paper], maxSpeakers: Int = 3) -> String {
        let matches = papers
            .filter { $0.clusterIndex == cluster.id || cluster.memberPaperIDs.contains($0.id) }
            .prefix(maxSpeakers)

        guard !matches.isEmpty else { return "No papers available for \(cluster.name)." }

        var lines: [String] = ["Panel on \(cluster.name)"]
        for (idx, paper) in matches.enumerated() {
            let claim = paper.claims?.first?.statement ?? paper.summary
            lines.append("Author \(idx + 1) (\(paper.title)): \(claim)")
            if idx > 0 {
                lines.append("Author \(idx + 1) responds to Author \(idx): contrasts assumptions.")
            }
        }
        return lines.joined(separator: "\n")
    }

    static func simulateDebate(between first: Cluster, and second: Cluster, papers: [Paper]) -> String {
        let firstPaper = papers.first(where: { $0.clusterIndex == first.id }) ?? papers.first
        let secondPaper = papers.first(where: { $0.clusterIndex == second.id }) ?? papers.dropFirst().first

        var lines: [String] = []
        lines.append("Debate: \(first.name) vs \(second.name)")
        if let fp = firstPaper {
            let arg = fp.claims?.first?.statement ?? fp.summary
            lines.append("\(first.name) position: \(arg)")
        }
        if let sp = secondPaper {
            let arg = sp.claims?.first?.statement ?? sp.summary
            lines.append("\(second.name) responds: \(arg)")
        }
        lines.append("Each side highlights advantages, limitations, and regimes where it wins.")
        return lines.joined(separator: "\n")
    }

    // MARK: Misconception detection

    static func detectMisconception(answer: String, paper: Paper) -> MisconceptionReport {
        let answerLower = answer.lowercased()
        var missing: [String] = []
        var misaligned: [String] = []

        if let keywords = paper.keywords {
            for key in keywords where !answerLower.contains(key.lowercased()) {
                missing.append(key)
            }
        }
        if let assumptions = paper.assumptions {
            for assumption in assumptions where !answerLower.contains(assumption.lowercased()) {
                misaligned.append(assumption)
            }
        }

        if missing.isEmpty && misaligned.isEmpty {
            return MisconceptionReport(missingConcepts: [], misalignedAssumptions: [], narrative: "No obvious gaps detected.")
        }

        let narrative = "Missing concepts: \(missing.joined(separator: ", ")). Assumptions not mentioned: \(misaligned.joined(separator: ", "))."
        return MisconceptionReport(missingConcepts: missing, misalignedAssumptions: misaligned, narrative: narrative)
    }
}

// MARK: - Helper models

struct TopicEvolutionStream {
    let clusterID: Int
    let countsByYear: [Int: Int]
    let burstYears: [Int]
    let decayYears: [Int]
    let narrative: String
}

struct MethodTakeover: Equatable {
    let a: String
    let b: String
    let crossingYear: Int?
    let leadingAtEnd: String?
}

struct ReadingOverlayPoint: Equatable {
    let clusterID: Int?
    let publicationYear: Int
    let readYear: Int
}

struct ReadingLagStats {
    let averageLagYears: Double
    let realTimeClusterIDs: [Int]
    let lateClusterIDs: [Int]
    let overlay: [ReadingOverlayPoint]
}

struct PaperNoveltyScore: Equatable {
    let paperID: UUID
    let novelty: Float
    let saturation: Float
    let nearestNeighbors: [UUID]
}

struct HypotheticalPaper {
    let id: UUID
    let title: String
    let abstract: String
    let embedding: [Float]
    let anchorClusterIDs: [Int]
}

struct MisconceptionReport {
    let missingConcepts: [String]
    let misalignedAssumptions: [String]
    let narrative: String
}

// MARK: - Local helpers

private func normalizeAndAverage(_ vectors: [[Float]]) -> [Float] {
    guard let first = vectors.first, !first.isEmpty else { return [] }
    let dim = first.count
    var sum = [Float](repeating: 0, count: dim)
    var count: Float = 0
    for vec in vectors where vec.count == dim {
        let norm = sqrt(vDSP.dot(vec, vec))
        guard norm > 0 else { continue }
        var normalized = vec
        vDSP.divide(vec, norm, result: &normalized)
        vDSP.add(sum, normalized, result: &sum)
        count += 1
    }
    guard count > 0 else { return [] }
    vDSP.divide(sum, count, result: &sum)
    let norm = sqrt(vDSP.dot(sum, sum))
    guard norm > 0 else { return sum }
    var unit = sum
    vDSP.divide(sum, norm, result: &unit)
    return unit
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    if sorted.count % 2 == 0 {
        return (sorted[mid - 1] + sorted[mid]) / 2
    }
    return sorted[mid]
}

private extension Paper {
    func contains(methodTag: String) -> Bool {
        let haystacks = [
            methodSummary?.lowercased() ?? "",
            summary.lowercased(),
            keywords?.joined(separator: " ").lowercased() ?? ""
        ]
        return haystacks.contains { $0.contains(methodTag) }
    }
}
