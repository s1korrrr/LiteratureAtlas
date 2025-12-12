import Foundation
@preconcurrency import NaturalLanguage

// MARK: - Claim-level structures

struct EvaluationContext: Codable, Equatable {
    var dataset: String?
    var period: String?
    var metrics: [String]
}

struct PaperClaim: Identifiable, Codable, Equatable {
    var id: UUID
    var paperID: UUID
    var statement: String
    var assumptions: [String]
    var evaluation: EvaluationContext?
    var year: Int?
    var strength: Float // heuristic confidence 0-1
}

enum ClaimRelationKind: String, Codable, Equatable {
    case supports
    case extends
    case contradicts
    case comparesTo
}

struct ClaimEdge: Identifiable, Codable, Equatable {
    var id: UUID
    var sourceClaimID: UUID
    var targetClaimID: UUID
    var kind: ClaimRelationKind
    var rationale: String?

    init(id: UUID = UUID(), sourceClaimID: UUID, targetClaimID: UUID, kind: ClaimRelationKind, rationale: String? = nil) {
        self.id = id
        self.sourceClaimID = sourceClaimID
        self.targetClaimID = targetClaimID
        self.kind = kind
        self.rationale = rationale
    }
}

struct ClaimExtraction {
    let claims: [PaperClaim]
    let assumptions: [String]
    let evaluation: EvaluationContext?
}

// MARK: - Heuristic claim extraction (fast, offline)

enum ClaimExtractor {
    static func heuristicExtraction(summary: String, paperID: UUID, year: Int?) -> ClaimExtraction {
        let normalized = summary.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        let sentences = splitIntoSentences(text: normalized)

        var claims: [PaperClaim] = []
        var assumptions: Set<String> = []

        let assumptionLexicon = [
            "poisson arrivals",
            "poisson",
            "risk-neutral",
            "risk neutral",
            "infinite liquidity",
            "no permanent impact",
            "iid",
            "stationary",
            "gaussian noise",
            "bounded inventory"
        ]

        for lex in assumptionLexicon {
            if normalized.lowercased().contains(lex) {
                assumptions.insert(lex)
            }
        }

        let evaluation = inferEvaluation(from: normalized)

        // Pick 1-3 key sentences as claims (improves, reduces, achieves, propose, introduce)
        let keywords = ["improv", "reduce", "increase", "achieve", "propos", "introduc", "demonstrat", "show"]
        for sentence in sentences {
            let lower = sentence.lowercased()
            if keywords.contains(where: { lower.contains($0) }) {
                let claim = PaperClaim(
                    id: UUID(),
                    paperID: paperID,
                    statement: sentence.trimmingCharacters(in: .whitespacesAndNewlines),
                    assumptions: Array(assumptions),
                    evaluation: evaluation,
                    year: year,
                    strength: 0.6
                )
                claims.append(claim)
            }
        }

        if claims.isEmpty, let first = sentences.first {
            let claim = PaperClaim(
                id: UUID(),
                paperID: paperID,
                statement: first.trimmingCharacters(in: .whitespacesAndNewlines),
                assumptions: Array(assumptions),
                evaluation: evaluation,
                year: year,
                strength: 0.4
            )
            claims.append(claim)
        }

        return ClaimExtraction(claims: claims, assumptions: Array(assumptions), evaluation: evaluation)
    }

    private static func splitIntoSentences(text: String) -> [String] {
        var results: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { results.append(sentence) }
            return true
        }
        return results
    }

    private static func inferEvaluation(from text: String) -> EvaluationContext? {
        let lower = text.lowercased()
        // Dataset detection (simple lexicon)
        let datasets = ["cifar-10", "cifar10", "imagenet", "mnist", "lobster", "nyse", "limit order book", "synthetic"]
        let datasetMatch = datasets.first { lower.contains($0) }

        // Period detection 2019-2020 or single year
        var period: String?
        if let range = text.range(of: "[12][0-9]{3}-[12][0-9]{3}", options: .regularExpression) {
            period = String(text[range])
        } else if let range = text.range(of: "[12][0-9]{3}", options: .regularExpression) {
            period = String(text[range])
        }

        // Metrics detection
        let metricLexicon = ["accuracy", "f1", "auc", "sharpe", "pnl", "profit", "hit rate", "mse", "rmse", "precision", "recall"]
        var metrics: [String] = []
        for m in metricLexicon {
            if lower.contains(m) {
                metrics.append(m == "pnl" ? "PnL" : (m == "sharpe" ? "Sharpe ratio" : m.capitalized))
            }
        }
        metrics = Array(Set(metrics))

        if datasetMatch == nil && period == nil && metrics.isEmpty {
            return nil
        }

        let datasetName: String? = datasetMatch.map { token in
            if token == "cifar10" { return "CIFAR-10" }
            if token == "cifar-10" { return "CIFAR-10" }
            if token == "imagenet" { return "ImageNet" }
            if token == "mnist" { return "MNIST" }
            if token == "lobster" { return "LOBSTER" }
            if token == "nyse" { return "NYSE" }
            return token.capitalized
        }

        return EvaluationContext(dataset: datasetName, period: period, metrics: metrics)
    }
}

// MARK: - Claim relations

enum ClaimRelationInferencer {
    static func inferEdges(for claims: [PaperClaim]) -> [ClaimEdge] {
        guard claims.count >= 2 else { return [] }
        var edges: [ClaimEdge] = []

        for i in 0..<claims.count {
            for j in i+1..<claims.count {
                let a = claims[i]
                let b = claims[j]
                if let edge = relation(from: a, to: b) { edges.append(edge) }
                if let edge = relation(from: b, to: a) { edges.append(edge) }
            }
        }
        return edges
    }

    private static func relation(from a: PaperClaim, to b: PaperClaim) -> ClaimEdge? {
        let sharedDataset = a.evaluation?.dataset != nil && a.evaluation?.dataset == b.evaluation?.dataset
        let sharedMetric = !(Set(a.evaluation?.metrics ?? []).intersection(Set(b.evaluation?.metrics ?? [])).isEmpty)

        let aLower = a.statement.lowercased()
        let bLower = b.statement.lowercased()

        if sharedDataset && sharedMetric && (bLower.contains("does not") || bLower.contains("fail")) {
            return ClaimEdge(sourceClaimID: a.id, targetClaimID: b.id, kind: .contradicts, rationale: "Same dataset/metric but negative outcome")
        }

        if sharedDataset && sharedMetric && (bLower.contains("extends") || bLower.contains("builds on") || bLower.contains("further improves")) {
            return ClaimEdge(sourceClaimID: a.id, targetClaimID: b.id, kind: .extends, rationale: "Shared eval; target extends source")
        }

        if sharedDataset && sharedMetric && (aLower.contains("improve") && bLower.contains("improve")) {
            return ClaimEdge(sourceClaimID: a.id, targetClaimID: b.id, kind: .supports, rationale: "Both report improvements on same eval")
        }

        if sharedDataset || sharedMetric {
            return ClaimEdge(sourceClaimID: a.id, targetClaimID: b.id, kind: .comparesTo, rationale: "Comparable evaluation context")
        }

        return nil
    }
}

// MARK: - Assumption stress testing

struct AssumptionStressReport {
    let assumption: String
    let affectedClaims: [PaperClaim]
    let narrative: String
}

enum AssumptionStressTester {
    static func stressTest(assumption: String, papers: [Paper]) -> AssumptionStressReport {
        let target = assumption.lowercased()
        var affected: [PaperClaim] = []

        for paper in papers {
            for var claim in paper.claims ?? [] {
                // Normalize to the parent paper ID if the stored claim ID differs (legacy/heuristic cases).
                if claim.paperID != paper.id { claim.paperID = paper.id }
                if claim.assumptions.contains(where: { $0.lowercased().contains(target) }) {
                    affected.append(claim)
                }
            }
            if let paperAssumptions = paper.assumptions {
                if paperAssumptions.contains(where: { $0.lowercased().contains(target) }) {
                    // If paper has no explicit claims, create a lightweight proxy claim
                    if (paper.claims ?? []).isEmpty {
                        let proxy = PaperClaim(id: UUID(), paperID: paper.id, statement: "Paper depends on assumption: \(assumption)", assumptions: paperAssumptions, evaluation: paper.evaluationContext, year: paper.year, strength: 0.3)
                        affected.append(proxy)
                    }
                }
            }
        }

        let titles = papers.map { $0.title }
        let titleList = titles.joined(separator: ", ")
        let narrative: String
        if affected.isEmpty {
            narrative = "No claims explicitly depend on \(assumption) across papers: \(titleList)."
        } else {
            let claimCount = affected.count
            let citedPapers = Set(affected.map { $0.paperID }).count
            let affectedTitles = papers.filter { paper in
                affected.contains(where: { $0.paperID == paper.id })
            }.map { $0.title }.joined(separator: ", ")
            narrative = "Found \(claimCount) claim(s) across \(citedPapers) paper(s) that rely on \(assumption) (\(affectedTitles)). If this assumption fails, these results may weaken. Examples: \(affected.prefix(2).map { $0.statement }.joined(separator: " | "))."
        }

        return AssumptionStressReport(assumption: assumption, affectedClaims: affected, narrative: narrative)
    }
}

// MARK: - Method pipelines and blueprint generation

enum PipelineStage: String, Codable, Equatable {
    case data
    case preprocessing
    case model
    case objective
    case evaluation
}

struct PipelineStep: Codable, Equatable {
    let stage: PipelineStage
    let label: String
    let detail: String?
}

struct MethodPipeline: Codable, Equatable {
    let steps: [PipelineStep]
}

enum MethodPipelineExtractor {
    static func extract(from text: String) -> MethodPipeline {
        let lower = text.lowercased()
        var steps: [PipelineStep] = []

        // Data
        let dataLabel: String
        if lower.contains("order book") {
            dataLabel = "Limit order book data"
        } else if lower.contains("transaction") {
            dataLabel = "Transaction data"
        } else {
            dataLabel = "Input data"
        }
        steps.append(PipelineStep(stage: .data, label: dataLabel, detail: nil))

        // Preprocessing
        var prep = "Preprocessing"
        if lower.contains("normalize") || lower.contains("standardize") {
            prep = "Normalize volumes/prices"
        } else if lower.contains("feature") {
            prep = "Feature engineering"
        }
        steps.append(PipelineStep(stage: .preprocessing, label: prep, detail: nil))

        // Model
        var modelLabel = "Model"
        if lower.contains("dqn") {
            modelLabel = "DQN agent"
        } else if lower.contains("actor-critic") {
            modelLabel = "Actor-Critic RL"
        } else if lower.contains("transformer") {
            modelLabel = "Transformer"
        } else if lower.contains("hawkes") {
            modelLabel = "Hawkes process"
        }
        steps.append(PipelineStep(stage: .model, label: modelLabel, detail: nil))

        // Objective
        var objLabel = "Objective"
        var objDetail: String? = nil
        if lower.contains("sharpe") {
            objLabel = "Maximize Sharpe"
            objDetail = "Reward includes Sharpe / risk-adjusted PnL"
        } else if lower.contains("pnl") || lower.contains("profit") {
            objLabel = "Maximize PnL"
            objDetail = "Reward uses PnL with inventory penalty"
        } else if lower.contains("cross entropy") {
            objLabel = "Cross-entropy"
        }
        steps.append(PipelineStep(stage: .objective, label: objLabel, detail: objDetail))

        // Evaluation
        var evalLabel = "Evaluation"
        var evalDetail: String? = nil
        if lower.contains("sharpe") {
            evalLabel = "Sharpe ratio evaluation"
            evalDetail = "Compute Sharpe ratio on held-out period"
        } else if lower.contains("hit rate") {
            evalLabel = "Hit rate evaluation"
            evalDetail = "Hit rate / fill ratio"
        } else if lower.contains("accuracy") {
            evalLabel = "Accuracy"
        }
        steps.append(PipelineStep(stage: .evaluation, label: evalLabel, detail: evalDetail))

        return MethodPipeline(steps: steps)
    }
}

enum BlueprintGenerator {
    static func generate(for pipeline: MethodPipeline, title: String) -> String {
        var lines: [String] = []
        lines.append("Blueprint for \(title)")
        for (idx, step) in pipeline.steps.enumerated() {
            let line = "Step \(idx + 1): [\(step.stage.rawValue)] \(step.label)" + (step.detail != nil ? " — \(step.detail!)" : "")
            lines.append(line)
        }
        lines.append("Notes: Generated from method description; align reward/metrics with paper.")
        return lines.joined(separator: "\n")
    }
}
