import Foundation

struct Cluster: Identifiable, Equatable, Codable {
    var id: Int
    var name: String
    var metaSummary: String
    var tradingLens: String? = nil
    var centroid: [Float]
    var memberPaperIDs: [UUID]
    var layoutPosition: Point2D?
    var resolutionK: Int?
    var corpusVersion: String?
    var subclusters: [Cluster]?
}

struct BridgingResult: Identifiable, Equatable {
    var id: UUID { paper.id }
    var paper: Paper
    var combinedScore: Float
    var scoreToFirst: Float
    var scoreToSecond: Float
}

struct ScoredPaper: Identifiable, Equatable {
    var id: UUID { paper.id }
    var paper: Paper
    var score: Float
}

struct PaperChunk: Identifiable, Codable, Equatable {
    var id: UUID
    var paperID: UUID
    var text: String
    var embedding: [Float]
    var order: Int
    var pageHint: Int?
}

struct Point2D: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
}

struct ChunkEvidence: Identifiable, Equatable {
    var id: UUID { chunk.id }
    var chunk: PaperChunk
    var score: Float
    var paperTitle: String
}

struct ClusterSummary {
    let name: String
    let metaSummary: String
    let tradingLens: String?
}

struct ClusterInsights: Equatable {
    let topKeywords: [String]
    let topTitles: [String]
}

enum ClusterNameSource: String, Codable, CaseIterable, Equatable {
    case heuristic
    case manual
    case ai

    var label: String {
        switch self {
        case .heuristic: return "Heuristic"
        case .manual: return "Manual"
        case .ai: return "AI"
        }
    }
}

// Personalized planning
enum CurriculumStage: String, Codable, CaseIterable, Equatable {
    case foundation
    case bridge
    case advanced

    var label: String {
        switch self {
        case .foundation: return "Foundation"
        case .bridge: return "Bridge"
        case .advanced: return "Advanced"
        }
    }
}

struct CurriculumStep: Identifiable, Equatable, Codable {
    var id: UUID
    var paper: Paper
    var stage: CurriculumStage
    var score: Float
}

struct KnowledgeSnapshot: Equatable {
    let topic: String
    let known: [Paper]
    let missing: [Paper]
    let summary: String
}
