import Foundation

struct QuantKnowledgeGraphSnapshot: Codable, Equatable {
    struct Node: Codable, Equatable, Identifiable {
        var id: String
        var kind: NodeKind
        var label: String
        var paperID: UUID?
        var strategyID: UUID?
        var extra: [String: String]?

        enum CodingKeys: String, CodingKey {
            case id
            case kind
            case label
            case paperID = "paper_id"
            case strategyID = "strategy_id"
            case extra
        }
    }

    struct Edge: Codable, Equatable, Identifiable {
        var id: UUID
        var source: String
        var target: String
        var kind: EdgeKind
        var weight: Double?
        var note: String?

        enum CodingKeys: String, CodingKey {
            case id
            case source
            case target
            case kind
            case weight
            case note
        }
    }

    enum NodeKind: String, Codable, Equatable {
        case paper
        case strategy
        case idea
        case feature
        case model
        case trade
        case decision
        case outcome
        case code
    }

    enum EdgeKind: String, Codable, Equatable {
        case inspiredBy
        case defines
        case usesFeature
        case usesModel
        case makesDecision
        case leadsTo
        case implementedInCode
        case referencesPaper
        case related
    }

    var generatedAt: Date
    var nodes: [Node]
    var edges: [Edge]
}
