import Foundation

struct AnalyticsSummary: Codable, Equatable {
    struct TopicTrend: Codable, Equatable {
        let clusterID: Int
        let year: Int
        let count: Int

        enum CodingKeys: String, CodingKey {
            case clusterID = "cluster_id"
            case year
            case count
        }
    }

    struct NoveltyScore: Codable, Equatable {
        let paperID: UUID
        let clusterID: Int?
        let novelty: Double

        enum CodingKeys: String, CodingKey {
            case paperID = "paper_id"
            case clusterID = "cluster_id"
            case novelty
        }
    }

    struct CentralityEntry: Codable, Equatable {
        struct Neighbor: Codable, Equatable {
            let paperID: UUID
            let score: Double

            enum CodingKeys: String, CodingKey {
                case paperID = "paper_id"
                case score
            }
        }

        let paperID: UUID
        let weightedDegree: Double
        let averageSimilarity: Double
        let neighbors: [Neighbor]

        enum CodingKeys: String, CodingKey {
            case paperID = "paper_id"
            case weightedDegree = "weighted_degree"
            case averageSimilarity = "average_similarity"
            case neighbors
        }
    }

    struct UserEventStats: Codable, Equatable {
        let total: Int
        let byType: [String: Int]
        let lastSeen: Date?

        enum CodingKeys: String, CodingKey {
            case total
            case byType = "by_type"
            case lastSeen = "last_seen"
        }
    }

    struct DriftEntry: Codable, Equatable {
        let clusterID: Int
        let year: Int
        let drift: Double
        let dx: Double?
        let dy: Double?
        let fromYear: Int?
        let toYear: Int?

        enum CodingKeys: String, CodingKey {
            case clusterID = "cluster_id"
            case year
            case drift
            case dx
            case dy
            case fromYear = "from_year"
            case toYear = "to_year"
        }
    }

    struct FactorLoading: Codable, Equatable {
        let paperID: UUID
        let scores: [Double]

        enum CodingKeys: String, CodingKey {
            case paperID = "paper_id"
            case scores
        }
    }

    struct FactorExposure: Codable, Equatable {
        let year: Int
        let factor: Int
        let score: Double
    }

    struct DriftVolatility: Codable, Equatable {
        let clusterID: Int
        let volatility: Double
        let avgStep: Double

        enum CodingKeys: String, CodingKey {
            case clusterID = "cluster_id"
            case volatility
            case avgStep = "avg_step"
        }
    }

    struct InfluenceEntry: Codable, Equatable {
        let paperID: UUID
        let influence: Double

        enum CodingKeys: String, CodingKey {
            case paperID = "paper_id"
            case influence
        }
    }

    struct PaperMetric: Codable, Equatable {
        let paperID: UUID
        let novCluster: Double
        let novGlobal: Double
        let novDirectional: Double
        let novCombinatorial: Double
        let noveltyUncertainty: Double
        let zNovelty: Double
        let consensusStruct: Double
        let consensusClaim: Double
        let consensusTemporal: Double
        let consensusTotal: Double
        let zConsensus: Double
        let consensusUncertainty: Double
        let influenceAbs: Double
        let influencePos: Double
        let influenceNeg: Double
        let driftContrib: Double
        let roleSource: Double
        let roleBridge: Double
        let roleSink: Double

        enum CodingKeys: String, CodingKey {
            case paperID = "paper_id"
            case novCluster = "nov_cluster"
            case novGlobal = "nov_global"
            case novDirectional = "nov_directional"
            case novCombinatorial = "nov_combinatorial"
            case noveltyUncertainty = "novelty_uncertainty"
            case zNovelty = "z_novelty"
            case consensusStruct = "consensus_struct"
            case consensusClaim = "consensus_claim"
            case consensusTemporal = "consensus_temporal"
            case consensusTotal = "consensus_total"
            case zConsensus = "z_consensus"
            case consensusUncertainty = "consensus_uncertainty"
            case influenceAbs = "influence_abs"
            case influencePos = "influence_pos"
            case influenceNeg = "influence_neg"
            case driftContrib = "drift_contrib"
            case roleSource = "role_source"
            case roleBridge = "role_bridge"
            case roleSink = "role_sink"
        }
    }

    struct IdeaFlowEdge: Codable, Equatable {
        let src: UUID?
        let dst: UUID?
        let weight: Double?
        let fromYear: Int?
        let toYear: Int?
        let sign: Int?

        enum CodingKeys: String, CodingKey {
            case src
            case dst
            case weight
            case fromYear = "from_year"
            case toYear = "to_year"
            case sign
        }
    }

    struct Counterfactual: Codable, Equatable {
        let name: String
        let paperCount: Int
        let avgCentrality: Double

        enum CodingKeys: String, CodingKey {
            case name
            case paperCount = "paper_count"
            case avgCentrality = "avg_centrality"
        }
    }

    let generatedAt: Date
    let paperCount: Int
    let vectorDim: Int
    let topicTrends: [TopicTrend]
    let novelty: [NoveltyScore]
    let centrality: [CentralityEntry]
    let drift: [DriftEntry]
    let factors: [[Double]]
    let factorLoadings: [FactorLoading]
    let factorExposures: [FactorExposure]
    let userFactorExposures: [FactorExposure]
    let factorLabels: [String]
    let influence: [InfluenceEntry]
    let influencePos: [InfluenceEntry]
    let influenceNeg: [InfluenceEntry]
    let ideaFlowEdges: [IdeaFlowEdge]
    let paperMetrics: [PaperMetric]
    let driftVolatility: [DriftVolatility]
    let recommendations: [UUID]
    let answerConfidence: Double?
    let counterfactuals: [Counterfactual]
    let userEvents: UserEventStats?
    let notes: String?

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case paperCount = "paper_count"
        case vectorDim = "vector_dim"
        case topicTrends = "topic_trends"
        case novelty
        case centrality
        case drift
        case driftVolatility = "drift_volatility"
        case factors
        case factorLoadings = "factor_loadings"
        case factorExposures = "factor_exposures"
        case userFactorExposures = "factor_exposures_user"
        case factorLabels = "factor_labels"
        case influence
        case influencePos = "influence_pos"
        case influenceNeg = "influence_neg"
        case ideaFlowEdges = "idea_flow_edges"
        case paperMetrics = "paper_metrics"
        case recommendations
        case answerConfidence = "answer_confidence"
        case counterfactuals
        case userEvents = "user_events"
        case notes
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        generatedAt = try container.decode(Date.self, forKey: .generatedAt)
        paperCount = try container.decode(Int.self, forKey: .paperCount)
        vectorDim = try container.decode(Int.self, forKey: .vectorDim)
        topicTrends = try container.decodeIfPresent([TopicTrend].self, forKey: .topicTrends) ?? []
        novelty = try container.decodeIfPresent([NoveltyScore].self, forKey: .novelty) ?? []
        centrality = try container.decodeIfPresent([CentralityEntry].self, forKey: .centrality) ?? []
        drift = try container.decodeIfPresent([DriftEntry].self, forKey: .drift) ?? []
        driftVolatility = try container.decodeIfPresent([DriftVolatility].self, forKey: .driftVolatility) ?? []
        factors = try container.decodeIfPresent([[Double]].self, forKey: .factors) ?? []
        factorLoadings = try container.decodeIfPresent([FactorLoading].self, forKey: .factorLoadings) ?? []
        factorExposures = try container.decodeIfPresent([FactorExposure].self, forKey: .factorExposures) ?? []
        userFactorExposures = try container.decodeIfPresent([FactorExposure].self, forKey: .userFactorExposures) ?? []
        factorLabels = try container.decodeIfPresent([String].self, forKey: .factorLabels) ?? []
        influence = try container.decodeIfPresent([InfluenceEntry].self, forKey: .influence) ?? []
        influencePos = try container.decodeIfPresent([InfluenceEntry].self, forKey: .influencePos) ?? []
        influenceNeg = try container.decodeIfPresent([InfluenceEntry].self, forKey: .influenceNeg) ?? []
        ideaFlowEdges = try container.decodeIfPresent([IdeaFlowEdge].self, forKey: .ideaFlowEdges) ?? []
        paperMetrics = try container.decodeIfPresent([PaperMetric].self, forKey: .paperMetrics) ?? []
        recommendations = try container.decodeIfPresent([UUID].self, forKey: .recommendations) ?? []
        answerConfidence = try container.decodeIfPresent(Double.self, forKey: .answerConfidence)
        counterfactuals = try container.decodeIfPresent([Counterfactual].self, forKey: .counterfactuals) ?? []
        userEvents = try container.decodeIfPresent(UserEventStats.self, forKey: .userEvents)
        notes = try container.decodeIfPresent(String.self, forKey: .notes)
    }
}

enum AnalyticsStore {
    /// Loads analytics summary JSON produced by the Python/Rust backend.
    /// Returns nil when the file is missing; throws when the payload is malformed.
    static func loadSummary(from url: URL) throws -> AnalyticsSummary? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(AnalyticsSummary.self, from: data)
    }
}
