import Foundation

struct AnalyticsSummary: Codable, Equatable {
    struct QualitySection: Codable, Equatable {
        struct MapQuality: Codable, Equatable {
            struct LocalDistortion: Codable, Equatable {
                let clusterID: Int
                let neighborOverlap: Double?
                let distortion: Double?

                enum CodingKeys: String, CodingKey {
                    case clusterID = "cluster_id"
                    case neighborOverlap = "neighbor_overlap"
                    case distortion
                }
            }

            let available: Bool?
            let reason: String?
            let galaxyVersion: String?
            let level: String?
            let k: Int?
            let trustworthiness: Double?
            let continuity: Double?
            let avgNeighborOverlap: Double?
            let localDistortion: [LocalDistortion]?

            enum CodingKeys: String, CodingKey {
                case available
                case reason
                case galaxyVersion = "galaxy_version"
                case level
                case k
                case trustworthiness
                case continuity
                case avgNeighborOverlap = "avg_neighbor_overlap"
                case localDistortion = "local_distortion"
            }
        }

        struct PaperMapQuality: Codable, Equatable {
            struct DistortionQuantiles: Codable, Equatable {
                let p50: Double?
                let p90: Double?
                let p95: Double?
                let p99: Double?
            }

            struct GridCell: Codable, Equatable {
                let xBin: Int
                let yBin: Int
                let count: Int
                let avgDistortion: Double?

                enum CodingKeys: String, CodingKey {
                    case xBin = "x_bin"
                    case yBin = "y_bin"
                    case count
                    case avgDistortion = "avg_distortion"
                }
            }

            let available: Bool?
            let reason: String?
            let level: String?
            let method: String?
            let k: Int?
            let trustworthiness: Double?
            let continuity: Double?
            let avgNeighborOverlap: Double?
            let distortionQuantiles: DistortionQuantiles?
            let gridBins: Int?
            let grid: [GridCell]?

            enum CodingKeys: String, CodingKey {
                case available
                case reason
                case level
                case method
                case k
                case trustworthiness
                case continuity
                case avgNeighborOverlap = "avg_neighbor_overlap"
                case distortionQuantiles = "distortion_quantiles"
                case gridBins = "grid_bins"
                case grid
            }
        }

        struct IngestionQuality: Codable, Equatable {
            struct Issue: Codable, Equatable {
                let paperID: UUID
                let flags: [String]
                let chunkTextLen: Int?
                let pageCount: Int?

                enum CodingKeys: String, CodingKey {
                    case paperID = "paper_id"
                    case flags
                    case chunkTextLen = "chunk_text_len"
                    case pageCount = "page_count"
                }
            }

            let available: Bool?
            let missingYear: Int?
            let missingTitle: Int?
            let missingChunks: Int?
            let lowTextYield: Int?
            let issues: [Issue]?
            let reason: String?

            enum CodingKeys: String, CodingKey {
                case available
                case missingYear = "missing_year"
                case missingTitle = "missing_title"
                case missingChunks = "missing_chunks"
                case lowTextYield = "low_text_yield"
                case issues
                case reason
            }
        }

        let map: MapQuality?
        let paperMap: PaperMapQuality?
        let ingestion: IngestionQuality?

        enum CodingKeys: String, CodingKey {
            case map
            case paperMap = "paper_map"
            case ingestion
        }
    }

    struct StabilitySection: Codable, Equatable {
        struct PaperStability: Codable, Equatable {
            let paperID: UUID
            let clusterConfidence: Double?
            let ambiguity: Double?
            let top1Cluster: Int?
            let top2Cluster: Int?
            let margin: Double?
            let distMargin: Double?

            enum CodingKeys: String, CodingKey {
                case paperID = "paper_id"
                case clusterConfidence = "cluster_confidence"
                case ambiguity
                case top1Cluster = "top1_cluster"
                case top2Cluster = "top2_cluster"
                case margin
                case distMargin = "dist_margin"
            }
        }

        struct ClusterStability: Codable, Equatable {
            let clusterID: Int
            let size: Int?
            let cohesion: Double?
            let avgConfidence: Double?
            let avgAmbiguity: Double?

            enum CodingKeys: String, CodingKey {
                case clusterID = "cluster_id"
                case size
                case cohesion
                case avgConfidence = "avg_confidence"
                case avgAmbiguity = "avg_ambiguity"
            }
        }

        let available: Bool?
        let reason: String?
        let trials: Int?
        let noise: Double?
        let clusters: [Int]?
        let perPaper: [PaperStability]?
        let perCluster: [ClusterStability]?

        enum CodingKeys: String, CodingKey {
            case available
            case reason
            case trials
            case noise
            case clusters
            case perPaper = "per_paper"
            case perCluster = "per_cluster"
        }
    }

    struct LifecycleSection: Codable, Equatable {
        struct ClusterLifecycle: Codable, Equatable {
            let clusterID: Int
            let phase: String?
            let burstYears: [Int]?
            let changepoints: [Int]?
            let latestYear: Int?
            let latestCount: Int?
            let latestBurstZ: Double?
            let latestWeightedBurstZ: Double?

            enum CodingKeys: String, CodingKey {
                case clusterID = "cluster_id"
                case phase
                case burstYears = "burst_years"
                case changepoints
                case latestYear = "latest_year"
                case latestCount = "latest_count"
                case latestBurstZ = "latest_burst_z"
                case latestWeightedBurstZ = "latest_weighted_burst_z"
            }
        }

        let available: Bool?
        let reason: String?
        let perCluster: [ClusterLifecycle]?

        enum CodingKeys: String, CodingKey {
            case available
            case reason
            case perCluster = "per_cluster"
        }
    }

    struct BridgesSection: Codable, Equatable {
        struct TopicGraph: Codable, Equatable {
            struct Edge: Codable, Equatable {
                let src: Int
                let dst: Int
                let weight: Double
            }

            let available: Bool?
            let reason: String?
            let nodes: [Int]?
            let edges: [Edge]?
        }

        struct TopicMetric: Codable, Equatable {
            let clusterID: Int
            let betweenness: Double?
            let clusteringCoeff: Double?
            let bridgingCentrality: Double?
            let degree: Int?
            let constraint: Double?
            let effectiveSize: Double?

            enum CodingKeys: String, CodingKey {
                case clusterID = "cluster_id"
                case betweenness
                case clusteringCoeff = "clustering_coeff"
                case bridgingCentrality = "bridging_centrality"
                case degree
                case constraint
                case effectiveSize = "effective_size"
            }
        }

        struct PaperRecombination: Codable, Equatable {
            struct PaperEntry: Codable, Equatable {
                let paperID: UUID
                let recombination: Double?

                enum CodingKeys: String, CodingKey {
                    case paperID = "paper_id"
                    case recombination
                }
            }

            let available: Bool?
            let reason: String?
            let k: Int?
            let perPaper: [PaperEntry]?

            enum CodingKeys: String, CodingKey {
                case available
                case reason
                case k
                case perPaper = "per_paper"
            }
        }

        let topicGraph: TopicGraph?
        let topicMetrics: [TopicMetric]?
        let paperRecombination: PaperRecombination?

        enum CodingKeys: String, CodingKey {
            case topicGraph = "topic_graph"
            case topicMetrics = "topic_metrics"
            case paperRecombination = "paper_recombination"
        }
    }

    struct CitationsSection: Codable, Equatable {
        struct Graph: Codable, Equatable {
            struct RankEntry: Codable, Equatable {
                let paperID: UUID
                let pagerank: Double?
                let inDegree: Int?
                let outDegree: Int?

                enum CodingKeys: String, CodingKey {
                    case paperID = "paper_id"
                    case pagerank
                    case inDegree = "in_degree"
                    case outDegree = "out_degree"
                }
            }

            let available: Bool?
            let reason: String?
            let edgeCount: Int?
            let pagerank: [RankEntry]?
            let topInDegree: [RankEntry]?

            enum CodingKeys: String, CodingKey {
                case available
                case reason
                case edgeCount = "edge_count"
                case pagerank
                case topInDegree = "top_in_degree"
            }
        }

        let refsExtracted: Int?
        let inCorpusCites: Int?
        let graph: Graph?

        enum CodingKeys: String, CodingKey {
            case refsExtracted = "refs_extracted"
            case inCorpusCites = "in_corpus_cites"
            case graph
        }
    }

    struct ClaimsSection: Codable, Equatable {
        struct ControversyCluster: Codable, Equatable {
            let clusterID: Int
            let contradictionRate: Double?
            let claimDiversity: Double?
            let consensusScore: Double?

            enum CodingKeys: String, CodingKey {
                case clusterID = "cluster_id"
                case contradictionRate = "contradiction_rate"
                case claimDiversity = "claim_diversity"
                case consensusScore = "consensus_score"
            }
        }

        struct MaturityCluster: Codable, Equatable {
            let clusterID: Int
            let avgMaturity: Double?
            let claimClusterCount: Int?

            enum CodingKeys: String, CodingKey {
                case clusterID = "cluster_id"
                case avgMaturity = "avg_maturity"
                case claimClusterCount = "claim_cluster_count"
            }
        }

        struct StressTestGaps: Codable, Equatable {
            struct ClusterEntry: Codable, Equatable {
                struct AssumptionCount: Codable, Equatable {
                    let assumption: String
                    let count: Int?
                }

                struct SuggestedTestCount: Codable, Equatable {
                    let text: String
                    let count: Int?
                }

                let clusterID: Int
                let eventCount: Int?
                let affectedPapers: Int?
                let affectedClaims: Int?
                let topAssumptions: [AssumptionCount]?
                let topSuggestedTests: [SuggestedTestCount]?

                enum CodingKeys: String, CodingKey {
                    case clusterID = "cluster_id"
                    case eventCount = "event_count"
                    case affectedPapers = "affected_papers"
                    case affectedClaims = "affected_claims"
                    case topAssumptions = "top_assumptions"
                    case topSuggestedTests = "top_suggested_tests"
                }
            }

            let available: Bool?
            let eventCount: Int?
            let perCluster: [ClusterEntry]?

            enum CodingKeys: String, CodingKey {
                case available
                case eventCount = "event_count"
                case perCluster = "per_cluster"
            }
        }

        let available: Bool?
        let controversyByCluster: [ControversyCluster]?
        let maturityByCluster: [MaturityCluster]?
        let stressTestGaps: StressTestGaps?

        enum CodingKeys: String, CodingKey {
            case available
            case controversyByCluster = "controversy_by_cluster"
            case maturityByCluster = "maturity_by_cluster"
            case stressTestGaps = "stress_test_gaps"
        }
    }

    struct WorkflowSection: Codable, Equatable {
        struct Coverage: Codable, Equatable {
            struct Blindspot: Codable, Equatable {
                let clusterID: Int
                let interestSimilarity: Double?
                let coverage: Double?
                let blindspotScore: Double?

                enum CodingKeys: String, CodingKey {
                    case clusterID = "cluster_id"
                    case interestSimilarity = "interest_similarity"
                    case coverage
                    case blindspotScore = "blindspot_score"
                }
            }

            struct ClusterCoverage: Codable, Equatable {
                let clusterID: Int
                let coverage: Double?
                let paperCount: Int?

                enum CodingKeys: String, CodingKey {
                    case clusterID = "cluster_id"
                    case coverage
                    case paperCount = "paper_count"
                }
            }

            let available: Bool?
            let clusterCoverage: [ClusterCoverage]?
            let blindspots: [Blindspot]?

            enum CodingKeys: String, CodingKey {
                case available
                case clusterCoverage = "cluster_coverage"
                case blindspots
            }
        }

        struct QAGaps: Codable, Equatable {
            struct Question: Codable, Equatable {
                let question: String
                let asked: Int?
                let answered: Int?
                let topScore: Double?
                let margin: Double?
                let supportBreadth: Double?
                let unanswered: Bool?

                enum CodingKeys: String, CodingKey {
                    case question
                    case asked
                    case answered
                    case topScore = "top_score"
                    case margin
                    case supportBreadth = "support_breadth"
                    case unanswered
                }
            }

            let available: Bool?
            let questions: [Question]?
        }

        struct MIGRecommendations: Codable, Equatable {
            struct Entry: Codable, Equatable {
                let paperID: UUID
                let marginalGain: Double?

                enum CodingKeys: String, CodingKey {
                    case paperID = "paper_id"
                    case marginalGain = "marginal_gain"
                }
            }

            let available: Bool?
            let selected: [Entry]?
        }

        let coverage: Coverage?
        let qaGaps: QAGaps?
        let recommendationsMIG: MIGRecommendations?

        enum CodingKeys: String, CodingKey {
            case coverage
            case qaGaps = "qa_gaps"
            case recommendationsMIG = "recommendations_mig"
        }
    }

    struct HygieneSection: Codable, Equatable {
        struct Duplicates: Codable, Equatable {
            let available: Bool?
            let groups: [[UUID]]?
        }

        let duplicates: Duplicates?
    }

    struct TradingSection: Codable, Equatable {
        struct TagCount: Codable, Equatable {
            let tag: String
            let count: Int
        }

        struct AssetClassCount: Codable, Equatable {
            let assetClass: String
            let count: Int

            enum CodingKeys: String, CodingKey {
                case assetClass = "asset_class"
                case count
            }
        }

        struct HorizonCount: Codable, Equatable {
            let horizon: String
            let count: Int
        }

        struct RiskFlagCount: Codable, Equatable {
            let riskFlag: String
            let count: Int

            enum CodingKeys: String, CodingKey {
                case riskFlag = "risk_flag"
                case count
            }
        }

        struct TagTrend: Codable, Equatable {
            let tag: String
            let year: Int
            let count: Int
        }

        struct ScorePoint: Codable, Equatable {
            let paperID: UUID
            let year: Int?
            let clusterID: Int?
            let novelty: Double
            let usability: Double
            let strategyImpact: Double?
            let confidence: Double?
            let priority: Double?
            let primaryTag: String?
            let primaryAssetClass: String?
            let primaryHorizon: String?
            let oneLineVerdict: String?
            let hasStrategyBlueprint: Bool?
            let hasBacktestAudit: Bool?

            enum CodingKeys: String, CodingKey {
                case paperID = "paper_id"
                case year
                case clusterID = "cluster_id"
                case novelty
                case usability
                case strategyImpact = "strategy_impact"
                case confidence
                case priority
                case primaryTag = "primary_tag"
                case primaryAssetClass = "primary_asset_class"
                case primaryHorizon = "primary_horizon"
                case oneLineVerdict = "one_line_verdict"
                case hasStrategyBlueprint = "has_strategy_blueprint"
                case hasBacktestAudit = "has_backtest_audit"
            }
        }

        struct PriorityEntry: Codable, Equatable {
            let paperID: UUID
            let priority: Double?
            let oneLineVerdict: String?

            enum CodingKeys: String, CodingKey {
                case paperID = "paper_id"
                case priority
                case oneLineVerdict = "one_line_verdict"
            }
        }

        let available: Bool?
        let reason: String?
        let paperCountWithLens: Int?
        let coveragePct: Double?
        let tagCounts: [TagCount]?
        let assetClassCounts: [AssetClassCount]?
        let horizonCounts: [HorizonCount]?
        let riskFlagCounts: [RiskFlagCount]?
        let tagTrends: [TagTrend]?
        let scorePoints: [ScorePoint]?
        let topPriority: [PriorityEntry]?

        enum CodingKeys: String, CodingKey {
            case available
            case reason
            case paperCountWithLens = "paper_count_with_lens"
            case coveragePct = "coverage_pct"
            case tagCounts = "tag_counts"
            case assetClassCounts = "asset_class_counts"
            case horizonCounts = "horizon_counts"
            case riskFlagCounts = "risk_flag_counts"
            case tagTrends = "tag_trends"
            case scorePoints = "score_points"
            case topPriority = "top_priority"
        }
    }

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
        let clusterConfidence: Double?
        let clusterAmbiguity: Double?
        let clusterTop1: Int?
        let clusterTop2: Int?
        let clusterMargin: Double?
        let clusterDistMargin: Double?
        let recombination: Double?
        let rigorProxy: Double?
        let opennessScore: Double?
        let hasCodeLink: Bool?
        let hasDataLink: Bool?
        let citationPagerank: Double?
        let citationInDegree: Int?
        let readScore: Double?
        let layoutDistortion: Double?
        let paperLayoutDistortion: Double?
        let duplicateGroup: Int?
        let ingestionFlags: [String]?

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
            case clusterConfidence = "cluster_confidence"
            case clusterAmbiguity = "cluster_ambiguity"
            case clusterTop1 = "cluster_top1"
            case clusterTop2 = "cluster_top2"
            case clusterMargin = "cluster_margin"
            case clusterDistMargin = "cluster_dist_margin"
            case recombination
            case rigorProxy = "rigor_proxy"
            case opennessScore = "openness_score"
            case hasCodeLink = "has_code_link"
            case hasDataLink = "has_data_link"
            case citationPagerank = "citation_pagerank"
            case citationInDegree = "citation_in_degree"
            case readScore = "read_score"
            case layoutDistortion = "layout_distortion"
            case paperLayoutDistortion = "paper_layout_distortion"
            case duplicateGroup = "duplicate_group"
            case ingestionFlags = "ingestion_flags"
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
    let recommendationsSimple: [UUID]
    let answerConfidence: Double?
    let counterfactuals: [Counterfactual]
    let userEvents: UserEventStats?
    let notes: String?
    let quality: QualitySection?
    let stability: StabilitySection?
    let lifecycle: LifecycleSection?
    let bridges: BridgesSection?
    let citations: CitationsSection?
    let claims: ClaimsSection?
    let trading: TradingSection?
    let workflow: WorkflowSection?
    let hygiene: HygieneSection?

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
        case recommendationsSimple = "recommendations_simple"
        case answerConfidence = "answer_confidence"
        case counterfactuals
        case userEvents = "user_events"
        case notes
        case quality
        case stability
        case lifecycle
        case bridges
        case citations
        case claims
        case trading
        case workflow
        case hygiene
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
        recommendationsSimple = try container.decodeIfPresent([UUID].self, forKey: .recommendationsSimple) ?? []
        answerConfidence = try container.decodeIfPresent(Double.self, forKey: .answerConfidence)
        counterfactuals = try container.decodeIfPresent([Counterfactual].self, forKey: .counterfactuals) ?? []
        userEvents = try container.decodeIfPresent(UserEventStats.self, forKey: .userEvents)
        notes = try container.decodeIfPresent(String.self, forKey: .notes)
        quality = try container.decodeIfPresent(QualitySection.self, forKey: .quality)
        stability = try container.decodeIfPresent(StabilitySection.self, forKey: .stability)
        lifecycle = try container.decodeIfPresent(LifecycleSection.self, forKey: .lifecycle)
        bridges = try container.decodeIfPresent(BridgesSection.self, forKey: .bridges)
        citations = try container.decodeIfPresent(CitationsSection.self, forKey: .citations)
        claims = try container.decodeIfPresent(ClaimsSection.self, forKey: .claims)
        trading = try container.decodeIfPresent(TradingSection.self, forKey: .trading)
        workflow = try container.decodeIfPresent(WorkflowSection.self, forKey: .workflow)
        hygiene = try container.decodeIfPresent(HygieneSection.self, forKey: .hygiene)
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
