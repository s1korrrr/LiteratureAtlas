import Foundation

struct StrategyProject: Identifiable, Codable, Equatable {
    // Order here drives JSON field order (encoder keeps declaration order).
    var version: Int
    var id: UUID
    var title: String
    var createdAt: Date
    var updatedAt: Date
    var paperIDs: [UUID]
    var idea: StrategyIdea?
    var features: [QuantFeature]
    var model: QuantModel?
    var tradePlan: QuantTradePlan?
    var decisions: [QuantDecision]
    var outcomes: [QuantOutcome]
    var feedback: [QuantFeedback]
    var codeRefs: [CodeReference]?
    var tags: [String]?
    var archived: Bool?

    enum CodingKeys: String, CodingKey {
        case version
        case id
        case title
        case createdAt
        case updatedAt
        case paperIDs = "paper_ids"
        case idea
        case features
        case model
        case tradePlan = "trade_plan"
        case decisions
        case outcomes
        case feedback
        case codeRefs = "code_refs"
        case tags
        case archived
    }

    init(
        version: Int = 1,
        id: UUID = UUID(),
        title: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        paperIDs: [UUID] = [],
        idea: StrategyIdea? = nil,
        features: [QuantFeature] = [],
        model: QuantModel? = nil,
        tradePlan: QuantTradePlan? = nil,
        decisions: [QuantDecision] = [],
        outcomes: [QuantOutcome] = [],
        feedback: [QuantFeedback] = [],
        codeRefs: [CodeReference]? = nil,
        tags: [String]? = nil,
        archived: Bool? = nil
    ) {
        self.version = version
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.paperIDs = paperIDs
        self.idea = idea
        self.features = features
        self.model = model
        self.tradePlan = tradePlan
        self.decisions = decisions
        self.outcomes = outcomes
        self.feedback = feedback
        self.codeRefs = codeRefs
        self.tags = tags
        self.archived = archived
    }
}

struct StrategyIdea: Codable, Equatable {
    var text: String
    var hypotheses: [String]?
    var assumptions: [String]?
}

struct CodeReference: Identifiable, Codable, Equatable {
    var id: UUID
    var path: String
    var symbol: String?
    var note: String?

    init(id: UUID = UUID(), path: String, symbol: String? = nil, note: String? = nil) {
        self.id = id
        self.path = path
        self.symbol = symbol
        self.note = note
    }
}

struct QuantFeature: Identifiable, Codable, Equatable {
    var id: UUID
    var name: String
    var description: String?
    var dataSources: [String]?
    var transforms: [String]?
    var codeRefs: [CodeReference]?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case dataSources = "data_sources"
        case transforms
        case codeRefs = "code_refs"
    }

    init(
        id: UUID = UUID(),
        name: String,
        description: String? = nil,
        dataSources: [String]? = nil,
        transforms: [String]? = nil,
        codeRefs: [CodeReference]? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.dataSources = dataSources
        self.transforms = transforms
        self.codeRefs = codeRefs
    }
}

struct QuantModel: Identifiable, Codable, Equatable {
    var id: UUID
    var name: String
    var description: String?
    var featureIDs: [UUID]?
    var hyperparameters: [String: String]?
    var codeRefs: [CodeReference]?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case featureIDs = "feature_ids"
        case hyperparameters
        case codeRefs = "code_refs"
    }

    init(
        id: UUID = UUID(),
        name: String,
        description: String? = nil,
        featureIDs: [UUID]? = nil,
        hyperparameters: [String: String]? = nil,
        codeRefs: [CodeReference]? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.featureIDs = featureIDs
        self.hyperparameters = hyperparameters
        self.codeRefs = codeRefs
    }
}

struct QuantTradePlan: Identifiable, Codable, Equatable {
    var id: UUID
    var universe: String?
    var horizon: String?
    var signalDefinition: String?
    var portfolioConstruction: String?
    var costsAndSlippage: String?
    var constraints: String?
    var executionNotes: String?
    var codeRefs: [CodeReference]?

    enum CodingKeys: String, CodingKey {
        case id
        case universe
        case horizon
        case signalDefinition = "signal_definition"
        case portfolioConstruction = "portfolio_construction"
        case costsAndSlippage = "costs_and_slippage"
        case constraints
        case executionNotes = "execution_notes"
        case codeRefs = "code_refs"
    }

    init(
        id: UUID = UUID(),
        universe: String? = nil,
        horizon: String? = nil,
        signalDefinition: String? = nil,
        portfolioConstruction: String? = nil,
        costsAndSlippage: String? = nil,
        constraints: String? = nil,
        executionNotes: String? = nil,
        codeRefs: [CodeReference]? = nil
    ) {
        self.id = id
        self.universe = universe
        self.horizon = horizon
        self.signalDefinition = signalDefinition
        self.portfolioConstruction = portfolioConstruction
        self.costsAndSlippage = costsAndSlippage
        self.constraints = constraints
        self.executionNotes = executionNotes
        self.codeRefs = codeRefs
    }
}

enum QuantDecisionKind: String, Codable, CaseIterable, Equatable {
    case build
    case backtest
    case trade
    case iterate
    case ship
    case shelve
    case kill

    var label: String {
        switch self {
        case .build: return "Build"
        case .backtest: return "Backtest"
        case .trade: return "Trade"
        case .iterate: return "Iterate"
        case .ship: return "Ship"
        case .shelve: return "Shelve"
        case .kill: return "Kill"
        }
    }
}

struct QuantDecision: Identifiable, Codable, Equatable {
    var id: UUID
    var madeAt: Date
    var kind: QuantDecisionKind
    var rationale: String
    var relatedFeatureIDs: [UUID]?
    var relatedModelID: UUID?
    var expectedOutcome: String?

    enum CodingKeys: String, CodingKey {
        case id
        case madeAt
        case kind
        case rationale
        case relatedFeatureIDs = "related_feature_ids"
        case relatedModelID = "related_model_id"
        case expectedOutcome = "expected_outcome"
    }

    init(
        id: UUID = UUID(),
        madeAt: Date = Date(),
        kind: QuantDecisionKind,
        rationale: String,
        relatedFeatureIDs: [UUID]? = nil,
        relatedModelID: UUID? = nil,
        expectedOutcome: String? = nil
    ) {
        self.id = id
        self.madeAt = madeAt
        self.kind = kind
        self.rationale = rationale
        self.relatedFeatureIDs = relatedFeatureIDs
        self.relatedModelID = relatedModelID
        self.expectedOutcome = expectedOutcome
    }
}

enum QuantOutcomeKind: String, Codable, CaseIterable, Equatable {
    case backtest
    case paperTrade
    case live
    case analysis

    var label: String {
        switch self {
        case .backtest: return "Backtest"
        case .paperTrade: return "Paper trade"
        case .live: return "Live"
        case .analysis: return "Analysis"
        }
    }
}

struct QuantBacktestMetrics: Codable, Equatable {
    var pnl: Double?
    var sharpe: Double?
    var cagr: Double?
    var maxDrawdown: Double?
    var turnover: Double?
    var hitRate: Double?
}

struct QuantTimeSeriesPoint: Codable, Equatable {
    var t: Double
    var v: Double
}

struct QuantPnLSeries: Codable, Equatable {
    var kind: String
    var points: [QuantTimeSeriesPoint]
}

struct QuantOutcome: Identifiable, Codable, Equatable {
    var id: UUID
    var measuredAt: Date
    var kind: QuantOutcomeKind
    var metrics: QuantBacktestMetrics?
    var pnlSeries: QuantPnLSeries?
    var artifactPaths: [String]?
    var notes: String?

    enum CodingKeys: String, CodingKey {
        case id
        case measuredAt
        case kind
        case metrics
        case pnlSeries = "pnl_series"
        case artifactPaths = "artifact_paths"
        case notes
    }

    init(
        id: UUID = UUID(),
        measuredAt: Date = Date(),
        kind: QuantOutcomeKind,
        metrics: QuantBacktestMetrics? = nil,
        pnlSeries: QuantPnLSeries? = nil,
        artifactPaths: [String]? = nil,
        notes: String? = nil
    ) {
        self.id = id
        self.measuredAt = measuredAt
        self.kind = kind
        self.metrics = metrics
        self.pnlSeries = pnlSeries
        self.artifactPaths = artifactPaths
        self.notes = notes
    }
}

struct QuantFeedback: Identifiable, Codable, Equatable {
    var id: UUID
    var at: Date
    var text: String
    var sentiment: String?

    init(id: UUID = UUID(), at: Date = Date(), text: String, sentiment: String? = nil) {
        self.id = id
        self.at = at
        self.text = text
        self.sentiment = sentiment
    }
}
