import Foundation

struct PaperTradingLens: Codable, Equatable {
    var title: String?
    var tradingTags: [String]?
    var assetClasses: [String]?
    var horizons: [String]?
    var signalArchetypes: [String]?
    var whereItFits: TradingLensWhereItFits?
    var alphaHypotheses: [TradingLensAlphaHypothesis]?
    var dataRequirements: TradingLensDataRequirements?
    var evaluationNotes: TradingLensEvaluationNotes?
    var riskFlags: [String]?
    var scores: TradingLensScores?
    var oneLineVerdict: String?

    enum CodingKeys: String, CodingKey {
        case title
        case tradingTags = "trading_tags"
        case assetClasses = "asset_classes"
        case horizons
        case signalArchetypes = "signal_archetypes"
        case whereItFits = "where_it_fits"
        case alphaHypotheses = "alpha_hypotheses"
        case dataRequirements = "data_requirements"
        case evaluationNotes = "evaluation_notes"
        case riskFlags = "risk_flags"
        case scores
        case oneLineVerdict = "one_line_verdict"
    }

    init(
        title: String? = nil,
        tradingTags: [String]? = nil,
        assetClasses: [String]? = nil,
        horizons: [String]? = nil,
        signalArchetypes: [String]? = nil,
        whereItFits: TradingLensWhereItFits? = nil,
        alphaHypotheses: [TradingLensAlphaHypothesis]? = nil,
        dataRequirements: TradingLensDataRequirements? = nil,
        evaluationNotes: TradingLensEvaluationNotes? = nil,
        riskFlags: [String]? = nil,
        scores: TradingLensScores? = nil,
        oneLineVerdict: String? = nil
    ) {
        self.title = title
        self.tradingTags = tradingTags
        self.assetClasses = assetClasses
        self.horizons = horizons
        self.signalArchetypes = signalArchetypes
        self.whereItFits = whereItFits
        self.alphaHypotheses = alphaHypotheses
        self.dataRequirements = dataRequirements
        self.evaluationNotes = evaluationNotes
        self.riskFlags = riskFlags
        self.scores = scores
        self.oneLineVerdict = oneLineVerdict
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        title = try container.decodeIfPresent(String.self, forKey: .title)
        tradingTags = container.decodeLossyStringArrayIfPresent(forKey: .tradingTags)
        assetClasses = container.decodeLossyStringArrayIfPresent(forKey: .assetClasses)
        horizons = container.decodeLossyStringArrayIfPresent(forKey: .horizons)
        signalArchetypes = container.decodeLossyStringArrayIfPresent(forKey: .signalArchetypes)
        do {
            whereItFits = try container.decodeIfPresent(TradingLensWhereItFits.self, forKey: .whereItFits)
        } catch {
            whereItFits = nil
        }

        do {
            alphaHypotheses = try container.decodeIfPresent([TradingLensAlphaHypothesis].self, forKey: .alphaHypotheses)
        } catch {
            do {
                let single = try container.decodeIfPresent(TradingLensAlphaHypothesis.self, forKey: .alphaHypotheses)
                alphaHypotheses = single.map { [$0] }
            } catch {
                alphaHypotheses = nil
            }
        }

        do {
            dataRequirements = try container.decodeIfPresent(TradingLensDataRequirements.self, forKey: .dataRequirements)
        } catch {
            dataRequirements = nil
        }

        do {
            evaluationNotes = try container.decodeIfPresent(TradingLensEvaluationNotes.self, forKey: .evaluationNotes)
        } catch {
            evaluationNotes = nil
        }

        riskFlags = container.decodeLossyStringArrayIfPresent(forKey: .riskFlags)
        do {
            scores = try container.decodeIfPresent(TradingLensScores.self, forKey: .scores)
        } catch {
            scores = nil
        }
        oneLineVerdict = try container.decodeIfPresent(String.self, forKey: .oneLineVerdict)
    }
}

struct TradingLensWhereItFits: Codable, Equatable {
    var pipelineStage: [String]?
    var primaryUse: String?

    enum CodingKeys: String, CodingKey {
        case pipelineStage = "pipeline_stage"
        case primaryUse = "primary_use"
    }

    init(pipelineStage: [String]? = nil, primaryUse: String? = nil) {
        self.pipelineStage = pipelineStage
        self.primaryUse = primaryUse
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        pipelineStage = container.decodeLossyStringArrayIfPresent(forKey: .pipelineStage)
        primaryUse = try container.decodeIfPresent(String.self, forKey: .primaryUse)
    }
}

struct TradingLensAlphaHypothesis: Codable, Equatable {
    var hypothesis: String?
    var features: [String]?
    var target: String?
    var horizon: String?

    enum CodingKeys: String, CodingKey {
        case hypothesis, features, target, horizon
    }

    init(hypothesis: String? = nil, features: [String]? = nil, target: String? = nil, horizon: String? = nil) {
        self.hypothesis = hypothesis
        self.features = features
        self.target = target
        self.horizon = horizon
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hypothesis = try container.decodeIfPresent(String.self, forKey: .hypothesis)
        features = container.decodeLossyStringArrayIfPresent(forKey: .features)
        target = try container.decodeIfPresent(String.self, forKey: .target)
        horizon = try container.decodeIfPresent(String.self, forKey: .horizon)
    }
}

struct TradingLensDataRequirements: Codable, Equatable {
    var mustHave: [String]?
    var niceToHave: [String]?

    enum CodingKeys: String, CodingKey {
        case mustHave = "must_have"
        case niceToHave = "nice_to_have"
    }

    init(mustHave: [String]? = nil, niceToHave: [String]? = nil) {
        self.mustHave = mustHave
        self.niceToHave = niceToHave
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        mustHave = container.decodeLossyStringArrayIfPresent(forKey: .mustHave)
        niceToHave = container.decodeLossyStringArrayIfPresent(forKey: .niceToHave)
    }
}

struct TradingLensEvaluationNotes: Codable, Equatable {
    var recommendedMetrics: [String]?
    var mustCheck: [String]?

    enum CodingKeys: String, CodingKey {
        case recommendedMetrics = "recommended_metrics"
        case mustCheck = "must_check"
    }

    init(recommendedMetrics: [String]? = nil, mustCheck: [String]? = nil) {
        self.recommendedMetrics = recommendedMetrics
        self.mustCheck = mustCheck
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        recommendedMetrics = container.decodeLossyStringArrayIfPresent(forKey: .recommendedMetrics)
        mustCheck = container.decodeLossyStringArrayIfPresent(forKey: .mustCheck)
    }
}

struct TradingLensScores: Codable, Equatable {
    var novelty: Double?
    var usability: Double?
    var strategyImpact: Double?
    var confidence: Double?

    enum CodingKeys: String, CodingKey {
        case novelty
        case usability
        case strategyImpact = "strategy_impact"
        case confidence
    }

    init(novelty: Double? = nil, usability: Double? = nil, strategyImpact: Double? = nil, confidence: Double? = nil) {
        self.novelty = novelty
        self.usability = usability
        self.strategyImpact = strategyImpact
        self.confidence = confidence
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        novelty = container.decodeLossyDoubleIfPresent(forKey: .novelty)
        usability = container.decodeLossyDoubleIfPresent(forKey: .usability)
        strategyImpact = container.decodeLossyDoubleIfPresent(forKey: .strategyImpact)
        confidence = container.decodeLossyDoubleIfPresent(forKey: .confidence)
    }
}

private extension KeyedDecodingContainer {
    func decodeLossyStringArrayIfPresent(forKey key: Key) -> [String]? {
        do {
            if let arr = try decodeIfPresent([String].self, forKey: key) {
                let cleaned = arr
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                return cleaned.isEmpty ? nil : cleaned
            }
        } catch {
            // fall through
        }

        do {
            if let str = try decodeIfPresent(String.self, forKey: key) {
                let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return nil }

                let parts = trimmed
                    .split(whereSeparator: { $0 == "\n" || $0 == "," || $0 == ";" })
                    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }

                if parts.count >= 2 {
                    return parts
                }
                return [trimmed]
            }
        } catch {
            // fall through
        }

        return nil
    }

    func decodeLossyDoubleIfPresent(forKey key: Key) -> Double? {
        do {
            if let d = try decodeIfPresent(Double.self, forKey: key) {
                return d
            }
        } catch {
            // fall through
        }

        do {
            if let str = try decodeIfPresent(String.self, forKey: key) {
                return str.extractFirstDouble()
            }
        } catch {
            // fall through
        }

        return nil
    }
}

private extension String {
    func extractFirstDouble() -> Double? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let pattern = #"-?\d+(?:\.\d+)?"#
        guard let range = trimmed.range(of: pattern, options: .regularExpression) else { return nil }
        return Double(trimmed[range])
    }
}
