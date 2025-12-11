import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class ClaimGraphTests: XCTestCase {

    func testHeuristicClaimExtractionFindsAssumptionsAndEvaluation() throws {
        let paperID = UUID()
        let summary = """
        The paper shows that a reinforcement learning market maker improves execution cost under Poisson arrivals and risk-neutral traders.
        Using the LOBSTER dataset (2019-2020) the authors report Sharpe ratio and PnL improvements.
        """

        let extraction = ClaimExtractor.heuristicExtraction(summary: summary, paperID: paperID, year: 2020)

        XCTAssertFalse(extraction.claims.isEmpty, "At least one claim should be produced")
        XCTAssertTrue(extraction.assumptions.contains(where: { $0.lowercased().contains("poisson arrivals") }))
        XCTAssertTrue(extraction.assumptions.contains(where: { $0.lowercased().contains("risk-neutral") }))

        XCTAssertEqual(extraction.evaluation?.dataset, "LOBSTER")
        XCTAssertEqual(extraction.evaluation?.period, "2019-2020")
        XCTAssertTrue(extraction.evaluation?.metrics.contains("Sharpe ratio") ?? false)
        XCTAssertTrue(extraction.evaluation?.metrics.contains(where: { $0.lowercased().contains("pnl") }) ?? false)
    }

    func testRelationInferenceDetectsContradictionAndExtension() {
        let a = PaperClaim(
            id: UUID(),
            paperID: UUID(),
            statement: "Model A improves accuracy on CIFAR-10 by 3% compared to baseline.",
            assumptions: ["iid"],
            evaluation: EvaluationContext(dataset: "CIFAR-10", period: nil, metrics: ["accuracy"]),
            year: 2021,
            strength: 0.5
        )

        let b = PaperClaim(
            id: UUID(),
            paperID: UUID(),
            statement: "Model A does not improve accuracy on CIFAR-10 when labels are corrupted by 40% noise.",
            assumptions: ["label noise"],
            evaluation: EvaluationContext(dataset: "CIFAR-10", period: nil, metrics: ["accuracy"]),
            year: 2022,
            strength: 0.4
        )

        let c = PaperClaim(
            id: UUID(),
            paperID: UUID(),
            statement: "Model B extends Model A with adversarial training to further improve CIFAR-10 accuracy.",
            assumptions: ["iid"],
            evaluation: EvaluationContext(dataset: "CIFAR-10", period: nil, metrics: ["accuracy"]),
            year: 2023,
            strength: 0.6
        )

        let edges = ClaimRelationInferencer.inferEdges(for: [a, b, c])

        let contradiction = edges.first(where: { $0.kind == .contradicts && $0.sourceClaimID == a.id && $0.targetClaimID == b.id })
        XCTAssertNotNil(contradiction, "a should contradict b")

        let extensionEdge = edges.first(where: { $0.kind == .extends && $0.sourceClaimID == a.id && $0.targetClaimID == c.id })
        XCTAssertNotNil(extensionEdge, "a should extend to c")
    }

    func testAssumptionStressTestReturnsDependentClaims() {
        let assumption = "infinite liquidity"
        let paperA = Paper(version: 1, filePath: "a", id: UUID(), originalFilename: "a.pdf", title: "A", introSummary: nil, summary: "", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [1, 0], clusterIndex: nil, claims: [
            PaperClaim(id: UUID(), paperID: UUID(), statement: "Strategy relies on infinite liquidity to hedge inventory instantly.", assumptions: [assumption], evaluation: nil, year: 2021, strength: 0.7)
        ], assumptions: [assumption], evaluationContext: nil, methodPipeline: nil)

        let paperB = Paper(version: 1, filePath: "b", id: UUID(), originalFilename: "b.pdf", title: "B", introSummary: nil, summary: "", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2020, embedding: [0, 1], clusterIndex: nil, claims: [
            PaperClaim(id: UUID(), paperID: UUID(), statement: "Approach remains stable without assuming infinite liquidity.", assumptions: ["bounded inventory"], evaluation: nil, year: 2020, strength: 0.9)
        ], assumptions: ["bounded inventory"], evaluationContext: nil, methodPipeline: nil)

        let report = AssumptionStressTester.stressTest(assumption: assumption, papers: [paperA, paperB])

        XCTAssertEqual(report.affectedClaims.count, 1)
        XCTAssertTrue(report.narrative.lowercased().contains("infinite liquidity"))
        XCTAssertTrue(report.narrative.contains("A"))
    }

    func testMethodPipelineExtractionAndBlueprint() {
        let text = "We take limit order book trades, normalize volumes, train a DQN agent with an inventory penalty, and evaluate using Sharpe ratio and hit rate."
        let pipeline = MethodPipelineExtractor.extract(from: text)

        XCTAssertEqual(pipeline.steps.count, 5)
        XCTAssertEqual(pipeline.steps.map { $0.stage }, [.data, .preprocessing, .model, .objective, .evaluation])
        XCTAssertTrue(pipeline.steps[2].label.contains("DQN"))
        XCTAssertTrue(pipeline.steps[4].detail?.contains("Sharpe") ?? false)

        let code = BlueprintGenerator.generate(for: pipeline, title: "RL market making")
        XCTAssertTrue(code.contains("DQN"))
        XCTAssertTrue(code.lowercased().contains("sharpe"))
        XCTAssertTrue(code.lowercased().contains("step 5"))
    }

    func testAppModelBuildsClaimEdges() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let baseID = UUID()
        let claim1 = PaperClaim(id: UUID(), paperID: baseID, statement: "Improves accuracy on CIFAR-10", assumptions: ["iid"], evaluation: EvaluationContext(dataset: "CIFAR-10", period: nil, metrics: ["accuracy"]), year: 2021, strength: 0.5)
        let claim2 = PaperClaim(id: UUID(), paperID: baseID, statement: "Does not improve accuracy on CIFAR-10 when labels are corrupted", assumptions: ["label noise"], evaluation: EvaluationContext(dataset: "CIFAR-10", period: nil, metrics: ["accuracy"]), year: 2022, strength: 0.5)
        let paper = Paper(version: 1, filePath: "a", id: baseID, originalFilename: "a.pdf", title: "A", introSummary: nil, summary: "", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2022, embedding: [1, 0], clusterIndex: nil, claims: [claim1, claim2], assumptions: ["iid"], evaluationContext: nil, methodPipeline: nil)
        await MainActor.run { model.papers = [paper] }

        let edges = await MainActor.run { model.claimGraphEdges() }
        XCTAssertTrue(edges.contains(where: { $0.kind == .contradicts }))
    }

    func testAppModelStressReportUsesAssumptions() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let assumption = "bounded inventory"
        let paper = Paper(version: 1, filePath: "a", id: UUID(), originalFilename: "a.pdf", title: "A", introSummary: nil, summary: "", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [1, 0], clusterIndex: nil, claims: nil, assumptions: [assumption], evaluationContext: nil, methodPipeline: nil)
        await MainActor.run { model.papers = [paper] }

        let report = await MainActor.run { model.assumptionStressReport(for: assumption) }
        XCTAssertEqual(report.affectedClaims.count, 1)
        XCTAssertTrue(report.narrative.contains("A"))
    }

    func testAppModelBlueprintGeneration() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let pipeline = MethodPipeline(steps: [
            PipelineStep(stage: .data, label: "Input data", detail: nil),
            PipelineStep(stage: .model, label: "DQN agent", detail: nil),
            PipelineStep(stage: .evaluation, label: "Sharpe", detail: nil)
        ])
        let paperID = UUID()
        let paper = Paper(version: 1, filePath: "a", id: paperID, originalFilename: "a.pdf", title: "A", introSummary: nil, summary: "", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [1, 0], clusterIndex: nil, claims: nil, assumptions: nil, evaluationContext: nil, methodPipeline: pipeline)
        await MainActor.run { model.papers = [paper] }

        let blueprint = await MainActor.run { model.methodBlueprint(for: paperID) }
        XCTAssertNotNil(blueprint)
        XCTAssertTrue(blueprint?.contains("DQN") ?? false)
    }
}
