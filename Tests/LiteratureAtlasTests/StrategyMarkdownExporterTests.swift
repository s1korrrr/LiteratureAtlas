import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class StrategyMarkdownExporterTests: XCTestCase {

    func testWriteCreatesMarkdownAndPreservesNotesTail() throws {
        let tmpRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpRoot, withIntermediateDirectories: true)

        let paperID = UUID()
        let projectID = UUID()

        var project = StrategyProject(
            version: 1,
            id: projectID,
            title: "Test Strategy: A/B",
            createdAt: Date(timeIntervalSince1970: 0),
            updatedAt: Date(timeIntervalSince1970: 10),
            paperIDs: [paperID],
            idea: StrategyIdea(text: "Original idea", hypotheses: ["H1"], assumptions: ["A1"]),
            features: [
                QuantFeature(name: "Feature 1", description: "Desc 1"),
                QuantFeature(name: "Feature 2", description: nil),
            ],
            model: QuantModel(name: "Model 1", description: "M desc"),
            tradePlan: QuantTradePlan(
                universe: "SPY",
                horizon: "Daily",
                signalDefinition: "Signal",
                portfolioConstruction: "Portfolio",
                costsAndSlippage: "1bp",
                constraints: "Constraints",
                executionNotes: "Execution"
            ),
            decisions: [
                QuantDecision(madeAt: Date(timeIntervalSince1970: 20), kind: .backtest, rationale: "Run backtest"),
            ],
            outcomes: [
                QuantOutcome(
                    measuredAt: Date(timeIntervalSince1970: 30),
                    kind: .backtest,
                    metrics: QuantBacktestMetrics(pnl: 1.2, sharpe: 2.3, cagr: nil, maxDrawdown: 0.1, turnover: nil, hitRate: nil),
                    pnlSeries: nil,
                    artifactPaths: nil,
                    notes: "Outcome notes"
                ),
            ],
            feedback: [
                QuantFeedback(at: Date(timeIntervalSince1970: 40), text: "Feedback note", sentiment: nil),
            ],
            tags: ["tag1"],
            archived: false
        )

        let url = try StrategyMarkdownExporter.write(
            project: project,
            outputRoot: tmpRoot,
            paperTitlesByID: [paperID: "Paper Title"]
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        let original = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(original.contains("<!-- atlas:begin -->"))
        XCTAssertTrue(original.contains("<!-- atlas:end -->"))
        XCTAssertTrue(original.contains(projectID.uuidString))
        XCTAssertTrue(original.contains("## Papers"))
        XCTAssertTrue(original.contains("Paper Title"))
        XCTAssertTrue(original.contains("## Idea"))
        XCTAssertTrue(original.contains("Original idea"))
        XCTAssertTrue(original.contains("## Trade Plan"))
        XCTAssertTrue(original.contains("Universe: SPY"))
        XCTAssertTrue(original.contains("## Outcomes"))
        XCTAssertTrue(original.contains("Sharpe 2.300"))

        // User edits after the managed block should survive future writes.
        let userTail = "\nCustom user line\n"
        try (original + userTail).write(to: url, atomically: true, encoding: .utf8)

        project.idea = StrategyIdea(text: "Updated idea", hypotheses: nil, assumptions: nil)
        _ = try StrategyMarkdownExporter.write(project: project, outputRoot: tmpRoot, paperTitlesByID: [paperID: "Paper Title"])

        let updated = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(updated.contains("Updated idea"))
        XCTAssertTrue(updated.contains("Custom user line"))
    }
}

