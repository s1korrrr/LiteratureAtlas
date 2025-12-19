import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class PaperMarkdownExporterTests: XCTestCase {

    func testWriteCreatesMarkdownAndPreservesNotesTail() throws {
        let tmpRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpRoot, withIntermediateDirectories: true)

        let paperID = UUID()
        var paper = Paper(
            version: 1,
            filePath: "/tmp/sample.pdf",
            id: paperID,
            originalFilename: "sample.pdf",
            title: "Test Paper: A/B",
            introSummary: "Intro bullets",
            summary: "Summary bullets",
            methodSummary: "Method bullets",
            resultsSummary: "Results bullets",
            takeaways: ["- T1", "* T2", "3) T3"],
            keywords: ["k1", "k2"],
            userNotes: "My in-app notes",
            userTags: ["important", "tag2"],
            readingStatus: .done,
            noteEmbedding: nil,
            userQuestions: ["Q1", "Q2"],
            flashcards: [
                Flashcard(id: UUID(), question: "Q?", answer: "A.", lastReviewedAt: nil, reviewCount: nil)
            ],
            year: 2021,
            embedding: [1, 0],
            clusterIndex: 2
        )

        paper.tradingLens = PaperTradingLens(
            title: nil,
            tradingTags: ["PURE_ALPHA"],
            assetClasses: ["EQUITIES"],
            horizons: ["DAILY"],
            signalArchetypes: ["TIME_SERIES_FORECAST"],
            whereItFits: nil,
            alphaHypotheses: nil,
            dataRequirements: nil,
            evaluationNotes: nil,
            riskFlags: ["LEAKAGE_RISK"],
            scores: TradingLensScores(novelty: 7, usability: 8, strategyImpact: 9, confidence: 0.5),
            oneLineVerdict: "Strong idea"
        )
        paper.strategyBlueprint = """
        # Prototype 1
        ## Alpha hypothesis
        - Use embeddings to detect regime shifts.

        ```python
        # inside code fence
        ```
        """
        paper.backtestAudit = """
        # Leakage & Bias Risks
        - Watch for lookahead in feature windows.
        """

        let url = try PaperMarkdownExporter.write(paper: paper, outputRoot: tmpRoot)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        let original = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(original.contains("<!-- atlas:begin -->"))
        XCTAssertTrue(original.contains("<!-- atlas:end -->"))
        XCTAssertTrue(original.contains("## Notes"))
        XCTAssertTrue(original.contains(paperID.uuidString))
        XCTAssertTrue(original.contains("obsidian_format_version: 2"))
        XCTAssertTrue(original.contains("aliases:"))
        XCTAssertTrue(original.contains("cssclass: atlas-paper"))

        XCTAssertTrue(original.contains("> [!info] Meta"))
        XCTAssertTrue(original.contains("file:///tmp/sample.pdf"))
        XCTAssertTrue(original.contains("- Status: Done"))

        XCTAssertTrue(original.contains("> [!summary] Summary"))
        XCTAssertTrue(original.contains("Summary bullets"))

        XCTAssertTrue(original.contains("> [!abstract] Takeaways"))
        XCTAssertTrue(original.contains("> - T1"))
        XCTAssertTrue(original.contains("> - T2"))
        XCTAssertTrue(original.contains("> - T3"))
        XCTAssertFalse(original.contains("- -"))

        XCTAssertTrue(original.contains("> [!tip] Trading Lens"))
        XCTAssertTrue(original.contains("> | Field | Value |"))
        XCTAssertTrue(original.contains("novelty=7.0"))
        XCTAssertTrue(original.contains("confidence=0.50"))

        XCTAssertTrue(original.contains("> [!todo]- Strategy Blueprint"))
        XCTAssertTrue(original.contains("### Prototype 1"))
        XCTAssertTrue(original.contains("#### Alpha hypothesis"))
        XCTAssertTrue(original.contains("```python"))
        XCTAssertTrue(original.contains("# inside code fence"))

        XCTAssertTrue(original.contains("> [!warning]- Backtest Audit"))
        XCTAssertTrue(original.contains("### Leakage & Bias Risks"))

        // User edits after the managed block should survive future writes.
        let userTail = "\nCustom user line\n"
        try (original + userTail).write(to: url, atomically: true, encoding: .utf8)

        paper.summary = "Updated Summary bullets"
        _ = try PaperMarkdownExporter.write(paper: paper, outputRoot: tmpRoot)

        let updated = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(updated.contains("Updated Summary bullets"))
        XCTAssertTrue(updated.contains("Custom user line"))
    }
}
