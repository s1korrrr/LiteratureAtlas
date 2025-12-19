import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class ModelJSONTests: XCTestCase {
    func testExtractFirstJSONFindsObjectInPreamble() throws {
        let text = """
        Sure — here's the payload:
        { "a": 1, "b": [2, 3] }
        Thanks.
        """

        let extracted = try XCTUnwrap(ModelJSON.extractFirstJSON(from: text))
        let data = try XCTUnwrap(extracted.data(using: .utf8))
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])

        let a = (obj["a"] as? NSNumber)?.intValue
        let bAny = obj["b"] as? [Any]
        let b = bAny?.compactMap { ($0 as? NSNumber)?.intValue } ?? []

        XCTAssertEqual(a, 1)
        XCTAssertEqual(b, [2, 3])
    }

    func testDecodeFirstJSONHandlesCodeFenceSmartQuotesAndTrailingCommas() throws {
        let text = """
        Here you go:
        ```json
        {“title”: “Foo”, “trading_tags”: [“PURE_ALPHA”,], “scores”: {“novelty”: 7, “usability”: 8, “strategy_impact”: 9, “confidence”: 0.5,},}
        ```
        """

        let lens = try ModelJSON.decodeFirstJSON(PaperTradingLens.self, from: text)

        XCTAssertEqual(lens.title, "Foo")
        XCTAssertEqual(lens.tradingTags ?? [], ["PURE_ALPHA"])
        XCTAssertEqual(lens.scores?.novelty, 7)
        XCTAssertEqual(lens.scores?.usability, 8)
        XCTAssertEqual(lens.scores?.strategyImpact, 9)
        XCTAssertEqual(lens.scores?.confidence, 0.5)
    }

    func testDecodeFirstJSONThrowsWhenMissingJSON() {
        XCTAssertThrowsError(try ModelJSON.decodeFirstJSON(PaperTradingLens.self, from: "no json here"))
    }

    func testDecodeFirstJSONIsTolerantOfCommonSchemaMismatches() throws {
        let text = """
        {
          "title": "Foo",
          "trading_tags": "PURE_ALPHA, MODELING",
          "where_it_fits": { "pipeline_stage": "Feature engineering", "primary_use": "Signal research" },
          "alpha_hypotheses": { "hypothesis": "X", "features": "a, b", "target": "y", "horizon": "DAILY" },
          "scores": { "novelty": "7", "usability": "8/10", "strategy_impact": 9, "confidence": "0.5" }
        }
        """

        let lens = try ModelJSON.decodeFirstJSON(PaperTradingLens.self, from: text)

        XCTAssertEqual(lens.title, "Foo")
        XCTAssertEqual(lens.tradingTags ?? [], ["PURE_ALPHA", "MODELING"])
        XCTAssertEqual(lens.whereItFits?.pipelineStage ?? [], ["Feature engineering"])
        XCTAssertEqual(lens.whereItFits?.primaryUse, "Signal research")
        XCTAssertEqual(lens.alphaHypotheses?.count, 1)
        XCTAssertEqual(lens.alphaHypotheses?.first?.features ?? [], ["a", "b"])
        XCTAssertEqual(lens.scores?.novelty, 7)
        XCTAssertEqual(lens.scores?.usability, 8)
        XCTAssertEqual(lens.scores?.strategyImpact, 9)
        XCTAssertEqual(lens.scores?.confidence, 0.5)
    }
}
