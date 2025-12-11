import XCTest
@testable import LiteratureAtlas

final class VectorIndexTests: XCTestCase {

    func testQueryReturnsTopResultsByCosine() {
        let vectors: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

        guard let index = VectorIndex(vectors: vectors) else {
            return XCTFail("Expected index to initialize with non-empty vectors")
        }

        let results = index.query([1, 0, 0], k: 2)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results.first?.index, 0)
        if results.count == 2 {
            XCTAssertGreaterThan(results[0].score, results[1].score)
        }
    }

    func testQueryReturnsEmptyOnDimensionMismatch() {
        let vectors: [[Float]] = [
            [1, 0],
            [0, 1]
        ]
        guard let index = VectorIndex(vectors: vectors) else {
            return XCTFail("Expected index to initialize")
        }

        let results = index.query([1, 0, 0], k: 1) // different dimension
        XCTAssertTrue(results.isEmpty)
    }
}
