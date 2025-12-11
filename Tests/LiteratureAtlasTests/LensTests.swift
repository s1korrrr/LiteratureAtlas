import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class LensTests: XCTestCase {

    func testInterestLensUsesProfileSimilarity() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }

        let cluster1 = Cluster(
            id: 0,
            name: "Near profile",
            metaSummary: "",
            centroid: [1, 0],
            memberPaperIDs: [],
            layoutPosition: nil,
            resolutionK: nil,
            corpusVersion: nil,
            subclusters: nil
        )
        let cluster2 = Cluster(
            id: 1,
            name: "Far from profile",
            metaSummary: "",
            centroid: [0, 1],
            memberPaperIDs: [],
            layoutPosition: nil,
            resolutionK: nil,
            corpusVersion: nil,
            subclusters: nil
        )

        await MainActor.run {
            model.readingProfileVector = [1, 0]
        }

        let adjusted = await MainActor.run {
            model.lensAdjustedClusters([cluster1, cluster2], lens: .interest)
        }

        guard let p1 = adjusted.first(where: { $0.id == 0 })?.layoutPosition,
              let p2 = adjusted.first(where: { $0.id == 1 })?.layoutPosition else {
            return XCTFail("Expected layout positions for both clusters")
        }

        func radius(_ point: Point2D) -> Double {
            let dx = point.x - 0.5
            let dy = point.y - 0.5
            return (dx * dx + dy * dy).squareRoot()
        }

        XCTAssertLessThan(radius(p1), radius(p2), "Profile-aligned cluster should be placed closer to center")
    }

    func testInterestLensFallsBackOnDimensionMismatch() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }

        let clusterA = Cluster(
            id: 0,
            name: "Dim3-A",
            metaSummary: "",
            centroid: [1, 0, 0],
            memberPaperIDs: [],
            layoutPosition: nil,
            resolutionK: nil,
            corpusVersion: nil,
            subclusters: nil
        )
        let clusterB = Cluster(
            id: 1,
            name: "Dim3-B",
            metaSummary: "",
            centroid: [0, 1, 0],
            memberPaperIDs: [],
            layoutPosition: nil,
            resolutionK: nil,
            corpusVersion: nil,
            subclusters: nil
        )

        await MainActor.run {
            model.readingProfileVector = [1, 0] // Different dimension (2 vs 3)
        }

        let adjusted = await MainActor.run {
            model.lensAdjustedClusters([clusterA, clusterB], lens: .interest)
        }

        for cluster in adjusted {
            guard let pos = cluster.layoutPosition else {
                return XCTFail("Expected layout position for cluster \(cluster.id)")
            }
            XCTAssertFalse(pos.x.isNaN || pos.y.isNaN, "Layout position should be finite")
            XCTAssertGreaterThanOrEqual(pos.x, 0)
            XCTAssertLessThanOrEqual(pos.x, 1)
            XCTAssertGreaterThanOrEqual(pos.y, 0)
            XCTAssertLessThanOrEqual(pos.y, 1)
        }
    }
}
