import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class AnalyticsStoreTests: XCTestCase {

    func testLoadSummaryDecodesISO8601() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let file = tmp.appendingPathComponent("analytics.json")

        let paperID = UUID()
        let payload: [String: Any] = [
            "generated_at": "2025-01-02T03:04:05Z",
            "paper_count": 2,
            "vector_dim": 256,
            "topic_trends": [["cluster_id": 1, "year": 2020, "count": 3]],
            "novelty": [["paper_id": paperID.uuidString, "cluster_id": 1, "novelty": 0.42]],
            "centrality": [[
                "paper_id": paperID.uuidString,
                "weighted_degree": 1.5,
                "average_similarity": 0.5,
                "neighbors": [["paper_id": paperID.uuidString, "score": 1.0]]
            ]],
            "drift": [["cluster_id": 1, "year": 2021, "drift": 0.12]],
            "factors": [[0.1, 0.2], [0.3, 0.4]],
            "factor_loadings": [["paper_id": paperID.uuidString, "scores": [0.5, -0.1]]],
            "factor_exposures": [["year": 2021, "factor": 0, "score": 0.5]],
            "influence": [["paper_id": paperID.uuidString, "influence": 0.9]],
            "idea_flow_edges": [["src": paperID.uuidString, "dst": paperID.uuidString, "weight": 1.0, "from_year": 2020, "to_year": 2021]],
            "recommendations": [paperID.uuidString],
            "answer_confidence": 0.66,
            "counterfactuals": [["name": "year>=2015", "paper_count": 1, "avg_centrality": 0.5]],
            "user_events": [
                "total": 4,
                "by_type": ["opened": 3, "starred": 1],
                "last_seen": "2025-01-02T03:04:05Z"
            ],
            "notes": "example"
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        try data.write(to: file)

        let summary = try AnalyticsStore.loadSummary(from: file)
        XCTAssertNotNil(summary)
        XCTAssertEqual(summary?.paperCount, 2)
        XCTAssertEqual(summary?.vectorDim, 256)
        XCTAssertEqual(summary?.topicTrends.first?.clusterID, 1)
        XCTAssertEqual(summary?.topicTrends.first?.year, 2020)
        XCTAssertEqual(summary?.novelty.first?.paperID, paperID)
        XCTAssertEqual(summary?.centrality.first?.weightedDegree, 1.5)
        XCTAssertEqual(summary?.drift.first?.drift, 0.12)
        XCTAssertEqual(summary?.factorLoadings.first?.scores.first, 0.5)
        XCTAssertEqual(summary?.factorExposures.first?.factor, 0)
        XCTAssertEqual(summary?.influence.first?.influence, 0.9)
        XCTAssertEqual(summary?.ideaFlowEdges.first?.fromYear, 2020)
        XCTAssertEqual(summary?.recommendations.first, paperID)
        XCTAssertEqual(summary?.answerConfidence, 0.66)
        XCTAssertEqual(summary?.counterfactuals.first?.paperCount, 1)
        XCTAssertEqual(summary?.userEvents?.total, 4)
    }

    func testAppModelReloadsAnalyticsFromCustomRoot() async throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp.appendingPathComponent("analytics"), withIntermediateDirectories: true)
        let file = tmp.appendingPathComponent("analytics/analytics.json")

        let paperID = UUID()
        let payload: [String: Any] = [
            "generated_at": "2025-01-02T03:04:05Z",
            "paper_count": 1,
            "vector_dim": 128,
            "topic_trends": [["cluster_id": 0, "year": 2024, "count": 1]],
            "novelty": [["paper_id": paperID.uuidString, "cluster_id": 0, "novelty": 0.9]],
            "notes": "fixture"
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        try data.write(to: file)

        let model = await MainActor.run { AppModel(skipInitialLoad: true, customOutputRoot: tmp) }
        await MainActor.run {
            model.reloadAnalyticsSummary()
            XCTAssertNotNil(model.analyticsSummary)
            XCTAssertEqual(model.analyticsSummary?.paperCount, 1)
            XCTAssertNil(model.analyticsLoadError)
        }
    }

    func testLoadSummaryDecodesExtendedSections() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let file = tmp.appendingPathComponent("analytics.json")

        let paperID = UUID()
        let payload: [String: Any] = [
            "generated_at": "2025-01-02T03:04:05Z",
            "paper_count": 1,
            "vector_dim": 512,
            "paper_metrics": [[
                "paper_id": paperID.uuidString,
                "nov_cluster": 0.1,
                "nov_global": 0.2,
                "nov_directional": 0.0,
                "nov_combinatorial": 0.0,
                "novelty_uncertainty": 0.0,
                "z_novelty": 0.0,
                "consensus_struct": 0.0,
                "consensus_claim": 0.0,
                "consensus_temporal": 0.0,
                "consensus_total": 0.0,
                "z_consensus": 0.0,
                "consensus_uncertainty": 0.0,
                "influence_abs": 0.0,
                "influence_pos": 0.0,
                "influence_neg": 0.0,
                "drift_contrib": 0.0,
                "role_source": 0.0,
                "role_bridge": 0.0,
                "role_sink": 0.0,
                "paper_layout_distortion": 0.42
            ]],
            "quality": [
                "paper_map": [
                    "available": true,
                    "level": "paper",
                    "method": "pca2",
                    "k": 5,
                    "trustworthiness": 0.9,
                    "continuity": 0.85,
                    "avg_neighbor_overlap": 0.7,
                    "distortion_quantiles": ["p50": 0.05, "p90": 0.15, "p95": 0.2, "p99": 0.3],
                    "grid_bins": 10,
                    "grid": [["x_bin": 0, "y_bin": 0, "count": 1, "avg_distortion": 0.1]]
                ]
            ],
            "bridges": [
                "topic_metrics": [[
                    "cluster_id": 7,
                    "betweenness": 0.1,
                    "clustering_coeff": 0.2,
                    "bridging_centrality": 0.3,
                    "degree": 2,
                    "constraint": 0.55,
                    "effective_size": 1.2
                ]]
            ],
            "claims": [
                "available": true,
                "stress_test_gaps": [
                    "available": true,
                    "event_count": 3,
                    "per_cluster": [[
                        "cluster_id": 7,
                        "event_count": 2,
                        "affected_papers": 4,
                        "affected_claims": 5,
                        "top_assumptions": [["assumption": "iid", "count": 2]],
                        "top_suggested_tests": [["text": "Test under distribution shift", "count": 2]]
                    ]]
                ]
            ]
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        try data.write(to: file)

        let summary = try AnalyticsStore.loadSummary(from: file)
        XCTAssertNotNil(summary)
        XCTAssertEqual(summary?.quality?.paperMap?.available, true)
        XCTAssertEqual(summary?.quality?.paperMap?.distortionQuantiles?.p95 ?? -1, 0.2, accuracy: 1e-9)
        XCTAssertEqual(summary?.paperMetrics.first?.paperLayoutDistortion ?? -1, 0.42, accuracy: 1e-9)
        XCTAssertEqual(summary?.bridges?.topicMetrics?.first?.constraint ?? -1, 0.55, accuracy: 1e-9)
        XCTAssertEqual(summary?.bridges?.topicMetrics?.first?.effectiveSize ?? -1, 1.2, accuracy: 1e-9)
        XCTAssertEqual(summary?.claims?.stressTestGaps?.available, true)
        XCTAssertEqual(summary?.claims?.stressTestGaps?.perCluster?.first?.clusterID, 7)
    }

    func testLoadSummaryReturnsNilWhenFileMissing() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let missingFile = tmp.appendingPathComponent("analytics/analytics.json")
        let summary = try AnalyticsStore.loadSummary(from: missingFile)
        XCTAssertNil(summary, "Missing analytics.json should return nil, not throw")
    }
}
