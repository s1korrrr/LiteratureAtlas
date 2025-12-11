import XCTest
import Accelerate
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class TemporalAnalyticsTests: XCTestCase {

    private func makePaper(id: UUID = UUID(),
                           year: Int,
                           methodSummary: String = "",
                           summary: String = "",
                           clusterIndex: Int? = nil,
                           readYear: Int? = nil,
                           centroidDim: Int = 2,
                           vector: [Float]? = nil) -> Paper {
        let readDate: Date? = readYear.flatMap { year in
            var comps = DateComponents()
            comps.year = year
            comps.month = 1
            comps.day = 1
            return Calendar.current.date(from: comps)
        }

        let embedding: [Float]
        if let vector { embedding = vector } else {
            // simple axis-aligned embedding by year for tests
            embedding = Array(repeating: 0, count: centroidDim)
                .enumerated()
                .map { idx, _ in idx == 0 ? Float(year) : 0 }
        }

        return Paper(
            version: 1,
            filePath: "/tmp/\(id).pdf",
            id: id,
            originalFilename: "\(id).pdf",
            title: "Paper \(year)",
            introSummary: nil,
            summary: summary.isEmpty ? "Summary \(year)" : summary,
            methodSummary: methodSummary,
            resultsSummary: nil,
            takeaways: nil,
            keywords: nil,
            userNotes: nil,
            userTags: nil,
            isImportant: nil,
            readingStatus: nil,
            noteEmbedding: nil,
            userQuestions: nil,
            flashcards: nil,
            year: year,
            embedding: embedding,
            clusterIndex: clusterIndex,
            claims: nil,
            assumptions: nil,
            evaluationContext: nil,
            methodPipeline: nil,
            firstReadAt: readDate,
            ingestedAt: nil
        )
    }

    func testTopicEvolutionDetectsBurstAndDecay() {
        let c = Cluster(id: 1, name: "RL MM", metaSummary: "", centroid: [1, 0], memberPaperIDs: [], layoutPosition: nil, resolutionK: 3, corpusVersion: nil, subclusters: nil)
        let p2018 = makePaper(year: 2018, clusterIndex: c.id)
        let p2019a = makePaper(year: 2019, clusterIndex: c.id)
        let p2019b = makePaper(year: 2019, clusterIndex: c.id)
        let p2021 = makePaper(year: 2021, clusterIndex: c.id)

        let stream = TemporalAnalytics.topicEvolution(for: c, papers: [p2018, p2019a, p2019b, p2021])

        XCTAssertEqual(stream.countsByYear[2018], 1)
        XCTAssertEqual(stream.countsByYear[2019], 2)
        XCTAssertTrue(stream.burstYears.contains(2019), "2019 should be flagged as a burst year")
        XCTAssertTrue(stream.decayYears.contains(2020), "Drop to zero after 2019 should be marked as decay in 2020")
        XCTAssertTrue(stream.narrative.lowercased().contains("burst"), "Narrative should mention burst")
    }

    func testMethodTakeoverFindsCrossingYear() {
        let p2018 = makePaper(year: 2018, methodSummary: "Q-learning baseline", clusterIndex: 0)
        let p2019 = makePaper(year: 2019, methodSummary: "Uses PPO policy gradient", clusterIndex: 0)
        let p2020 = makePaper(year: 2020, methodSummary: "Improved PPO with entropy bonus", clusterIndex: 0)
        let p2021 = makePaper(year: 2021, methodSummary: "Q-learning variant", clusterIndex: 0)

        let takeovers = TemporalAnalytics.methodTakeovers(
            papers: [p2018, p2019, p2020, p2021],
            methodTags: ["q-learning", "ppo"]
        )

        guard let qp = takeovers.first(where: { $0.a == "q-learning" && $0.b == "ppo" }) else {
            return XCTFail("Expected q-learning vs ppo result")
        }
        XCTAssertEqual(qp.crossingYear, 2020, "PPO should overtake Q-learning by 2020")
        XCTAssertEqual(qp.leadingAtEnd, "ppo")
    }

    func testReadingLagStatsIdentifyRealTimeAndLateClusters() {
        let clusterFast = Cluster(id: 0, name: "Fast", metaSummary: "", centroid: [1, 0], memberPaperIDs: [], layoutPosition: nil, resolutionK: 2, corpusVersion: nil, subclusters: nil)
        let clusterSlow = Cluster(id: 1, name: "Slow", metaSummary: "", centroid: [0, 1], memberPaperIDs: [], layoutPosition: nil, resolutionK: 2, corpusVersion: nil, subclusters: nil)

        let pFast = makePaper(year: 2021, clusterIndex: clusterFast.id, readYear: 2021)
        let pSlow1 = makePaper(year: 2018, clusterIndex: clusterSlow.id, readYear: 2022)
        let pSlow2 = makePaper(year: 2019, clusterIndex: clusterSlow.id, readYear: 2021)

        let stats = TemporalAnalytics.readingLagStats(
            papers: [pFast, pSlow1, pSlow2],
            clusters: [clusterFast, clusterSlow]
        )

        XCTAssertEqual(round(stats.averageLagYears * 10) / 10, 2.0)
        XCTAssertTrue(stats.realTimeClusterIDs.contains(clusterFast.id))
        XCTAssertTrue(stats.lateClusterIDs.contains(clusterSlow.id))
        XCTAssertEqual(stats.overlay.count, 3)
    }

    func testNoveltyAndSaturationScoresHighlightOutliers() {
        let cluster = Cluster(id: 0, name: "Cluster", metaSummary: "", centroid: [1, 0], memberPaperIDs: [], layoutPosition: nil, resolutionK: 2, corpusVersion: nil, subclusters: nil)
        let anchor = makePaper(year: 2020, clusterIndex: cluster.id, vector: [1, 0])
        let outlier = makePaper(year: 2020, clusterIndex: cluster.id, vector: [-1, 0])
        let mid = makePaper(year: 2020, clusterIndex: cluster.id, vector: [0.2, 0.1])

        let scores = TemporalAnalytics.noveltyScores(papers: [anchor, outlier, mid], clusters: [cluster], neighbors: 2)
        guard let anchorScore = scores.first(where: { $0.paperID == anchor.id }),
              let outlierScore = scores.first(where: { $0.paperID == outlier.id }) else {
            return XCTFail("Missing scores")
        }

        XCTAssertGreaterThan(outlierScore.novelty, anchorScore.novelty)
        XCTAssertGreaterThan(anchorScore.saturation, outlierScore.saturation)
    }

    func testHypotheticalPaperGeneratorBlendsClusters() {
        let c1 = Cluster(id: 1, name: "RL market making", metaSummary: "RL approaches", centroid: [1, 0], memberPaperIDs: [], layoutPosition: nil, resolutionK: 2, corpusVersion: nil, subclusters: nil)
        let c2 = Cluster(id: 2, name: "Stochastic control", metaSummary: "Control theory baselines", centroid: [0, 1], memberPaperIDs: [], layoutPosition: nil, resolutionK: 2, corpusVersion: nil, subclusters: nil)

        let hypothetical = TemporalAnalytics.generateHypotheticalPaper(from: [c1, c2], papers: [])

        XCTAssertTrue(hypothetical.title.contains("RL market making"))
        XCTAssertTrue(hypothetical.title.contains("Stochastic control"))
        XCTAssertEqual(hypothetical.embedding.count, 2)
        let norm = sqrt(vDSP.dot(hypothetical.embedding, hypothetical.embedding))
        XCTAssertLessThan(abs(norm - 1.0), 0.0001, "Embedding should be normalized")
    }

    func testAuthorPanelSimulatorMentionsPapers() {
        let cluster = Cluster(id: 7, name: "Panel cluster", metaSummary: "A test cluster", centroid: [1, 0], memberPaperIDs: [], layoutPosition: nil, resolutionK: 2, corpusVersion: nil, subclusters: nil)
        let claim1 = PaperClaim(id: UUID(), paperID: UUID(), statement: "Argues X", assumptions: [], evaluation: nil, year: 2020, strength: 0.5)
        let claim2 = PaperClaim(id: UUID(), paperID: UUID(), statement: "Responds with Y", assumptions: [], evaluation: nil, year: 2021, strength: 0.5)

        let p1 = Paper(version: 1, filePath: "a", id: claim1.paperID, originalFilename: "a.pdf", title: "Alpha", introSummary: nil, summary: "Summary A", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2020, embedding: [1, 0], clusterIndex: cluster.id, claims: [claim1], assumptions: nil, evaluationContext: nil, methodPipeline: nil, firstReadAt: nil, ingestedAt: nil)
        let p2 = Paper(version: 1, filePath: "b", id: claim2.paperID, originalFilename: "b.pdf", title: "Beta", introSummary: nil, summary: "Summary B", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [1, 0], clusterIndex: cluster.id, claims: [claim2], assumptions: nil, evaluationContext: nil, methodPipeline: nil, firstReadAt: nil, ingestedAt: nil)

        let transcript = TemporalAnalytics.simulatePanel(for: cluster, papers: [p1, p2], maxSpeakers: 2)

        XCTAssertTrue(transcript.contains("Alpha"))
        XCTAssertTrue(transcript.contains("Beta"))
        XCTAssertTrue(transcript.contains("Author 1"))
    }

    func testMisconceptionDetectorFindsMissingConcepts() {
        let paper = Paper(version: 1, filePath: "a", id: UUID(), originalFilename: "a.pdf", title: "Gamma", introSummary: nil, summary: "The paper assumes risk-neutral agents and uses order book data.", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: ["risk-neutral", "order book"], userNotes: nil, userTags: nil, isImportant: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2020, embedding: [1, 0], clusterIndex: 1, claims: nil, assumptions: ["risk-neutral agents"], evaluationContext: nil, methodPipeline: nil, firstReadAt: nil, ingestedAt: nil)

        let report = TemporalAnalytics.detectMisconception(answer: "It studies agents", paper: paper)

        XCTAssertTrue(report.missingConcepts.contains(where: { $0.contains("risk") }))
        XCTAssertFalse(report.misalignedAssumptions.isEmpty)
        XCTAssertTrue(report.narrative.lowercased().contains("missing"))
    }
}
