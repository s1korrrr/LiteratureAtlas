import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class AppModelTests: XCTestCase {

    func testUpsertReplacesByFilePath() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }

        await MainActor.run {
            model.papers = []
        }

        let p1 = Paper(
            version: 1,
            filePath: "/tmp/sample.pdf",
            id: UUID(),
            originalFilename: "sample.pdf",
            title: "Title 1",
            introSummary: nil,
            summary: "Summary one",
            methodSummary: nil,
            resultsSummary: nil,
            takeaways: nil,
            keywords: ["a"],
            userNotes: nil,
            userTags: nil,
            readingStatus: nil,
            noteEmbedding: nil,
            userQuestions: nil,
            flashcards: nil,
            year: nil,
            embedding: [1, 0],
            clusterIndex: nil
        )

        let p2 = Paper(
            version: 1,
            filePath: "/tmp/sample.pdf",
            id: UUID(),
            originalFilename: "sample.pdf",
            title: "Title 2",
            introSummary: nil,
            summary: "Summary two",
            methodSummary: nil,
            resultsSummary: nil,
            takeaways: nil,
            keywords: ["b"],
            userNotes: nil,
            userTags: nil,
            readingStatus: nil,
            noteEmbedding: nil,
            userQuestions: nil,
            flashcards: nil,
            year: nil,
            embedding: [0, 1],
            clusterIndex: nil
        )

        await MainActor.run {
            model.testUpsert(p1)
            model.testUpsert(p2)
        }

        await MainActor.run {
            XCTAssertEqual(model.papers.count, 1)
            XCTAssertEqual(model.papers.first?.title, "Title 2")
            XCTAssertEqual(model.papers.first?.summary, "Summary two")
        }
    }

    func testClusteringAssignsClusters() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = await MainActor.run { AppModel(skipInitialLoad: true, customOutputRoot: tmp) }

        let papers: [Paper] = [
            Paper(version: 1, filePath: "a", id: UUID(), originalFilename: "a.pdf", title: "A", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: nil, embedding: [1, 0], clusterIndex: nil),
            Paper(version: 1, filePath: "b", id: UUID(), originalFilename: "b.pdf", title: "B", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: nil, embedding: [0, 1], clusterIndex: nil),
            Paper(version: 1, filePath: "c", id: UUID(), originalFilename: "c.pdf", title: "C", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, readingStatus: nil, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: nil, embedding: [0.9, 0.1], clusterIndex: nil)
        ]

        await MainActor.run {
            model.papers = papers
        }

        await model.testRunClustering(k: 2)

        await MainActor.run {
            XCTAssertEqual(model.clusters.count, 2)
            XCTAssertTrue(model.papers.allSatisfy { $0.clusterIndex != nil })
            XCTAssertTrue(model.explorationPapers.allSatisfy { $0.clusterIndex != nil })
        }
    }

    func testExplorationPapersInClusterRespectsMembershipAndYearFilter() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }

        let p1 = Paper(version: 1, filePath: "a", id: UUID(), originalFilename: "a.pdf", title: "Alpha", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: ["x"], userNotes: nil, userTags: nil, readingStatus: .unread, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2010, embedding: [1, 0], clusterIndex: 1)
        let p2 = Paper(version: 1, filePath: "b", id: UUID(), originalFilename: "b.pdf", title: "Beta", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: ["y"], userNotes: nil, userTags: nil, readingStatus: .done, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2020, embedding: [0, 1], clusterIndex: 999)
        let p3 = Paper(version: 1, filePath: "c", id: UUID(), originalFilename: "c.pdf", title: "Gamma", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: ["z"], userNotes: nil, userTags: nil, readingStatus: .unread, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2015, embedding: [0.9, 0.1], clusterIndex: nil)

        await MainActor.run {
            model.papers = [p1, p2, p3]
        }

        let cluster = Cluster(id: 42, name: "C", metaSummary: "", centroid: [0, 0], memberPaperIDs: [p2.id, p3.id], layoutPosition: nil, resolutionK: 1, corpusVersion: "v", subclusters: nil)

        let all = await MainActor.run { model.explorationPapers(in: cluster).map(\.id) }
        XCTAssertEqual(Set(all), Set([p2.id, p3.id]))

        await MainActor.run {
            model.yearFilterEnabled = true
            model.yearFilterStart = 2020
            model.yearFilterEnd = 2020
        }

        let filtered = await MainActor.run { model.explorationPapers(in: cluster).map(\.id) }
        XCTAssertEqual(filtered, [p2.id])
    }

    func testUserEventsPersistAsJSONLines() async throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = await MainActor.run { AppModel(skipInitialLoad: true, customOutputRoot: tmp) }

        await MainActor.run {
            model.recordQuestionAsked("Test question?")
        }

        let logURL = tmp.appendingPathComponent("analytics/user_events.jsonl")
        let contents = try String(contentsOf: logURL, encoding: .utf8)
        XCTAssertTrue(contents.contains("\"event_type\":\"qa_question\""))
        XCTAssertTrue(contents.contains("Test question?"))
    }
}
