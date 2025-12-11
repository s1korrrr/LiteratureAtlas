import XCTest
@testable import LiteratureAtlas

@available(macOS 26, iOS 26, *)
final class GalaxyTests: XCTestCase {

    func testMultiScaleGalaxyAssignsClusterIndices() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let papers: [Paper] = (0..<12).map { idx in
            Paper(
                version: 1,
                filePath: "/tmp/\(idx).pdf",
                id: UUID(),
                originalFilename: "\(idx).pdf",
                title: "Paper \(idx)",
                introSummary: nil,
                summary: "Summary \(idx)",
                methodSummary: nil,
                resultsSummary: nil,
                takeaways: nil,
                keywords: nil,
                userNotes: nil,
                userTags: nil,
                isImportant: nil,
                readingStatus: idx % 3 == 0 ? .done : .unread,
                noteEmbedding: nil,
                userQuestions: nil,
                flashcards: nil,
                year: 2020 + idx % 4,
                embedding: idx % 2 == 0 ? [1, 0] : [0, 1],
                clusterIndex: nil
            )
        }

        await MainActor.run { model.papers = papers }
        await model.buildMultiScaleGalaxy(level0Range: 2...3, level1Range: 2...4)

        await MainActor.run {
            XCTAssertFalse(model.megaClusters.isEmpty, "Expected mega-topics")
            XCTAssertFalse(model.clusters.isEmpty, "Expected subtopics")
            XCTAssertTrue(model.papers.allSatisfy { $0.clusterIndex != nil }, "All papers should be assigned to a subtopic")
        }
    }

    func testBlindSpotsUseReadingProfile() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let p1 = Paper(version: 1, filePath: "a", id: UUID(), originalFilename: "a.pdf", title: "Close to profile", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: .done, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [1, 0], clusterIndex: nil)
        let p2 = Paper(version: 1, filePath: "b", id: UUID(), originalFilename: "b.pdf", title: "Central but far", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: .unread, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [0, 1], clusterIndex: nil)

        await MainActor.run {
            model.papers = [p1, p2]
            model.readingProfileVector = [1, 0] // pretend profile
        }

        let blind = await MainActor.run { model.blindSpots(limit: 1) }
        XCTAssertEqual(blind.first?.title, "Central but far")
    }

    func testKnowledgeSnapshotReturnsSummary() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let readPaper = Paper(
            version: 1,
            filePath: "c",
            id: UUID(),
            originalFilename: "c.pdf",
            title: "Read Paper",
            introSummary: nil,
            summary: "summary",
            methodSummary: nil,
            resultsSummary: nil,
            takeaways: nil,
            keywords: ["topic"],
            userNotes: nil,
            userTags: nil,
            isImportant: true,
            readingStatus: .done,
            noteEmbedding: nil,
            userQuestions: nil,
            flashcards: nil,
            year: 2022,
            embedding: Array(repeating: 0.05, count: 256),
            clusterIndex: nil
        )
        await MainActor.run { model.papers = [readPaper] }

        let snap = await MainActor.run { model.knowledgeSnapshot(for: "topic", maxKnown: 2, maxMissing: 2) }
        XCTAssertNotNil(snap)
        XCTAssertEqual(snap?.known.first?.title, "Read Paper")
    }

    func testAdaptiveCurriculumProducesStages() async {
        let model = await MainActor.run { AppModel(skipInitialLoad: true) }
        let done = Paper(version: 1, filePath: "p0", id: UUID(), originalFilename: "p0.pdf", title: "Done", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: .done, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2020, embedding: [1, 0], clusterIndex: 0)
        let bridge = Paper(version: 1, filePath: "p1", id: UUID(), originalFilename: "p1.pdf", title: "Bridge", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: .unread, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2021, embedding: [0.7, 0.3], clusterIndex: 1)
        let advanced = Paper(version: 1, filePath: "p2", id: UUID(), originalFilename: "p2.pdf", title: "Advanced", introSummary: nil, summary: "s", methodSummary: nil, resultsSummary: nil, takeaways: nil, keywords: nil, userNotes: nil, userTags: nil, isImportant: nil, readingStatus: .unread, noteEmbedding: nil, userQuestions: nil, flashcards: nil, year: 2022, embedding: [0, 1], clusterIndex: 2)

        await MainActor.run {
            model.papers = [done, bridge, advanced]
            model.readingProfileVector = [1, 0] // close to done
            model.clusters = [
                Cluster(id: 0, name: "C0", metaSummary: "", centroid: [1, 0], memberPaperIDs: [done.id], layoutPosition: nil, resolutionK: 1, corpusVersion: "v", subclusters: nil),
                Cluster(id: 1, name: "C1", metaSummary: "", centroid: [0.7, 0.3], memberPaperIDs: [bridge.id], layoutPosition: nil, resolutionK: 1, corpusVersion: "v", subclusters: nil),
                Cluster(id: 2, name: "C2", metaSummary: "", centroid: [0, 1], memberPaperIDs: [advanced.id], layoutPosition: nil, resolutionK: 1, corpusVersion: "v", subclusters: nil)
            ]
        }

        let steps = await MainActor.run { model.adaptiveCurriculum() }
        XCTAssertFalse(steps.isEmpty)
        XCTAssertTrue(steps.contains { $0.stage == .foundation })
        XCTAssertTrue(steps.contains { $0.stage == .bridge })
        XCTAssertTrue(steps.contains { $0.stage == .advanced })
    }
}
