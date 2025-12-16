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
            takeaways: ["T1", "T2"],
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

        let url = try PaperMarkdownExporter.write(paper: paper, outputRoot: tmpRoot)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        let original = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(original.contains("<!-- atlas:begin -->"))
        XCTAssertTrue(original.contains("<!-- atlas:end -->"))
        XCTAssertTrue(original.contains("## Notes"))
        XCTAssertTrue(original.contains(paperID.uuidString))

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

