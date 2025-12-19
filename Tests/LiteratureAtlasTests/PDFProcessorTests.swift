import XCTest
@testable import LiteratureAtlas

final class PDFProcessorTests: XCTestCase {

    func testInferYearFromTextFindsFourDigitYear() {
        let text = "This study (2017) investigates..."
        let year = PDFProcessor().inferYear(fromText: text)
        XCTAssertEqual(year, 2017)
    }

    func testInferYearFromURLParsesArxivIDsAndAvoidsEmbeddedDigits() {
        // "2512.12039" is an arXiv YYMM.NNNNN identifier (Dec 2025).
        // Older logic incorrectly matched "2039" inside "12039".
        let url = URL(fileURLWithPath: "/tmp/2512.12039v1_Some_Paper_Title.pdf")
        let year = PDFProcessor().inferYear(from: url)
        XCTAssertEqual(year, 2025)
    }
}
