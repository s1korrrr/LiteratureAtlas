import XCTest
@testable import LiteratureAtlas

final class PDFProcessorTests: XCTestCase {

    func testInferYearFromTextFindsFourDigitYear() {
        let text = "This study (2017) investigates..."
        let year = PDFProcessor().inferYear(fromText: text)
        XCTAssertEqual(year, 2017)
    }
}
