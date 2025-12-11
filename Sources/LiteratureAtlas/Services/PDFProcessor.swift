import Foundation
import PDFKit

struct PDFProcessor {
    func extractFirstPagesText(from url: URL, maxPages: Int) throws -> String {
        guard let doc = PDFDocument(url: url) else {
            throw PDFError.failedToOpen
        }
        let pageCount = doc.pageCount
        let pagesToRead = min(maxPages, pageCount)
        guard pagesToRead > 0 else { throw PDFError.noPages }

        var text = ""
        for index in 0..<pagesToRead {
            if let page = doc.page(at: index), let pageText = page.string {
                text.append(pageText)
                text.append("\n\n")
            }
        }
        return text
    }

    func pageCount(for url: URL) -> Int? {
        guard let doc = PDFDocument(url: url) else { return nil }
        let count = doc.pageCount
        return count > 0 ? count : nil
    }

    func inferTitle(for url: URL, text: String) -> String {
        if let doc = PDFDocument(url: url),
           let attrs = doc.documentAttributes,
           let metaTitle = attrs[PDFDocumentAttribute.titleAttribute] as? String,
           !metaTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return metaTitle
        }
        if let firstLine = text.split(separator: "\n").first {
            let trimmed = firstLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                return String(trimmed.prefix(120))
            }
        }
        return url.deletingPathExtension().lastPathComponent
    }

    func inferYear(from url: URL) -> Int? {
        if let doc = PDFDocument(url: url),
           let attrs = doc.documentAttributes {
            if let created = attrs[PDFDocumentAttribute.creationDateAttribute] as? Date {
                return Calendar.current.dateComponents([.year], from: created).year
            }
            if let modified = attrs[PDFDocumentAttribute.modificationDateAttribute] as? Date {
                return Calendar.current.dateComponents([.year], from: modified).year
            }
        }
        // Fallback: parse a 4-digit year in the filename
        let name = url.deletingPathExtension().lastPathComponent
        let pattern = #"20\d{2}|19\d{2}"#
        if let range = name.range(of: pattern, options: .regularExpression) {
            return Int(name[range])
        }
        return nil
    }

    func inferYear(fromText text: String) -> Int? {
        let pattern = #"20\d{2}|19\d{2}"#
        if let range = text.range(of: pattern, options: .regularExpression) {
            return Int(text[range])
        }
        return nil
    }

    enum PDFError: LocalizedError {
        case failedToOpen
        case noPages

        var errorDescription: String? {
            switch self {
            case .failedToOpen: return "Failed to open PDF."
            case .noPages: return "PDF has no pages."
            }
        }
    }
}
