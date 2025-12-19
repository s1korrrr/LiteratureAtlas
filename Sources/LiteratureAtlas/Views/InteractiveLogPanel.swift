import SwiftUI

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

@available(macOS 26, iOS 26, *)
struct InteractiveLogPanel: View {
    let title: String
    @Binding var text: String
    var minHeight: CGFloat = 220

    @State private var query: String = ""
    @State private var showOnlyErrors: Bool = false
    @State private var showOnlyAnalytics: Bool = false
    @State private var followTail: Bool = true
    @State private var showAllLines: Bool = false

    private let maxCollapsedLines = 500

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(filteredLines) { line in
                            LogLineRow(line: line)
                                .id(line.id)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 2)
                }
                .frame(minHeight: minHeight)
                .onChange(of: text) { _, _ in
                    guard followTail else { return }
                    guard let lastID = filteredLines.last?.id else { return }
                    DispatchQueue.main.async {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
                .onChange(of: query) { _, _ in
                    guard followTail else { return }
                    guard let lastID = filteredLines.last?.id else { return }
                    DispatchQueue.main.async {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text(title)
                    .font(.headline)

                Spacer()

                Toggle(isOn: $followTail) {
                    Label("Follow", systemImage: followTail ? "dot.radiowaves.left.and.right" : "pause")
                        .labelStyle(.iconOnly)
                }
                .toggleStyle(.switch)
                .help("Auto-scroll to newest log lines")

                Button {
                    PlatformClipboard.copy(text)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)
                .help("Copy log to clipboard")

                Button(role: .destructive) {
                    text = ""
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .help("Clear log")
            }

            HStack(spacing: 10) {
                TextField("Search log…", text: $query)
                    .textFieldStyle(.roundedBorder)

                Toggle("Errors", isOn: $showOnlyErrors)
                    .toggleStyle(.switch)
                    .onChange(of: showOnlyErrors) { _, newValue in
                        if newValue { showOnlyAnalytics = false }
                    }

                Toggle("Analytics", isOn: $showOnlyAnalytics)
                    .toggleStyle(.switch)
                    .onChange(of: showOnlyAnalytics) { _, newValue in
                        if newValue { showOnlyErrors = false }
                    }

                Toggle("All lines", isOn: $showAllLines)
                    .toggleStyle(.switch)
                    .help(showAllLines ? "Show full log" : "Show only the most recent lines")
            }
            .font(.caption)
        }
    }

    private var filteredLines: [LogLine] {
        let rawLines = text
            .split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
            .map(String.init)

        let limited: ArraySlice<String> = {
            if showAllLines { return rawLines[rawLines.startIndex..<rawLines.endIndex] }
            let n = rawLines.count
            let start = max(0, n - maxCollapsedLines)
            return rawLines[start..<n]
        }()

        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        var out: [LogLine] = []
        out.reserveCapacity(limited.count)
        for (offset, line) in limited.enumerated() {
            let parsed = LogLine(
                id: offset,
                text: line,
                kind: LogLine.Kind.detect(from: line),
                fileURL: LogLine.extractFileURL(from: line)
            )

            if showOnlyErrors, parsed.kind != .error { continue }
            if showOnlyAnalytics, parsed.kind != .analytics { continue }
            if !q.isEmpty, !parsed.text.lowercased().contains(q) { continue }

            out.append(parsed)
        }

        return out
    }
}

@available(macOS 26, iOS 26, *)
private struct LogLine: Identifiable, Hashable {
    enum Kind: Hashable {
        case info
        case analytics
        case warning
        case error

        static func detect(from line: String) -> Kind {
            let lower = line.lowercased()
            if lower.hasPrefix("[analytics]") { return .analytics }
            if lower.contains("warning") { return .warning }
            if lower.contains("failed") || lower.contains("error") { return .error }
            return .info
        }
    }

    let id: Int
    let text: String
    let kind: Kind
    let fileURL: URL?

    static func extractFileURL(from line: String) -> URL? {
        // Common patterns: "/Users/.../Output/..." or "Output/..."
        // Find first absolute path token.
        if let range = line.range(of: #"/[^ \t\n\r\)\]]+"#, options: .regularExpression) {
            let raw = String(line[range])
            let trimmed = raw.trimmingCharacters(in: CharacterSet(charactersIn: ".,;:"))
            let url = URL(fileURLWithPath: trimmed)
            return url
        }

        if let range = line.range(of: #"Output/[^ \t\n\r\)\]]+"#, options: .regularExpression) {
            let raw = String(line[range])
            let trimmed = raw.trimmingCharacters(in: CharacterSet(charactersIn: ".,;:"))
            let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let url = cwd.appendingPathComponent(trimmed)
            return url
        }

        return nil
    }
}

@available(macOS 26, iOS 26, *)
private struct LogLineRow: View {
    let line: LogLine

    @State private var isHovering = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Circle()
                .fill(dotColor)
                .frame(width: 6, height: 6)
                .padding(.top, 4)

            Text(line.text.isEmpty ? " " : line.text)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(textColor)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let url = line.fileURL, isHovering {
                Button {
                    PlatformOpen.open(url: url)
                } label: {
                    Image(systemName: "arrow.up.right.square")
                }
                .buttonStyle(.borderless)
                .help("Open \(url.lastPathComponent)")
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(rowBackground, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
#if os(macOS)
        .onHover { hovering in
            isHovering = hovering
        }
#endif
        .contextMenu {
            Button("Copy line") {
                PlatformClipboard.copy(line.text)
            }
            if let url = line.fileURL {
                Divider()
                Button("Copy path") {
                    PlatformClipboard.copy(url.path)
                }
                Button("Open") {
                    PlatformOpen.open(url: url)
                }
#if os(macOS)
                Button("Reveal in Finder") {
                    PlatformOpen.revealInFinder(url: url)
                }
#endif
            }
        }
    }

    private var rowBackground: Color {
        if isHovering { return Color.white.opacity(0.07) }
        return Color.white.opacity(0.03)
    }

    private var dotColor: Color {
        switch line.kind {
        case .info: return .secondary.opacity(0.7)
        case .analytics: return .mint
        case .warning: return .orange
        case .error: return .red
        }
    }

    private var textColor: Color {
        switch line.kind {
        case .info: return .primary.opacity(0.92)
        case .analytics: return .primary.opacity(0.92)
        case .warning: return .orange.opacity(0.95)
        case .error: return .red.opacity(0.95)
        }
    }
}

enum PlatformClipboard {
    static func copy(_ text: String) {
#if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
#elseif os(iOS)
        UIPasteboard.general.string = text
#else
        _ = text
#endif
    }
}

enum PlatformOpen {
    static func open(url: URL) {
#if os(macOS)
        NSWorkspace.shared.open(url)
#else
        // Best-effort; a Link in the UI can be used on iOS.
        _ = url
#endif
    }

#if os(macOS)
    static func revealInFinder(url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
#endif
}

