import SwiftUI

@available(macOS 26, iOS 26, *)
struct PaperActionRow: View {
    @EnvironmentObject private var model: AppModel

    let paper: Paper
    var title: String? = nil
    var subtitle: String? = nil
    var leadingBadge: String? = nil
    var trailingPill: String? = nil
    var trailingPillTint: Color = .mint
    var onOpen: (() -> Void)? = nil

    @State private var isHovering = false

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            if let leadingBadge {
                Text(leadingBadge)
                    .font(.caption2.bold())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.orange.opacity(0.14), in: Capsule())
                    .foregroundStyle(Color.orange)
                    .lineLimit(1)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(title ?? paper.title)
                    .font(.subheadline.bold())
                    .foregroundStyle(.primary)
                    .lineLimit(2)

                if let subtitle, !subtitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }

            Spacer(minLength: 8)

            if let trailingPill {
                Text(trailingPill)
                    .font(.caption2.bold())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(trailingPillTint.opacity(0.16), in: Capsule())
                    .foregroundStyle(trailingPillTint)
            }

#if os(macOS)
            if isHovering {
                quickActions
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
            }
#endif
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(background, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.white.opacity(isHovering ? 0.22 : 0.12), lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .onTapGesture {
            onOpen?()
        }
#if os(macOS)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.12)) {
                isHovering = hovering
            }
        }
#endif
        .contextMenu {
            Button("Open details") { onOpen?() }
            Divider()
            Button("Open PDF") { PlatformOpen.open(url: paper.fileURL) }
            if let noteURL = model.obsidianNoteURL(for: paper.id) {
                Button("Open Obsidian note") { PlatformOpen.open(url: noteURL) }
            }
            if isMissingTradingLens {
                Button("Generate trading lens") { model.generateTradingLens(for: paper.id) }
            }
            Divider()
            Menu("Mark status") {
                Button("Unread") { setStatus(.unread) }
                Button("In progress") { setStatus(.inProgress) }
                Button("Done") { setStatus(.done) }
                Divider()
                Button("Clear") { setStatus(nil) }
            }
            Button(isStarred ? "Unstar" : "Star") { toggleStar() }
        }
        .accessibilityAddTraits(.isButton)
    }

    private var background: Color {
        if isHovering { return Color.white.opacity(0.08) }
        return Color.white.opacity(0.04)
    }

    private var isStarred: Bool {
        if paper.isImportant == true { return true }
        let tags = paper.userTags ?? []
        return tags.contains(where: { tag in
            let norm = tag.lowercased()
            return norm == "important" || norm == "starred" || norm == "fav" || norm == "favorite"
        })
    }

    private var isMissingTradingLens: Bool {
        if paper.tradingLens == nil { return true }
        return (paper.tradingScores ?? paper.tradingLens?.scores) == nil
    }

#if os(macOS)
    private var quickActions: some View {
        HStack(spacing: 6) {
            if isMissingTradingLens {
                Button {
                    model.generateTradingLens(for: paper.id)
                } label: {
                    Image(systemName: "sparkles")
                }
                .buttonStyle(.borderless)
                .help("Generate trading lens")
            }

            Menu {
                Button("Unread") { setStatus(.unread) }
                Button("In progress") { setStatus(.inProgress) }
                Button("Done") { setStatus(.done) }
                Divider()
                Button("Clear") { setStatus(nil) }
            } label: {
                Image(systemName: "checkmark.circle")
            }
            .menuStyle(.borderlessButton)
            .help("Set reading status")

            Button {
                toggleStar()
            } label: {
                Image(systemName: isStarred ? "star.fill" : "star")
            }
            .buttonStyle(.borderless)
            .help(isStarred ? "Unstar" : "Star")

            Button {
                PlatformOpen.open(url: paper.fileURL)
            } label: {
                Image(systemName: "doc.richtext")
            }
            .buttonStyle(.borderless)
            .help("Open PDF")

            if let noteURL = model.obsidianNoteURL(for: paper.id) {
                Button {
                    PlatformOpen.open(url: noteURL)
                } label: {
                    Image(systemName: "note.text")
                }
                .buttonStyle(.borderless)
                .help("Open Obsidian note")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
#endif

    private func setStatus(_ status: ReadingStatus?) {
        let current = model.papers.first(where: { $0.id == paper.id }) ?? paper
        model.updatePaperUserData(
            id: paper.id,
            notes: current.userNotes ?? "",
            tags: current.userTags ?? [],
            status: status
        )
    }

    private func toggleStar() {
        let current = model.papers.first(where: { $0.id == paper.id }) ?? paper
        let notes = current.userNotes ?? ""
        var tags = current.userTags ?? []
        let importantKeys: Set<String> = ["important", "starred", "fav", "favorite"]

        if isStarred {
            tags.removeAll(where: { importantKeys.contains($0.lowercased()) })
        } else {
            tags.append("important")
        }
        tags = Array(NSOrderedSet(array: tags)).compactMap { $0 as? String }
        model.updatePaperUserData(id: paper.id, notes: notes, tags: tags, status: current.readingStatus)
    }
}
