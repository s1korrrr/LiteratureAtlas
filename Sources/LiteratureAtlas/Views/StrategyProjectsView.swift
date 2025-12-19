import SwiftUI

@available(macOS 26, iOS 26, *)
struct StrategyProjectsView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var nav: AppNavigation

    @State private var searchQuery: String = ""
    @State private var showArchived: Bool = false
    @State private var selectedProject: StrategyProject?
    @State private var stageFilter: StageFilter = .all
    @State private var sort: ProjectSort = .recent

    private enum StageFilter: String, CaseIterable, Identifiable {
        case all
        case needsIdea
        case needsFeatures
        case needsModel
        case needsTradePlan
        case needsOutcome
        case hasOutcome

        var id: String { rawValue }

        var label: String {
            switch self {
            case .all: return "All"
            case .needsIdea: return "Needs idea"
            case .needsFeatures: return "Needs features"
            case .needsModel: return "Needs model"
            case .needsTradePlan: return "Needs trade plan"
            case .needsOutcome: return "Needs outcome"
            case .hasOutcome: return "Has outcome"
            }
        }
    }

    private enum ProjectSort: String, CaseIterable, Identifiable {
        case recent
        case title
        case sharpe
        case outcomes

        var id: String { rawValue }

        var label: String {
            switch self {
            case .recent: return "Recent"
            case .title: return "Title"
            case .sharpe: return "Sharpe"
            case .outcomes: return "Outcomes"
            }
        }
    }

    private var projects: [StrategyProject] {
        model.strategyProjects
    }

    private var filtered: [StrategyProject] {
        let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var base = projects
        if !showArchived {
            base = base.filter { !($0.archived ?? false) }
        }
        if !query.isEmpty {
            base = base.filter { project in
                if project.title.lowercased().contains(query) { return true }
                if project.idea?.text.lowercased().contains(query) == true { return true }
                if project.tags?.joined(separator: " ").lowercased().contains(query) == true { return true }
                return false
            }
        }

        base = base.filter(matchesStageFilter(_:))
        return base.sorted(by: comparator)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            if filtered.isEmpty {
                GlassCard {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("No strategy projects yet").font(.headline)
                        Text("Create one from a paper (Paper → Create project) or start an empty project here.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(filtered) { project in
                            Button {
                                selectedProject = project
                            } label: {
                                StrategyProjectRow(project: project, onOpen: { selectedProject = project })
                                    .environmentObject(model)
                            }
                            .buttonStyle(.plain)
                            .contextMenu {
                                if let noteURL = model.obsidianStrategyNoteURL(for: project.id) {
                                    Button("Open Obsidian note") { PlatformOpen.open(url: noteURL) }
                                }
#if os(macOS)
                                if let noteURL = model.obsidianStrategyNoteURL(for: project.id) {
                                    Button("Reveal note in Finder") { PlatformOpen.revealInFinder(url: noteURL) }
                                }
#endif
                                Divider()
                                Button(project.archived ?? false ? "Unarchive" : "Archive") {
                                    var updated = project
                                    updated.archived = !(project.archived ?? false)
                                    model.updateStrategyProject(updated)
                                }
                                Divider()
                                Button("Export KG snapshot") {
                                    model.exportQuantKnowledgeGraphSnapshot()
                                }
                                Divider()
                                Button("Delete", role: .destructive) {
                                    model.deleteStrategyProject(project.id)
                                }
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .padding(16)
        .sheet(item: $selectedProject) { project in
            StrategyProjectDetailView(strategyID: project.id)
                .environmentObject(model)
        }
        .onChange(of: nav.requestedStrategyProjectID) { requestedID in
            guard let requestedID else { return }

            if let project = model.strategyProjects.first(where: { $0.id == requestedID }) {
                selectedProject = project
                nav.requestedStrategyProjectID = nil
                return
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                if let project = model.strategyProjects.first(where: { $0.id == requestedID }) {
                    selectedProject = project
                }
                nav.requestedStrategyProjectID = nil
            }
        }
    }

    private var header: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Projects").font(.title2.bold())
                    Spacer()
                    Button {
                        let project = model.createEmptyStrategyProject()
                        selectedProject = project
                    } label: {
                        Label("New", systemImage: "plus")
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        model.exportQuantKnowledgeGraphSnapshot()
                    } label: {
                        Label("Export KG", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.bordered)
                }

                HStack(spacing: 10) {
                    TextField("Search title / idea / tags", text: $searchQuery)
                        .textFieldStyle(.roundedBorder)
                    Toggle("Show archived", isOn: $showArchived)
                        .toggleStyle(.switch)
                }

                HStack(spacing: 10) {
                    Picker("Stage", selection: $stageFilter) {
                        ForEach(StageFilter.allCases) { option in
                            Text(option.label).tag(option)
                        }
                    }
                    .pickerStyle(.menu)

                    Picker("Sort", selection: $sort) {
                        ForEach(ProjectSort.allCases) { option in
                            Text(option.label).tag(option)
                        }
                    }
                    .pickerStyle(.menu)

                    Spacer()

                    Text(summaryLine)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var summaryLine: String {
        let active = projects.filter { !($0.archived ?? false) }.count
        let archived = projects.count - active
        let withOutcome = projects.filter { !$0.outcomes.isEmpty }.count
        let sharpeAvg = averageLatestSharpe(in: projects.filter { !($0.archived ?? false) })
        let sharpeText = sharpeAvg == nil ? "Sharpe n/a" : String(format: "Sharpe %.2f", sharpeAvg ?? 0)
        return "\(filtered.count) shown · \(projects.count) total · \(active) active · \(archived) archived · \(withOutcome) w/ outcomes · \(sharpeText)"
    }

    private func matchesStageFilter(_ project: StrategyProject) -> Bool {
        guard stageFilter != .all else { return true }
        let hasIdea: Bool = {
            let text = project.idea?.text.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let hyps = project.idea?.hypotheses ?? []
            return !text.isEmpty || !hyps.isEmpty
        }()
        let hasFeatures = !project.features.isEmpty
        let hasModel = project.model != nil
        let hasTradePlan = project.tradePlan != nil
        let hasOutcome = !project.outcomes.isEmpty

        switch stageFilter {
        case .all:
            return true
        case .needsIdea:
            return !hasIdea
        case .needsFeatures:
            return hasIdea && !hasFeatures
        case .needsModel:
            return hasIdea && hasFeatures && !hasModel
        case .needsTradePlan:
            return hasIdea && hasFeatures && hasModel && !hasTradePlan
        case .needsOutcome:
            return hasIdea && hasFeatures && hasModel && hasTradePlan && !hasOutcome
        case .hasOutcome:
            return hasOutcome
        }
    }

    private var comparator: (StrategyProject, StrategyProject) -> Bool {
        { lhs, rhs in
            switch sort {
            case .recent:
                return lhs.updatedAt > rhs.updatedAt
            case .title:
                return lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
            case .outcomes:
                if lhs.outcomes.count != rhs.outcomes.count { return lhs.outcomes.count > rhs.outcomes.count }
                return lhs.updatedAt > rhs.updatedAt
            case .sharpe:
                let ls = latestSharpe(lhs) ?? -Double.infinity
                let rs = latestSharpe(rhs) ?? -Double.infinity
                if ls != rs { return ls > rs }
                return lhs.updatedAt > rhs.updatedAt
            }
        }
    }

    private func latestSharpe(_ project: StrategyProject) -> Double? {
        project.outcomes
            .max(by: { $0.measuredAt < $1.measuredAt })?
            .metrics?.sharpe
    }

    private func averageLatestSharpe(in projects: [StrategyProject]) -> Double? {
        let vals = projects.compactMap { latestSharpe($0) }
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / Double(vals.count)
    }
}
